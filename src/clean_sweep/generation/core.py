from __future__ import annotations

import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from clean_sweep.artifacts import TraceRecord, compact_trace_row
from clean_sweep.config import FullConfig
from clean_sweep.eval import check_trace_correctness
from clean_sweep.generation.methods import AntidistillationLogitsProcessor, ProductOfExpertsLogitsProcessor
from clean_sweep.generation.methods_strategic import (
    StrategicAntidistillationLogitsProcessor,
    StrategicProductOfExpertsLogitsProcessor,
)


ANSWER_FORCE_SUFFIX_DEFAULT = "\n\n**Final Answer**\n\\[\\boxed{"
RESPONSE_STRINGS = {
    "llama": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "qwen": "<|im_start|>assistant\n",
    "r1": "<｜Assistant｜>",
}
LLAMA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|start_header_id|>system<|end_header_id|>\\n\\n' + (message['content']|trim) + '<|eot_id|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + (message['content']|trim) + '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + (message['content']|trim) + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


def get_dtype(name: str) -> torch.dtype:
    return getattr(torch, name, torch.bfloat16)


def ensure_chat_template(tokenizer: Any) -> Any:
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer
    default = getattr(tokenizer, "default_chat_template", None)
    if default:
        tokenizer.chat_template = default
        return tokenizer
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "llama" in name and "instruct" in name:
        tokenizer.chat_template = LLAMA_CHAT_TEMPLATE
    return tokenizer


def align_tokenizer_to_model(tokenizer: Any, model: Any) -> Any:
    model_eos = getattr(model.config, "eos_token_id", None)
    model_pad = getattr(model.config, "pad_token_id", None)
    model_bos = getattr(model.config, "bos_token_id", None)
    if model_eos is not None:
        tokenizer.eos_token_id = model_eos
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(model_eos) or tokenizer.eos_token
    if model_pad is not None:
        tokenizer.pad_token_id = model_pad
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(model_pad) or tokenizer.pad_token
    else:
        tokenizer.pad_token_id = model_eos if model_eos is not None else tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id) or "[PAD]"
    if model_bos is not None:
        tokenizer.bos_token_id = model_bos
        tokenizer.bos_token = tokenizer.convert_ids_to_tokens(model_bos) or tokenizer.bos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model_and_tokenizer(model_name: str, tokenizer_name: str | None, cfg: FullConfig, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True, padding_side="left")
    tokenizer = ensure_chat_template(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=get_dtype(cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    tokenizer = align_tokenizer_to_model(tokenizer, model)
    model = model.to(device)
    return model, tokenizer


class CachedModelWrapper:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.past_key_values = None
        self.last_position = 0

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.past_key_values is None or input_ids.shape[1] <= self.last_position:
            out = self.model(input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            self.past_key_values = out.past_key_values
            self.last_position = input_ids.shape[1]
            return out.logits
        new_token = input_ids[:, -1:]
        out = self.model(new_token, attention_mask=attention_mask, use_cache=True, past_key_values=self.past_key_values, return_dict=True)
        self.past_key_values = out.past_key_values
        self.last_position += 1
        return out.logits

    def reset_cache(self) -> None:
        self.past_key_values = None
        self.last_position = 0


def build_proxy_plus_minus(cfg: FullConfig, tokenizer: Any, device: torch.device, grad_dict: dict[str, torch.Tensor]) -> tuple[CachedModelWrapper, CachedModelWrapper]:
    # Build each perturbed proxy from a fresh `from_pretrained` load so the
    # weights see exactly one `add_(g, alpha=sign*eps)` in the target dtype —
    # bit-identical to the previous base + clone_with_sign pattern (which
    # round-tripped through fp32 CPU, a no-op for finite bf16 values), while
    # peaking at 2 proxy copies on GPU instead of 3.
    import gc

    eps = cfg.generation.eps
    dtype = get_dtype(cfg.model.torch_dtype)

    def _build(sign: float) -> Any:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.proxy_student,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=cfg.model.attn_implementation,
        ).to(device)
        with torch.no_grad():
            for name, param in model.named_parameters():
                g = grad_dict.get(name)
                if g is not None and g.shape == param.shape:
                    param.add_(g.to(device=param.device, dtype=param.dtype), alpha=sign * eps)
        return model

    plus = _build(1.0)
    minus = _build(-1.0)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return CachedModelWrapper(plus), CachedModelWrapper(minus)


def load_proxy_for_poe(cfg: FullConfig, tokenizer: Any, device: torch.device) -> CachedModelWrapper:
    proxy = AutoModelForCausalLM.from_pretrained(
        cfg.model.proxy_student,
        trust_remote_code=True,
        torch_dtype=get_dtype(cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    proxy = proxy.to(device)
    return CachedModelWrapper(proxy)


def _to_ids(raw: Any) -> list[int]:
    if hasattr(raw, "input_ids"):
        return list(raw["input_ids"])
    return list(raw)


def _extract_assistant_reply(full_text: str, prompt_text: str) -> str:
    return full_text[len(prompt_text):].strip() if full_text.startswith(prompt_text) else full_text.strip()


def _af_tail(raw_full_text: str, af_full_text: str) -> str:
    if af_full_text.startswith(raw_full_text):
        return af_full_text[len(raw_full_text):].strip()
    return af_full_text.strip()


def _extract_af_final_answer_only(af_tail_text: str) -> str:
    if not af_tail_text:
        return ""
    marker = "**Final Answer**"
    if marker in af_tail_text:
        return af_tail_text[af_tail_text.index(marker) :].strip()
    return af_tail_text.strip()


def _has_complete_boxed(text: str) -> bool:
    matches = list(re.finditer(r"\\boxed\{", text, re.IGNORECASE))
    if not matches:
        return False
    for m in matches:
        start = m.end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return True
    return False


def _answer_force(
    traces: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    cfg: FullConfig,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    suffix = getattr(cfg.generation, "answer_force_suffix", ANSWER_FORCE_SUFFIX_DEFAULT)
    response_string = None
    if traces:
        for _, marker in RESPONSE_STRINGS.items():
            if marker in traces[0]["full_text"]:
                response_string = marker
                break
    if response_string is None:
        raise ValueError("Response string not found in tokenizer chat template")

    trace_strings = [t["full_text"] for t in traces]
    pad_tok = tokenizer.pad_token or ""
    bos_tok = tokenizer.bos_token or ""
    eos_tok = tokenizer.eos_token or ""
    traces_af = []
    for text in trace_strings:
        after_assistant = text.split(response_string)[-1] if response_string in text else ""
        if "</think>" in after_assistant:
            traces_af.append(text + suffix)
        else:
            traces_af.append(text + "\n</think>" + suffix)
    af_batch_size = max(1, batch_size // 2)
    final_traces: list[str] = []
    for start in range(0, len(traces_af), af_batch_size):
        batch_af = traces_af[start : start + af_batch_size]
        batch_af = [t.replace(bos_tok, "", 1).replace(eos_tok, "") for t in batch_af]
        inputs_af = tokenizer(
            batch_af,
            return_tensors="pt",
            padding=True,
        )
        inputs_af = {k: v.to(device) for k, v in inputs_af.items()}
        with torch.inference_mode():
            out_af = model.generate(
                **inputs_af,
                max_new_tokens=cfg.generation.answer_force_max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                logits_processor=None,
                renormalize_logits=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded_af = tokenizer.batch_decode(out_af, skip_special_tokens=False)
        for text in decoded_af:
            while pad_tok and text.startswith(pad_tok):
                text = text[len(pad_tok) :].lstrip()
            text = text.replace(pad_tok, "")
            if eos_tok and not text.endswith(eos_tok):
                text = text + eos_tok
            final_traces.append(text)
    return final_traces


def generate_teacher_traces(
    *,
    cfg: FullConfig,
    dataset: Any,
    format_prompt: Callable[[str], list[dict[str, str]]],
    method_name: str,
    device: torch.device,
    model: Any,
    tokenizer: Any,
    grad_dict: dict[str, torch.Tensor] | None = None,
    lam: float | None = None,
    penalty_transform: str = "identity",
    beta_teacher: float = 1.0,
    gamma: float | None = None,
    ads_proxy_wrappers: tuple[CachedModelWrapper, CachedModelWrapper] | None = None,
    poe_proxy_wrapper: CachedModelWrapper | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    messages_list = [format_prompt(dataset[i]["problem"]) for i in range(len(dataset))]
    input_ids_list = [
        _to_ids(tokenizer.apply_chat_template(msgs, add_generation_prompt=True, truncation=True, max_length=cfg.generation.max_prompt_tokens))
        for msgs in messages_list
    ]
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    batch = collator([{"input_ids": ids} for ids in input_ids_list])
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    batch_size = cfg.generation.batch_size
    traces: list[dict[str, Any]] = []
    proxy_wrappers: list[CachedModelWrapper] = []
    poe_wrapper: CachedModelWrapper | None = None
    logits_builder = None
    poe_logits = None

    if method_name in ("antidistillation", "strategic_antidistillation"):
        if ads_proxy_wrappers is not None:
            plus, minus = ads_proxy_wrappers
        else:
            if grad_dict is None:
                raise ValueError(f"{method_name} requires proxy gradients")
            plus, minus = build_proxy_plus_minus(cfg, tokenizer, device, grad_dict)
        proxy_wrappers = [plus, minus]

        def get_logits_plus_minus(i: torch.Tensor, a: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            return plus(i, a)[:, -1], minus(i, a)[:, -1]

        logits_builder = get_logits_plus_minus
    elif method_name in ("poe", "strategic_poe"):
        if poe_proxy_wrapper is not None:
            poe_wrapper = poe_proxy_wrapper
        else:
            if cfg.model.proxy_student is None:
                raise ValueError(f"{method_name} requires proxy_student")
            poe_wrapper = load_proxy_for_poe(cfg, tokenizer, device)

        def get_proxy_logits(i: torch.Tensor, a: torch.Tensor | None) -> torch.Tensor:
            assert poe_wrapper is not None
            return poe_wrapper(i, a)[:, -1]

        poe_logits = get_proxy_logits

    n = input_ids.size(0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_input_ids = input_ids[start:end]
        batch_attention_mask = attention_mask[start:end]
        logits_processor = None
        if method_name == "antidistillation":
            for wrapper in proxy_wrappers:
                wrapper.reset_cache()
            tau = cfg.generation.temperature if cfg.generation.temperature > 0 else 1.0
            logits_processor = LogitsProcessorList([
                AntidistillationLogitsProcessor(
                    lam=lam or 0.0,
                    eps=cfg.generation.eps,
                    get_logits_plus_minus=logits_builder,
                    penalty_transform=penalty_transform,
                    beta_teacher=beta_teacher,
                    attention_mask=batch_attention_mask,
                    temperature=tau,
                )
            ])
        elif method_name == "poe" and gamma is not None and gamma != 0.0:
            assert poe_logits is not None
            poe_wrapper.reset_cache()
            logits_processor = LogitsProcessorList([
                ProductOfExpertsLogitsProcessor(gamma=gamma, get_proxy_logits=poe_logits, attention_mask=batch_attention_mask)
            ])
        elif method_name == "strategic_antidistillation":
            for wrapper in proxy_wrappers:
                wrapper.reset_cache()
            tau = cfg.generation.temperature if cfg.generation.temperature > 0 else 1.0
            logits_processor = LogitsProcessorList([
                StrategicAntidistillationLogitsProcessor(
                    lam=lam or 0.0,
                    eta_prefix=cfg.generation.strategic_eta_prefix,
                    lambda_min=(lam or 0.0) * tau,
                    lambda_max=cfg.generation.strategic_lambda_max,
                    eps=cfg.generation.eps,
                    get_logits_plus_minus=logits_builder,
                    penalty_transform=penalty_transform,
                    beta_teacher=beta_teacher,
                    attention_mask=batch_attention_mask,
                    temperature=tau,
                    teacher_sign=cfg.distill.teacher_sign,
                    ignore_token_ids={
                        tok
                        for tok in [tokenizer.pad_token_id, tokenizer.eos_token_id]
                        if tok is not None
                    },
                    debug_every=getattr(cfg.generation, "strategic_debug_every", 0),
                )
            ])
        elif method_name == "strategic_poe" and gamma is not None and gamma != 0.0:
            assert poe_logits is not None
            poe_wrapper.reset_cache()
            logits_processor = LogitsProcessorList([
                StrategicProductOfExpertsLogitsProcessor(
                    gamma=gamma,
                    eta_prefix=cfg.generation.strategic_eta_prefix,
                    gamma_max=cfg.generation.strategic_gamma_max,
                    gamma_min=gamma,
                    get_proxy_logits=poe_logits,
                    attention_mask=batch_attention_mask,
                    ignore_token_ids={
                        tok
                        for tok in [tokenizer.pad_token_id, tokenizer.eos_token_id]
                        if tok is not None
                    },
                    debug_every=getattr(cfg.generation, "strategic_debug_every", 0),
                )
            ])

        with torch.inference_mode():
            out = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=(not cfg.generation.greedy) and cfg.generation.temperature > 0,
                temperature=None if cfg.generation.greedy or cfg.generation.temperature <= 0 else cfg.generation.temperature,
                top_p=None if cfg.generation.greedy or cfg.generation.temperature <= 0 else cfg.generation.top_p,
                logits_processor=logits_processor,
                renormalize_logits=logits_processor is not None,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=False)
        pad_tok = tokenizer.pad_token or ""
        for i, text in enumerate(decoded):
            text = text.replace(pad_tok, "") if pad_tok else text
            idx = start + i
            prompt_text = tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)
            reasoning = _extract_assistant_reply(text, prompt_text)
            traces.append({
                "example_id": dataset[idx]["example_id"],
                "problem": dataset[idx]["problem"],
                "solution": dataset[idx]["solution"],
                "prompt_text": prompt_text,
                "reasoning": reasoning,
                "full_text": text,
            })

    # Free proxy KV caches before answer-forcing. `_answer_force` only uses
    # the teacher, but `plus`/`minus` (ADS) and `poe_wrapper` (PoE) still hold
    # their KV caches from the last main-loop batch (~11 GB combined at
    # batch_size=100). Reclaiming them prevents OOM during the teacher's
    # longer AF prefill (prompt + trace + suffix).
    for wrapper in proxy_wrappers:
        wrapper.reset_cache()
    if poe_wrapper is not None:
        poe_wrapper.reset_cache()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg.generation.answer_force:
        final_texts = _answer_force(traces, model=model, tokenizer=tokenizer, cfg=cfg, batch_size=batch_size, device=device)
        for item, final_text in zip(traces, final_texts):
            item["af_full_text"] = final_text
            item["af_final_answer_only"] = _extract_af_final_answer_only(_af_tail(item["full_text"], final_text))

    compact_rows: list[dict[str, Any]] = []
    inspection_rows: list[dict[str, Any]] = []
    for item in traces:
        raw_correctness = check_trace_correctness(item["full_text"], item["solution"])
        af_correctness = check_trace_correctness(item["af_full_text"], item["solution"]) if item.get("af_full_text") else None
        record = TraceRecord(
            example_id=item["example_id"],
            split=item["example_id"].split("_")[0],
            method=method_name if method_name != "poe" else f"poe_gamma_{gamma}",
            prompt_text=item["problem"],
            raw_trace_text=item["reasoning"],
            af_final_answer_only=item.get("af_final_answer_only"),
            raw_correct=bool(raw_correctness["correct"]),
            af_correct=(bool(af_correctness["correct"]) if af_correctness is not None else None),
            raw_extracted_answer=raw_correctness["extracted_answer"],
            af_extracted_answer=(af_correctness["extracted_answer"] if af_correctness is not None else None),
        )
        compact_rows.append(compact_trace_row(record))
        inspection_rows.append({
            "example_id": item["example_id"],
            "method": record.method,
            "prompt": item["problem"],
            "solution": item["solution"],
            "trace": item["reasoning"],
            "af_final_answer_only": item.get("af_final_answer_only"),
            "raw_extracted_answer": raw_correctness["extracted_answer"],
            "af_extracted_answer": (af_correctness["extracted_answer"] if af_correctness is not None else None),
            "raw_correct": bool(raw_correctness["correct"]),
            "af_correct": (bool(af_correctness["correct"]) if af_correctness is not None else None),
            "extracted_answer": (af_correctness["extracted_answer"] if af_correctness is not None else raw_correctness["extracted_answer"]),
            "correct": (bool(af_correctness["correct"]) if af_correctness is not None else bool(raw_correctness["correct"])),
        })
    return compact_rows, inspection_rows
