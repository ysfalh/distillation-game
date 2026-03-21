from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from clean_sweep.config import FullConfig
from clean_sweep.data import format_prompt_gsm8k
from clean_sweep.generation.core import align_tokenizer_to_model, ensure_chat_template, get_dtype
from clean_sweep.generation.methods import apply_weight_transform
from clean_sweep.train.collator import DataCollatorForCompletionOnlyLM
from clean_sweep.utils import set_seed


def _to_input_ids(raw: Any) -> list[int]:
    if hasattr(raw, "input_ids"):
        return list(raw["input_ids"])
    return list(raw)


class WeightedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weight = inputs.pop("weight", None)
        if weight is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        mask = shift_labels != -100
        loss_per_token = F.cross_entropy(shift_logits, shift_labels, reduction="none", ignore_index=-100)
        B = logits.size(0)
        loss_per_token = loss_per_token.view(B, -1)
        mask = mask.view(B, -1)
        per_sample = (loss_per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        weight = weight.to(per_sample.device, dtype=per_sample.dtype)
        loss = (weight * per_sample).sum() / max(B, 1)
        return (loss, outputs) if return_outputs else loss


def _response_template_for_model(model_name: str, tokenizer: Any) -> list[int]:
    response_str = "<|im_start|>assistant\n" if "qwen" in model_name.lower() else "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return tokenizer.encode(response_str, add_special_tokens=False)


def compute_student_holdout_grad(
    model: torch.nn.Module,
    tokenizer: Any,
    holdout_traces: list[dict[str, Any]],
    device: torch.device,
    response_template: list[int],
    max_length: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)
    model.train()
    grads = {name: torch.zeros_like(p.data) for name, p in model.named_parameters() if p.requires_grad}
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n = 0
    for start in range(0, len(holdout_traces), batch_size):
        batch_traces = holdout_traces[start : start + batch_size]
        batch_inputs = []
        for ex in batch_traces:
            prompt = ex.get("problem", ex.get("prompt", ""))
            reasoning = ex.get("af_trace", ex.get("reasoning_text", ex.get("trace", "")))
            messages = format_prompt_gsm8k(prompt) + [{"role": "assistant", "content": reasoning}]
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, truncation=True, max_length=max_length)
            batch_inputs.append({"input_ids": _to_input_ids(tokens)})
        batch = collator(batch_inputs)
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        out = model(**batch)
        (out.loss * len(batch_traces)).backward()
        n += len(batch_traces)
        for name, p in trainable:
            if p.grad is not None:
                grads[name].add_(p.grad)
        model.zero_grad()
    for name in grads:
        grads[name] = grads[name] / max(n, 1)
    return grads


def compute_trace_weights_fd(
    model: torch.nn.Module,
    tokenizer: Any,
    traces: list[dict[str, Any]],
    g_s: dict[str, torch.Tensor],
    device: torch.device,
    response_template: list[int],
    max_length: int,
    batch_size: int,
    eps: float,
) -> list[float]:
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)
    model.eval()
    trainable = [(n, p) for n, p in model.named_parameters() if n in g_s]
    g_s = {n: g.to(device) for n, g in g_s.items() if n in dict(trainable)}

    all_inputs = []
    for ex in traces:
        prompt = ex.get("problem", ex.get("prompt", ""))
        reasoning = ex.get("af_trace", ex.get("reasoning_text", ex.get("trace", "")))
        messages = format_prompt_gsm8k(prompt) + [{"role": "assistant", "content": reasoning}]
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, truncation=True, max_length=max_length)
        all_inputs.append({"input_ids": _to_input_ids(tokens)})

    def per_sample_nll(batched: dict[str, torch.Tensor]) -> list[float]:
        with torch.no_grad():
            out = model(**batched)
        logits = out.logits
        labels = batched["labels"]
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        mask = shift_labels != -100
        loss_per_token = F.cross_entropy(shift_logits, shift_labels, reduction="none", ignore_index=-100)
        B = logits.size(0)
        loss_per_token = loss_per_token.view(B, -1)
        mask = mask.view(B, -1)
        per_sample = (loss_per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return per_sample.tolist()

    def forward_all() -> list[float]:
        nlls = []
        for start in range(0, len(all_inputs), batch_size):
            batch = collator(all_inputs[start : start + batch_size])
            batch = {k: v.to(device) for k, v in batch.items()}
            nlls.extend(per_sample_nll(batch))
        return nlls

    with torch.no_grad():
        for name, p in trainable:
            p.data.add_(g_s[name], alpha=eps)
    nll_plus = forward_all()

    with torch.no_grad():
        for name, p in trainable:
            p.data.sub_(g_s[name], alpha=2 * eps)
    nll_minus = forward_all()

    with torch.no_grad():
        for name, p in trainable:
            p.data.add_(g_s[name], alpha=eps)

    return [(nm - np_) / (2 * eps) for nm, np_ in zip(nll_minus, nll_plus)]


def load_student_model(cfg: FullConfig, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        getattr(cfg.model, "student_tokenizer", None) or cfg.model.student,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer = ensure_chat_template(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.student,
        trust_remote_code=True,
        torch_dtype=get_dtype(cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    tokenizer = align_tokenizer_to_model(tokenizer, model)
    model = model.to(device)
    lora_config = LoraConfig(
        r=cfg.distill.lora_r,
        lora_alpha=cfg.distill.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=cfg.distill.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def run_distill(
    *,
    cfg: FullConfig,
    train_traces: list[dict[str, Any]],
    holdout_traces: list[dict[str, Any]],
    output_dir: str | Path,
    device: torch.device,
    mode: str,
    beta_s: float,
) -> tuple[dict[str, float], Any, Any]:
    set_seed(cfg.run.seed)
    model, tokenizer = load_student_model(cfg, device)
    response_template = _response_template_for_model(cfg.model.student, tokenizer)
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False, max_length=cfg.distill.max_length)

    def to_text(ex: dict[str, Any]) -> list[dict[str, str]]:
        prompt = ex.get("problem", ex.get("prompt", ""))
        reasoning = ex.get("af_trace", ex.get("reasoning_text", ex.get("trace", "")))
        return format_prompt_gsm8k(prompt) + [{"role": "assistant", "content": reasoning}]

    train_texts = [tokenizer.decode(tokenizer.apply_chat_template(to_text(ex), add_generation_prompt=False, truncation=True, max_length=cfg.distill.max_length)) for ex in train_traces]
    train_dataset = Dataset.from_dict({"text": train_texts})
    stats: dict[str, float] = {}

    if mode == "strategic_fd":
        g_s = compute_student_holdout_grad(
            model,
            tokenizer,
            holdout_traces,
            device,
            response_template,
            cfg.distill.max_length,
            cfg.distill.holdout_grad_batch_size,
        )
        raw_weights = compute_trace_weights_fd(
            model,
            tokenizer,
            train_traces,
            g_s,
            device,
            response_template,
            cfg.distill.max_length,
            cfg.distill.trace_weights_fd_batch_size,
            cfg.generation.eps,
        )
        weights = apply_weight_transform(raw_weights, beta_s, cfg.distill.penalty_transform)
        total = sum(weights)
        n = len(weights)
        norm = total / max(n, 1) if total > 0 else 1.0
        weights = [w / norm for w in weights]
        mean_a = sum(raw_weights) / max(len(raw_weights), 1)
        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in raw_weights) / max(len(raw_weights), 1))
        k_top = max(1, int(0.2 * len(weights)))
        sorted_by_w = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        frac_mass_top20 = (sum(weights[i] for i in sorted_by_w[:k_top]) / max(sum(weights), 1e-12)) if weights else 0.0
        stats = {"mean_a": mean_a, "std_a": std_a, "frac_mass_top20": frac_mass_top20}
        train_dataset = Dataset.from_dict({"text": train_texts, "weight": weights})

    trainer_cls = WeightedSFTTrainer if mode == "strategic_fd" else SFTTrainer
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.distill.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.distill.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.distill.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=cfg.distill.num_epochs,
        learning_rate=cfg.distill.lr,
        weight_decay=cfg.distill.weight_decay,
        max_grad_norm=cfg.distill.max_grad_norm,
        warmup_ratio=cfg.distill.warmup_ratio,
        lr_scheduler_type=cfg.distill.lr_scheduler_type,
        optim="adamw_torch_fused",
        bf16=(cfg.model.torch_dtype == "bfloat16"),
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        logging_strategy="no",
        disable_tqdm=True,
        report_to=[],
        save_strategy="no",
        do_eval=False,
        seed=cfg.run.seed,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=cfg.distill.max_length,
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    gc.collect()
    torch.cuda.empty_cache()
    return stats, model, tokenizer
