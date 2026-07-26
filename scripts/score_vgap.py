#!/usr/bin/env python3
"""Score saved teacher traces with v_gap = log P_teacher(y|x) - log P_proxy(y|x).

This is the sequence-level version of the quantity the PoE defense accumulates
token by token in `StrategicProductOfExpertsLogitsProcessor`:

    s_t = s_{t-1} + [log p_teacher(y_t|h_t) - log p_proxy(y_t|h_t)]

so summing over a whole response gives log P_teacher - log P_proxy. Both models
score the *same* token ids, produced by the teacher tokenizer, which is exactly
what PoE does at generation time (the proxy is fed the teacher's ids and its
logits are reconciled with `_match_vocab_dim`). Only response tokens are scored;
the prompt is excluded.

The scored text is the trace text the student is actually fine-tuned on, so the
scores line up with what the attacker learns from.

Models come from the run config: `model.teacher` and `model.proxy_student`. The
two are loaded one at a time, so peak memory is one 7B model in bf16.

Usage:
    python scripts/score_vgap.py --input-dir gsm8k_output_small --source standard
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich.console import Console

from clean_sweep.config import FullConfig


console = Console()


def _trace_text(row: dict[str, Any]) -> str:
    """The assistant text, matching what `run_distill` trains the student on."""
    return row.get("af_trace", row.get("reasoning_text", row.get("trace", "")))


def _problem_text(row: dict[str, Any]) -> str:
    return row.get("problem", row.get("prompt", ""))


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_examples(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    """Tokenize each (prompt, trace) pair and mark where the response starts.

    The prompt is rendered with the chat template and the trace is appended as
    raw text rather than as an assistant message, because R1-style templates
    rewrite or drop `<think>` spans and 89% of these traces contain one. String
    concatenation keeps the scored response byte-identical to the stored trace.
    """
    from clean_sweep.data import format_prompt_gsm8k

    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt_text = tokenizer.apply_chat_template(
            format_prompt_gsm8k(_problem_text(row)),
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt_text + _trace_text(row), add_special_tokens=False)["input_ids"]

        # A token can span the boundary, so trust the shared prefix instead of
        # len(prompt_ids), which would leak a prompt token into the response.
        boundary = min(_common_prefix_len(prompt_ids, full_ids), len(prompt_ids))
        truncated = len(full_ids) > max_length
        if truncated:
            full_ids = full_ids[:max_length]
        examples.append({
            "index": index,
            "example_id": row.get("example_id", f"row_{index}"),
            "ids": full_ids,
            "boundary": boundary,
            "n_response_tokens": max(len(full_ids) - boundary, 0),
            "truncated": truncated,
            "correct": bool(row.get("correct", False)),
        })
    return examples


def _sum_logprobs(
    logits: Any,
    input_ids: Any,
    response_mask: Any,
    *,
    row_chunk: int,
) -> Any:
    """Summed log p(token) over the masked positions of each sequence.

    The softmax is taken in fp32 over a bounded number of positions at a time:
    a full (batch, seq, 152k) fp32 tensor would be tens of gigabytes.
    """
    import torch

    flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    flat_targets = input_ids[:, 1:].reshape(-1)
    flat_mask = response_mask[:, 1:].reshape(-1)

    per_token = torch.zeros(flat_targets.shape[0], dtype=torch.float32, device=logits.device)
    selected = flat_mask.nonzero(as_tuple=True)[0]
    for chunk in selected.split(row_chunk):
        chunk_logits = flat_logits[chunk].float()
        chunk_logprobs = chunk_logits.log_softmax(dim=-1)
        per_token[chunk] = chunk_logprobs.gather(1, flat_targets[chunk].unsqueeze(1)).squeeze(1)
    return per_token.view(input_ids.shape[0], -1).sum(dim=1)


def score_with_model(
    model: Any,
    examples: list[dict[str, Any]],
    *,
    pad_id: int,
    batch_size: int,
    row_chunk: int,
    label: str,
) -> tuple[list[float], int]:
    """Teacher-forced summed response log-prob for every example.

    Examples are batched shortest-first to cut padding waste, then the results
    are returned in the caller's order. Ids past the model's vocabulary are
    clamped and counted, which matters only if the two tokenizers disagree on
    more than the padding of the embedding matrix.
    """
    import torch

    device = next(model.parameters()).device
    vocab = model.get_input_embeddings().weight.shape[0]
    model.eval()

    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["ids"]))
    totals = [0.0] * len(examples)
    clamped = 0
    started = time.perf_counter()

    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        batch = [examples[i] for i in batch_idx]
        width = max(len(ex["ids"]) for ex in batch)

        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        response_mask = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, ex in enumerate(batch):
            ids = ex["ids"]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
            response_mask[row, ex["boundary"] : len(ids)] = True

        out_of_vocab = int((input_ids >= vocab).sum())
        if out_of_vocab:
            clamped += out_of_vocab
            input_ids = input_ids.clamp_max(vocab - 1)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        response_mask = response_mask.to(device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            sums = _sum_logprobs(logits, input_ids, response_mask, row_chunk=row_chunk)
        del logits

        for row, i in enumerate(batch_idx):
            totals[i] = float(sums[row])

        done = start + len(batch)
        if start == 0 or done % (batch_size * 20) == 0 or done == len(order):
            rate = done / max(time.perf_counter() - started, 1e-9)
            console.print(f"    {label}: {done}/{len(order)} traces, {rate:.1f}/s")

    return totals, clamped


def _load_model(name: str, cfg: FullConfig, device: Any) -> Any:
    from transformers import AutoModelForCausalLM
    from clean_sweep.generation.core import get_dtype

    model = AutoModelForCausalLM.from_pretrained(
        name,
        trust_remote_code=True,
        torch_dtype=get_dtype(cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    return model.to(device)


def _free(model: Any) -> None:
    import gc
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="gsm8k_output_small", help="Saved-trace directory.")
    parser.add_argument("--source", default="standard", help="Trace stem to score, e.g. standard.")
    parser.add_argument("--config", default="configs/gsm8k.yaml", help="Run config naming teacher and proxy.")
    parser.add_argument("--output-dir", default=None, help="Defaults to vgap/<input dir name>.")
    parser.add_argument("--batch-size", default=8, type=int, help="Sequences per forward pass.")
    parser.add_argument(
        "--row-chunk",
        default=2048,
        type=int,
        help="Token positions softmaxed at once in fp32. Lower this if scoring OOMs.",
    )
    parser.add_argument("--max-length", default=None, type=int, help="Defaults to distill.max_length.")
    parser.add_argument("--limit", default=0, type=int, help="Score only the first N traces, 0 for all.")
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Report the tokenization and the prompt/response split without loading any model.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    trace_path = input_dir / f"train_{args.source}.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"No such trace file: {trace_path}")

    cfg = FullConfig.from_yaml(Path(args.config))
    if cfg.model.proxy_student is None:
        raise ValueError(f"{args.config} has no model.proxy_student, so v_gap is undefined")
    max_length = args.max_length or cfg.distill.max_length
    output_dir = Path(args.output_dir or f"vgap/{input_dir.name}")

    rows = json.loads(trace_path.read_text())
    if args.limit:
        rows = rows[: args.limit]

    console.rule(f"v_gap scoring: {trace_path}")
    console.print(f"  Teacher:    {cfg.model.teacher}")
    console.print(f"  Proxy:      {cfg.model.proxy_student}")
    console.print(f"  Traces:     {len(rows)}")
    console.print(f"  Max length: {max_length}")
    console.print(f"  Output:     {output_dir}")

    from transformers import AutoTokenizer
    from clean_sweep.generation.core import ensure_chat_template

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer or cfg.model.teacher,
        trust_remote_code=True,
    )
    tokenizer = ensure_chat_template(tokenizer)

    t0 = time.perf_counter()
    examples = build_examples(rows, tokenizer, max_length)
    n_truncated = sum(ex["truncated"] for ex in examples)
    response_lengths = [ex["n_response_tokens"] for ex in examples]
    console.print(
        f"  Tokenized in {time.perf_counter() - t0:.0f}s | "
        f"response tokens: mean={sum(response_lengths) / max(len(examples), 1):.0f}, "
        f"max={max(response_lengths, default=0)} | truncated={n_truncated}"
    )
    if n_truncated:
        console.print(
            f"  [yellow]{n_truncated} traces hit --max-length; their v_gap covers the kept prefix only.[/yellow]"
        )
    empty = [ex for ex in examples if ex["n_response_tokens"] == 0]
    if empty:
        console.print(f"  [yellow]{len(empty)} traces have no response tokens and will score 0.[/yellow]")

    if args.tokenize_only:
        console.print("  --tokenize-only: no models loaded, nothing written.")
        return

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    console.print(f"\n  Loading teacher {cfg.model.teacher}")
    teacher = _load_model(cfg.model.teacher, cfg, device)
    logp_teacher, teacher_clamped = score_with_model(
        teacher,
        examples,
        pad_id=pad_id,
        batch_size=args.batch_size,
        row_chunk=args.row_chunk,
        label="teacher",
    )
    _free(teacher)

    console.print(f"\n  Loading proxy {cfg.model.proxy_student}")
    proxy = _load_model(cfg.model.proxy_student, cfg, device)
    logp_proxy, proxy_clamped = score_with_model(
        proxy,
        examples,
        pad_id=pad_id,
        batch_size=args.batch_size,
        row_chunk=args.row_chunk,
        label="proxy",
    )
    _free(proxy)

    if teacher_clamped or proxy_clamped:
        console.print(
            f"  [yellow]Clamped out-of-vocabulary ids: teacher={teacher_clamped}, proxy={proxy_clamped}. "
            "Those tokens' log-probs are meaningless.[/yellow]"
        )

    scores = []
    n_non_finite = 0
    for ex, lt, lp in zip(examples, logp_teacher, logp_proxy):
        n_tokens = ex["n_response_tokens"]
        gap = lt - lp
        finite = math.isfinite(lt) and math.isfinite(lp) and n_tokens > 0
        if not finite:
            n_non_finite += 1
        scores.append({
            "index": ex["index"],
            "example_id": ex["example_id"],
            "n_response_tokens": n_tokens,
            "truncated": ex["truncated"],
            "correct": ex["correct"],
            "finite": finite,
            "logp_teacher": lt,
            "logp_proxy": lp,
            "vgap_sum": gap,
            "vgap_mean": gap / n_tokens if n_tokens else 0.0,
        })
    if n_non_finite:
        console.print(
            f"  [yellow]{n_non_finite} traces scored non-finite or empty. The binning step "
            "drops them rather than letting them sort arbitrarily.[/yellow]"
        )

    from clean_sweep.utils import ensure_dir, write_json

    ensure_dir(output_dir)
    write_json(scores, output_dir / f"scores_{args.source}.json")
    write_json(
        {
            "input_dir": str(input_dir),
            "source": args.source,
            "trace_file": str(trace_path),
            "n_traces": len(scores),
            "teacher": cfg.model.teacher,
            "proxy": cfg.model.proxy_student,
            "tokenizer": cfg.model.tokenizer or cfg.model.teacher,
            "max_length": max_length,
            "n_truncated": n_truncated,
            "n_non_finite": n_non_finite,
            "clamped_ids": {"teacher": teacher_clamped, "proxy": proxy_clamped},
            "definition": "vgap = sum over response tokens of log p_teacher - log p_proxy, both on teacher-tokenized ids",
            "created_at": datetime.now().isoformat(),
        },
        output_dir / f"scores_{args.source}_manifest.json",
    )

    usable = [s for s in scores if s["finite"]]
    gaps = [s["vgap_sum"] for s in usable]
    means = [s["vgap_mean"] for s in usable]
    console.rule("Result")
    console.print(f"  Wrote {output_dir / f'scores_{args.source}.json'}")
    console.print(f"  Usable traces: {len(usable)}/{len(scores)}")
    # Percentiles rather than min/max/mean: v_gap has a long tail, and the tail
    # is exactly what should not be setting anyone's expectations.
    for name, values in (("vgap_sum ", gaps), ("vgap_mean", means)):
        if not values:
            continue
        console.print(
            f"  {name}: p1={_percentile(values, 0.01):.3f}, "
            f"p25={_percentile(values, 0.25):.3f}, "
            f"median={_percentile(values, 0.5):.3f}, "
            f"p75={_percentile(values, 0.75):.3f}, "
            f"p99={_percentile(values, 0.99):.3f} | "
            f"min={min(values):.3f}, max={max(values):.3f}"
        )


if __name__ == "__main__":
    main()
