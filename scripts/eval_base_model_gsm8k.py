#!/usr/bin/env python3
"""Test accuracy of an untrained base model on GSM8K.

No teachers, no traces, no distillation: the model is loaded, sampled on the
GSM8K-platinum test split, and graded with the same generation settings and
correctness checker the distilled students are evaluated with, so the number
is directly comparable to the student accuracies.

Generation hyperparameters come from the run config, `configs/gsm8k.yaml` by
default, and `--batch-size` overrides the one setting that depends on the GPU.

Usage:
    python scripts/eval_base_model_gsm8k.py --model Qwen/Qwen2.5-3B \
        --tokenizer Qwen/Qwen2.5-3B-Instruct --seed 456
"""
from __future__ import annotations

import argparse
import json
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


def _accuracy(rows: list[dict[str, Any]], key: str) -> float | None:
    """Fraction correct under one grading key, or None if it was not graded."""
    graded = [row for row in rows if row.get(key) is not None]
    if not graded:
        return None
    return sum(1 for row in graded if row[key]) / len(graded)


def evaluate_base_model(
    *,
    cfg: FullConfig,
    model_name: str,
    tokenizer_name: str | None = None,
    limit: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sample the GSM8K test split with a base model and grade every trace.

    Returns the accuracy summary and the trace rows, which have the same
    schema as the saved teacher traces and the student test outputs. Base
    models rarely close a `\\boxed{}` on their own, so the config's answer
    forcing matters here: `accuracy` is the answer-forced number when forcing
    is on, and both components are reported alongside it.
    """
    import torch
    from clean_sweep.data import format_prompt_gsm8k, load_dataset_splits
    from clean_sweep.generation import generate_teacher_traces, load_model_and_tokenizer
    from clean_sweep.utils import set_seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.run.seed)

    splits = load_dataset_splits(
        cfg.data.dataset_name,
        seed=cfg.run.seed,
        train_size=cfg.data.train_size,
        holdout_size=cfg.data.holdout_size,
        test_size=cfg.data.test_size,
    )
    test_split = splits["test"]
    if limit:
        test_split = test_split.select(range(min(limit, len(test_split))))
    del splits

    console.print(f"  Loading {model_name}")
    t0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_name, cfg, device)
    console.print(f"  Loaded in {time.perf_counter() - t0:.0f}s")

    console.print(f"  Generating {len(test_split)} test traces, batch={cfg.generation.batch_size}")
    t0 = time.perf_counter()
    set_seed(cfg.run.seed)
    traces, _ = generate_teacher_traces(
        cfg=cfg,
        dataset=test_split,
        format_prompt=format_prompt_gsm8k,
        method_name="standard",
        device=device,
        model=model,
        tokenizer=tokenizer,
    )
    elapsed = time.perf_counter() - t0

    summary = {
        "model": model_name,
        "tokenizer": tokenizer_name or model_name,
        "dataset": cfg.data.dataset_name,
        "split": "test",
        "n": len(traces),
        "seed": cfg.run.seed,
        "accuracy": _accuracy(traces, "correct"),
        "accuracy_raw": _accuracy(traces, "raw_correct"),
        "accuracy_answer_forced": _accuracy(traces, "af_correct"),
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "max_new_tokens": cfg.generation.max_new_tokens,
        "answer_force": cfg.generation.answer_force,
        "batch_size": cfg.generation.batch_size,
        "generation_seconds": round(elapsed, 1),
        "created_at": datetime.now().isoformat(),
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model to evaluate, e.g. Qwen/Qwen2.5-3B.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer supplying the chat template. Base checkpoints often ship "
            "without one, so pass the matching Instruct tokenizer."
        ),
    )
    parser.add_argument("--config", default="configs/gsm8k.yaml", help="Run config for generation settings.")
    parser.add_argument("--seed", default=None, type=int, help="Override the config seed.")
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Override generation.batch_size, which is what has to shrink on a smaller GPU.",
    )
    parser.add_argument("--limit", default=0, type=int, help="Evaluate only the first N test problems, 0 for all.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to outputs/base_eval/seed_<seed>/<model with / replaced by _>.",
    )
    args = parser.parse_args()

    cfg = FullConfig.from_yaml(Path(args.config))
    if cfg.data.dataset_name != "gsm8k":
        raise ValueError(f"This evaluator expects a GSM8K config, got {cfg.data.dataset_name!r}")
    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.batch_size is not None:
        cfg.generation.batch_size = args.batch_size

    output_dir = Path(
        args.output_dir
        or f"outputs/base_eval/seed_{cfg.run.seed}/{args.model.replace('/', '_')}"
    )

    console.rule(f"Base model on GSM8K test: {args.model}")
    console.print(f"  Config:  {args.config}")
    console.print(f"  Seed:    {cfg.run.seed}")
    console.print(f"  Output:  {output_dir}")

    t0 = time.perf_counter()
    summary, traces = evaluate_base_model(
        cfg=cfg,
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        limit=args.limit,
    )

    from clean_sweep.utils import ensure_dir, write_json, write_markdown_examples

    ensure_dir(output_dir)
    write_json(summary, output_dir / "results.json")
    write_json(traces, output_dir / "test_traces.json")
    write_markdown_examples(
        traces[: cfg.artifacts.save_inspection_samples],
        output_dir / "samples.md",
        title=f"{args.model} on GSM8K test (seed {cfg.run.seed})",
    )

    console.rule("Result")
    console.print(json.dumps(summary, indent=2))
    console.print(
        f"\nACCURACY {args.model} seed={cfg.run.seed} "
        f"n={summary['n']} acc={summary['accuracy']:.4f} "
        f"(raw={summary['accuracy_raw']:.4f}) "
        f"in {time.perf_counter() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
