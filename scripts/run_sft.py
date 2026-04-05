#!/usr/bin/env python3
"""Standalone SFT — train on Q&A pairs or pre-generated traces."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import disable_progress_bars

disable_progress_bars()

from clean_sweep.config import FullConfig
from clean_sweep.data import format_prompt_gsm8k, format_prompt_math, load_dataset_splits
from clean_sweep.generation import generate_teacher_traces
from clean_sweep.train import run_distill
from clean_sweep.utils import ensure_dir, set_seed, write_json

_PROMPT_FN = {"gsm8k": format_prompt_gsm8k, "math": format_prompt_math}


def _extract_answer(row: dict, dataset: str) -> str:
    sol = row["solution"]
    if dataset == "gsm8k" and "####" in sol:
        return f"\\boxed{{{sol.split('####')[-1].strip()}}}"
    return sol


def _build_traces(data, dataset: str) -> list[dict]:
    return [
        {"problem": r["problem"], "trace": _extract_answer(r, dataset),
         "solution": r["solution"], "example_id": r["example_id"]}
        for r in data
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Standalone SFT: train on Q&A pairs or traces.")
    p.add_argument("--config", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["gsm8k", "math"])
    src.add_argument("--traces", type=str)
    p.add_argument("--holdout-traces", type=str, default=None)
    p.add_argument("--mode", choices=["naive", "strategic_fd"], default="naive")
    p.add_argument("--beta-s", type=float, default=1.0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    cfg = FullConfig.from_yaml(args.config)
    set_seed(cfg.run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_tag = args.dataset or Path(args.traces).stem
    out_dir = ensure_dir(
        Path(args.output_dir or cfg.run.output_dir)
        / f"sft_{source_tag}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"Output → {out_dir}")

    # ── Build traces ──────────────────────────────────────────────────
    splits = None
    dataset_name = args.dataset or cfg.data.dataset_name

    if args.dataset:
        splits = load_dataset_splits(
            args.dataset, seed=cfg.run.seed,
            train_size=cfg.data.train_size,
            holdout_size=cfg.data.holdout_size,
            test_size=cfg.data.test_size,
        )
        train_traces = _build_traces(splits["train"], args.dataset)
        holdout_traces = _build_traces(splits["holdout"], args.dataset)
    else:
        with open(args.traces) as f:
            train_traces = json.load(f)
        holdout_traces = []
        if args.holdout_traces:
            with open(args.holdout_traces) as f:
                holdout_traces = json.load(f)

    if args.mode == "strategic_fd" and not holdout_traces:
        raise SystemExit("strategic_fd requires holdout traces")

    # ── Train ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    stats, model, tokenizer = run_distill(
        cfg=cfg, train_traces=train_traces, holdout_traces=holdout_traces,
        output_dir=ensure_dir(out_dir / "model"), device=device,
        mode=args.mode, beta_s=args.beta_s,
    )
    print(f"Training done in {time.perf_counter() - t0:.1f}s")
    if stats:
        write_json(stats, out_dir / "train_stats.json")

    # ── Eval ──────────────────────────────────────────────────────────
    if args.eval:
        if splits is None:
            splits = load_dataset_splits(
                dataset_name, seed=cfg.run.seed,
                train_size=cfg.data.train_size,
                holdout_size=cfg.data.holdout_size,
                test_size=cfg.data.test_size,
            )
        set_seed(cfg.run.seed)
        compact, _ = generate_teacher_traces(
            cfg=cfg, dataset=splits["test"],
            format_prompt=_PROMPT_FN[dataset_name],
            method_name="standard", device=device,
            model=model, tokenizer=tokenizer,
        )
        acc = sum(r["correct"] for r in compact) / max(len(compact), 1)
        print(f"Accuracy: {acc:.4f}")
        write_json(compact, out_dir / "eval_results.json")
        write_json({"accuracy": acc, "n": len(compact)}, out_dir / "eval_summary.json")

    # ── Cleanup ───────────────────────────────────────────────────────
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
