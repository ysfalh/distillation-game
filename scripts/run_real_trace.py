#!/usr/bin/env python3
"""SFT on real-world LLM traces — trains Q&A, naive, and strategic_fd models."""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import disable_progress_bars

disable_progress_bars()

from clean_sweep.config import FullConfig
from clean_sweep.data import format_prompt_gsm8k, format_prompt_math, load_dataset_splits
from clean_sweep.generation import generate_teacher_traces, load_model_and_tokenizer
from clean_sweep.train import run_distill
from clean_sweep.utils import ensure_dir, set_seed, write_json

_PROMPT_FN = {"gsm8k": format_prompt_gsm8k, "math": format_prompt_math}


def _split_traces(traces: list[dict], train_size: int, holdout_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    traces = list(traces)
    rng.shuffle(traces)
    n = len(traces)
    if n >= train_size + holdout_size:
        return traces[:train_size], traces[train_size:train_size + holdout_size]
    ratio = train_size / (train_size + holdout_size)
    cut = int(n * ratio)
    return traces[:cut], traces[cut:]


def _extract_answer(solution: str, dataset_name: str) -> str:
    if dataset_name == "gsm8k" and "####" in solution:
        return f"\\boxed{{{solution.split('####')[-1].strip()}}}"
    return solution


def _build_qa_traces(dataset, dataset_name: str) -> list[dict]:
    return [
        {"problem": r["problem"], "trace": _extract_answer(r["solution"], dataset_name),
         "solution": r["solution"], "example_id": r["example_id"]}
        for r in dataset
    ]


def _eval_model(cfg, model, tokenizer, test_data, dataset_name, device):
    set_seed(cfg.run.seed)
    compact, _ = generate_teacher_traces(
        cfg=cfg, dataset=test_data,
        format_prompt=_PROMPT_FN[dataset_name],
        method_name="standard", device=device,
        model=model, tokenizer=tokenizer,
    )
    return sum(r["correct"] for r in compact) / max(len(compact), 1)


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_and_eval(cfg, train_traces, holdout_traces, out_dir, tag, device,
                    mode, beta_s, test_data, dataset_name, source):
    set_seed(cfg.run.seed)
    tag_dir = ensure_dir(out_dir / tag)
    t0 = time.perf_counter()
    stats, model, tokenizer = run_distill(
        cfg=cfg, train_traces=train_traces, holdout_traces=holdout_traces,
        output_dir=tag_dir, device=device,
        mode=mode, beta_s=beta_s,
    )
    train_time = time.perf_counter() - t0
    print(f"  Training: {train_time:.1f}s")

    acc = _eval_model(cfg, model, tokenizer, test_data, dataset_name, device)
    print(f"  Accuracy: {acc:.4f}")

    del model, tokenizer
    _free()

    report_beta = beta_s if mode == "strategic_fd" else None
    train_info = {
        "source": source, "mode": mode, "beta_s": report_beta,
        "n_train": len(train_traces), "n_holdout": len(holdout_traces),
        "train_time_s": round(train_time, 1),
        "accuracy": acc,
        "seed": cfg.run.seed,
        "student": cfg.model.student,
        "lr": cfg.distill.lr, "num_epochs": cfg.distill.num_epochs,
        "per_device_train_batch_size": cfg.distill.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.distill.gradient_accumulation_steps,
        "lora_r": cfg.distill.lora_r,
    }
    if stats:
        train_info["stats"] = stats
    write_json(train_info, tag_dir / "train_info.json")

    entry = {"source": source, "mode": mode, "beta_s": report_beta, "accuracy": acc}
    if stats:
        entry["stats"] = stats
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description="SFT on real-world LLM traces.")
    p.add_argument("--config", required=True)
    p.add_argument("--trace-name", required=True, help="e.g. gsm8k_gpt → llm_trace/gsm8k_gpt.json")
    p.add_argument("--trace-dir", type=str, default="llm_trace", help="Directory containing trace JSON files")
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    cfg = FullConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.run.seed = args.seed
    set_seed(cfg.run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg.data.dataset_name

    out_dir = ensure_dir(
        Path(args.output_dir or cfg.run.output_dir)
        / f"real_{args.trace_name}_seed{cfg.run.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"Output → {out_dir}")

    # ── Load dataset splits (for Q&A training + eval) ─────────────────
    splits = load_dataset_splits(
        dataset_name, seed=cfg.run.seed,
        train_size=cfg.data.train_size,
        holdout_size=cfg.data.holdout_size,
        test_size=cfg.data.test_size,
    )
    test_data = splits["test"]

    # ── Load and split LLM traces ─────────────────────────────────────
    trace_path = Path(args.trace_dir) / f"{args.trace_name}.json"
    with open(trace_path) as f:
        all_traces = json.load(f)
    trace_train, trace_holdout = _split_traces(
        all_traces, cfg.data.train_size, cfg.data.holdout_size, cfg.run.seed,
    )
    print(f"Traces: {len(all_traces)} total → {len(trace_train)} train / {len(trace_holdout)} holdout")

    # ── Build Q&A traces from dataset ─────────────────────────────────
    qa_train = _build_qa_traces(splits["train"], dataset_name)
    qa_holdout = _build_qa_traces(splits["holdout"], dataset_name)
    print(f"Q&A: {len(qa_train)} train / {len(qa_holdout)} holdout")

    # ── Base model eval ───────────────────────────────────────────────
    base_model, base_tokenizer = load_model_and_tokenizer(
        cfg.model.student,
        cfg.model.student_tokenizer or cfg.model.student,
        cfg, device,
    )
    base_acc = _eval_model(cfg, base_model, base_tokenizer, test_data, dataset_name, device)
    print(f"Base model accuracy: {base_acc:.4f}")
    del base_model, base_tokenizer
    _free()

    results = [{"source": "base", "mode": None, "beta_s": None, "accuracy": base_acc}]

    # ── Q&A SFT (naive only) ─────────────────────────────────────────
    print("\n── qa_naive ──")
    results.append(_train_and_eval(
        cfg, qa_train, qa_holdout, out_dir, "qa_naive", device,
        mode="naive", beta_s=1.0,
        test_data=test_data, dataset_name=dataset_name,
        source="qa",
    ))

    # ── Trace SFT (all modes/betas from config) ──────────────────────
    for mode in cfg.distill.student_modes:
        if mode == "strategic_fd":
            for beta_s in cfg.distill.beta_s_values:
                tag = f"traces_strategic_fd_beta_{beta_s}"
                print(f"\n── {tag} ──")
                results.append(_train_and_eval(
                    cfg, trace_train, trace_holdout, out_dir, tag, device,
                    mode=mode, beta_s=beta_s,
                    test_data=test_data, dataset_name=dataset_name,
                    source="traces",
                ))
        else:
            print(f"\n── traces_naive ──")
            results.append(_train_and_eval(
                cfg, trace_train, trace_holdout, out_dir, "traces_naive", device,
                mode="naive", beta_s=1.0,
                test_data=test_data, dataset_name=dataset_name,
                source="traces",
            ))

    # ── Save results ──────────────────────────────────────────────────
    write_json(results, out_dir / "results.json")
    print(f"\nDone → {out_dir}")
    print("Results:")
    for r in results:
        if r["source"] == "base":
            label = "base"
        elif r["beta_s"] is not None:
            label = f"{r['source']}/{r['mode']} (β={r['beta_s']})"
        else:
            label = f"{r['source']}/{r['mode']}"
        print(f"  {label}: {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()