#!/usr/bin/env python3
"""Compute per-trace importance weights from saved teacher traces and dump to JSON.

Reuses the holdout + training trace files already on disk.  Does NOT retrain
the student — only computes g_s and the finite-difference advantage scores,
using the same functions as the main pipeline (run_pipeline.py → distill.py).

Output JSON: one object per training trace with fields
    example_id, prompt, trace, correct, raw_weight (advantage a_i), weight (transformed + normalised).

Example
-------
python scripts/compute_weights.py \
    --config gsm8k_output/gsm8k_ads_456_20260324_203021/config_snapshot.yaml \
    --holdout gsm8k_output/gsm8k_ads_456_20260324_203021/teacher/holdout_standard_internal.json \
    --train gsm8k_output/gsm8k_ads_456_20260324_203021/teacher/train_antidistillation_lam_0.055.json \
    --beta-s 0.5 \
    --out gsm8k_output/analysis/weights_ads_lam0.055_beta0.5.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import torch

from clean_sweep.config import FullConfig
from clean_sweep.generation.methods import apply_weight_transform
from clean_sweep.train.distill import (
    _response_template_for_model,
    compute_student_holdout_grad,
    compute_trace_weights_fd,
    load_student_model,
)
from clean_sweep.utils import set_seed, write_json


def _normalise(weights: list[float]) -> list[float]:
    """Shift-and-scale so the mean weight equals 1 (same logic as run_distill)."""
    n = len(weights)
    total = sum(weights)
    norm = total / max(n, 1) if total > 0 else 1.0
    return [w / norm for w in weights]


def _summary_stats(raw: list[float], weights: list[float]) -> dict[str, float]:
    """Compute the same diagnostics that run_distill logs."""
    n = len(raw)
    mean_a = sum(raw) / max(n, 1)
    std_a = math.sqrt(sum((a - mean_a) ** 2 for a in raw) / max(n, 1))
    k_top = max(1, int(0.2 * n))
    top_idx = sorted(range(n), key=lambda i: weights[i], reverse=True)
    frac_top20 = sum(weights[i] for i in top_idx[:k_top]) / max(sum(weights), 1e-12)
    return {"mean_a": mean_a, "std_a": std_a, "frac_top20": frac_top20}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",  required=True, help="config_snapshot.yaml from the run")
    ap.add_argument("--holdout", required=True, help="Holdout standard traces JSON")
    ap.add_argument("--train",   required=True, help="Training traces JSON to score")
    ap.add_argument("--beta-s",  type=float, default=0.5)
    ap.add_argument("--out",     required=True, help="Output JSON path")
    args = ap.parse_args()

    # ── Config & reproducibility ──────────────────────────────────────
    # Load the exact config used by the original run so every hyper-
    # parameter (LoRA rank, eps, batch sizes, …) matches.
    cfg = FullConfig.from_yaml(args.config)
    set_seed(cfg.run.seed)
    device = torch.device("cuda")
    print(f"seed={cfg.run.seed}  device={device}  transform={cfg.distill.penalty_transform}")

    # ── Student model (fresh init, same as training time) ─────────────
    # load_student_model creates the base model + LoRA adapter at the
    # *initial* (untrained) checkpoint — identical to what run_distill
    # starts from.
    print("Loading student model …")
    model, tokenizer = load_student_model(cfg, device)
    response_template = _response_template_for_model(cfg.model.student, tokenizer)

    # ── Load traces from disk ─────────────────────────────────────────
    with open(args.holdout) as f:
        holdout_traces = json.load(f)
    with open(args.train) as f:
        train_traces = json.load(f)
    print(f"holdout={len(holdout_traces)}  train={len(train_traces)}")

    # ── Step 1: holdout gradient g_s ──────────────────────────────────
    # Average gradient of the student's NLL on the holdout set (standard
    # teacher traces).  This is the "desired learning direction".
    print("Computing holdout gradient g_s …")
    g_s = compute_student_holdout_grad(
        model, tokenizer, holdout_traces, device,
        response_template, cfg.distill.max_length,
        cfg.distill.holdout_grad_batch_size,
    )

    # ── Step 2: per-trace advantage scores via finite differences ─────
    # For each training trace i, approximates  a_i ≈ ⟨∇ℓ_i, g_s⟩  by
    # evaluating ℓ_i at θ ± ε·g_s  (two forward passes, no per-sample
    # backward needed).
    print("Computing per-trace weights (finite differences) …")
    raw_weights = compute_trace_weights_fd(
        model, tokenizer, train_traces, g_s, device,
        response_template, cfg.distill.max_length,
        cfg.distill.trace_weights_fd_batch_size, cfg.generation.eps,
    )

    # ── Step 3: transform + normalise ─────────────────────────────────
    # apply_weight_transform maps raw advantages through exp(β·a) (or
    # identity/softplus depending on config), then we mean-normalise so
    # the average weight is 1.
    weights = _normalise(
        apply_weight_transform(raw_weights, args.beta_s, cfg.distill.penalty_transform)
    )

    stats = _summary_stats(raw_weights, weights)
    print(f"mean_a={stats['mean_a']:.6f}  std_a={stats['std_a']:.6f}  "
          f"frac_top20={stats['frac_top20']:.4f}  "
          f"w_range=[{min(weights):.6f}, {max(weights):.6f}]")

    # ── Write output ──────────────────────────────────────────────────
    rows = [
        {
            "example_id": tr.get("example_id", f"train_{i}"),
            "prompt":     tr.get("prompt"),
            "trace":      tr.get("trace"),
            "correct":    tr.get("correct"),
            "raw_weight": raw_weights[i],
            "weight":     weights[i],
        }
        for i, tr in enumerate(train_traces)
    ]
    write_json(rows, args.out)
    print(f"Wrote {len(rows)} weights → {args.out}")


if __name__ == "__main__":
    main()
