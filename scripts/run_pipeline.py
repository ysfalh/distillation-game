#!/usr/bin/env python3
"""
Strategic distillation pipeline.

Orchestrates the full teacher → student distillation workflow:

  STAGE 1 — Standard teacher generation
      Load the teacher LM and generate traces on holdout (always) and train
      (if cfg.teachers.standard).  The holdout traces are needed later for
      proxy-gradient computation.

  STAGE 2 — Proxy-student gradients
      Load the proxy student, compute  ∇_θ L_holdout(proxy), then free the
      proxy to reclaim VRAM.  These gradients drive antidistillation.

  STAGE 3 — Antidistillation (ADS) family teachers
      Build θ± = θ ± ε·∇  once, then sweep over:
        • Regular ADS  — penalty_transform="identity", no beta_teacher
        • Strategic ADS — prefix-aware penalty with beta_teacher and the
                          configured penalty_transform (e.g. "exp")

  STAGE 4 — Product-of-Experts (PoE) family teachers
      Load the proxy once, then sweep regular and strategic PoE gammas.

  STAGE 5 — Student distillation & evaluation
      Free the teacher model (no longer needed).  For each
      (teacher-source × student-mode × beta_s), fine-tune a LoRA student
      and evaluate it on the test split.

  STAGE 6 — Final results
      Aggregate all rows into results.json and RESULTS.md.

GPU memory layout (approximate, bf16):
  Stage 1:  teacher 7B  (~15 GB) + KV cache
  Stage 2:  teacher 7B  + proxy 3B (~6 GB) + gradients
  Stage 3:  teacher 7B  + proxy_plus 3B + proxy_minus 3B  (~27 GB + KV)
  Stage 4:  teacher 7B  + proxy 3B  (~21 GB + KV)
  Stage 5:  student 3B  + LoRA + optimizer  (~8 GB)
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from rich.console import Console
from datasets import disable_progress_bars as _datasets_disable_progress_bars

try:
    from transformers.utils.logging import disable_progress_bar as _hf_disable_progress_bar
except ImportError:  # pragma: no cover
    try:
        from transformers.logging import disable_progress_bar as _hf_disable_progress_bar  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover
        _hf_disable_progress_bar = None  # type: ignore[assignment]

from clean_sweep.config import FullConfig
from clean_sweep.data import format_prompt_gsm8k, format_prompt_math, load_dataset_splits
from clean_sweep.generation import generate_teacher_traces, load_model_and_tokenizer
from clean_sweep.generation.core import build_proxy_plus_minus, load_proxy_for_poe
from clean_sweep.summary import build_results_report
from clean_sweep.train import run_distill
from clean_sweep.train.distill import compute_student_holdout_grad, _response_template_for_model
from clean_sweep.utils import ensure_dir, set_seed, write_json, write_markdown_examples

console = Console()
_datasets_disable_progress_bars()
if _hf_disable_progress_bar is not None:
    _hf_disable_progress_bar()


# ── Logging helpers ───────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    t = torch.cuda.get_device_properties(0).total_mem / 1e9
    return f"GPU mem: {a:.1f}GB alloc / {r:.1f}GB reserved / {t:.1f}GB total"


def _separator(title: str) -> None:
    console.print(f"\n{'=' * 70}")
    console.print(f"  {title}")
    console.print(f"{'=' * 70}")


def _stage_start(label: str) -> float:
    console.print(f"[{_now()}] {label}...")
    return time.perf_counter()


def _stage_end(label: str, t0: float, extra: str | None = None) -> None:
    dur = _fmt_dur(time.perf_counter() - t0)
    tail = f" | {extra}" if extra else ""
    console.print(f"[{_now()}] {label} done in {dur}{tail}")


def _trace_stats(compact: list[dict]) -> str:
    """One-line summary: count, correct, accuracy, avg response length."""
    n = len(compact)
    n_correct = sum(1 for r in compact if r.get("correct", False))
    avg_len = sum(len(r.get("response", "")) for r in compact) / max(n, 1)
    return f"n={n}, correct={n_correct}, acc={n_correct / max(n, 1):.4f}, avg_len={avg_len:.0f}"


def _flush_vram(label: str = "") -> None:
    """Run GC + empty CUDA cache, optionally log a label."""
    gc.collect()
    torch.cuda.empty_cache()
    if label:
        console.print(f"[{_now()}] Freed {label} | {_gpu_mem()}")


# ── Result-recording helpers ──────────────────────────────────────────────

def _save_teacher_result(
    compact: list[dict],
    inspection: list[dict],
    *,
    file_tag: str,
    title: str,
    out_dir: Path,
    n_samples: int,
    teacher_train_sources: dict[str, list[dict]],
    teacher_rows: list[dict],
    stage_label: str,
    t0: float,
) -> None:
    """Persist teacher traces, update accumulators, log summary stats.

    file_tag drives both the output filename (train_{file_tag}.json) and
    the method key in teacher_train_sources (teacher_{file_tag}).
    """
    write_json(compact, out_dir / "teacher" / f"train_{file_tag}.json")
    write_markdown_examples(
        compact[:n_samples],
        out_dir / "teacher" / f"train_{file_tag}.md",
        title=title,
    )
    method_key = f"teacher_{file_tag}"
    teacher_train_sources[method_key] = inspection
    acc = sum(1 for r in compact if r.get("correct", False)) / max(len(compact), 1)
    _stage_end(stage_label, t0, extra=_trace_stats(compact))
    teacher_rows.append({
        "train_source": "-",
        "eval_model": f"{method_key}_train",
        "accuracy": acc,
        "notes": "train",
    })


def _log_teacher_summary(
    teacher_rows: list[dict], teacher_train_sources: dict[str, list[dict]],
) -> None:
    _separator("TEACHER GENERATION SUMMARY")
    console.print(f"  Methods completed: {len(teacher_train_sources)}")
    console.print(f"  Sources for student distillation:")
    for name, traces in teacher_train_sources.items():
        row = next((r for r in teacher_rows if name in r.get("eval_model", "")), None)
        acc_s = f", acc={row['accuracy']:.4f}" if row else ""
        console.print(f"    - {name}: {len(traces)} traces{acc_s}")
    console.print(f"  {_gpu_mem()}")
    console.print()


def _log_student_summary(student_rows: list[dict]) -> None:
    _separator("STUDENT DISTILLATION SUMMARY")
    console.print(f"  Runs completed: {len(student_rows)}")
    console.print(f"  {'Source':<50} {'Mode':<16} {'Test Acc':>10}")
    console.print(f"  {'-' * 50} {'-' * 16} {'-' * 10}")
    for r in student_rows:
        console.print(f"  {r['train_source']:<50} {r['eval_model']:<16} {r['accuracy']:>10.4f}")
    console.print()


def _log_config(cfg: FullConfig, config_path: str, run_id: str, out_dir: Path) -> None:
    _separator("PIPELINE CONFIG")
    console.print(f"  Config:           {config_path}")
    console.print(f"  Run ID:           {run_id}")
    console.print(f"  Output:           {out_dir}")
    console.print(f"  Dataset:          {cfg.data.dataset_name} "
                   f"(train={cfg.data.train_size}, holdout={cfg.data.holdout_size}, "
                   f"test={cfg.data.test_size})")
    console.print(f"  Seed:             {cfg.run.seed}")
    console.print(f"  Teacher model:    {cfg.model.teacher}")
    console.print(f"  Proxy student:    {cfg.model.proxy_student}")
    console.print(f"  Student model:    {cfg.model.student}")
    console.print(f"  Teacher methods:")
    console.print(f"    standard:               {cfg.teachers.standard}")
    console.print(f"    antidistillation_lams:   {cfg.teachers.antidistillation_lams}")
    console.print(f"    strategic_ads_lams:      {cfg.teachers.strategic_antidistillation_lams}")
    console.print(f"    poe_gammas:              {cfg.teachers.poe_gammas}")
    console.print(f"    strategic_poe_gammas:    {cfg.teachers.strategic_poe_gammas}")
    console.print(f"    strategic_beta_teachers: {cfg.teachers.strategic_beta_teachers}")
    console.print(f"  Student modes:    {cfg.distill.student_modes}")
    console.print(f"  beta_s_values:    {cfg.distill.beta_s_values}")
    console.print(f"  penalty_transform: {cfg.distill.penalty_transform}")
    console.print(f"  teacher_sign:     {cfg.distill.teacher_sign}")
    console.print(f"  Generation:       temp={cfg.generation.temperature}, "
                   f"top_p={cfg.generation.top_p}, eps={cfg.generation.eps}, "
                   f"batch={cfg.generation.batch_size}")
    console.print(f"  Strategic gen:    eta_prefix={cfg.generation.strategic_eta_prefix}, "
                   f"lambda_max={cfg.generation.strategic_lambda_max}")
    if torch.cuda.is_available():
        console.print(f"  GPU:              {torch.cuda.get_device_name(0)}")
        console.print(f"  {_gpu_mem()}")
    else:
        console.print(f"  Device:           CPU")
    console.print()


# ── Main pipeline ─────────────────────────────────────────────────────────

def main() -> None:
    pipeline_t0 = time.perf_counter()

    parser = argparse.ArgumentParser(description="Strategic distillation pipeline.")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = FullConfig.from_yaml(args.config)
    set_seed(cfg.run.seed)

    run_id = f"{cfg.run.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(cfg.run.output_dir) / run_id)

    _log_config(cfg, args.config, run_id, out_dir)

    if cfg.run.save_config_snapshot:
        cfg.to_yaml(out_dir / "config_snapshot.yaml")

    # ── Load dataset splits ───────────────────────────────────────────────

    t = _stage_start(f"Loading {cfg.data.dataset_name.upper()} splits")
    splits = load_dataset_splits(
        cfg.data.dataset_name,
        seed=cfg.run.seed,
        train_size=cfg.data.train_size,
        holdout_size=cfg.data.holdout_size,
        test_size=cfg.data.test_size,
    )
    format_prompt = (
        format_prompt_gsm8k
        if cfg.data.dataset_name in {"gsm8k", "gsm_hard", "svamp"}
        else format_prompt_math
    )
    _stage_end(
        f"Loading {cfg.data.dataset_name.upper()} splits", t,
        extra=f"train={len(splits['train'])}, holdout={len(splits['holdout'])}, test={len(splits['test'])}",
    )

    # Save run manifest and optional prompt dictionary
    manifest = {
        "run_id": run_id,
        "dataset": cfg.data.dataset_name,
        "teacher": cfg.model.teacher,
        "proxy_student": cfg.model.proxy_student,
        "student": cfg.model.student,
        "seed": cfg.run.seed,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "planned_methods": {
            "teachers": {
                "standard": cfg.teachers.standard,
                "antidistillation_lams": cfg.teachers.antidistillation_lams,
                "poe_gammas": cfg.teachers.poe_gammas,
                "strategic_beta_teachers": cfg.teachers.strategic_beta_teachers,
                "strategic_antidistillation_lams": cfg.teachers.strategic_antidistillation_lams,
                "strategic_poe_gammas": cfg.teachers.strategic_poe_gammas,
            },
            "students": {
                "modes": cfg.distill.student_modes,
                "beta_s_values": cfg.distill.beta_s_values,
            },
        },
    }
    write_json(manifest, out_dir / "run_manifest.json")

    if cfg.artifacts.save_prompt_dictionary:
        prompt_dict = {
            split: [
                {"example_id": row["example_id"], "problem": row["problem"], "solution": row["solution"]}
                for row in ds
            ]
            for split, ds in splits.items()
        }
        write_json(prompt_dict, out_dir / "prompts.json")
        console.print(f"[{_now()}] Saved prompt dictionary")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_times: dict[str, float] = {}
    teacher_rows: list[dict] = []
    teacher_train_sources: dict[str, list[dict]] = {}
    n_samples = cfg.artifacts.save_inspection_samples

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Standard teacher generation
    #   Always generate on holdout (needed for proxy gradients).
    #   Generate on train only when cfg.teachers.standard is True.
    # ══════════════════════════════════════════════════════════════════════
    _separator("STAGE 1: STANDARD TEACHER GENERATION")
    phase_t0 = time.perf_counter()

    t = _stage_start(f"Loading teacher model on {device}")
    teacher_model, teacher_tok = load_model_and_tokenizer(
        cfg.model.teacher, cfg.model.tokenizer, cfg, device,
    )
    _stage_end("Loading teacher model", t, extra=f"{cfg.model.teacher} | {_gpu_mem()}")

    holdout_source: list[dict] | None = None

    for split_name in ["holdout"] + (["train"] if cfg.teachers.standard else []):
        set_seed(cfg.run.seed)
        label = f"Teacher gen | standard | split={split_name}"
        t = _stage_start(label)
        compact, inspection = generate_teacher_traces(
            cfg=cfg, dataset=splits[split_name], format_prompt=format_prompt,
            method_name="standard", device=device,
            model=teacher_model, tokenizer=teacher_tok,
        )
        # Holdout-only traces are tagged "internal" when standard train is disabled
        stem = f"{split_name}_standard"
        if split_name == "holdout" and not cfg.teachers.standard:
            stem += "_internal"
        write_json(compact, out_dir / "teacher" / f"{stem}.json")
        write_markdown_examples(
            compact[:n_samples],
            out_dir / "teacher" / f"{stem}.md",
            title=f"{split_name} standard",
        )

        if split_name == "holdout":
            holdout_source = inspection
        elif split_name == "train":
            teacher_train_sources["teacher_standard"] = inspection

        acc = sum(1 for r in compact if r["correct"]) / max(len(compact), 1)
        extra = _trace_stats(compact)
        if split_name == "holdout" and not cfg.teachers.standard:
            extra += ", internal_only=true"
        _stage_end(label, t, extra=extra)

        if split_name != "holdout":
            teacher_rows.append({
                "train_source": "-",
                "eval_model": f"teacher_standard_{split_name}",
                "accuracy": acc,
                "notes": split_name,
            })

    if holdout_source is None:
        raise RuntimeError("standard holdout traces were not generated")
    holdout_full = holdout_source

    holdout_acc = sum(1 for r in holdout_full if r.get("correct", False)) / max(len(holdout_full), 1)
    console.print(f"[{_now()}] Teacher holdout accuracy: {holdout_acc:.4f} (n={len(holdout_full)})")
    phase_times["1_teacher_gen"] = time.perf_counter() - phase_t0
    console.print(f"[{_now()}] STAGE 1 completed in {_fmt_dur(phase_times['1_teacher_gen'])}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Proxy-student gradients
    #   Compute  g = ∇_θ L_holdout(proxy)  on the holdout teacher traces.
    #   These gradients are used to build θ± for antidistillation.
    #   The proxy is freed immediately after to reclaim ~6 GB VRAM.
    # ══════════════════════════════════════════════════════════════════════
    _separator("STAGE 2: PROXY GRADIENTS")
    phase_t0 = time.perf_counter()

    t = _stage_start("Loading proxy model")
    proxy_model, proxy_tok = load_model_and_tokenizer(
        cfg.model.proxy_student, cfg.model.tokenizer or cfg.model.proxy_student,
        cfg, device,
    )
    _stage_end("Loading proxy model", t, extra=f"{cfg.model.proxy_student} | {_gpu_mem()}")

    response_tpl = _response_template_for_model(cfg.model.proxy_student or "", proxy_tok)
    proxy_model.train()
    set_seed(cfg.run.seed)

    label = "Proxy gradients | source=standard_holdout"
    t = _stage_start(label)
    proxy_grads = compute_student_holdout_grad(
        proxy_model, proxy_tok, holdout_full, device,
        response_tpl, cfg.distill.max_length, cfg.distill.holdout_grad_batch_size,
    )
    _stage_end(label, t, extra=f"n_holdout={len(holdout_full)}")

    del proxy_model, proxy_tok
    _flush_vram("proxy model")

    phase_times["2_proxy_grads"] = time.perf_counter() - phase_t0
    console.print(f"[{_now()}] STAGE 2 completed in {_fmt_dur(phase_times['2_proxy_grads'])}")

    # Precompute which families are needed so we skip building wrappers when
    # no methods of that family are configured.
    beta_teacher_values = cfg.teachers.strategic_beta_teachers
    need_ads = bool(cfg.teachers.antidistillation_lams or cfg.teachers.strategic_antidistillation_lams)
    need_poe = bool(cfg.teachers.poe_gammas or cfg.teachers.strategic_poe_gammas)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Antidistillation-family teachers
    #   Build  proxy_plus = θ + ε·g  and  proxy_minus = θ − ε·g  once,
    #   then run all ADS variants that share these perturbed proxies.
    #
    #   Regular ADS:
    #     - When beta_teacher_values is empty → penalty_transform="identity"
    #       (raw gradient penalty, no nonlinear transform).
    #     - When non-empty → sweep each beta with the configured transform.
    #
    #   Strategic ADS (prefix-aware):
    #     - Defaults to beta_teacher=1.0 when strategic_beta_teachers is [].
    #     - Always uses cfg.distill.penalty_transform (e.g. "exp").
    # ══════════════════════════════════════════════════════════════════════
    _separator("STAGE 3: ANTIDISTILLATION-FAMILY TEACHERS")
    phase_t0 = time.perf_counter()

    ads_wrappers = None
    if need_ads:
        t = _stage_start("Building ADS proxy +/-")
        ads_wrappers = build_proxy_plus_minus(cfg, teacher_tok, device, proxy_grads)
        _stage_end("Building ADS proxy +/-", t, extra=_gpu_mem())

    # ── Regular ADS ───────────────────────────────────────────────────────
    for lam in cfg.teachers.antidistillation_lams:
        if beta_teacher_values:
            for bt in beta_teacher_values:
                set_seed(cfg.run.seed)
                label = f"Teacher gen | ADS | lam={lam} | beta={bt}"
                t = _stage_start(label)
                compact, inspection = generate_teacher_traces(
                    cfg=cfg, dataset=splits["train"], format_prompt=format_prompt,
                    method_name="antidistillation", device=device,
                    model=teacher_model, tokenizer=teacher_tok,
                    grad_dict=proxy_grads, lam=lam,
                    penalty_transform=cfg.distill.penalty_transform,
                    beta_teacher=bt, ads_proxy_wrappers=ads_wrappers,
                )
                _save_teacher_result(
                    compact, inspection,
                    file_tag=f"antidistillation_lam_{lam}_beta_teacher_{bt}",
                    title=f"train antidistillation lam={lam} beta_teacher={bt}",
                    out_dir=out_dir, n_samples=n_samples,
                    teacher_train_sources=teacher_train_sources,
                    teacher_rows=teacher_rows, stage_label=label, t0=t,
                )
        else:
            # No beta → use identity transform (plain gradient penalty)
            set_seed(cfg.run.seed)
            label = f"Teacher gen | ADS | lam={lam}"
            t = _stage_start(label)
            compact, inspection = generate_teacher_traces(
                cfg=cfg, dataset=splits["train"], format_prompt=format_prompt,
                method_name="antidistillation", device=device,
                model=teacher_model, tokenizer=teacher_tok,
                grad_dict=proxy_grads, lam=lam,
                penalty_transform="identity",
                ads_proxy_wrappers=ads_wrappers,
            )
            _save_teacher_result(
                compact, inspection,
                file_tag=f"antidistillation_lam_{lam}",
                title=f"train antidistillation lam={lam}",
                out_dir=out_dir, n_samples=n_samples,
                teacher_train_sources=teacher_train_sources,
                teacher_rows=teacher_rows, stage_label=label, t0=t,
            )

    # ── Strategic ADS (prefix-aware) ──────────────────────────────────────
    for lam in cfg.teachers.strategic_antidistillation_lams:
        for bt in (beta_teacher_values or [1.0]):
            set_seed(cfg.run.seed)
            label = f"Teacher gen | strategic ADS | lam={lam} | beta={bt}"
            t = _stage_start(label)
            compact, inspection = generate_teacher_traces(
                cfg=cfg, dataset=splits["train"], format_prompt=format_prompt,
                method_name="strategic_antidistillation", device=device,
                model=teacher_model, tokenizer=teacher_tok,
                grad_dict=proxy_grads, lam=lam,
                penalty_transform=cfg.distill.penalty_transform,
                beta_teacher=bt, ads_proxy_wrappers=ads_wrappers,
            )
            _save_teacher_result(
                compact, inspection,
                file_tag=f"strategic_ads_lam_{lam}_beta_{bt}",
                title=f"train strategic ADS lam={lam} beta={bt}",
                out_dir=out_dir, n_samples=n_samples,
                teacher_train_sources=teacher_train_sources,
                teacher_rows=teacher_rows, stage_label=label, t0=t,
            )

    # Free ADS proxy wrappers and grad dict (no longer needed)
    if ads_wrappers is not None:
        del ads_wrappers
    del proxy_grads
    _flush_vram("ADS proxies + grad dict")

    phase_times["3_ads_family"] = time.perf_counter() - phase_t0
    console.print(f"[{_now()}] STAGE 3 completed in {_fmt_dur(phase_times['3_ads_family'])}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4: PoE-family teachers
    #   Load the proxy once, then sweep regular and strategic PoE gammas.
    # ══════════════════════════════════════════════════════════════════════
    _separator("STAGE 4: POE-FAMILY TEACHERS")
    phase_t0 = time.perf_counter()

    poe_wrapper = None
    if need_poe:
        t = _stage_start("Building PoE proxy")
        poe_wrapper = load_proxy_for_poe(cfg, teacher_tok, device)
        _stage_end("Building PoE proxy", t, extra=_gpu_mem())

    # ── Regular PoE ───────────────────────────────────────────────────────
    for gamma in cfg.teachers.poe_gammas:
        set_seed(cfg.run.seed)
        label = f"Teacher gen | PoE | gamma={gamma}"
        t = _stage_start(label)
        compact, inspection = generate_teacher_traces(
            cfg=cfg, dataset=splits["train"], format_prompt=format_prompt,
            method_name="poe", device=device,
            model=teacher_model, tokenizer=teacher_tok,
            gamma=gamma, poe_proxy_wrapper=poe_wrapper,
        )
        _save_teacher_result(
            compact, inspection,
            file_tag=f"poe_gamma_{gamma}",
            title=f"train PoE gamma={gamma}",
            out_dir=out_dir, n_samples=n_samples,
            teacher_train_sources=teacher_train_sources,
            teacher_rows=teacher_rows, stage_label=label, t0=t,
        )

    # ── Strategic PoE (prefix-aware) ──────────────────────────────────────
    for gamma in cfg.teachers.strategic_poe_gammas:
        set_seed(cfg.run.seed)
        label = f"Teacher gen | strategic PoE | gamma={gamma}"
        t = _stage_start(label)
        compact, inspection = generate_teacher_traces(
            cfg=cfg, dataset=splits["train"], format_prompt=format_prompt,
            method_name="strategic_poe", device=device,
            model=teacher_model, tokenizer=teacher_tok,
            gamma=gamma, poe_proxy_wrapper=poe_wrapper,
        )
        _save_teacher_result(
            compact, inspection,
            file_tag=f"strategic_poe_gamma_{gamma}",
            title=f"train strategic PoE gamma={gamma}",
            out_dir=out_dir, n_samples=n_samples,
            teacher_train_sources=teacher_train_sources,
            teacher_rows=teacher_rows, stage_label=label, t0=t,
        )

    if poe_wrapper is not None:
        del poe_wrapper
        _flush_vram("PoE proxy wrapper")

    phase_times["4_poe_family"] = time.perf_counter() - phase_t0
    console.print(f"[{_now()}] STAGE 4 completed in {_fmt_dur(phase_times['4_poe_family'])}")

    _log_teacher_summary(teacher_rows, teacher_train_sources)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: Student distillation & evaluation
    #   Teacher model is no longer needed — free it to make room.
    #   For each (teacher source × student mode × beta_s):
    #     1. Fine-tune a LoRA student on the teacher traces
    #     2. Evaluate the student on the test split
    # ══════════════════════════════════════════════════════════════════════
    _separator("STAGE 5: STUDENT DISTILLATION & EVALUATION")
    phase_t0 = time.perf_counter()

    del teacher_model, teacher_tok
    _flush_vram("teacher model (no longer needed)")

    student_rows: list[dict] = []

    for source_name, train_traces in teacher_train_sources.items():
        for mode in cfg.distill.student_modes:
            # strategic_fd uses configurable beta_s; naive always uses 1.0
            betas = cfg.distill.beta_s_values if mode == "strategic_fd" else [1.0]

            for beta_s in betas:
                set_seed(cfg.run.seed)

                # ── Train ─────────────────────────────────────────────
                label = f"Student train | src={source_name} | {mode} | beta_s={beta_s}"
                t = _stage_start(label)
                model_dir = ensure_dir(out_dir / "student" / source_name / f"{mode}_beta_{beta_s}")

                n_tr = len(train_traces)
                n_ok = sum(1 for x in train_traces if x.get("correct", False))
                console.print(
                    f"[{_now()}]   Train traces: {n_tr} total, {n_ok} correct "
                    f"({n_ok / max(n_tr, 1):.4f})"
                )

                stats, student_model, student_tok = run_distill(
                    cfg=cfg, train_traces=train_traces, holdout_traces=holdout_full,
                    output_dir=model_dir, device=device, mode=mode, beta_s=beta_s,
                )

                parts = []
                if stats:
                    parts.append(f"mean_a={stats.get('mean_a', 0.0):.4f}")
                    parts.append(f"frac_top20={stats.get('frac_mass_top20', 0.0):.4f}")
                    if "train_loss" in stats:
                        parts.append(f"loss={stats['train_loss']:.4f}")
                parts.append(_gpu_mem())
                _stage_end(label, t, extra=", ".join(parts))

                # ── Evaluate ──────────────────────────────────────────
                set_seed(cfg.run.seed)
                label = f"Student eval | src={source_name} | {mode} | beta_s={beta_s} | test"
                t = _stage_start(label)
                compact, inspection = generate_teacher_traces(
                    cfg=cfg, dataset=splits["test"], format_prompt=format_prompt,
                    method_name="standard", device=device,
                    model=student_model, tokenizer=student_tok,
                )
                out_base = out_dir / "student" / source_name
                write_json(compact, out_base / f"test_{mode}_beta_{beta_s}.json")
                write_markdown_examples(
                    compact[:n_samples],
                    out_base / f"test_{mode}_beta_{beta_s}.md",
                    title=f"test {mode} beta_s={beta_s} on {source_name}",
                )
                acc = sum(1 for r in compact if r["correct"]) / max(len(compact), 1)
                _stage_end(label, t, extra=_trace_stats(compact))

                # Record result row
                notes = "" if mode == "naive" else f"beta_s={beta_s}"
                if stats:
                    sn = (f"mean_a={stats.get('mean_a', 0.0):.4f}, "
                          f"frac_top20={stats.get('frac_mass_top20', 0.0):.4f}")
                    notes = f"{notes}, {sn}" if notes else sn
                student_rows.append({
                    "train_source": source_name,
                    "eval_model": f"student_{mode}",
                    "accuracy": acc,
                    "notes": notes,
                })
                write_json(
                    inspection[:n_samples],
                    out_dir / "inspection" / f"{source_name}_{mode}_beta_{beta_s}.json",
                )

                # Free student VRAM before the next iteration
                del student_model, student_tok
                _flush_vram()

    phase_times["5_student"] = time.perf_counter() - phase_t0
    console.print(f"[{_now()}] STAGE 5 completed in {_fmt_dur(phase_times['5_student'])}")
    _log_student_summary(student_rows)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 6: Final results
    # ══════════════════════════════════════════════════════════════════════
    _separator("STAGE 6: FINAL RESULTS")
    results_rows = teacher_rows + student_rows
    t = _stage_start("Writing results")
    write_json(results_rows, out_dir / "results.json")
    report = build_results_report(
        results_rows,
        dataset=cfg.data.dataset_name,
        teacher=cfg.model.teacher,
        student=cfg.model.student,
    )
    (out_dir / "RESULTS.md").write_text(report)
    _stage_end("Writing results", t, extra=f"rows={len(results_rows)}")

    # ── Final summary ─────────────────────────────────────────────────────
    total = time.perf_counter() - pipeline_t0
    _separator("PIPELINE COMPLETE")
    console.print(f"  Output:           {out_dir}")
    console.print(f"  Wall time:        {_fmt_dur(total)}")
    console.print(f"  Teacher methods:  {len(teacher_train_sources)}")
    console.print(f"  Student runs:     {len(student_rows)}")
    if torch.cuda.is_available():
        console.print(f"  Peak GPU mem:     {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    console.print()
    console.print(f"  Time breakdown:")
    for name, dur in phase_times.items():
        pct = dur / total * 100 if total > 0 else 0
        console.print(f"    {name:<25} {_fmt_dur(dur):>10}  ({pct:5.1f}%)")
    console.print()
    console.print(report)


if __name__ == "__main__":
    main()
