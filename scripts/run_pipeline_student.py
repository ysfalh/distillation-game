#!/usr/bin/env python3
"""
Strategic distillation — student pipeline (Stages 5 & 6).

Consumes the output of run_pipeline_teachers.py:
  <input-dir>/
    config_snapshot.yaml             full config (may be an override snapshot)
    TEACHERS_DONE                    completion marker (non-fatal if missing)
    cache/
      holdout_full.json              standard-holdout inspection rows
                                     (symlinked/copied from the generation dir)
      teacher_train_sources.json     dict[method_key -> inspection rows]
      teacher_rows.json              teacher-side result rows (Stages 1 + 3 + 4)
    phase_times.json                 timings from Stages 1–2 + 3–4

Runs:
  STAGE 5 — Student distillation & evaluation (one run per
            (source, mode, beta_s) triple from cfg.distill)
  STAGE 6 — Final results (results.json + RESULTS.md)

Writes to --output-dir (defaults to --input-dir):
  student/<source_name>/<mode>_beta_<beta_s>/    trained student checkpoints
  student/<source_name>/test_<mode>_beta_<beta_s>.{json,md}   eval traces
  inspection/<source_name>_<mode>_beta_<beta_s>.json          inspection samples
  results.json                                                teacher + student rows
  RESULTS.md                                                  markdown report
  phase_times.json                                            with 5_student added

Usage:
    # Simplest: run after `run_pipeline_teachers.py --input-dir X`.
    python scripts/run_pipeline_student.py --input-dir X

    # Sweep variant: share the teacher output, override distill.* via a
    # custom config, and write student artifacts to a per-task dir.
    python scripts/run_pipeline_student.py \
        --input-dir  /shared/teacher_out \
        --config     /tmp/sweep_task_3.yaml \
        --output-dir /shared/teacher_out/student_sweep_task_3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

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
from clean_sweep.generation import generate_teacher_traces
from clean_sweep.summary import build_results_report
from clean_sweep.train import run_distill
from clean_sweep.utils import ensure_dir, set_seed, write_json, write_markdown_examples

console = Console()
_datasets_disable_progress_bars()
if _hf_disable_progress_bar is not None:
    _hf_disable_progress_bar()


# Cache file layout — keep in sync with run_pipeline_generation.py and
# run_pipeline_teachers.py.
CACHE_HOLDOUT_FULL = "holdout_full.json"
CACHE_TEACHER_ROWS = "teacher_rows.json"
CACHE_TEACHER_TRAIN_SOURCES = "teacher_train_sources.json"
CACHE_PHASE_TIMES = "phase_times.json"
TEACHERS_DONE_MARKER = "TEACHERS_DONE"
CONFIG_SNAPSHOT_NAME = "config_snapshot.yaml"


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
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, "total_memory", getattr(props, "total_mem", None))
    if total_mem is None:
        return f"GPU mem: {a:.1f}GB alloc / {r:.1f}GB reserved"
    t = total_mem / 1e9
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
    n = len(compact)
    n_correct = sum(1 for r in compact if r.get("correct", False))
    avg_len = sum(len(r.get("response", "")) for r in compact) / max(n, 1)
    return f"n={n}, correct={n_correct}, acc={n_correct / max(n, 1):.4f}, avg_len={avg_len:.0f}"


def _flush_vram(label: str = "") -> None:
    gc.collect()
    torch.cuda.empty_cache()
    if label:
        console.print(f"[{_now()}] Freed {label} | {_gpu_mem()}")


def _log_student_summary(student_rows: list[dict]) -> None:
    _separator("STUDENT DISTILLATION SUMMARY")
    console.print(f"  Runs completed: {len(student_rows)}")
    console.print(f"  {'Source':<50} {'Mode':<16} {'Test Acc':>10}")
    console.print(f"  {'-' * 50} {'-' * 16} {'-' * 10}")
    for r in student_rows:
        console.print(f"  {r['train_source']:<50} {r['eval_model']:<16} {r['accuracy']:>10.4f}")
    console.print()


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    pipeline_t0 = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Strategic distillation — Stages 5 & 6 (student distill + eval + results)."
    )
    parser.add_argument("--input-dir", required=True, type=str,
                        help="Directory produced by run_pipeline_teachers.py "
                             "(contains config_snapshot.yaml, cache/holdout_full.json, "
                             "cache/teacher_train_sources.json, cache/teacher_rows.json).")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config override. Defaults to "
                             "<input-dir>/config_snapshot.yaml. Must remain "
                             "compatible on seed/data/student/generation.* for "
                             "the cached teacher traces to be valid.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Destination for student/, inspection/, results.json, "
                             "RESULTS.md. Defaults to --input-dir. Use a per-task "
                             "dir when multiple student jobs share one teacher cache.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    cache_dir = in_dir / "cache"
    snapshot_path = Path(args.config) if args.config else in_dir / CONFIG_SNAPSHOT_NAME
    done_marker = in_dir / TEACHERS_DONE_MARKER

    required = [
        in_dir,
        cache_dir,
        snapshot_path,
        cache_dir / CACHE_HOLDOUT_FULL,
        cache_dir / CACHE_TEACHER_TRAIN_SOURCES,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")
    if not done_marker.exists():
        console.print(f"[warn] {done_marker} not found — teacher stage may be incomplete; proceeding anyway.")

    cfg = FullConfig.from_yaml(snapshot_path)
    set_seed(cfg.run.seed)

    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else in_dir)

    _separator("PIPELINE CONFIG (student, Stages 5–6)")
    console.print(f"  Input (cache):    {in_dir}")
    console.print(f"  Output:           {out_dir}")
    console.print(f"  Config:           {snapshot_path}"
                  + ("  (override)" if args.config else "  (from snapshot)"))
    console.print(f"  Dataset:          {cfg.data.dataset_name} "
                  f"(train={cfg.data.train_size}, holdout={cfg.data.holdout_size}, "
                  f"test={cfg.data.test_size})")
    console.print(f"  Seed:             {cfg.run.seed}")
    console.print(f"  Student model:    {cfg.model.student}")
    console.print(f"  Student modes:    {cfg.distill.student_modes}")
    console.print(f"  beta_s_values:    {cfg.distill.beta_s_values}")
    console.print(f"  penalty_transform: {cfg.distill.penalty_transform}")
    console.print(f"  teacher_sign:     {cfg.distill.teacher_sign}")
    if torch.cuda.is_available():
        console.print(f"  GPU:              {torch.cuda.get_device_name(0)}")
        console.print(f"  {_gpu_mem()}")
    else:
        console.print(f"  Device:           CPU")
    console.print()

    # ── Rebuild splits ───────────────────────────────────────────────────
    # Only `test` is consumed in Stage 5 (eval), but keeping the full split
    # construction mirrors generation/teachers and keeps determinism guarantees
    # identical (load_dataset_splits is seed-deterministic).
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

    # ── Load cache from teachers ─────────────────────────────────────────
    t = _stage_start("Loading teacher cache")
    holdout_full: list[dict] = _load_json(cache_dir / CACHE_HOLDOUT_FULL)
    teacher_train_sources: dict[str, list[dict]] = _load_json(
        cache_dir / CACHE_TEACHER_TRAIN_SOURCES
    )
    teacher_rows_path = cache_dir / CACHE_TEACHER_ROWS
    teacher_rows: list[dict] = _load_json(teacher_rows_path) if teacher_rows_path.exists() else []
    phase_times_path = in_dir / CACHE_PHASE_TIMES
    phase_times: dict[str, float] = _load_json(phase_times_path) if phase_times_path.exists() else {}
    total_traces = sum(len(v) for v in teacher_train_sources.values())
    _stage_end(
        "Loading teacher cache", t,
        extra=(f"holdout={len(holdout_full)}, methods={len(teacher_train_sources)}, "
               f"traces={total_traces}, teacher_rows={len(teacher_rows)}"),
    )

    if not teacher_train_sources:
        raise RuntimeError(
            f"No teacher sources found in {cache_dir / CACHE_TEACHER_TRAIN_SOURCES}. "
            "Ensure the teacher stage produced at least one method."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = cfg.artifacts.save_inspection_samples

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: Student distillation & evaluation
    # ══════════════════════════════════════════════════════════════════════
    _separator("STAGE 5: STUDENT DISTILLATION & EVALUATION")
    phase_t0 = time.perf_counter()

    student_rows: list[dict] = []

    for source_name, train_traces in teacher_train_sources.items():
        for mode in cfg.distill.student_modes:
            # strategic_fd uses configurable beta_s; naive always uses 1.0.
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

                notes = "" if mode == "naive" else f"beta_s={beta_s}"
                if stats:
                    sn = (f"mean_a={stats.get('mean_a', 0.0):.4f}, "
                          f"frac_mass_top20={stats.get('frac_mass_top20', 0.0):.4f}")
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
    # Persist the updated phase_times (now including Stage 5) alongside this
    # task's outputs, so array jobs sharing one input-dir do not race on the
    # upstream cache file.
    write_json(phase_times, out_dir / CACHE_PHASE_TIMES)
    _stage_end("Writing results", t, extra=f"rows={len(results_rows)}")

    # ── Final summary ─────────────────────────────────────────────────────
    total = time.perf_counter() - pipeline_t0
    _separator("PIPELINE COMPLETE")
    console.print(f"  Output:           {out_dir}")
    console.print(f"  Wall time:        {_fmt_dur(total)} (student only)")
    console.print(f"  Teacher methods:  {len(teacher_train_sources)}")
    console.print(f"  Student runs:     {len(student_rows)}")
    if torch.cuda.is_available():
        console.print(f"  Peak GPU mem:     {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    console.print()
    console.print(f"  Time breakdown (incl. cached Stages 1–4):")
    for name, dur in phase_times.items():
        pct = dur / total * 100 if total > 0 else 0
        console.print(f"    {name:<25} {_fmt_dur(dur):>10}  ({pct:5.1f}%)")
    console.print()
    console.print(report)


if __name__ == "__main__":
    main()
