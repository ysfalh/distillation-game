#!/usr/bin/env python3
"""
Student-only distillation from a flat saved-trace folder.

This runner consumes legacy artifacts like `gsm8k_output_small/` directly:

  config_snapshot.yaml
  holdout_standard_internal.json
  train_standard.json
  train_antidistillation_lam_0.055.json
  train_poe_gamma_0.7.json

It reuses the existing distillation/evaluation code and does not regenerate
teacher-side traces. Strategic-FD student weights are recomputed for the
configured student model.
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
from typing import Any

_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich.console import Console

from clean_sweep.config import FullConfig


console = Console()
_TORCH: Any | None = None
_TORCH_IMPORT_ERROR: Exception | None = None


CONFIG_SNAPSHOT_NAME = "config_snapshot.yaml"
HOLDOUT_FILE = "holdout_standard_internal.json"
SOURCE_FILES = {
    "teacher_standard": "train_standard.json",
    "teacher_antidistillation_lam_0.055": "train_antidistillation_lam_0.055.json",
    "teacher_poe_gamma_0.7": "train_poe_gamma_0.7.json",
}


def _get_torch() -> Any | None:
    global _TORCH, _TORCH_IMPORT_ERROR
    if _TORCH is not None:
        return _TORCH
    if _TORCH_IMPORT_ERROR is not None:
        return None
    try:
        import torch as torch_mod
    except Exception as exc:  # pragma: no cover - depends on runtime env
        _TORCH_IMPORT_ERROR = exc
        return None
    _TORCH = torch_mod
    return _TORCH


def _require_torch() -> Any:
    torch_mod = _get_torch()
    if torch_mod is None:
        raise RuntimeError(
            "torch is required for a full student run. Use --dry-run for trace/config validation, "
            "or run from the training environment with project dependencies installed."
        ) from _TORCH_IMPORT_ERROR
    return torch_mod


def _disable_runtime_progress_bars() -> None:
    try:
        from datasets import disable_progress_bars as datasets_disable_progress_bars
        datasets_disable_progress_bars()
    except ImportError:  # pragma: no cover
        pass

    try:
        from transformers.utils.logging import disable_progress_bar as hf_disable_progress_bar
    except ImportError:  # pragma: no cover
        try:
            from transformers.logging import (  # type: ignore[attr-defined]
                disable_progress_bar as hf_disable_progress_bar,
            )
        except ImportError:
            hf_disable_progress_bar = None  # type: ignore[assignment]
    if hf_disable_progress_bar is not None:
        hf_disable_progress_bar()


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
    torch_mod = _get_torch()
    if torch_mod is None or not torch_mod.cuda.is_available():
        return "cpu"
    a = torch_mod.cuda.memory_allocated() / 1e9
    r = torch_mod.cuda.memory_reserved() / 1e9
    props = torch_mod.cuda.get_device_properties(0)
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


def _trace_stats(compact: list[dict[str, Any]]) -> str:
    n = len(compact)
    n_correct = sum(1 for r in compact if r.get("correct", False))
    avg_len = sum(len(r.get("trace", "")) for r in compact) / max(n, 1)
    return (
        f"n={n}, correct={n_correct}, "
        f"acc={n_correct / max(n, 1):.4f}, avg_len={avg_len:.0f}"
    )


def _flush_vram(label: str = "") -> None:
    gc.collect()
    torch_mod = _get_torch()
    if torch_mod is not None and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()
    if label:
        console.print(f"[{_now()}] Freed {label} | {_gpu_mem()}")


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_trace_list(path: Path) -> list[dict[str, Any]]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of trace rows in {path}")
    return rows


def _source_accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(1 for r in rows if r.get("correct", False)) / max(len(rows), 1)


def _load_saved_traces(
    input_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    required = [input_dir / CONFIG_SNAPSHOT_NAME, input_dir / HOLDOUT_FILE]
    required.extend(input_dir / filename for filename in SOURCE_FILES.values())
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required saved-trace artifact missing: {path}")

    holdout = _load_trace_list(input_dir / HOLDOUT_FILE)
    train_sources = {
        source_name: _load_trace_list(input_dir / filename)
        for source_name, filename in SOURCE_FILES.items()
    }
    return holdout, train_sources


def _log_config(
    *,
    cfg: FullConfig,
    input_dir: Path,
    output_dir: Path,
    holdout: list[dict[str, Any]],
    train_sources: dict[str, list[dict[str, Any]]],
    dry_run: bool,
) -> None:
    _separator("PIPELINE CONFIG (saved-trace student)")
    console.print(f"  Input traces:      {input_dir}")
    console.print(f"  Output:            {output_dir}")
    console.print(f"  Dry run:           {dry_run}")
    console.print(f"  Dataset:           {cfg.data.dataset_name} "
                  f"(train={cfg.data.train_size}, holdout={cfg.data.holdout_size}, "
                  f"test={cfg.data.test_size})")
    console.print(f"  Seed:              {cfg.run.seed}")
    console.print(f"  Teacher model:     {cfg.model.teacher}")
    console.print(f"  Proxy student:     {cfg.model.proxy_student}")
    console.print(f"  Student model:     {cfg.model.student}")
    console.print(f"  Student tokenizer: {cfg.model.student_tokenizer}")
    console.print(f"  Attention backend: {cfg.model.attn_implementation}")
    console.print(f"  Student modes:     {cfg.distill.student_modes}")
    console.print(f"  beta_s_values:     {cfg.distill.beta_s_values}")
    console.print(f"  penalty_transform: {cfg.distill.penalty_transform}")
    console.print(f"  Holdout traces:    {len(holdout)}")
    console.print("  Train sources:")
    for source_name, rows in train_sources.items():
        console.print(
            f"    - {source_name}: {len(rows)} traces, "
            f"acc={_source_accuracy(rows):.4f}"
        )
    torch_mod = _get_torch()
    if torch_mod is not None and torch_mod.cuda.is_available():
        console.print(f"  GPU:               {torch_mod.cuda.get_device_name(0)}")
        console.print(f"  {_gpu_mem()}")
    else:
        console.print("  Device:            CPU")
    console.print()


def _teacher_rows_from_sources(train_sources: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, traces in train_sources.items():
        rows.append({
            "train_source": "-",
            "eval_model": f"{source_name}_train",
            "accuracy": _source_accuracy(traces),
            "notes": "saved_train_source",
        })
    return rows


def _write_manifest(
    *,
    cfg: FullConfig,
    input_dir: Path,
    output_dir: Path,
    train_sources: dict[str, list[dict[str, Any]]],
) -> None:
    from clean_sweep.utils import write_json

    cfg.to_yaml(output_dir / CONFIG_SNAPSHOT_NAME)
    manifest = {
        "stage": "saved_trace_student",
        "input_dir": str(input_dir),
        "created_at": datetime.now().isoformat(),
        "dataset": cfg.data.dataset_name,
        "seed": cfg.run.seed,
        "teacher": cfg.model.teacher,
        "proxy_student": cfg.model.proxy_student,
        "student": cfg.model.student,
        "student_tokenizer": cfg.model.student_tokenizer,
        "train_sources": {name: len(rows) for name, rows in train_sources.items()},
        "holdout_source": HOLDOUT_FILE,
        "teacher_generation_reused": True,
    }
    write_json(manifest, output_dir / "run_manifest.json")


def _log_student_summary(student_rows: list[dict[str, Any]]) -> None:
    _separator("STUDENT DISTILLATION SUMMARY")
    console.print(f"  Runs completed: {len(student_rows)}")
    console.print(f"  {'Source':<50} {'Mode':<16} {'Test Acc':>10}")
    console.print(f"  {'-' * 50} {'-' * 16} {'-' * 10}")
    for row in student_rows:
        console.print(
            f"  {row['train_source']:<50} "
            f"{row['eval_model']:<16} {row['accuracy']:>10.4f}"
        )
    console.print()


def main() -> None:
    pipeline_t0 = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Run Llama/student distillation from flat saved teacher traces without regenerating teachers."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Flat saved-trace directory, e.g. gsm8k_output_small.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Destination for student artifacts and results.",
    )
    parser.add_argument("--student", default="meta-llama/Llama-3.2-1B", help="Student base model to train.")
    parser.add_argument(
        "--student-tokenizer",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Tokenizer/chat template for the student.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend for the student run. Defaults to sdpa.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and saved traces without loading models or writing outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == input_resolved or input_resolved in output_resolved.parents:
        raise ValueError(
            "--output-dir must be outside --input-dir so saved teacher traces are never mutated"
        )

    snapshot_path = input_dir / CONFIG_SNAPSHOT_NAME
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Required config snapshot missing: {snapshot_path}")

    cfg = FullConfig.from_yaml(snapshot_path)
    if cfg.data.dataset_name != "gsm8k":
        raise ValueError(f"This saved-trace runner expects GSM8K traces, got {cfg.data.dataset_name!r}")
    cfg.model.student = args.student
    cfg.model.student_tokenizer = args.student_tokenizer
    cfg.model.attn_implementation = args.attn_implementation
    cfg.run.output_dir = str(output_dir.parent)
    cfg.run.run_name = output_dir.name

    t = _stage_start("Loading saved teacher traces")
    holdout_full, teacher_train_sources = _load_saved_traces(input_dir)
    _stage_end(
        "Loading saved teacher traces",
        t,
        extra=(
            f"holdout={len(holdout_full)}, methods={len(teacher_train_sources)}, "
            f"traces={sum(len(v) for v in teacher_train_sources.values())}"
        ),
    )
    _flush_vram("saved trace load")

    _log_config(
        cfg=cfg,
        input_dir=input_dir,
        output_dir=output_dir,
        holdout=holdout_full,
        train_sources=teacher_train_sources,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        _separator("DRY RUN COMPLETE")
        console.print("  Saved traces and config are valid.")
        console.print("  No output files were written and no models were loaded.")
        return

    torch = _require_torch()
    _disable_runtime_progress_bars()
    from clean_sweep.data import format_prompt_gsm8k, load_dataset_splits
    from clean_sweep.generation import generate_teacher_traces
    from clean_sweep.summary import build_results_report
    from clean_sweep.train import run_distill
    from clean_sweep.utils import ensure_dir, set_seed, write_json, write_markdown_examples

    set_seed(cfg.run.seed)
    output_dir = ensure_dir(output_dir)
    _write_manifest(
        cfg=cfg,
        input_dir=input_dir,
        output_dir=output_dir,
        train_sources=teacher_train_sources,
    )

    _separator("LOADING TEST SPLIT")
    t = _stage_start(f"Loading {cfg.data.dataset_name.upper()} test split")
    splits = load_dataset_splits(
        cfg.data.dataset_name,
        seed=cfg.run.seed,
        train_size=cfg.data.train_size,
        holdout_size=cfg.data.holdout_size,
        test_size=cfg.data.test_size,
    )
    test_split = splits["test"]
    del splits
    _stage_end("Loading GSM8K test split", t, extra=f"test={len(test_split)}")
    _flush_vram("dataset split load")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = cfg.artifacts.save_inspection_samples
    phase_times: dict[str, float] = {}
    teacher_rows = _teacher_rows_from_sources(teacher_train_sources)
    student_rows: list[dict[str, Any]] = []

    _separator("STAGE 5: STUDENT DISTILLATION & EVALUATION")
    phase_t0 = time.perf_counter()

    if cfg.distill.eval_batch_size is not None:
        cfg.generation.batch_size = cfg.distill.eval_batch_size
        console.print(
            f"[{_now()}]   Using eval_batch_size={cfg.distill.eval_batch_size} "
            "for student test-set generation"
        )

    for source_name, train_traces in teacher_train_sources.items():
        for mode in cfg.distill.student_modes:
            betas = cfg.distill.beta_s_values if mode == "strategic_fd" else [1.0]
            for beta_s in betas:
                set_seed(cfg.run.seed)
                model_dir = ensure_dir(
                    output_dir / "student" / source_name / f"{mode}_beta_{beta_s}"
                )
                label = f"Student train | src={source_name} | {mode} | beta_s={beta_s}"
                t = _stage_start(label)
                n_tr = len(train_traces)
                n_ok = sum(1 for x in train_traces if x.get("correct", False))
                console.print(
                    f"[{_now()}]   Train traces: {n_tr} total, {n_ok} correct "
                    f"({n_ok / max(n_tr, 1):.4f})"
                )
                if mode == "strategic_fd":
                    console.print(
                        f"[{_now()}]   Strategic weights are recomputed for {cfg.model.student}"
                    )
                else:
                    console.print(f"[{_now()}]   Uniform trace weights")

                stats, student_model, student_tok = run_distill(
                    cfg=cfg,
                    train_traces=train_traces,
                    holdout_traces=holdout_full,
                    output_dir=model_dir,
                    device=device,
                    mode=mode,
                    beta_s=beta_s,
                )

                train_parts = []
                if stats:
                    train_parts.append(f"mean_a={stats.get('mean_a', 0.0):.4f}")
                    train_parts.append(f"std_a={stats.get('std_a', 0.0):.4f}")
                    train_parts.append(f"frac_top20={stats.get('frac_mass_top20', 0.0):.4f}")
                train_parts.append(_gpu_mem())
                _stage_end(label, t, extra=", ".join(train_parts))
                _flush_vram(f"post-train cleanup ({source_name}, {mode})")

                set_seed(cfg.run.seed)
                label = f"Student eval | src={source_name} | {mode} | beta_s={beta_s} | test"
                t = _stage_start(label)
                compact, inspection = generate_teacher_traces(
                    cfg=cfg,
                    dataset=test_split,
                    format_prompt=format_prompt_gsm8k,
                    method_name="standard",
                    device=device,
                    model=student_model,
                    tokenizer=student_tok,
                )
                out_base = output_dir / "student" / source_name
                write_json(compact, out_base / f"test_{mode}_beta_{beta_s}.json")
                write_markdown_examples(
                    compact[:n_samples],
                    out_base / f"test_{mode}_beta_{beta_s}.md",
                    title=f"test {mode} beta_s={beta_s} on {source_name} ({cfg.model.student})",
                )
                write_json(
                    inspection[:n_samples],
                    output_dir / "inspection" / f"{source_name}_{mode}_beta_{beta_s}.json",
                )
                acc = sum(1 for r in compact if r.get("correct", False)) / max(len(compact), 1)
                _stage_end(label, t, extra=_trace_stats(compact))

                notes = f"student={cfg.model.student}"
                if mode == "strategic_fd":
                    notes += f", beta_s={beta_s}"
                if stats:
                    notes += (
                        f", mean_a={stats.get('mean_a', 0.0):.4f}, "
                        f"frac_mass_top20={stats.get('frac_mass_top20', 0.0):.4f}"
                    )
                student_rows.append({
                    "train_source": source_name,
                    "eval_model": f"student_{mode}",
                    "accuracy": acc,
                    "notes": notes,
                })

                del compact, inspection
                del student_model, student_tok
                _flush_vram(f"student model ({source_name}, {mode}, beta={beta_s})")

    phase_times["5_student"] = time.perf_counter() - phase_t0
    console.print(f"[{_now()}] STAGE 5 completed in {_fmt_dur(phase_times['5_student'])}")
    _log_student_summary(student_rows)

    _separator("STAGE 6: FINAL RESULTS")
    t = _stage_start("Writing results")
    results_rows = teacher_rows + student_rows
    write_json(results_rows, output_dir / "results.json")
    report = build_results_report(
        results_rows,
        dataset=cfg.data.dataset_name,
        teacher=cfg.model.teacher,
        student=cfg.model.student,
    )
    (output_dir / "RESULTS.md").write_text(report)
    write_json(phase_times, output_dir / "phase_times.json")
    _stage_end("Writing results", t, extra=f"rows={len(results_rows)}")

    total = time.perf_counter() - pipeline_t0
    _separator("PIPELINE COMPLETE")
    console.print(f"  Output:           {output_dir}")
    console.print(f"  Wall time:        {_fmt_dur(total)}")
    console.print(f"  Teacher sources:  {len(teacher_train_sources)} (reused from {input_dir})")
    console.print(f"  Student runs:     {len(student_rows)}")
    if torch.cuda.is_available():
        console.print(f"  Peak GPU mem:     {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    console.print()
    console.print("  Time breakdown:")
    for name, dur in phase_times.items():
        pct = dur / total * 100 if total > 0 else 0
        console.print(f"    {name:<25} {_fmt_dur(dur):>10}  ({pct:5.1f}%)")
    console.print()
    console.print(report)
    _flush_vram("final cleanup")


if __name__ == "__main__":
    main()
