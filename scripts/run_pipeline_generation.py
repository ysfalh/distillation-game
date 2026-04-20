#!/usr/bin/env python3
"""
Strategic distillation — generation pipeline (Stages 1 & 2 only).

Produces artifacts consumed by run_pipeline_teachers.py (and, transitively,
run_pipeline_student.py):
  - Standard teacher traces on holdout (always) and on train (if teachers.standard).
  - Proxy-student holdout gradients  g = mean ∇_θ L_holdout(proxy).
  - Config snapshot + cache manifests so later stages resume deterministically.

Intended for cluster use: for a fixed (seed, dataset, teacher, proxy_student,
generation.*) tuple, the artifacts here are shared across many teacher/student
sweeps. Run once per such tuple; then invoke run_pipeline_teachers.py any
number of times against the resulting --output-dir.

Usage:
    python scripts/run_pipeline_generation.py \
        --config configs/gsm8k_strategic_teacher.yaml \
        --output-dir /path/to/store/results
"""
from __future__ import annotations

import argparse
import gc
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
from clean_sweep.generation import generate_teacher_traces, load_model_and_tokenizer
from clean_sweep.train.distill import compute_student_holdout_grad, _response_template_for_model
from clean_sweep.utils import ensure_dir, set_seed, write_json, write_markdown_examples

console = Console()
_datasets_disable_progress_bars()
if _hf_disable_progress_bar is not None:
    _hf_disable_progress_bar()


# Cache file layout (under <output-dir>/cache/) — keep in sync with
# run_pipeline_teachers.py and run_pipeline_student.py.
CACHE_PROXY_GRADS = "proxy_grads.pt"
CACHE_HOLDOUT_FULL = "holdout_full.json"
CACHE_TEACHER_STANDARD_TRAIN = "teacher_standard_train.json"
CACHE_TEACHER_ROWS = "teacher_rows.json"
CACHE_PHASE_TIMES = "phase_times.json"
GENERATION_DONE_MARKER = "GENERATION_DONE"


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


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    pipeline_t0 = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Strategic distillation — Stages 1 & 2 (teacher standard generation + proxy gradients)."
    )
    parser.add_argument("--config", required=True, type=str,
                        help="YAML config (same schema as scripts/run_pipeline.py).")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Destination directory for generation artifacts. "
                             "Consumed by run_pipeline_teachers.py.")
    args = parser.parse_args()

    cfg = FullConfig.from_yaml(args.config)
    set_seed(cfg.run.seed)

    out_dir = ensure_dir(Path(args.output_dir))
    cache_dir = ensure_dir(out_dir / "cache")

    _separator("PIPELINE CONFIG (generation)")
    console.print(f"  Config:           {args.config}")
    console.print(f"  Output:           {out_dir}")
    console.print(f"  Dataset:          {cfg.data.dataset_name} "
                  f"(train={cfg.data.train_size}, holdout={cfg.data.holdout_size}, "
                  f"test={cfg.data.test_size})")
    console.print(f"  Seed:             {cfg.run.seed}")
    console.print(f"  Teacher model:    {cfg.model.teacher}")
    console.print(f"  Proxy student:    {cfg.model.proxy_student}")
    console.print(f"  Standard train:   {cfg.teachers.standard}")
    console.print(f"  Generation:       temp={cfg.generation.temperature}, "
                  f"top_p={cfg.generation.top_p}, eps={cfg.generation.eps}, "
                  f"batch={cfg.generation.batch_size}")
    if torch.cuda.is_available():
        console.print(f"  GPU:              {torch.cuda.get_device_name(0)}")
        console.print(f"  {_gpu_mem()}")
    else:
        console.print(f"  Device:           CPU")
    console.print()

    # Always snapshot the config — distillation recovers it from here.
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

    manifest = {
        "stage": "generation",
        "dataset": cfg.data.dataset_name,
        "teacher": cfg.model.teacher,
        "proxy_student": cfg.model.proxy_student,
        "student": cfg.model.student,
        "seed": cfg.run.seed,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "standard_train_generated": bool(cfg.teachers.standard),
        "created_at": datetime.now().isoformat(),
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
        # Holdout-only traces get an "_internal" tag when standard train is off.
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

    # Free the teacher before Stage 2 — the proxy does not need it co-resident.
    # Distillation will reload the teacher for Stages 3 & 4.
    del teacher_model, teacher_tok
    _flush_vram("teacher model (will be reloaded in distillation)")

    phase_times["1_teacher_gen"] = time.perf_counter() - phase_t0
    console.print(f"[{_now()}] STAGE 1 completed in {_fmt_dur(phase_times['1_teacher_gen'])}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Proxy-student gradients
    #   Compute  g = ∇_θ L_holdout(proxy)  on the holdout teacher traces.
    #   Freed immediately after; only the gradient dict survives.
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

    # ══════════════════════════════════════════════════════════════════════
    # Persist artifacts consumed by run_pipeline_teachers.py
    # ══════════════════════════════════════════════════════════════════════
    _separator("WRITING CACHE FOR TEACHERS")
    t = _stage_start("Saving cache")
    # Move grads to CPU before pickling so downstream can choose its own device.
    proxy_grads_cpu = {k: v.detach().to("cpu").contiguous() for k, v in proxy_grads.items()}
    torch.save(proxy_grads_cpu, cache_dir / CACHE_PROXY_GRADS)
    write_json(holdout_full, cache_dir / CACHE_HOLDOUT_FULL)
    if "teacher_standard" in teacher_train_sources:
        write_json(
            teacher_train_sources["teacher_standard"],
            cache_dir / CACHE_TEACHER_STANDARD_TRAIN,
        )
    write_json(teacher_rows, cache_dir / CACHE_TEACHER_ROWS)
    write_json(phase_times, cache_dir / CACHE_PHASE_TIMES)
    (out_dir / GENERATION_DONE_MARKER).write_text(datetime.now().isoformat() + "\n")
    _stage_end("Saving cache", t, extra=f"files in {cache_dir}")

    # ── Final summary ─────────────────────────────────────────────────────
    total = time.perf_counter() - pipeline_t0
    _separator("GENERATION COMPLETE")
    console.print(f"  Output:           {out_dir}")
    console.print(f"  Wall time:        {_fmt_dur(total)}")
    console.print(f"  Standard train:   {'generated' if cfg.teachers.standard else 'holdout-only (internal)'}")
    if torch.cuda.is_available():
        console.print(f"  Peak GPU mem:     {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    console.print()
    console.print(f"  Time breakdown:")
    for name, dur in phase_times.items():
        pct = dur / total * 100 if total > 0 else 0
        console.print(f"    {name:<25} {_fmt_dur(dur):>10}  ({pct:5.1f}%)")
    console.print()
    console.print(f"  Next step:")
    console.print(f"    python scripts/run_pipeline_teachers.py --input-dir {out_dir}")


if __name__ == "__main__":
    main()
