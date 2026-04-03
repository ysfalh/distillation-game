#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
from clean_sweep.summary import build_results_report
from clean_sweep.train import run_distill
from clean_sweep.utils import ensure_dir, set_seed, write_json, write_markdown_examples
import torch


console = Console()
_datasets_disable_progress_bars()
if _hf_disable_progress_bar is not None:
    _hf_disable_progress_bar()


def _now_stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


def _log_stage_start(label: str) -> float:
    console.print(f"[{_now_stamp()}] {label}...")
    return time.perf_counter()


def _log_stage_end(label: str, start_time: float, extra: str | None = None) -> None:
    elapsed = _fmt_duration(time.perf_counter() - start_time)
    suffix = f" | {extra}" if extra else ""
    console.print(f"[{_now_stamp()}] {label} done in {elapsed}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean antidistillation pipeline.")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = FullConfig.from_yaml(args.config)
    set_seed(cfg.run.seed)

    run_id = f"{cfg.run.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(cfg.run.output_dir) / run_id)
    console.print(f"[{_now_stamp()}] Run initialized | output={out_dir} | seed={cfg.run.seed}")

    if cfg.run.save_config_snapshot:
        cfg.to_yaml(out_dir / "config_snapshot.yaml")

    stage = _log_stage_start(f"Loading {cfg.data.dataset_name.upper()} splits")
    splits = load_dataset_splits(
        cfg.data.dataset_name,
        seed=cfg.run.seed,
        train_size=cfg.data.train_size,
        holdout_size=cfg.data.holdout_size,
        test_size=cfg.data.test_size,
    )
    format_prompt = format_prompt_gsm8k if cfg.data.dataset_name in {"gsm8k", "gsm_hard", "svamp"} else format_prompt_math
    _log_stage_end(
        f"Loading {cfg.data.dataset_name.upper()} splits",
        stage,
        extra=f"train={len(splits['train'])}, holdout={len(splits['holdout'])}, test={len(splits['test'])}",
    )

    manifest = {
        "run_id": run_id,
        "dataset": cfg.data.dataset_name,
        "teacher": cfg.model.teacher,
        "proxy_student": cfg.model.proxy_student,
        "student": cfg.model.student,
        "seed": cfg.run.seed,
        "split_sizes": {name: len(ds) for name, ds in splits.items()},
        "planned_methods": {
            "teachers": {
                "standard": cfg.teachers.standard,
                "antidistillation_lams": cfg.teachers.antidistillation_lams,
                "poe_gammas": cfg.teachers.poe_gammas,
                "strategic_beta_teachers": cfg.teachers.strategic_beta_teachers,
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
        console.print(f"[{_now_stamp()}] Saved prompt dictionary")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = _log_stage_start(f"Loading teacher model on {device}")
    teacher_model, teacher_tokenizer = load_model_and_tokenizer(cfg.model.teacher, cfg.model.tokenizer, cfg, device)
    _log_stage_end("Loading teacher model", stage, extra=cfg.model.teacher)

    teacher_rows: list[dict] = []
    teacher_train_sources: dict[str, list[dict]] = {}
    teacher_holdout_source: list[dict] | None = None

    standard_splits = ["holdout"] + (["train", "test"] if cfg.teachers.standard else [])
    for split_name in standard_splits:
        set_seed(cfg.run.seed)
        stage_label = f"Teacher generation | method=standard | split={split_name}"
        stage = _log_stage_start(stage_label)
        compact, inspection = generate_teacher_traces(
            cfg=cfg,
            dataset=splits[split_name],
            format_prompt=format_prompt,
            method_name="standard",
            device=device,
            model=teacher_model,
            tokenizer=teacher_tokenizer,
        )
        output_stem = f"{split_name}_standard"
        if split_name == "holdout" and not cfg.teachers.standard:
            output_stem = f"{split_name}_standard_internal"
        write_json(compact, out_dir / "teacher" / f"{output_stem}.json")
        write_markdown_examples(
            compact[: cfg.artifacts.save_inspection_samples],
            out_dir / "teacher" / f"{output_stem}.md",
            title=f"{split_name} standard",
        )
        if split_name == "holdout":
            teacher_holdout_source = inspection
        elif split_name == "train":
            teacher_train_sources["teacher_standard"] = inspection
        acc = sum(1 for row in compact if row["correct"]) / max(len(compact), 1)
        extra = f"n={len(compact)}, accuracy={acc:.4f}"
        if split_name == "holdout" and not cfg.teachers.standard:
            extra += ", internal_only=true"
        _log_stage_end(stage_label, stage, extra=extra)
        if split_name != "holdout":
            teacher_rows.append({"train_source": "-", "eval_model": f"teacher_standard_{split_name}", "accuracy": acc, "notes": split_name})

    if teacher_holdout_source is None:
        raise RuntimeError("standard holdout traces were not generated")
    holdout_full = teacher_holdout_source

    # Proxy gradients are computed from standard holdout traces, matching the original pipeline.
    from clean_sweep.train.distill import compute_student_holdout_grad, _response_template_for_model
    stage = _log_stage_start("Loading proxy model")
    proxy_model, proxy_tokenizer = load_model_and_tokenizer(cfg.model.proxy_student, cfg.model.tokenizer or cfg.model.proxy_student, cfg, device)
    _log_stage_end("Loading proxy model", stage, extra=str(cfg.model.proxy_student))
    response_template = _response_template_for_model(cfg.model.proxy_student or "", proxy_tokenizer)
    proxy_model.train()
    set_seed(cfg.run.seed)
    stage_label = "Proxy gradients | source=standard_holdout"
    stage = _log_stage_start(stage_label)
    proxy_grads = compute_student_holdout_grad(
        proxy_model,
        proxy_tokenizer,
        holdout_full,
        device,
        response_template,
        cfg.distill.max_length,
        cfg.distill.holdout_grad_batch_size,
    )
    _log_stage_end(stage_label, stage, extra=f"n_holdout={len(holdout_full)}")

    beta_teacher_values = cfg.teachers.strategic_beta_teachers
    for lam in cfg.teachers.antidistillation_lams:
        if beta_teacher_values:
            for beta_teacher in beta_teacher_values:
                for split_name in ("train", "test"):
                    set_seed(cfg.run.seed)
                    stage_label = (
                        f"Teacher generation | method=antidistillation | lam={lam} | beta_teacher={beta_teacher} | split={split_name}"
                    )
                    stage = _log_stage_start(stage_label)
                    compact, inspection = generate_teacher_traces(
                        cfg=cfg,
                        dataset=splits[split_name],
                        format_prompt=format_prompt,
                        method_name="antidistillation",
                        device=device,
                        model=teacher_model,
                        tokenizer=teacher_tokenizer,
                        grad_dict=proxy_grads,
                        lam=lam,
                        beta_teacher=beta_teacher,
                    )
                    method_key = f"teacher_antidistillation_lam_{lam}_beta_teacher_{beta_teacher}"
                    write_json(compact, out_dir / "teacher" / f"{split_name}_antidistillation_lam_{lam}_beta_teacher_{beta_teacher}.json")
                    write_markdown_examples(
                        compact[: cfg.artifacts.save_inspection_samples],
                        out_dir / "teacher" / f"{split_name}_antidistillation_lam_{lam}_beta_teacher_{beta_teacher}.md",
                        title=f"{split_name} antidistillation lam={lam} beta_teacher={beta_teacher}",
                    )
                    if split_name == "train":
                        teacher_train_sources[method_key] = inspection
                    acc = sum(1 for row in compact if row["correct"]) / max(len(compact), 1)
                    _log_stage_end(
                        stage_label,
                        stage,
                        extra=f"n={len(compact)}, accuracy={acc:.4f}",
                    )
                    teacher_rows.append({"train_source": "-", "eval_model": f"{method_key}_{split_name}", "accuracy": acc, "notes": split_name})
        else:
            for split_name in ("train", "test"):
                set_seed(cfg.run.seed)
                stage_label = f"Teacher generation | method=antidistillation | lam={lam} | split={split_name}"
                stage = _log_stage_start(stage_label)
                compact, inspection = generate_teacher_traces(
                    cfg=cfg,
                    dataset=splits[split_name],
                    format_prompt=format_prompt,
                    method_name="antidistillation",
                    device=device,
                    model=teacher_model,
                    tokenizer=teacher_tokenizer,
                    grad_dict=proxy_grads,
                    lam=lam,
                )
                method_key = f"teacher_antidistillation_lam_{lam}"
                write_json(compact, out_dir / "teacher" / f"{split_name}_antidistillation_lam_{lam}.json")
                write_markdown_examples(
                    compact[: cfg.artifacts.save_inspection_samples],
                    out_dir / "teacher" / f"{split_name}_antidistillation_lam_{lam}.md",
                    title=f"{split_name} antidistillation lam={lam}",
                )
                if split_name == "train":
                    teacher_train_sources[method_key] = inspection
                acc = sum(1 for row in compact if row["correct"]) / max(len(compact), 1)
                _log_stage_end(
                    stage_label,
                    stage,
                    extra=f"n={len(compact)}, accuracy={acc:.4f}",
                )
                teacher_rows.append({"train_source": "-", "eval_model": f"{method_key}_{split_name}", "accuracy": acc, "notes": split_name})

    for gamma in cfg.teachers.poe_gammas:
        for split_name in ("train", "test"):
            set_seed(cfg.run.seed)
            stage_label = f"Teacher generation | method=poe | gamma={gamma} | split={split_name}"
            stage = _log_stage_start(stage_label)
            compact, inspection = generate_teacher_traces(
                cfg=cfg,
                dataset=splits[split_name],
                format_prompt=format_prompt,
                method_name="poe",
                device=device,
                model=teacher_model,
                tokenizer=teacher_tokenizer,
                gamma=gamma,
            )
            method_key = f"teacher_poe_gamma_{gamma}"
            write_json(compact, out_dir / "teacher" / f"{split_name}_poe_gamma_{gamma}.json")
            write_markdown_examples(
                compact[: cfg.artifacts.save_inspection_samples],
                out_dir / "teacher" / f"{split_name}_poe_gamma_{gamma}.md",
                title=f"{split_name} poe gamma={gamma}",
            )
            if split_name == "train":
                teacher_train_sources[method_key] = inspection
            acc = sum(1 for row in compact if row["correct"]) / max(len(compact), 1)
            _log_stage_end(
                stage_label,
                stage,
                extra=f"n={len(compact)}, accuracy={acc:.4f}",
            )
            teacher_rows.append({"train_source": "-", "eval_model": f"{method_key}_{split_name}", "accuracy": acc, "notes": split_name})

    student_rows: list[dict] = []
    for teacher_source_name, teacher_train_traces in teacher_train_sources.items():
        for mode in cfg.distill.student_modes:
            beta_s_values = cfg.distill.beta_s_values if mode == "strategic_fd" else [1.0]
            for beta_s in beta_s_values:
                set_seed(cfg.run.seed)
                stage_label = f"Student training | source={teacher_source_name} | mode={mode} | beta_s={beta_s}"
                stage = _log_stage_start(stage_label)
                model_dir = ensure_dir(out_dir / "student" / teacher_source_name / f"{mode}_beta_{beta_s}")
                stats, student_model, student_tokenizer = run_distill(
                    cfg=cfg,
                    train_traces=teacher_train_traces,
                    holdout_traces=holdout_full,
                    output_dir=model_dir,
                    device=device,
                    mode=mode,
                    beta_s=beta_s,
                )
                _log_stage_end(
                    stage_label,
                    stage,
                    extra=(f"mean_a={stats.get('mean_a', 0.0):.4f}, frac_mass_top20={stats.get('frac_mass_top20', 0.0):.4f}" if stats else None),
                )
                set_seed(cfg.run.seed)
                stage_label = f"Student eval | source={teacher_source_name} | mode={mode} | beta_s={beta_s} | split=test"
                stage = _log_stage_start(stage_label)
                compact, inspection = generate_teacher_traces(
                    cfg=cfg,
                    dataset=splits["test"],
                    format_prompt=format_prompt,
                    method_name="standard",
                    device=device,
                    model=student_model,
                    tokenizer=student_tokenizer,
                )
                out_base = out_dir / "student" / teacher_source_name
                write_json(compact, out_base / f"test_{mode}_beta_{beta_s}.json")
                write_markdown_examples(
                    compact[: cfg.artifacts.save_inspection_samples],
                    out_base / f"test_{mode}_beta_{beta_s}.md",
                    title=f"test {mode} beta={beta_s} trained on {teacher_source_name}",
                )
                acc = sum(1 for row in compact if row["correct"]) / max(len(compact), 1)
                _log_stage_end(
                    stage_label,
                    stage,
                    extra=f"n={len(compact)}, accuracy={acc:.4f}",
                )
                notes = "" if mode == "naive" else f"beta_s={beta_s}"
                if stats:
                    stats_note = f"mean_a={stats.get('mean_a', 0.0):.4f}, frac_mass_top20={stats.get('frac_mass_top20', 0.0):.4f}"
                    notes = f"{notes}, {stats_note}" if notes else stats_note
                student_rows.append({"train_source": teacher_source_name, "eval_model": f"student_{mode}", "accuracy": acc, "notes": notes})
                write_json(
                    inspection[: cfg.artifacts.save_inspection_samples],
                    out_dir / "inspection" / f"{teacher_source_name}_{mode}_beta_{beta_s}.json",
                )

    results_rows = teacher_rows + student_rows
    stage = _log_stage_start("Writing final results")
    write_json(results_rows, out_dir / "results.json")
    report = build_results_report(results_rows, dataset=cfg.data.dataset_name, teacher=cfg.model.teacher, student=cfg.model.student)
    (out_dir / "RESULTS.md").write_text(report)
    _log_stage_end("Writing final results", stage, extra=f"rows={len(results_rows)}")

    console.print(f"Finished clean pipeline run at {out_dir}")
    console.print(report)


if __name__ == "__main__":
    main()
