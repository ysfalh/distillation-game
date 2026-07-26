#!/usr/bin/env python3
"""Aggregate the filtered-trace student runs into one markdown table.

Walks ``<root>/<level>/seed_<seed>/<source>/results.json`` as written by
``slurm_scripts/run-student-filtered-traces-lowm.sbatch``. Matched and
mismatched-proxy teachers land in the same table, one row each, with the
number of traces the degeneracy filter dropped next to the accuracies.

Usage:
    python scripts/aggregate_filtered_results.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_ROOT = Path("outputs/filtered_traces")
REPORT_NAME = "RESULTS.md"
MISSING = "-"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _filter_manifest(input_dir: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-condition drop counts for the trace directory a run trained on.

    The run manifest records which filtered directory was used, so the counts
    come from that directory rather than from a mapping kept in step with it
    here.
    """
    if input_dir not in cache:
        cache[input_dir] = _read_json(Path(input_dir) / "filter_manifest.json")
    return cache[input_dir]


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Collect one record per (level, seed, source, student mode)."""
    runs = []
    manifest_cache: dict[str, dict[str, Any]] = {}
    for results_path in sorted(root.glob("*/seed_*/*/results.json")):
        source_dir = results_path.parent
        seed_dir = source_dir.parent
        level = seed_dir.parent.name
        seed = int(seed_dir.name.removeprefix("seed_"))

        manifest = _read_json(source_dir / "run_manifest.json")
        input_dir = str(manifest.get("input_dir", ""))
        train_sources = manifest.get("train_sources", {})
        conditions = _filter_manifest(input_dir, manifest_cache).get("conditions", {})
        # The mismatched teachers used a proxy that is not the student model.
        proxy = "mismatched" if "mismatch" in f"{input_dir} {level}" else "matched"

        for row in json.loads(results_path.read_text()):
            eval_model = str(row.get("eval_model", ""))
            if not eval_model.startswith("student_"):
                continue
            source = str(row["train_source"])
            counts = conditions.get(source.removeprefix("teacher_"), {})
            runs.append({
                "level": level,
                "proxy": proxy,
                "seed": seed,
                "source": source,
                "mode": eval_model.removeprefix("student_"),
                "accuracy": float(row["accuracy"]),
                "kept": train_sources.get(source, counts.get("n_after")),
                "dropped": counts.get("n_dropped"),
                "n_before": counts.get("n_before"),
                "student": manifest.get("student", ""),
            })
    return runs


def row_order(key: tuple[str, str, str]) -> tuple[int, int, str, str]:
    """Matched teachers first, each group led by the standard baseline."""
    proxy, source, level = key
    stem = source.removeprefix("teacher_")
    return (proxy == "mismatched", stem != "standard", stem, level)


def dropped_cell(run: dict[str, Any]) -> str:
    dropped, before = run.get("dropped"), run.get("n_before")
    if dropped is None:
        return MISSING
    if not before:
        return str(dropped)
    return f"{dropped} ({dropped / before:.1%})"


def format_table(runs: list[dict[str, Any]], seeds: list[int]) -> list[str]:
    headers = (
        ["teacher", "proxy", "filter", "traces", "dropped"]
        + [f"seed {seed}" for seed in seeds]
        + ["mean", "std err"]
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    keys = sorted({(r["proxy"], r["source"], r["level"]) for r in runs}, key=row_order)
    for proxy, source, level in keys:
        by_seed = {
            run["seed"]: run for run in runs
            if (run["proxy"], run["source"], run["level"]) == (proxy, source, level)
        }
        sample = next(iter(by_seed.values()))
        kept = sample.get("kept")
        cells = [
            source.removeprefix("teacher_"),
            proxy,
            level,
            str(kept) if kept else MISSING,
            dropped_cell(sample),
        ]
        cells += [
            f"{by_seed[seed]['accuracy']:.4f}" if seed in by_seed else MISSING
            for seed in seeds
        ]
        accuracies = [by_seed[seed]["accuracy"] for seed in seeds if seed in by_seed]
        if accuracies:
            cells.append(f"{mean(accuracies):.4f}")
            cells.append(
                f"{stdev(accuracies) / len(accuracies) ** 0.5:.4f}"
                if len(accuracies) > 1
                else MISSING
            )
        else:
            cells += [MISSING, MISSING]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def missing_runs(runs: list[dict[str, Any]], seeds: list[int]) -> list[str]:
    """Cells with no result, so a failed array task is not read as a gap."""
    gaps = []
    for mode in sorted({run["mode"] for run in runs}):
        in_mode = [run for run in runs if run["mode"] == mode]
        keys = sorted({(r["proxy"], r["source"], r["level"]) for r in in_mode}, key=row_order)
        for proxy, source, level in keys:
            done = {
                run["seed"] for run in in_mode
                if (run["proxy"], run["source"], run["level"]) == (proxy, source, level)
            }
            for seed in seeds:
                if seed not in done:
                    gaps.append(
                        f"- `{source.removeprefix('teacher_')}` ({proxy}, {level}), "
                        f"seed {seed}, mode {mode}"
                    )
    return gaps


def build_report(runs: list[dict[str, Any]], root: Path) -> str:
    seeds = sorted({run["seed"] for run in runs})
    students = sorted({run["student"] for run in runs if run["student"]})
    lines = [
        "# Filtered-trace student results",
        "",
        f"- Runs: `{root}`",
        "- Student: " + ", ".join(f"`{s}`" for s in students),
        "- Seeds: " + ", ".join(str(seed) for seed in seeds),
        f"- Completed runs: {len(runs)}",
        "",
        "Test accuracy on the GSM8K-platinum test split, which is identical "
        "across seeds. `proxy` says whether the teacher shaped its traces "
        "against the student model (matched) or against a different model "
        "(mismatched), the latter coming from `gsm8k_output_small_mismatch`. "
        "`traces` is how many traces survived the degeneracy filter and were "
        "trained on, and `dropped` is how many the filter removed. Standard "
        "error is the sample standard deviation over seeds divided by the "
        "square root of the seed count.",
        "",
    ]
    for mode in sorted({run["mode"] for run in runs}):
        selected = [run for run in runs if run["mode"] == mode]
        lines += [f"## Student mode: {mode}", ""]
        lines += format_table(selected, seeds)
        lines.append("")

    gaps = missing_runs(runs, seeds)
    if gaps:
        lines += [
            "## Missing runs",
            "",
            "These cells are blank in the tables above because no "
            "`results.json` was found for them.",
            "",
            *gaps,
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"No student results found under {args.root}")

    # One cell holds one run. Several beta_s values under strategic_fd would
    # collapse into the same cell, so say so instead of hiding it.
    keys = [(run["level"], run["seed"], run["source"], run["mode"]) for run in runs]
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        print(
            f"warning: {len(duplicates)} level/seed/source/mode groups have more than "
            "one run (e.g. several beta_s values); only the last one is tabulated"
        )

    out_path = args.out or args.root / REPORT_NAME
    out_path.write_text(build_report(runs, args.root))
    print(f"Aggregated {len(runs)} runs into {out_path}")


if __name__ == "__main__":
    main()
