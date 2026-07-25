#!/usr/bin/env python3
"""Aggregate the filtered-trace student array into one markdown table.

Walks ``<root>/<level>/seed_<seed>/<source>/results.json`` as written by
``slurm_scripts/run-student-filtered-traces.sbatch`` and reports test accuracy
per seed plus the mean and standard error for every filter level.

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
REPORT_NAME = "RESULTS_FILTERED.md"


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Collect one record per (level, seed, source, student mode)."""
    runs = []
    for results_path in sorted(root.glob("*/seed_*/*/results.json")):
        source_dir = results_path.parent
        seed_dir = source_dir.parent
        level = seed_dir.parent.name
        seed = int(seed_dir.name.removeprefix("seed_"))

        manifest_path = source_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        train_sources = manifest.get("train_sources", {})

        for row in json.loads(results_path.read_text()):
            eval_model = str(row.get("eval_model", ""))
            if not eval_model.startswith("student_"):
                continue
            runs.append({
                "level": level,
                "seed": seed,
                "source": str(row["train_source"]),
                "mode": eval_model.removeprefix("student_"),
                "accuracy": float(row["accuracy"]),
                "n_traces": train_sources.get(str(row["train_source"])),
                "student": manifest.get("student", ""),
            })
    return runs


def level_order(level: str) -> tuple[int, str]:
    return (level != "unfiltered", level)


def format_table(runs: list[dict[str, Any]], seeds: list[int]) -> list[str]:
    levels = sorted({run["level"] for run in runs}, key=level_order)
    headers = ["level", "traces"] + [f"seed {seed}" for seed in seeds] + ["mean", "std err"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for level in levels:
        by_seed = {run["seed"]: run for run in runs if run["level"] == level}
        counts = {run["n_traces"] for run in by_seed.values() if run["n_traces"]}
        cells = [level, str(counts.pop()) if len(counts) == 1 else "-"]
        cells += [
            f"{by_seed[seed]['accuracy']:.4f}" if seed in by_seed else "-"
            for seed in seeds
        ]
        accuracies = [by_seed[seed]["accuracy"] for seed in seeds if seed in by_seed]
        if accuracies:
            cells.append(f"{mean(accuracies):.4f}")
            cells.append(
                f"{stdev(accuracies) / len(accuracies) ** 0.5:.4f}"
                if len(accuracies) > 1
                else "-"
            )
        else:
            cells += ["-", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def build_report(runs: list[dict[str, Any]], root: Path) -> str:
    seeds = sorted({run["seed"] for run in runs})
    students = sorted({run["student"] for run in runs if run["student"]})
    lines = [
        "# Filtered-trace student results",
        "",
        f"- Runs: `{root}`",
        f"- Student: " + ", ".join(f"`{s}`" for s in students),
        f"- Seeds: " + ", ".join(str(seed) for seed in seeds),
        f"- Completed runs: {len(runs)}",
        "",
        "Test accuracy on the GSM8K-platinum test split, which is identical "
        "across seeds. `traces` is the number of teacher traces the student "
        "was trained on, so unequal counts across levels are visible next to "
        "the accuracies. Standard error is the sample standard deviation over "
        "seeds divided by the square root of the seed count.",
        "",
    ]
    for mode in sorted({run["mode"] for run in runs}):
        for source in sorted({run["source"] for run in runs}):
            selected = [
                run for run in runs
                if run["mode"] == mode and run["source"] == source
            ]
            if not selected:
                continue
            lines += [f"## {source} (student mode: {mode})", ""]
            lines += format_table(selected, seeds)
            lines.append("")
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
