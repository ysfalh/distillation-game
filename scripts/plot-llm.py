#!/usr/bin/env python3
"""Plot LLM-trace results for GSM8K and MATH.

This script:
1. scans `outputs_from_cluster/outputs`
2. picks the newest run with `results.json` for each (provider, dataset, seed)
3. uses hardcoded fallback values for missing Gemini/MATH runs
4. plots one figure per dataset in the style requested by the user

Bars shown in each figure:
- Base
- SFT on Q&A
- Gemini
- Claude
- GPT

`Gemini`, `Claude`, and `GPT` correspond to trace-SFT runs (`traces/naive`).
`Base` and `SFT on Q&A` are averaged per seed across available providers, then
aggregated across the three seeds with standard error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_RE = re.compile(
    r"^real_(?P<provider>[^_]+)_(?P<dataset>gsm8k|math)_traces_seed(?P<seed>\d+)_(?P<timestamp>\d{8}_\d{6})$"
)

PROVIDERS = ["gemini", "claude", "openai"]
PROVIDER_BAR_LABELS = {"gemini": "Gemini", "claude": "Claude", "openai": "GPT"}
PROVIDER_PRINT_LABELS = {"gemini": "gemini", "claude": "claude", "openai": "openai"}
DATASET_LABELS = {"gsm8k": "GSM8K", "math": "MATH"}
EXPECTED_SEEDS = [42, 123, 456]

ROW_TO_LABEL = {
    ("base", None, None): "Base",
    ("qa", "naive", None): "SFT on Q&A",
    ("traces", "naive", None): "Trace SFT",
}

PLOT_BAR_ORDER = ["Base", "SFT on Q&A", "Gemini", "Claude", "GPT"]
BAR_COLOR = "#79B984"
EDGE_COLOR = "#335B3E"

# Hardcoded fallbacks supplied by the user for missing Gemini/MATH runs.
HARDCODED_FALLBACKS = {
    ("math", "gemini", 42): {
        "Base": 0.0104,
        "SFT on Q&A": 0.0968,
        "Trace SFT": 0.1248,
    },
    ("math", "gemini", 123): {
        "Base": 0.0098,
        "SFT on Q&A": 0.1040,
        "Trace SFT": 0.1182,
    },
    ("math", "gemini", 456): {
        "Base": 0.0106,
        "SFT on Q&A": 0.1026,
        "Trace SFT": 0.1198,
    },
}


plt.rcParams.update(
    {
        "text.usetex": shutil.which("latex") is not None,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "Latin Modern Roman",
            "DejaVu Serif",
            "cmr10",
        ],
        "font.size": 16,
        "axes.unicode_minus": False,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot latest LLM-trace accuracy figures.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs_from_cluster/outputs"),
        help="Directory containing real_* run folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_from_cluster"),
        help="Directory for generated plot PDFs and summary JSON.",
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math", "all"],
        default="all",
        help="Which dataset(s) to plot.",
    )
    return parser.parse_args()


def mean_stderr(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    sample_var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    stderr = math.sqrt(sample_var) / math.sqrt(len(values))
    return mean, stderr


def format_percent(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def find_runs(
    runs_root: Path,
) -> tuple[dict[tuple[str, str, int], dict], dict[tuple[str, str, int], list[dict]]]:
    latest_completed: dict[tuple[str, str, int], dict] = {}
    observed: dict[tuple[str, str, int], list[dict]] = defaultdict(list)

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue

        meta = match.groupdict()
        key = (meta["provider"], meta["dataset"], int(meta["seed"]))
        candidate = {
            "provider": meta["provider"],
            "dataset": meta["dataset"],
            "seed": int(meta["seed"]),
            "timestamp": meta["timestamp"],
            "run_dir": run_dir,
            "results_path": run_dir / "results.json",
            "has_results": (run_dir / "results.json").exists(),
        }
        observed[key].append(candidate)

        if not candidate["has_results"]:
            continue

        current = latest_completed.get(key)
        if current is None or candidate["timestamp"] > current["timestamp"]:
            latest_completed[key] = candidate

    return latest_completed, observed


def load_actual_results(latest_completed: dict[tuple[str, str, int], dict]) -> dict:
    per_seed_provider: dict[str, dict[str, dict[int, dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))

    for (provider, dataset, seed), run in latest_completed.items():
        with run["results_path"].open() as f:
            results = json.load(f)

        values: dict[str, float] = {}
        for row in results:
            label = ROW_TO_LABEL.get((row.get("source"), row.get("mode"), row.get("beta_s")))
            if label is not None:
                values[label] = float(row["accuracy"])
        per_seed_provider[dataset][provider][seed] = values

    return per_seed_provider


def apply_fallbacks(per_seed_provider: dict) -> list[dict[str, str | int]]:
    used_fallbacks: list[dict[str, str | int]] = []
    for (dataset, provider, seed), values in HARDCODED_FALLBACKS.items():
        if seed not in per_seed_provider[dataset][provider]:
            per_seed_provider[dataset][provider][seed] = dict(values)
            used_fallbacks.append(
                {
                    "dataset": dataset,
                    "provider": provider,
                    "seed": seed,
                }
            )
    return used_fallbacks


def compute_plot_stats(per_seed_provider: dict) -> dict:
    plot_stats: dict[str, dict[str, dict[str, float | int | None]]] = {}

    for dataset in DATASET_LABELS:
        dataset_stats: dict[str, dict[str, float | int | None]] = {}

        # Baselines: average within each seed across providers, then aggregate over seeds.
        for baseline_label in ("Base", "SFT on Q&A"):
            seed_level_values: list[float] = []
            for seed in EXPECTED_SEEDS:
                values_for_seed = []
                for provider in PROVIDERS:
                    provider_seed = per_seed_provider.get(dataset, {}).get(provider, {}).get(seed, {})
                    if baseline_label in provider_seed:
                        values_for_seed.append(provider_seed[baseline_label])
                if values_for_seed:
                    seed_level_values.append(sum(values_for_seed) / len(values_for_seed))
            mean, stderr = mean_stderr(seed_level_values)
            dataset_stats[baseline_label] = {"mean": mean, "stderr": stderr, "n": len(seed_level_values)}

        # Provider bars: use trace SFT over the three seeds.
        for provider in PROVIDERS:
            trace_values = []
            for seed in EXPECTED_SEEDS:
                provider_seed = per_seed_provider.get(dataset, {}).get(provider, {}).get(seed, {})
                if "Trace SFT" in provider_seed:
                    trace_values.append(provider_seed["Trace SFT"])
            mean, stderr = mean_stderr(trace_values)
            dataset_stats[PROVIDER_BAR_LABELS[provider]] = {"mean": mean, "stderr": stderr, "n": len(trace_values)}

        plot_stats[dataset] = dataset_stats

    return plot_stats


def compute_missing_without_fallback(
    observed: dict[tuple[str, str, int], list[dict]],
    latest_completed: dict[tuple[str, str, int], dict],
) -> list[dict[str, str | int | None]]:
    missing: list[dict[str, str | int | None]] = []
    for dataset in DATASET_LABELS:
        for provider in PROVIDERS:
            for seed in EXPECTED_SEEDS:
                actual_key = (provider, dataset, seed)
                fallback_key = (dataset, provider, seed)
                if actual_key in latest_completed or fallback_key in HARDCODED_FALLBACKS:
                    continue

                observed_runs = sorted(observed.get(actual_key, []), key=lambda item: item["timestamp"])
                newest = observed_runs[-1] if observed_runs else None
                missing.append(
                    {
                        "dataset": dataset,
                        "provider": provider,
                        "seed": seed,
                        "newest_timestamp": None if newest is None else newest["timestamp"],
                        "newest_run_dir": None if newest is None else newest["run_dir"].name,
                    }
                )
    return missing


def save_summary_json(
    out_path: Path,
    latest_completed: dict[tuple[str, str, int], dict],
    per_seed_provider: dict,
    plot_stats: dict,
    used_fallbacks: list[dict[str, str | int]],
) -> None:
    payload = {
        "selected_runs": {
            f"{dataset}/{provider}/seed{seed}": {
                "timestamp": run["timestamp"],
                "run_dir": str(run["run_dir"]),
            }
            for (provider, dataset, seed), run in sorted(latest_completed.items())
        },
        "used_fallbacks": used_fallbacks,
        "per_seed_provider": {
            dataset: {
                provider: {str(seed): values for seed, values in sorted(provider_data.items())}
                for provider, provider_data in sorted(dataset_data.items())
            }
            for dataset, dataset_data in sorted(per_seed_provider.items())
        },
        "plot_stats": plot_stats,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def plot_dataset(dataset: str, plot_stats: dict, out_dir: Path) -> Path:
    labels = PLOT_BAR_ORDER
    stats = plot_stats[dataset]
    means = [stats[label]["mean"] * 100.0 if stats[label]["mean"] is not None else np.nan for label in labels]
    errors = [stats[label]["stderr"] * 100.0 if stats[label]["stderr"] is not None else 0.0 for label in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 5.9))
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        capsize=4,
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
        zorder=3,
    )

    ax.set_title(f"Llama-3.2-3B Performance on {DATASET_LABELS[dataset]}")
    ax.set_ylabel(f"Test Accuracy on {DATASET_LABELS[dataset]} (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    ymax = max(value + err for value, err in zip(means, errors) if not np.isnan(value))
    ax.set_ylim(0, max(5.0, ymax * 1.28))

    for idx, (bar, value, err) in enumerate(zip(bars, means, errors)):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + err + max(ax.get_ylim()[1] * 0.015, 0.2),
            format_percent(value),
            ha="center",
            va="bottom",
            fontsize=15,
        )

    fig.tight_layout()

    pdf_path = out_dir / f"llm_accuracy_{dataset}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def print_summary(
    latest_completed: dict[tuple[str, str, int], dict],
    used_fallbacks: list[dict[str, str | int]],
    missing_without_fallback: list[dict[str, str | int | None]],
    plot_stats: dict,
) -> None:
    print("Selected latest completed runs:")
    for (provider, dataset, seed), run in sorted(latest_completed.items()):
        print(
            f"  {dataset:5s} | {provider:6s} | seed={seed:3d} | "
            f"{run['timestamp']} | {run['run_dir'].name}"
        )

    if used_fallbacks:
        print("\nUsing hardcoded fallbacks:")
        for item in used_fallbacks:
            print(
                f"  {item['dataset']:5s} | {item['provider']:6s} | seed={item['seed']:3d}"
            )

    if missing_without_fallback:
        print("\nMissing results.json with no fallback:")
        for item in missing_without_fallback:
            if item["newest_run_dir"] is None:
                print(
                    f"  {item['dataset']:5s} | {item['provider']:6s} | seed={item['seed']:3d} | no run directory found"
                )
            else:
                print(
                    f"  {item['dataset']:5s} | {item['provider']:6s} | seed={item['seed']:3d} | "
                    f"newest incomplete run: {item['newest_timestamp']} | {item['newest_run_dir']}"
                )

    print("\nPlot values:")
    for dataset in ("gsm8k", "math"):
        print(f"  {DATASET_LABELS[dataset]}:")
        for label in PLOT_BAR_ORDER:
            stat = plot_stats[dataset][label]
            mean = stat["mean"]
            stderr = stat["stderr"]
            n = stat["n"]
            if mean is None:
                print(f"    {label:10s} | missing")
            else:
                print(
                    f"    {label:10s} | mean={mean * 100:.2f}% | stderr={stderr * 100:.2f}% | n={n}"
                )


def main() -> None:
    args = parse_args()
    latest_completed, observed = find_runs(args.runs_root)
    if not latest_completed and not HARDCODED_FALLBACKS:
        raise SystemExit(f"No completed runs with results.json found under {args.runs_root}")

    per_seed_provider = load_actual_results(latest_completed)
    used_fallbacks = apply_fallbacks(per_seed_provider)
    plot_stats = compute_plot_stats(per_seed_provider)
    missing_without_fallback = compute_missing_without_fallback(observed, latest_completed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "llm_accuracy_summary.json"
    save_summary_json(summary_path, latest_completed, per_seed_provider, plot_stats, used_fallbacks)

    datasets = ["gsm8k", "math"] if args.dataset == "all" else [args.dataset]
    generated_paths: list[Path] = []
    for dataset in datasets:
        pdf_path = plot_dataset(dataset, plot_stats, args.out_dir)
        generated_paths.append(pdf_path)

    for path in generated_paths:
        print(f"Saved plot: {path}")
    print(f"Saved summary: {summary_path}\n")
    print_summary(latest_completed, used_fallbacks, missing_without_fallback, plot_stats)


if __name__ == "__main__":
    main()
