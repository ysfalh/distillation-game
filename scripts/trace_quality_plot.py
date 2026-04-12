"""
Plot PMF of trace-quality scores (1–5) for Standard vs PoE vs ADS.

Usage:
    python trace_quality_plot.py [gsm8k] [math]
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

SEEDS = ("42", "43", "44")
BINS = np.arange(1, 6)
METHODS = (
    ("standard", "Standard", "#55A868"),
    ("poe", "PoE", "#4C72B0"),
    ("ads", "ADS", "#DD8452"),
)

DATASETS = {
    "gsm8k": {
        "out": "outputs/trace-quality-plot/gsm8k_trace_quality_pmf.pdf",
    },
    "math": {
        "out": "outputs/trace-quality-plot/math_trace_quality_pmf.pdf",
    },
}


def load_scores(path):
    with open(path) as f:
        data = json.load(f)
    return [item["score"] for item in data if item["score"] is not None]


def compute_pmf(scores, bins=BINS):
    counts = np.array([scores.count(score) for score in bins], dtype=float)
    return counts / counts.sum()


def resolve_seed_dir(seed):
    candidates = [
        f"plot-quality-seed{seed}",
        f"trace-quality-seed{seed}",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a seed directory for seed {seed}. "
        f"Tried: {', '.join(candidates)}"
    )


def load_seed_scores(dataset_name, method_key):
    scores_by_seed = []
    for seed in SEEDS:
        seed_dir = resolve_seed_dir(seed)
        path = os.path.join(seed_dir, dataset_name, f"trace_quality_{method_key}.json")
        scores_by_seed.append(load_scores(path))
    return scores_by_seed


def summarize_method(dataset_name, method_key):
    seed_scores = load_seed_scores(dataset_name, method_key)
    seed_pmfs = np.array([compute_pmf(scores) for scores in seed_scores])
    mean_pmf = seed_pmfs.mean(axis=0)
    if len(seed_pmfs) > 1:
        se_pmf = seed_pmfs.std(axis=0, ddof=1) / np.sqrt(len(seed_pmfs))
    else:
        se_pmf = np.zeros_like(mean_pmf)

    seed_mean_scores = np.array([np.mean(scores) for scores in seed_scores], dtype=float)
    if len(seed_mean_scores) > 1:
        se_mean_score = seed_mean_scores.std(ddof=1) / np.sqrt(len(seed_mean_scores))
    else:
        se_mean_score = 0.0

    return {
        "seed_scores": seed_scores,
        "mean_pmf": mean_pmf,
        "se_pmf": se_pmf,
        "mean_score": seed_mean_scores.mean(),
        "se_mean_score": se_mean_score,
    }


def plot(dataset_name):
    paths = DATASETS[dataset_name]
    summaries = {
        method_key: summarize_method(dataset_name, method_key)
        for method_key, _, _ in METHODS
    }

    x = np.arange(1, 6)
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    offsets = (-width, 0.0, width)
    for offset, (method_key, label, color) in zip(offsets, METHODS):
        summary = summaries[method_key]
        ax.bar(
            x + offset,
            summary["mean_pmf"],
            width,
            yerr=summary["se_pmf"],
            capsize=3,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )

    ax.set_xlabel("Trace Quality Score")
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.legend()
    ax.set_ylim(0, None)

    fig.tight_layout()
    os.makedirs(os.path.dirname(paths["out"]), exist_ok=True)
    fig.savefig(paths["out"], bbox_inches="tight")
    print(f"Saved to {paths['out']}")
    for method_key, label, _ in METHODS:
        summary = summaries[method_key]
        seed_sizes = [len(scores) for scores in summary["seed_scores"]]
        print(
            f"  {label:<8} mean score={summary['mean_score']:.2f} ± {summary['se_mean_score']:.2f} "
            f"(seed sizes={seed_sizes})"
        )
        for score, pmf, se in zip(BINS, summary["mean_pmf"], summary["se_pmf"]):
            print(f"    score={score}: pmf={pmf:.4f} ± {se:.4f}")


if __name__ == "__main__":
    dataset_names = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS)
    invalid = [name for name in dataset_names if name not in DATASETS]
    if invalid:
        raise SystemExit(
            f"Unknown dataset(s): {', '.join(invalid)}. "
            f"Expected one or more of: {', '.join(DATASETS)}"
        )
    for name in dataset_names:
        plot(name)
