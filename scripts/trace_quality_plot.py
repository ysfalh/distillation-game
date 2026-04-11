"""
Plot PMF of trace-quality scores (1–5) for Standard vs PoE vs ADS.

Usage:
    python trace_quality_plot.py [gsm8k | math]
"""

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

DATASETS = {
    "gsm8k": {
        "standard": "gsm8k_output_small/analysis/trace_quality_standard.json",
        "poe": "gsm8k_output_small/analysis/trace_quality_poe.json",
        "ads": "gsm8k_output_small/analysis/trace_quality_ads.json",
        "out": "gsm8k_output_small/analysis/trace_quality_pmf.pdf",
    },
    "math": {
        "standard": "math_output_small/analysis/trace_quality_standard.json",
        "poe": "math_output_small/analysis/trace_quality_poe.json",
        "ads": "math_output_small/analysis/trace_quality_ads.json",
        "out": "math_output_small/analysis/trace_quality_pmf.pdf",
    },
}


def load_scores(path):
    with open(path) as f:
        data = json.load(f)
    return [item["score"] for item in data if item["score"] is not None]


def compute_pmf(scores, bins=range(1, 6)):
    counts = np.array([scores.count(b) for b in bins])
    return counts / counts.sum()


def plot(dataset_name):
    paths = DATASETS[dataset_name]

    std_scores = load_scores(paths["standard"])
    poe_scores = load_scores(paths["poe"])
    ads_scores = load_scores(paths["ads"])

    std_pmf = compute_pmf(std_scores)
    poe_pmf = compute_pmf(poe_scores)
    ads_pmf = compute_pmf(ads_scores)

    x = np.arange(1, 6)
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x - width, std_pmf, width, label="Standard", color="#55A868", edgecolor="black", linewidth=0.5)
    ax.bar(x,         poe_pmf, width, label="PoE",      color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, ads_pmf, width, label="ADS",      color="#DD8452", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Trace Quality Score")
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.legend()
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(paths["out"], bbox_inches="tight")
    print(f"Saved to {paths['out']}")
    print(f"  Standard  mean={np.mean(std_scores):.2f}  (n={len(std_scores)})")
    print(f"  PoE       mean={np.mean(poe_scores):.2f}  (n={len(poe_scores)})")
    print(f"  ADS       mean={np.mean(ads_scores):.2f}  (n={len(ads_scores)})")


if __name__ == "__main__":
    for name in DATASETS:
        plot(name)
