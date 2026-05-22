"""
Score reasoning traces using Claude API (absolute grading, 1-5) and plot results.

Reads raw train JSONs from (relative to repo root):
    plot-quality/gsm8k/train_standard.json
    plot-quality/gsm8k/train_poe_gamma_0.7.json
    plot-quality/gsm8k/train_antidistillation_lam_0.055.json
    plot-quality/math/train_*.json (same three methods)

Writes scored JSONs and PDFs under the same plot-quality/{gsm8k,math}/ directories.

Usage:
    python trace_quality_llm.py [--datasets gsm8k math] [--max-examples 0] [--plot-only]
"""

API_KEY = 'fill-me'

import os
import json
import time
import random
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_path(rel: str) -> str:
    """Resolve path relative to strategic-distillation repo root."""
    return str(REPO_ROOT / rel)

import numpy as np

np.random.seed(42)
random.seed(42)  # subsampling uses random.shuffle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anthropic import Anthropic

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

SYSTEM_PROMPT = (
    "You are a judge for the quality of reasoning traces. "
    "You will be given a math problem and a reasoning trace. "
    "Score the trace on a 1-5 scale using the rubric below.\n\n"
    "### Rubric: Auditability of Reasoning\n\n"
    "**Criteria**: Can a human reader audit the reasoning in this trace? "
    "For each step, assess whether it is (a) identifiable as a distinct "
    "reasoning step, (b) relevant to the problem, and (c) checkable against "
    "the previous step or the problem statement. Verbose but coherent traces "
    "and concise traces should score equally if both are auditable.\n\n"
    "**Score 1** – No auditable reasoning. The trace is dominated by "
    "non-reasoning content (repeated tokens, garbled text, or irrelevant "
    "material). A reader cannot identify any checkable steps.\n\n"
    "**Score 2** – Few auditable steps. Some reasoning is present but is "
    "interleaved with substantial non-reasoning content (filler tokens, "
    "irrelevant tangents, or corrupted text) making it unclear which parts "
    "to trust.\n\n"
    "**Score 3** – Partially auditable. The core reasoning steps are "
    "identifiable but some steps lack clear justification, or the reader "
    "must ignore non-trivial amounts of irrelevant content to follow the "
    "argument.\n\n"
    "**Score 4** – Mostly auditable. Nearly every step is identifiable, "
    "relevant, and checkable. Minor issues such as a redundant restatement "
    "or one unclear transition do not prevent verification.\n\n"
    "**Score 5** – Fully auditable. Every step is identifiable, relevant "
    "to the problem, and independently checkable. The trace may be long or "
    "short — what matters is that no step requires guesswork to verify.\n\n"
    "Use the score_trace tool to submit your auditability score."
)

TOOLS = [
    {
        "name": "score_trace",
        "description": "Submit your auditability score for the reasoning trace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feedback": {
                    "type": "string",
                    "description": "Short feedback paragraph.",
                },
                "score": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                    "description": "Auditability score (1-5).",
                },
            },
            "required": ["feedback", "score"],
        },
    }
]

DATASETS = {
    "gsm8k": {
        "standard": "plot-quality/gsm8k/train_standard.json",
        "poe":      "plot-quality/gsm8k/train_poe_gamma_0.7.json",
        "ads":      "plot-quality/gsm8k/train_antidistillation_lam_0.055.json",
    },
    "math": {
        "standard": "plot-quality/math/train_standard.json",
        "poe":      "plot-quality/math/train_poe_gamma_0.75.json",
        "ads":      "plot-quality/math/train_antidistillation_lam_0.08.json",
    },
}


def build_user_message(prompt: str, trace: str) -> str:
    return (
        f"### Problem\n{prompt}\n\n"
        f"### Reasoning Trace\n{trace}\n\n"
        "Please evaluate the auditability of this reasoning trace."
    )


def score_traces(
    client: Anthropic,
    dataset_name: str,
    method_name: str,
    input_path: str,
    output_path: str,
    model: str,
    max_examples: int,
):
    with open(input_path) as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    already_done = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        already_done = {r["example_id"] for r in results}
        print(f"  Resuming: {len(already_done)} already scored")

    if max_examples > 0:
        remaining = max_examples - len(already_done)
        if remaining <= 0:
            print(f"  Already have {len(already_done)} >= {max_examples}, nothing to do")
            return
        unscored = [item for item in data if item.get("example_id", None) not in already_done]
        random.shuffle(unscored)
        data = unscored[:remaining]
    else:
        data = [item for item in data if item.get("example_id", None) not in already_done]

    if not data:
        print(f"  All examples already scored, nothing to do")
        return

    new_scored = 0
    for i, item in enumerate(data):
        example_id = item.get("example_id", f"idx_{i}")

        user_msg = build_user_message(item["prompt"], item["trace"])

        for attempt in range(5):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                    tools=TOOLS,
                    tool_choice={"type": "tool", "name": "score_trace"},
                )
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"    API retry {attempt+1} after error: {e}  (wait {wait}s)")
                time.sleep(wait)
        else:
            print(f"    SKIP {example_id} after 5 API retries")
            continue

        block = next((b for b in response.content if b.type == "tool_use"), None)
        if block is None:
            print(f"    SKIP {example_id} — no tool_use block (stop_reason: {response.stop_reason})")
            continue

        score = block.input["score"]
        feedback = block.input["feedback"]

        results.append({
            "example_id": example_id,
            "method": item.get("method", method_name),
            "trace": item.get("trace", ""),
            "score": score,
            "feedback": feedback,
        })
        new_scored += 1

        print(f"  [{dataset_name}/{method_name}] {example_id} | score: {score}  ({i+1}/{len(data)})")

        if new_scored % 20 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    if new_scored > 0:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"  Wrote {len(results)} scores to {output_path} ({new_scored} new)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def load_scores(path):
    with open(path) as f:
        data = json.load(f)
    return [item["score"] for item in data if item["score"] is not None]


def compute_pmf(scores, bins=range(1, 6)):
    counts = np.array([scores.count(b) for b in bins])
    return counts / counts.sum()


def plot(dataset_name):
    score_dir = _repo_path(f"plot-quality/{dataset_name}")
    paths = {
        "standard": os.path.join(score_dir, "trace_quality_standard.json"),
        "poe":      os.path.join(score_dir, "trace_quality_poe.json"),
        "ads":      os.path.join(score_dir, "trace_quality_ads.json"),
    }
    out_path = os.path.join(score_dir, "trace_quality_pmf.pdf")

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
    title = "GSM8K" if dataset_name == "gsm8k" else "MATH"
    ax.set_title(f"Trace Quality — {title} (Claude)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.legend()
    ax.set_ylim(0, None)

    fig.tight_layout()
    os.makedirs(score_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")
    print(f"  Standard  mean={np.mean(std_scores):.2f}  (n={len(std_scores)})")
    print(f"  PoE       mean={np.mean(poe_scores):.2f}  (n={len(poe_scores)})")
    print(f"  ADS       mean={np.mean(ads_scores):.2f}  (n={len(ads_scores)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["gsm8k", "math"],
        choices=["gsm8k", "math"],
    )
    parser.add_argument("--model", default="claude-opus-4-0-20250514")
    parser.add_argument(
        "--max-examples", type=int, default=0,
        help="Cap per file (0 = all)",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip scoring, only generate plots from existing JSONs",
    )
    args = parser.parse_args()

    if not args.plot_only:
        client = Anthropic(api_key=API_KEY)
        for ds in args.datasets:
            methods = DATASETS[ds]
            for method_key, rel_in in methods.items():
                input_path = _repo_path(rel_in)
                output_path = _repo_path(f"plot-quality/{ds}/trace_quality_{method_key}.json")
                print(f"\n=== {ds} / {method_key} ===")
                print(f"  Input:  {input_path}")
                print(f"  Output: {output_path}")
                score_traces(client, ds, method_key, input_path, output_path, args.model, args.max_examples)

    print("\n=== Plotting ===")
    for ds in args.datasets:
        plot(ds)


if __name__ == "__main__":
    main()
