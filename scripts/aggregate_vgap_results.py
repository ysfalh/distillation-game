#!/usr/bin/env python3
"""Report whether v_gap predicts what the Llama attacker learns.

Collects the per-bin student runs under `outputs/vgap_bins/`, averages each
bin over its seeds, subtracts the untrained base model's accuracy to get the
attacker's gain, and correlates bin rank with gain.

Writes RESULTS.md, a summary json, and a plot of gain against v_gap bin.

Usage:
    python scripts/aggregate_vgap_results.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vgap_stats import spearman, spearman_permutation_p


DEFAULT_RESULTS_DIR = Path("outputs/vgap_bins")
DEFAULT_BASELINE_MODEL = "meta-llama_Llama-3.2-3B"
# Matches make_vgap_bins.TOKEN_RATIO_WARN: above this the bins differ enough in
# training tokens that a trend across them is not attributable to v_gap alone.
TOKEN_RATIO_WARN = 1.10


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mu = _mean(values)
    return sqrt(sum((v - mu) ** 2 for v in values) / (n - 1) / n)


def student_accuracy(results_path: Path) -> float | None:
    rows = json.loads(results_path.read_text())
    for row in rows:
        if str(row.get("eval_model", "")).startswith("student_"):
            return float(row["accuracy"])
    return None


def load_runs(results_dir: Path) -> list[dict[str, Any]]:
    """One record per finished (metric, bin, seed) student run."""
    runs: list[dict[str, Any]] = []
    for results_path in sorted(results_dir.glob("*/bin_*/seed_*/results.json")):
        run_dir = results_path.parent
        accuracy = student_accuracy(results_path)
        if accuracy is None:
            continue
        bin_dir = run_dir.parent
        metric = bin_dir.parent.name
        manifest_path = run_dir / "run_manifest.json"
        bin_meta: dict[str, Any] = {}
        if manifest_path.exists():
            input_dir = Path(json.loads(manifest_path.read_text()).get("input_dir", ""))
            bin_manifest = input_dir / "bin_manifest.json"
            if bin_manifest.exists():
                bin_meta = json.loads(bin_manifest.read_text())
        runs.append({
            "metric": metric,
            "bin": int(bin_dir.name.removeprefix("bin_")),
            "seed": int(run_dir.name.removeprefix("seed_")),
            "accuracy": accuracy,
            "bin_meta": bin_meta,
        })
    return runs


def baseline_accuracy(base_eval_dir: Path, model_dir_name: str) -> tuple[float | None, list[int]]:
    """Untrained student accuracy, averaged over whatever seeds were evaluated."""
    accuracies: list[float] = []
    seeds: list[int] = []
    for results_path in sorted(base_eval_dir.glob(f"seed_*/{model_dir_name}/results.json")):
        summary = json.loads(results_path.read_text())
        if summary.get("accuracy") is None:
            continue
        accuracies.append(float(summary["accuracy"]))
        seeds.append(int(results_path.parent.parent.name.removeprefix("seed_")))
    if not accuracies:
        return None, []
    return _mean(accuracies), seeds


def summarize(runs: list[dict[str, Any]], baseline: float | None) -> dict[str, Any]:
    """Per-bin accuracy and gain, plus the rank correlation, for each metric."""
    metrics: dict[str, Any] = {}
    for metric in sorted({run["metric"] for run in runs}):
        metric_runs = [run for run in runs if run["metric"] == metric]
        bins = []
        for bin_index in sorted({run["bin"] for run in metric_runs}):
            bin_runs = sorted((r for r in metric_runs if r["bin"] == bin_index), key=lambda r: r["seed"])
            accuracies = [r["accuracy"] for r in bin_runs]
            meta = next((r["bin_meta"] for r in bin_runs if r["bin_meta"]), {})
            mean_accuracy = _mean(accuracies)
            bins.append({
                "bin": bin_index,
                "seeds": {r["seed"]: r["accuracy"] for r in bin_runs},
                "n_seeds": len(accuracies),
                "accuracy_mean": mean_accuracy,
                "accuracy_stderr": _stderr(accuracies),
                "gain": mean_accuracy - baseline if baseline is not None else float("nan"),
                "n_traces": meta.get("n"),
                "total_tokens": meta.get("total_response_tokens"),
                "vgap_median": meta.get("vgap_median"),
                "vgap_p25": meta.get("vgap_p25"),
                "vgap_p75": meta.get("vgap_p75"),
                "vgap_mean": meta.get("vgap_mean"),
                "vgap_min": meta.get("vgap_min"),
                "vgap_max": meta.get("vgap_max"),
                "trace_accuracy": meta.get("trace_accuracy"),
                "response_tokens_median": meta.get("response_tokens_median"),
            })

        ranks = [float(b["bin"]) for b in bins]
        accuracies = [b["accuracy_mean"] for b in bins]
        rho = spearman(ranks, accuracies)
        p_value, exact = spearman_permutation_p(ranks, accuracies)

        per_seed = {}
        for seed in sorted({run["seed"] for run in metric_runs}):
            points = [(r["bin"], r["accuracy"]) for r in metric_runs if r["seed"] == seed]
            if len(points) >= 3:
                per_seed[seed] = spearman([float(b) for b, _ in points], [a for _, a in points])

        pooled = [(r["bin"], r["accuracy"]) for r in metric_runs]
        any_meta = next((r["bin_meta"] for r in metric_runs if r["bin_meta"]), {})
        counts = [b["n_traces"] for b in bins if isinstance(b["n_traces"], int)]
        tokens = [b["total_tokens"] for b in bins if isinstance(b["total_tokens"], int)]
        metrics[metric] = {
            "bins": bins,
            "spearman_rho": rho,
            "spearman_p": p_value,
            "spearman_p_exact": exact,
            "spearman_rho_per_seed": per_seed,
            "spearman_rho_pooled": spearman([float(b) for b, _ in pooled], [a for _, a in pooled]),
            "n_points": len(bins),
            "equalize": any_meta.get("equalize"),
            "token_budget": any_meta.get("token_budget"),
            "aggregation": any_meta.get("metric", metric),
            "stratify": any_meta.get("stratify", "none"),
            "accuracy_ratio": _spread([b["trace_accuracy"] for b in bins]),
            "traces_equal": len(set(counts)) == 1 if counts else None,
            "token_ratio": max(tokens) / max(min(tokens), 1) if tokens else None,
        }
    return metrics


def missing_runs(runs: list[dict[str, Any]], expected_seeds: list[int]) -> list[str]:
    gaps: list[str] = []
    for metric in sorted({run["metric"] for run in runs}):
        bins = sorted({run["bin"] for run in runs if run["metric"] == metric})
        for bin_index in bins:
            have = {r["seed"] for r in runs if r["metric"] == metric and r["bin"] == bin_index}
            for seed in expected_seeds:
                if seed not in have:
                    gaps.append(f"{metric}/bin_{bin_index}/seed_{seed}")
    return gaps


def _spread(values: list[Any]) -> float | None:
    numeric = [v for v in values if isinstance(v, (int, float)) and v == v]
    if not numeric or min(numeric) <= 0:
        return None
    return max(numeric) / min(numeric)


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value != value:
        return "-"
    return format(value, spec)


def _fairness_notes(metrics: dict[str, Any]) -> list[str]:
    """State plainly whether the bins were comparable training sets."""
    lines: list[str] = []
    for metric, data in metrics.items():
        equalize = data.get("equalize")
        ratio = data.get("token_ratio")
        equal = data.get("traces_equal")
        matched = data.get("stratify", "none") != "none"
        if equalize is None or ratio is None:
            continue
        held = "length and teacher correctness" if data.get("stratify") == "length_correct" else "length"
        if matched and equal and ratio <= TOKEN_RATIO_WARN:
            lines.append(
                f"- `{metric}`: bins were matched on {held}, so they hold the same number of "
                f"traces, agree on tokens within {ratio:.2f}x, and carry the same share of "
                "correct traces. What differs between them is v_gap, which makes this arm the "
                "one to read causally."
            )
        elif equalize == "tokens":
            budget = data.get("token_budget")
            budget_note = f" at a common budget of {budget / 1000:.0f}k tokens" if budget else ""
            lines.append(
                f"- `{metric}`: bins were equalized by **response tokens**{budget_note}, so the "
                "amount of supervision is held fixed and the trace counts differ instead."
            )
        elif equal and ratio <= TOKEN_RATIO_WARN:
            lines.append(
                f"- `{metric}`: bins hold the same number of traces and their token totals "
                f"agree within {ratio:.2f}x, so the training budget is not a confound."
            )
        elif equal:
            note = (
                f"- `{metric}`: bins hold the same number of traces, but the fattest carries "
                f"**{ratio:.2f}x the tokens** of the leanest, because the summed gap grows with "
                "trace length. A trend across these bins mixes v_gap with training budget"
            )
            acc_ratio = data.get("accuracy_ratio")
            if acc_ratio and acc_ratio > 1.05:
                note += (
                    f", and their teacher accuracy also spans {acc_ratio:.2f}x, so supervision "
                    "quality differs too"
                )
            matched_arms = [m for m, d in metrics.items() if d.get("stratify", "none") != "none"]
            if matched_arms:
                note += f". Read it against {', '.join(f'`{m}`' for m in matched_arms)}, where both are held fixed."
            else:
                note += ". Treat the comparison as uncontrolled."
            lines.append(note)
        else:
            lines.append(
                f"- `{metric}`: trace counts differ across bins and token totals span "
                f"{ratio:.2f}x. Treat the comparison as uncontrolled."
            )
    if lines:
        lines.insert(0, "**Are the bins comparable?**")
        lines.insert(1, "")
        lines.append("")
    return lines


def format_report(
    *,
    metrics: dict[str, Any],
    baseline: float | None,
    baseline_seeds: list[int],
    expected_seeds: list[int],
    gaps: list[str],
    plot_name: str | None,
) -> str:
    lines = [
        "# Does v_gap predict what the attacker learns?",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.",
        "",
        "Every GSM8K training trace from the undefended teacher was scored with "
        "`v_gap = log P_teacher(y|x) - log P_proxy(y|x)`, the sequence-level form of "
        "the per-token gap the PoE defense mixes against, with the Qwen2.5-3B proxy "
        "from the run config. Traces were ranked by that score and cut into bins, "
        "and a fresh Llama-3.2-3B was LoRA-fine-tuned on each bin alone, so the "
        "students differ only in which part of the v_gap range they saw. Binning is "
        "on rank, so the tail of v_gap decides only which bin an extreme trace lands "
        "in, never where the splits fall.",
        "",
        "The summed gap grows with trace length, so splitting on it also splits on "
        "how many training tokens each student gets. Arms whose name ends in "
        "`matched` rank each trace only against others of its response-length decile "
        "and teacher correctness, which holds both fixed across bins and leaves v_gap "
        "as the only thing that varies. An unmatched arm answers the question as the "
        "defense poses it; a matched arm answers whether the gap itself is what "
        "matters. Where the two disagree, the difference is the length effect.",
        "",
    ]
    if baseline is not None:
        seed_note = ", ".join(str(s) for s in baseline_seeds)
        lines += [
            f"Gain is measured against the untrained Llama-3.2-3B, which scores "
            f"**{baseline:.4f}** on the same test split (seed {seed_note}).",
            "",
        ]
    else:
        lines += [
            "No base-model evaluation was found, so gains are blank. Produce one with "
            "`sbatch slurm_scripts/run-base-eval-gsm8k.sbatch` or pass `--baseline-acc`.",
            "",
        ]
    lines += [
        "Subtracting a constant baseline cannot change a rank correlation, so rho "
        "below is equally the correlation between bin rank and raw accuracy; the "
        "baseline only sets where zero sits on the plot.",
        "",
        "## Test performance by bin",
        "",
        "GSM8K test accuracy of the attacker trained on each bin, averaged over "
        "seeds. This is the headline result; the per-seed breakdown is further down.",
        "",
        "| arm | bin | v_gap median | traces | tokens | resp. tokens | test accuracy | s.e. | gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric, data in metrics.items():
        for b in data["bins"]:
            lines.append(
                "| " + " | ".join([
                    metric,
                    str(b["bin"] + 1),
                    _fmt(b["vgap_median"], ".2f"),
                    str(b["n_traces"]) if isinstance(b["n_traces"], int) else "-",
                    f"{b['total_tokens'] / 1000:.0f}k" if isinstance(b["total_tokens"], int) else "-",
                    _fmt(b["response_tokens_median"], ".0f"),
                    _fmt(b["accuracy_mean"]),
                    _fmt(b["accuracy_stderr"]),
                    _fmt(b["gain"], "+.4f"),
                ]) + " |"
            )
    lines.append("")
    lines += _fairness_notes(metrics)

    for metric, data in metrics.items():
        lines += [
            f"## Arm `{metric}`: {_arm_label(data, metric)}",
            "",
        ]
        seeds_seen = sorted({seed for b in data["bins"] for seed in b["seeds"]}) or expected_seeds
        header = "| bin | traces | v_gap median | v_gap IQR | trace acc | " + " | ".join(f"seed {s}" for s in seeds_seen)
        header += " | mean | stderr | gain |"
        lines.append(header)
        lines.append("| --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in seeds_seen) + " | ---: | ---: | ---: |")
        for b in data["bins"]:
            iqr = "-"
            if b["vgap_p25"] is not None and b["vgap_p75"] is not None:
                iqr = f"{b['vgap_p25']:.2f} to {b['vgap_p75']:.2f}"
            cells = [
                str(b["bin"]),
                _fmt(b["n_traces"], "d") if isinstance(b["n_traces"], int) else "-",
                _fmt(b["vgap_median"], ".2f"),
                iqr,
                _fmt(b["trace_accuracy"], ".3f"),
            ]
            cells += [_fmt(b["seeds"].get(seed)) for seed in seeds_seen]
            cells += [_fmt(b["accuracy_mean"]), _fmt(b["accuracy_stderr"]), _fmt(b["gain"], "+.4f")]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        p_note = ""
        if data["spearman_p"] == data["spearman_p"]:
            kind = "exact permutation" if data["spearman_p_exact"] else "permutation"
            p_note = f", {kind} p = {data['spearman_p']:.3f}"
        per_seed = ", ".join(f"seed {s}: {r:+.2f}" for s, r in data["spearman_rho_per_seed"].items())
        lines += [
            f"**Spearman rho between bin rank and gain: {data['spearman_rho']:+.3f}** "
            f"over {data['n_points']} bins{p_note}.",
            "",
            f"Per-seed rho ({per_seed}) shows how much of that is stable across seeds. "
            f"Pooling all bin-seed points gives rho = {data['spearman_rho_pooled']:+.3f}.",
            "",
        ]

    if plot_name:
        lines += [
            f"![attacker gain against median v_gap]({plot_name})",
            "",
            "The plot places each bin at its **median** v_gap on the x-axis, with "
            "horizontal whiskers spanning the interquartile range. Extremes in the "
            "tail therefore cannot stretch the axis. Vertical bars are the seed "
            "mean ± standard error; faint markers are the individual seeds. "
            "Spearman rho above each panel is still computed on bin *rank*, which "
            "is invariant to the v_gap scale.",
            "",
        ]

    lines += [
        "## Reading this",
        "",
        "A positive rho means students trained on higher-v_gap traces end up more "
        "accurate, which is the assumption PoE rests on: suppressing high-gap tokens "
        "should cost the attacker the most. A flat or negative rho means v_gap, as "
        "the proxy measures it, is not tracking what the attacker actually picks up.",
        "",
        f"With {len(next(iter(metrics.values()))['bins']) if metrics else 0} bins the "
        "permutation test is exact but weak: only a perfectly monotone ordering "
        "reaches p < 0.05. Treat a middling rho as suggestive, not established.",
        "",
        "Check `trace acc` before trusting the correlation. If it moves across bins, "
        "the students differ in how often their supervision was right, which is a "
        "confound for v_gap. Comparing the `sum` and `mean` binnings does the same "
        "job for trace length.",
        "",
        "Bins are cut on rank, so the tail of v_gap decides only which bin an "
        "extreme trace falls in, never where the splits are, and each bin is "
        "described by its median and interquartile range rather than a mean that "
        "one outlier could carry. Bin ranks are what rho is computed on, so the "
        "correlation is unaffected by the v_gap scale entirely.",
        "",
    ]

    if gaps:
        lines += [
            "## Missing runs",
            "",
            "These did not produce results.json, so their bins are averaged over "
            "fewer seeds:",
            "",
        ]
        lines += [f"- `{gap}`" for gap in gaps]
        lines.append("")
    return "\n".join(lines)


ARM_ORDER = ["sum", "mean", "sum_lenmatched", "mean_lenmatched", "sum_matched", "mean_matched"]
AGGREGATION_LABEL = {
    "sum": "v_gap summed over the response (the paper's V_gap)",
    "mean": "v_gap per response token (length-normalized)",
}
STRATIFY_LABEL = {
    "none": "ranked against all traces",
    "length": "ranked within a response-length decile",
    "length_correct": "ranked within a response-length decile and teacher correctness",
}
SEED_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def _arm_label(data: dict[str, Any], fallback: str) -> str:
    aggregation = AGGREGATION_LABEL.get(data.get("aggregation", ""), fallback)
    stratify = STRATIFY_LABEL.get(data.get("stratify", "none"), "")
    return f"{aggregation}, {stratify}" if stratify else aggregation


def _plot_title(data: dict[str, Any], arm: str) -> str:
    aggregation = AGGREGATION_LABEL.get(data.get("aggregation", ""), arm).replace(" (", "\n(")
    stratify = data.get("stratify", "none")
    if stratify == "none":
        return f"{aggregation}\nlength not controlled"
    held = "length + correctness matched" if stratify == "length_correct" else "length matched"
    return f"{aggregation}\n{held}"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value


def write_plot(metrics: dict[str, Any], path: Path, baseline: float | None) -> bool:
    """One panel per binning: median v_gap against attacker gain.

    X is the bin's median v_gap with IQR whiskers, so a single extreme score
    cannot stretch the axis. Y shows seed means ± s.e. with the individual
    seeds behind them. Spearman rho in the title is still on bin rank.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    order = [m for m in ARM_ORDER if m in metrics] + [m for m in metrics if m not in ARM_ORDER]
    fig, axes = plt.subplots(
        1,
        len(order),
        figsize=(6.0 * len(order), 5.0),
        sharey=True,
        squeeze=False,
    )
    seeds = sorted({
        seed for data in metrics.values() for b in data["bins"] for seed in b["seeds"]
    })

    for ax, metric in zip(axes[0], order):
        data = metrics[metric]
        bins = [b for b in data["bins"] if _finite(b.get("vgap_median"))]
        if not bins:
            # Fall back to bin rank if manifests were missing medians.
            bins = data["bins"]
            xs = [float(b["bin"] + 1) for b in bins]
            xerr_lo = xerr_hi = [0.0] * len(bins)
            xlabel = "v_gap bin (1 = lowest)"
        else:
            xs = [float(b["vgap_median"]) for b in bins]
            xerr_lo = [
                max(0.0, float(b["vgap_median"]) - float(b["vgap_p25"]))
                if _finite(b.get("vgap_p25")) else 0.0
                for b in bins
            ]
            xerr_hi = [
                max(0.0, float(b["vgap_p75"]) - float(b["vgap_median"]))
                if _finite(b.get("vgap_p75")) else 0.0
                for b in bins
            ]
            xlabel = "bin median v_gap (whiskers = IQR)"

        ys = [
            b["gain"] if baseline is not None else b["accuracy_mean"]
            for b in bins
        ]
        errs = [
            0.0 if not _finite(b["accuracy_stderr"]) else b["accuracy_stderr"]
            for b in bins
        ]
        offset = baseline if baseline is not None else 0.0

        # Seed jitter on x keeps overlapping markers readable without claiming
        # a different v_gap for each seed.
        span = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
        jitter = 0.012 * span
        for i, seed in enumerate(seeds):
            seed_x = []
            seed_y = []
            for j, b in enumerate(bins):
                if seed not in b["seeds"]:
                    continue
                seed_x.append(xs[j] + (i - (len(seeds) - 1) / 2) * jitter)
                seed_y.append(b["seeds"][seed] - offset)
            ax.scatter(
                seed_x,
                seed_y,
                s=28,
                alpha=0.55,
                color=SEED_COLORS[i % len(SEED_COLORS)],
                edgecolors="none",
                zorder=3,
                label=f"seed {seed}",
            )

        ax.errorbar(
            xs,
            ys,
            xerr=[xerr_lo, xerr_hi],
            yerr=errs,
            fmt="o-",
            markersize=8,
            linewidth=2.2,
            capsize=4,
            color="#1b1b1b",
            ecolor="#555555",
            elinewidth=1.2,
            zorder=5,
            label="bin mean ± s.e.",
        )
        for j, b in enumerate(bins):
            ax.annotate(
                f"bin {b['bin'] + 1}",
                (xs[j], ys[j]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="#333333",
            )

        if baseline is not None:
            ax.axhline(0.0, color="#888888", linewidth=0.9, linestyle="--", zorder=0)

        rho = data["spearman_rho"]
        subtitle = f"Spearman ρ = {rho:+.2f}"
        if _finite(data["spearman_p"]):
            subtitle += f"  (p = {data['spearman_p']:.3f})"
        ax.set_title(f"{_plot_title(data, metric)}\n{subtitle}", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="both", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0][0].set_ylabel(
        "GSM8K accuracy gain over\nuntrained Llama-3.2-3B"
        if baseline is not None
        else "GSM8K accuracy"
    )
    axes[0][-1].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Does v_gap predict what the Llama attacker learns?", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Root of the per-bin student runs.")
    parser.add_argument("--base-eval-dir", default="outputs/base_eval", help="Where the untrained-model eval landed.")
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL, help="Base-eval subdirectory to read.")
    parser.add_argument("--baseline-acc", default=None, type=float, help="Use this accuracy instead of the base eval.")
    parser.add_argument("--config", default="configs/gsm8k.yaml", help="Read the expected seeds from here.")
    parser.add_argument("--output", default=None, help="Defaults to <results-dir>/RESULTS.md.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No finished student runs under {results_dir}. Nothing to report.")

    expected_seeds: list[int] = []
    config_path = Path(args.config)
    if config_path.exists():
        try:
            import yaml

            expected_seeds = list(yaml.safe_load(config_path.read_text())["run"]["seeds"])
        except Exception:
            expected_seeds = []
    if not expected_seeds:
        expected_seeds = sorted({run["seed"] for run in runs})

    if args.baseline_acc is not None:
        baseline, baseline_seeds = args.baseline_acc, []
    else:
        baseline, baseline_seeds = baseline_accuracy(Path(args.base_eval_dir), args.baseline_model)

    metrics = summarize(runs, baseline)
    gaps = missing_runs(runs, expected_seeds)

    plot_path = results_dir / "vgap_gain.png"
    has_plot = write_plot(metrics, plot_path, baseline)

    report = format_report(
        metrics=metrics,
        baseline=baseline,
        baseline_seeds=baseline_seeds,
        expected_seeds=expected_seeds,
        gaps=gaps,
        plot_name=plot_path.name if has_plot else None,
    )
    output_path = Path(args.output) if args.output else results_dir / "RESULTS.md"
    output_path.write_text(report)
    (results_dir / "vgap_summary.json").write_text(
        json.dumps(
            {
                "baseline_accuracy": baseline,
                "baseline_seeds": baseline_seeds,
                "expected_seeds": expected_seeds,
                "n_runs": len(runs),
                "missing_runs": gaps,
                "metrics": metrics,
                "created_at": datetime.now().isoformat(),
            },
            indent=2,
        )
    )

    print(f"Wrote {output_path} from {len(runs)} student runs")
    if not has_plot:
        print("  matplotlib is not installed, so no plot was written")
    for metric, data in metrics.items():
        print(f"  v_gap {metric}: rho={data['spearman_rho']:+.3f} over {data['n_points']} bins")
    if gaps:
        print(f"  missing {len(gaps)} runs: {', '.join(gaps)}")


if __name__ == "__main__":
    main()
