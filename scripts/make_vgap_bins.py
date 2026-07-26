#!/usr/bin/env python3
"""Split scored traces into equal v_gap bins, one drop-in trace folder each.

Reads the scores written by `score_vgap.py` and emits, for every arm, K folders
that look exactly like `gsm8k_output_small`. Each holds only its bin's traces, so
the existing student runner trains on a bin with no changes:

    vgap/<dataset>/bins/<arm>/bin_0/train_standard.json
                                   /config_snapshot.yaml
                                   /holdout_standard_internal.json

An arm is a v_gap aggregation plus a choice of what to hold fixed across bins.
`sum` is the paper's V_gap, the total log-prob ratio over the response; `mean`
divides it by the response length. The summed gap grows with trace length, so
splitting on it also splits on length, and the top bin ends up with more training
tokens than the bottom one. The `_matched` arms rank each trace only against
others in its response-length decile (and, for `length_correct`, of the same
teacher correctness) and then draw equally from every group, which leaves the
bins with the same trace count, near-identical token totals, and the same share
of correct traces, so what remains between them is v_gap. Running an unmatched
arm alongside a matched one shows how much of any trend was length.

Bin 0 is the lowest v_gap.

Robustness, in the order it is applied:

1. Traces whose v_gap is not finite, whose response is empty, or (with
   --drop-truncated) whose scoring hit the length cap are excluded up front.
   A NaN would otherwise sort arbitrarily and silently corrupt a bin.
2. Traces are assigned by *rank*, never by value, so however heavy the tail of
   v_gap is, an extreme trace takes one slot in the top bin and cannot move a
   boundary.
3. Bins hold exactly the same number of traces, in the stratified arms too,
   since every stratum contributes the same count to each bin. The few leftover
   traces are dropped from the extremes of the ranking.
4. Bins are described by median and interquartile range, which one outlier
   cannot drag around, and the reported spread makes the tail visible.

Equal trace counts and equal token counts cannot both hold, since a bin's token
total is fixed once its traces are. `--equalize traces` (the default) keeps the
number of demonstrations and optimizer steps identical; `--equalize tokens`
instead subsamples each bin down to a common response-token budget, which keeps
the amount of supervision identical and lets the trace counts differ.

Usage:
    python scripts/make_vgap_bins.py --input-dir gsm8k_output_small --source standard
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vgap_stats import spearman


METRICS = {"sum": "vgap_sum", "mean": "vgap_mean"}
# An arm is a (metric, stratification) pair, and each one gets its own set of bin
# folders. Stratified arms rank traces only against others of similar length (and
# correctness), then take the same share of every stratum into every bin, which
# leaves the bins with matched length and correctness profiles while still
# differing in v_gap. That is the only way to hold trace count and token count
# fixed at the same time.
ARMS: dict[str, tuple[str, str]] = {
    "sum": ("sum", "none"),
    "mean": ("mean", "none"),
    "sum_lenmatched": ("sum", "length"),
    "mean_lenmatched": ("mean", "length"),
    "sum_matched": ("sum", "length_correct"),
    "mean_matched": ("mean", "length_correct"),
}
DEFAULT_ARMS = ["sum", "mean_matched"]
# Copied beside the train file so the student runner can read a bin folder the
# same way it reads the original trace directory.
PASSTHROUGH_FILES = ["config_snapshot.yaml", "holdout_standard_internal.json"]
# Above this ratio between the fattest and leanest bin, a trend across bins
# could be a training-budget effect rather than a v_gap effect.
TOKEN_RATIO_WARN = 1.10


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def filter_scores(
    scores: list[dict[str, Any]],
    *,
    drop_truncated: bool,
    min_response_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep only traces that can be ranked and trained on honestly."""
    kept: list[dict[str, Any]] = []
    excluded = {"non_finite": 0, "empty_response": 0, "too_short": 0, "truncated": 0}
    for score in scores:
        if not (_finite(score.get("vgap_sum")) and _finite(score.get("vgap_mean"))):
            excluded["non_finite"] += 1
            continue
        tokens = int(score.get("n_response_tokens", 0))
        if tokens <= 0:
            excluded["empty_response"] += 1
            continue
        # vgap_mean divides by the token count, so a handful of tokens produces a
        # wild per-token value off almost no evidence.
        if tokens < min_response_tokens:
            excluded["too_short"] += 1
            continue
        if drop_truncated and score.get("truncated"):
            excluded["truncated"] += 1
            continue
        kept.append(score)
    return kept, excluded


def stratum_keys(
    scores: list[dict[str, Any]],
    mode: str,
    n_strata: int,
) -> list[Any]:
    """Group traces so that ranking happens only within comparable groups.

    `length` puts each trace in a response-length decile; `length_correct` also
    splits on whether the teacher got the answer right. Bins then draw equally
    from every group, so they cannot differ in length or in supervision quality.
    """
    if mode == "none":
        return [0] * len(scores)
    tokens = [int(s["n_response_tokens"]) for s in scores]
    order = sorted(range(len(scores)), key=lambda i: tokens[i])
    decile = [0] * len(scores)
    for rank, index in enumerate(order):
        decile[index] = min(rank * n_strata // len(scores), n_strata - 1)
    if mode == "length":
        return decile
    return [(d, bool(s["correct"])) for d, s in zip(decile, scores)]


def assign_bins(
    values: list[float],
    n_bins: int,
    strata: list[Any] | None = None,
) -> tuple[list[int | None], int]:
    """Rank traces and cut them into bins of identical size.

    Returns the per-trace bin index, with None for traces left over by an uneven
    split, and how many were left over. Leftovers come off the two ends of the
    ranking rather than one side, so neither the lowest nor the highest bin is
    the one that absorbs the imbalance.

    With `strata`, the ranking and the cut happen inside each stratum and every
    bin takes the same number of traces from each, so bin sizes stay identical
    while the stratifying variable is held constant across bins.
    """
    n = len(values)
    keys = strata if strata is not None else [0] * n
    groups: dict[Any, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)

    assignment: list[int | None] = [None] * n
    remainder = 0
    for indices in groups.values():
        indices.sort(key=lambda i: values[i])
        per_bin = len(indices) // n_bins
        leftover = len(indices) - per_bin * n_bins
        remainder += leftover
        if per_bin == 0:
            continue
        low_drop = leftover // 2
        for slot, index in enumerate(indices[low_drop : low_drop + per_bin * n_bins]):
            assignment[index] = slot // per_bin

    if all(b is None for b in assignment):
        raise ValueError(f"{n} traces cannot fill {n_bins} bins")
    return assignment, remainder


def equalize_by_tokens(
    members: list[list[dict[str, Any]]],
    *,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], int]:
    """Subsample every bin down to the leanest bin's response-token total.

    Traces are considered in a shuffled order and kept while they fit, so the
    survivors are an unbiased sample of the bin rather than its short traces.
    """
    totals = [sum(int(s["n_response_tokens"]) for s in bin_rows) for bin_rows in members]
    budget = min(totals)
    rng = random.Random(seed)

    equalized: list[list[dict[str, Any]]] = []
    for bin_rows in members:
        shuffled = list(bin_rows)
        rng.shuffle(shuffled)
        running = 0
        keep: list[dict[str, Any]] = []
        for score in shuffled:
            tokens = int(score["n_response_tokens"])
            if running + tokens > budget:
                continue
            keep.append(score)
            running += tokens
        keep.sort(key=lambda s: s["index"])
        equalized.append(keep)
    return equalized, budget


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    """Linearly interpolated percentile, q in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stage(final: Path) -> Path:
    staging = final.parent / f".{final.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _publish(staging: Path, final: Path) -> None:
    if final.exists():
        shutil.rmtree(final)
    staging.replace(final)


def write_bin(
    *,
    folder: Path,
    input_dir: Path,
    source: str,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    staging = _stage(folder)
    (staging / f"train_{source}.json").write_text(json.dumps(rows, indent=2))
    (staging / "bin_manifest.json").write_text(json.dumps(manifest, indent=2))
    for filename in PASSTHROUGH_FILES:
        origin = input_dir / filename
        if origin.exists():
            shutil.copy2(origin, staging / filename)
    _publish(staging, folder)


def bin_summary(
    scores: list[dict[str, Any]],
    metric_key: str,
    bin_index: int,
) -> dict[str, Any]:
    values = [s[metric_key] for s in scores]
    tokens = [float(s["n_response_tokens"]) for s in scores]
    return {
        "bin": bin_index,
        "n": len(scores),
        "total_response_tokens": int(sum(tokens)),
        "vgap_min": min(values),
        "vgap_max": max(values),
        "vgap_median": _percentile(values, 0.5),
        "vgap_p25": _percentile(values, 0.25),
        "vgap_p75": _percentile(values, 0.75),
        "vgap_mean": _mean(values),
        "response_tokens_median": _percentile(tokens, 0.5),
        "response_tokens_mean": _mean(tokens),
        "trace_accuracy": _mean([float(s["correct"]) for s in scores]),
        "n_truncated": sum(1 for s in scores if s["truncated"]),
    }


def fairness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How comparable the bins are as training sets."""
    counts = [row["n"] for row in rows]
    tokens = [row["total_response_tokens"] for row in rows]
    return {
        "trace_counts": counts,
        "traces_equal": len(set(counts)) == 1,
        "total_tokens": tokens,
        "token_ratio": max(tokens) / max(min(tokens), 1),
    }


def format_report(
    *,
    source: str,
    input_dir: Path,
    n_bins: int,
    n_scored: int,
    excluded: dict[str, int],
    n_binned: int,
    remainders: dict[str, int],
    equalize: str,
    token_budget: int | None,
    summaries: dict[str, list[dict[str, Any]]],
    fairness_by_arm: dict[str, dict[str, Any]],
    agreement: dict[str, Any],
    spread: dict[str, list[float]],
) -> str:
    lines = [
        f"# v_gap bins for `{input_dir}/train_{source}.json`",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from {n_scored} scored traces.",
        "",
        "`v_gap = log P_teacher(y|x) - log P_proxy(y|x)`, summed over the response "
        "tokens of each trace. It is the sequence-level form of the per-token gap "
        "the PoE defense mixes against. Bin 0 holds the lowest v_gap traces.",
        "",
        "## Traces used",
        "",
        f"- Scored: {n_scored}",
        f"- Excluded, v_gap not finite: {excluded['non_finite']}",
        f"- Excluded, empty response: {excluded['empty_response']}",
        f"- Excluded, shorter than the minimum response length: {excluded['too_short']}",
        f"- Excluded, truncated at the scoring cap: {excluded['truncated']}",
        "- Left over after cutting into equal bins: "
        + ", ".join(f"{arm} {count}" for arm, count in remainders.items())
        + " (dropped from the ends of the ranking)",
        f"- **Binned: {n_binned} per arm**",
        "",
        "A non-finite score would sort arbitrarily and quietly poison a bin, so "
        "those traces are removed before ranking rather than after.",
        "",
        "## Spread of v_gap",
        "",
        "| metric | min | p1 | p25 | median | p75 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("sum", "mean"):
        values = spread.get(name)
        if not values:
            continue
        lines.append(
            "| " + " | ".join([
                name,
                f"{min(values):.2f}",
                f"{_percentile(values, 0.01):.2f}",
                f"{_percentile(values, 0.25):.2f}",
                f"{_percentile(values, 0.5):.2f}",
                f"{_percentile(values, 0.75):.2f}",
                f"{_percentile(values, 0.99):.2f}",
                f"{max(values):.2f}",
            ]) + " |"
        )
    lines += [
        "",
        "A wide gap between p99 and max is the tail to be careful with. Because "
        "traces are assigned by rank, that tail affects only the top bin's "
        "membership, never the split points.",
        "",
        "## Diagnostics",
        "",
        f"- Spearman(v_gap sum, response tokens) = {agreement['rho_sum_length']:+.3f}",
        f"- Spearman(v_gap mean, response tokens) = {agreement['rho_mean_length']:+.3f}",
        f"- Spearman(v_gap sum, v_gap mean) = {agreement['rho_sum_mean']:+.3f}",
        "- Traces landing in the same bin under "
        + " and ".join(f"`{a}`" for a in agreement["compared_arms"])
        + f": {agreement['same_bin']}/{agreement['same_bin_of']} "
        f"({agreement['same_bin'] / max(agreement['same_bin_of'], 1):.1%})",
        "",
        "The first line is the length confound. When it is strongly positive, "
        "ranking traces by the summed v_gap is close to ranking them by length, "
        "and no reweighting of an unstratified split can separate the two. The "
        "matched arms handle it by ranking traces only against others of the same "
        "length.",
        "",
        "## Comparability of the bins",
        "",
        f"Equalization mode: **{equalize}**"
        + (f", common token budget {token_budget}" if token_budget else ""),
        "",
        "| arm | ranked within | traces per bin | total tokens per bin | fattest/leanest tokens |",
        "| --- | --- | --- | --- | ---: |",
    ]
    stratify_label = {
        "none": "all traces",
        "length": "length decile",
        "length_correct": "length decile x correctness",
    }
    for arm, stats in fairness_by_arm.items():
        counts = stats["trace_counts"]
        count_cell = str(counts[0]) if stats["traces_equal"] else " / ".join(str(c) for c in counts)
        token_cell = " / ".join(f"{t / 1000:.0f}k" for t in stats["total_tokens"])
        lines.append(
            f"| {arm} | {stratify_label[ARMS[arm][1]]} | {count_cell} | "
            f"{token_cell} | {stats['token_ratio']:.2f}x |"
        )
    lines += [
        "",
        "An arm ranked within all traces answers the question as the defense poses "
        "it, but its bins may differ in length and so in how much supervision each "
        "student sees. A matched arm holds those fixed and isolates the gap itself. "
        "Reading the two together is the point: a trend that survives matching is a "
        "v_gap effect, a trend that disappears was a length effect.",
        "",
    ]
    for arm, stats in fairness_by_arm.items():
        if ARMS[arm][1] == "none" and stats["token_ratio"] > TOKEN_RATIO_WARN:
            lines += [
                f"> **`{arm}`: bins hold equal traces but the fattest carries "
                f"{stats['token_ratio']:.2f}x the response tokens of the leanest.** Any "
                "trend across these bins mixes v_gap with training budget. Compare it "
                "against the matched arm before drawing a conclusion.",
                "",
            ]
    if equalize == "tokens":
        lines += [
            "> Every bin was subsampled to a common response-token budget, so trace "
            "counts differ instead. Within an unstratified arm, equal traces and equal "
            "tokens cannot hold at once; only stratification gives both.",
            "",
        ]

    for arm, rows in summaries.items():
        metric, stratify = ARMS[arm]
        lines += [
            f"## Arm `{arm}`: v_gap {metric}, ranked within {stratify_label[stratify]}",
            "",
            "| bin | traces | tokens | v_gap median | v_gap IQR | v_gap range | v_gap mean | resp. tokens | trace acc |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['bin']} | {row['n']} | {row['total_response_tokens'] / 1000:.0f}k | "
                f"{row['vgap_median']:.3f} | "
                f"{row['vgap_p25']:.3f} to {row['vgap_p75']:.3f} | "
                f"{row['vgap_min']:.3f} to {row['vgap_max']:.3f} | {row['vgap_mean']:.3f} | "
                f"{row['response_tokens_median']:.0f} | {row['trace_accuracy']:.3f} |"
            )
        lines += [
            "",
            "Median and mean drifting apart within a bin means that bin is skewed; "
            "the median is what the report and the plot use.",
            "",
        ]
    lines += [
        "`trace acc` is the fraction of that bin's teacher traces that reach the "
        "right answer. A bin that is both high v_gap and low accuracy would give "
        "its student worse supervision for a reason unrelated to v_gap, so read "
        "the accuracy column before reading the correlation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="gsm8k_output_small", help="Directory holding the source traces.")
    parser.add_argument("--source", default="standard", help="Trace stem that was scored.")
    parser.add_argument("--scores-dir", default=None, help="Defaults to vgap/<input dir name>.")
    parser.add_argument("--n-bins", default=5, type=int, help="Number of equal bins.")
    parser.add_argument(
        "--arms",
        default=DEFAULT_ARMS,
        nargs="+",
        choices=sorted(ARMS),
        help=(
            "Which binnings to build. `sum` and `mean` rank every trace against "
            "every other; the `_lenmatched` and `_matched` variants rank only "
            "within a response-length decile, and `_matched` also within teacher "
            "correctness, so the bins cannot differ in length or supervision "
            "quality. Default: sum (V_gap as the defense computes it) plus "
            "mean_matched (the controlled comparison)."
        ),
    )
    parser.add_argument(
        "--length-strata",
        default=10,
        type=int,
        help="Number of response-length groups used by the matched arms.",
    )
    parser.add_argument(
        "--equalize",
        default="traces",
        choices=["traces", "tokens"],
        help=(
            "What to hold fixed across bins. `traces` gives every bin the same number "
            "of demonstrations; `tokens` subsamples each bin to a common response-token "
            "budget instead, letting the counts differ."
        ),
    )
    parser.add_argument(
        "--drop-truncated",
        action="store_true",
        help="Also exclude traces whose scoring hit the length cap, whose v_gap covers a prefix only.",
    )
    parser.add_argument(
        "--min-response-tokens",
        default=16,
        type=int,
        help=(
            "Exclude traces shorter than this. Guards the per-token metric, which "
            "divides by the length and so is unstable on a handful of tokens."
        ),
    )
    parser.add_argument("--seed", default=0, type=int, help="Seed for token-budget subsampling.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    scores_dir = Path(args.scores_dir or f"vgap/{input_dir.name}")
    scores_path = scores_dir / f"scores_{args.source}.json"
    traces_path = input_dir / f"train_{args.source}.json"
    for path in (scores_path, traces_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    all_scores = json.loads(scores_path.read_text())
    traces = json.loads(traces_path.read_text())
    if len(all_scores) > len(traces):
        raise ValueError(f"{scores_path} has {len(all_scores)} rows but {traces_path} has {len(traces)}")

    for score in all_scores:
        trace = traces[score["index"]]
        expected = trace.get("example_id")
        if expected is not None and expected != score["example_id"]:
            raise ValueError(
                f"Score row {score['index']} is {score['example_id']} but the trace file "
                f"has {expected}. The scores were written against different traces."
            )

    scores, excluded = filter_scores(
        all_scores,
        drop_truncated=args.drop_truncated,
        min_response_tokens=args.min_response_tokens,
    )
    dropped = sum(excluded.values())
    if dropped:
        print(
            f"  Excluded {dropped} traces before binning: "
            + ", ".join(f"{k}={v}" for k, v in excluded.items() if v)
        )
    if args.n_bins < 2 or args.n_bins > len(scores):
        raise ValueError(f"--n-bins must be between 2 and {len(scores)}, got {args.n_bins}")

    lengths = [float(s["n_response_tokens"]) for s in scores]
    sums = [s["vgap_sum"] for s in scores]
    means = [s["vgap_mean"] for s in scores]

    assignments: dict[str, list[int | None]] = {}
    members_by_arm: dict[str, list[list[dict[str, Any]]]] = {}
    remainders: dict[str, int] = {}
    token_budget: int | None = None
    for arm in args.arms:
        metric, stratify = ARMS[arm]
        strata = stratum_keys(scores, stratify, args.length_strata) if stratify != "none" else None
        assignment, remainder = assign_bins(
            [s[METRICS[metric]] for s in scores], args.n_bins, strata
        )
        assignments[arm] = assignment
        remainders[arm] = remainder
        members = [
            [s for s, b in zip(scores, assignment) if b == bin_index]
            for bin_index in range(args.n_bins)
        ]
        if args.equalize == "tokens":
            members, token_budget = equalize_by_tokens(members, seed=args.seed)
        members_by_arm[arm] = members

    summaries: dict[str, list[dict[str, Any]]] = {}
    fairness_by_arm: dict[str, dict[str, Any]] = {}
    for arm in args.arms:
        metric, stratify = ARMS[arm]
        metric_key = METRICS[metric]
        rows: list[dict[str, Any]] = []
        for bin_index, members in enumerate(members_by_arm[arm]):
            summary = bin_summary(members, metric_key, bin_index)
            rows.append(summary)
            folder = scores_dir / "bins" / arm / f"bin_{bin_index}"
            write_bin(
                folder=folder,
                input_dir=input_dir,
                source=args.source,
                rows=[traces[s["index"]] for s in members],
                manifest={
                    "source": args.source,
                    "input_dir": str(input_dir),
                    "arm": arm,
                    "metric": metric,
                    "metric_key": metric_key,
                    "stratify": stratify,
                    "length_strata": args.length_strata if stratify != "none" else 0,
                    "n_bins": args.n_bins,
                    "equalize": args.equalize,
                    "token_budget": token_budget,
                    **summary,
                    "example_ids": [s["example_id"] for s in members],
                    "created_at": datetime.now().isoformat(),
                },
            )
            print(
                f"  {arm} bin {bin_index}: n={summary['n']}, "
                f"tokens={summary['total_response_tokens'] / 1000:.0f}k, "
                f"v_gap median={summary['vgap_median']:.2f} "
                f"[{summary['vgap_min']:.2f}, {summary['vgap_max']:.2f}], "
                f"acc={summary['trace_accuracy']:.3f}, "
                f"resp tokens median={summary['response_tokens_median']:.0f}"
            )
        summaries[arm] = rows
        fairness_by_arm[arm] = fairness(rows)

    same_bin = shared = 0
    if len(args.arms) >= 2:
        first, second = assignments[args.arms[0]], assignments[args.arms[1]]
        pairs = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
        shared = len(pairs)
        same_bin = sum(1 for a, b in pairs if a == b)
    agreement = {
        "rho_sum_length": spearman(sums, lengths),
        "rho_mean_length": spearman(means, lengths),
        "rho_sum_mean": spearman(sums, means),
        "same_bin": same_bin,
        "same_bin_of": shared,
        "compared_arms": args.arms[:2],
    }

    n_binned = min(sum(row["n"] for row in rows) for rows in summaries.values())
    report = format_report(
        source=args.source,
        input_dir=input_dir,
        n_bins=args.n_bins,
        n_scored=len(all_scores),
        excluded=excluded,
        n_binned=n_binned,
        remainders=remainders,
        equalize=args.equalize,
        token_budget=token_budget,
        summaries=summaries,
        fairness_by_arm=fairness_by_arm,
        agreement=agreement,
        spread={"sum": sums, "mean": means},
    )
    (scores_dir / "BINS.md").write_text(report)
    (scores_dir / "bins_manifest.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "source": args.source,
                "n_bins": args.n_bins,
                "arms": {arm: {"metric": ARMS[arm][0], "stratify": ARMS[arm][1]} for arm in args.arms},
                "length_strata": args.length_strata,
                "equalize": args.equalize,
                "token_budget": token_budget,
                "n_scored": len(all_scores),
                "excluded": excluded,
                "n_binned": n_binned,
                "agreement": agreement,
                "fairness": fairness_by_arm,
                "summaries": summaries,
                "created_at": datetime.now().isoformat(),
            },
            indent=2,
        )
    )
    print(f"\nWrote {scores_dir / 'BINS.md'} and {len(args.arms) * args.n_bins} bin folders")
    print(
        f"  Spearman(v_gap sum, length)={agreement['rho_sum_length']:+.3f}, "
        f"Spearman(v_gap mean, length)={agreement['rho_mean_length']:+.3f}"
    )
    for arm, stats in fairness_by_arm.items():
        flag = ""
        if ARMS[arm][1] == "none" and stats["token_ratio"] > TOKEN_RATIO_WARN:
            flag = f"  <-- length is not held fixed here; compare against a matched arm"
        print(
            f"  {arm}: traces per bin {'equal' if stats['traces_equal'] else stats['trace_counts']}, "
            f"token ratio {stats['token_ratio']:.2f}x{flag}"
        )


if __name__ == "__main__":
    main()
