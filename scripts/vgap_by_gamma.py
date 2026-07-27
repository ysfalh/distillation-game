#!/usr/bin/env python3
"""Does token-level PoE actually suppress the teacher-proxy likelihood gap?

PoE samples each token from a blend of the two models,

    q(a|h) proportional to p_teacher(a|h)^(1-gamma) * p_proxy(a|h)^gamma

so it should pull the sampled text toward tokens the proxy also finds likely.
The sequence-level rule it is defending against scores a trace by

    v_gap(y) = log P_teacher(y) - log P_proxy(y)

and this script measures that quantity directly on traces already generated at
each gamma. If the implementation does what it claims, v_gap should fall as
gamma rises.

v_gap is one number per trace, defined by the equation above. Because the sum
grows with trace length -- on these traces its rank correlation with response
length is +0.88 -- a bare sum would mostly report how long the traces are, and
PoE changes trace length as a side effect. Dividing by the number of response
tokens gives the same per-trace quantity on a per-token scale, comparable across
traces of different lengths, so that is the headline number. The sum is reported
alongside it, and the median per-token gap is reported too, since one trace with
an extreme gap cannot move a median.

Every source covers the identical GSM8K problems in the same order, so each
gamma is compared to the gamma = 0 baseline *paired by problem*: the statistic
is the mean of v_gap(poe trace) - v_gap(standard trace) over problems. Pairing
cancels problem difficulty, which is the dominant source of variation in v_gap,
and it comes with a distribution-free companion, the fraction of problems whose
gap went down at all.

Both models score token ids produced by the teacher's tokenizer, which is what
PoE does at generation time, and only response tokens count.

Usage:
    python scripts/vgap_by_gamma.py --seed 123
    python scripts/vgap_by_gamma.py --report-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent
for _path in (str(_root / "src"), str(_root / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich.console import Console

from clean_sweep.config import FullConfig
from score_vgap import _free, _load_model, _percentile, build_examples, score_with_model

console = Console()

BASELINE = "standard"
# Ordered for the report: the baseline, then PoE by strength, then the other
# defense. Anything found on disk but not named here is appended.
PREFERRED_ORDER = [
    "standard",
    "poe_gamma_0.6",
    "poe_gamma_0.7",
    "antidistillation_lam_0.05",
    "antidistillation_lam_0.055",
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1) / n)


def _gamma_of(source: str) -> float | None:
    """The PoE mixing weight a source was generated with, if it is a PoE source."""
    if source == BASELINE:
        return 0.0
    if source.startswith("poe_gamma_"):
        try:
            return float(source.removeprefix("poe_gamma_"))
        except ValueError:
            return None
    return None


def discover_sources(input_dir: Path) -> list[str]:
    stems = {p.stem.removeprefix("train_") for p in input_dir.glob("train_*.json")}
    ordered = [s for s in PREFERRED_ORDER if s in stems]
    return ordered + sorted(stems - set(ordered))


def score_sources(
    sources: list[str],
    input_dir: Path,
    cfg: FullConfig,
    *,
    max_length: int,
    batch_size: int,
    row_chunk: int,
    seed: int,
    limit: int,
) -> dict[str, Any]:
    """Teacher and proxy log-probs for every trace of every source.

    Each model is loaded once and scores all sources before being freed, so peak
    memory is one model and the 7B is not paid for five times over.
    """
    import torch

    from transformers import AutoTokenizer
    from clean_sweep.generation.core import ensure_chat_template

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer or cfg.model.teacher,
        trust_remote_code=True,
    )
    tokenizer = ensure_chat_template(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    examples_by_source: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        rows = json.loads((input_dir / f"train_{source}.json").read_text())
        if limit:
            rows = rows[:limit]
        examples = build_examples(rows, tokenizer, max_length)
        examples_by_source[source] = examples
        lengths = [ex["n_response_tokens"] for ex in examples]
        console.print(
            f"  {source}: {len(examples)} traces, "
            f"response tokens mean={_mean([float(v) for v in lengths]):.0f}, "
            f"max={max(lengths, default=0)}, "
            f"truncated={sum(ex['truncated'] for ex in examples)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logprobs: dict[str, dict[str, list[float]]] = {s: {} for s in sources}
    clamped: dict[str, int] = {}

    for role, name in (("teacher", cfg.model.teacher), ("proxy", cfg.model.proxy_student)):
        console.print(f"\n  Loading {role} {name}")
        model = _load_model(name, cfg, device)
        total_clamped = 0
        for source in sources:
            values, n_clamped = score_with_model(
                model,
                examples_by_source[source],
                pad_id=pad_id,
                batch_size=batch_size,
                row_chunk=row_chunk,
                label=f"{role} {source}",
                shuffle_seed=seed,
            )
            logprobs[source][role] = values
            total_clamped += n_clamped
        clamped[role] = total_clamped
        _free(model)

    per_source: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        rows = []
        for ex, lt, lp in zip(
            examples_by_source[source],
            logprobs[source]["teacher"],
            logprobs[source]["proxy"],
        ):
            n_tokens = ex["n_response_tokens"]
            gap = lt - lp
            rows.append({
                "example_id": ex["example_id"],
                "n_response_tokens": n_tokens,
                "truncated": ex["truncated"],
                "correct": ex["correct"],
                "finite": math.isfinite(lt) and math.isfinite(lp) and n_tokens > 0,
                "logp_teacher": lt,
                "logp_proxy": lp,
                "vgap_sum": gap,
                "vgap_mean": gap / n_tokens if n_tokens else 0.0,
            })
        per_source[source] = rows

    return {"traces": per_source, "clamped": clamped}


def _usable(rows: list[dict[str, Any]], drop_truncated: bool) -> list[dict[str, Any]]:
    return [r for r in rows if r["finite"] and not (drop_truncated and r["truncated"])]


def summarize_source(rows: list[dict[str, Any]], drop_truncated: bool) -> dict[str, Any]:
    usable = _usable(rows, drop_truncated)
    means = [r["vgap_mean"] for r in usable]
    sums = [r["vgap_sum"] for r in usable]
    lengths = [float(r["n_response_tokens"]) for r in usable]
    return {
        "n": len(rows),
        "n_usable": len(usable),
        "n_truncated": sum(r["truncated"] for r in rows),
        "vgap_mean": _mean(means),
        "vgap_mean_stderr": _stderr(means),
        "vgap_mean_median": _percentile(means, 0.5),
        "vgap_mean_p25": _percentile(means, 0.25),
        "vgap_mean_p75": _percentile(means, 0.75),
        "vgap_sum": _mean(sums),
        "vgap_sum_median": _percentile(sums, 0.5),
        "response_tokens": _mean(lengths),
        "trace_accuracy": _mean([float(r["correct"]) for r in rows]),
    }


def compare_to_baseline(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    drop_truncated: bool,
) -> dict[str, Any] | None:
    """Paired per-problem change in v_gap against the gamma = 0 traces.

    Only problems usable on both sides are paired, so a trace truncated in one
    condition cannot masquerade as a change in the gap.
    """
    baseline_by_id = {r["example_id"]: r for r in baseline_rows}
    deltas: list[float] = []
    delta_sums: list[float] = []
    for row in rows:
        base = baseline_by_id.get(row["example_id"])
        if base is None:
            continue
        if not (row["finite"] and base["finite"]):
            continue
        if drop_truncated and (row["truncated"] or base["truncated"]):
            continue
        deltas.append(row["vgap_mean"] - base["vgap_mean"])
        delta_sums.append(row["vgap_sum"] - base["vgap_sum"])
    if not deltas:
        return None

    mean_delta = _mean(deltas)
    stderr = _stderr(deltas)
    n_down = sum(1 for d in deltas if d < 0)
    return {
        "n_paired": len(deltas),
        "delta_mean": mean_delta,
        "delta_stderr": stderr,
        "delta_median": _percentile(deltas, 0.5),
        "delta_sum_mean": _mean(delta_sums),
        "frac_suppressed": n_down / len(deltas),
        # How many standard errors the paired change sits from zero. With
        # thousands of pairs this is effectively a z statistic.
        "t_stat": mean_delta / stderr if stderr and math.isfinite(stderr) and stderr > 0 else float("nan"),
    }


def summarize_seed(traces: dict[str, list[dict[str, Any]]], drop_truncated: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    baseline_rows = traces.get(BASELINE)
    for source, rows in traces.items():
        entry = summarize_source(rows, drop_truncated)
        entry["gamma"] = _gamma_of(source)
        if baseline_rows is not None and source != BASELINE:
            entry["vs_baseline"] = compare_to_baseline(rows, baseline_rows, drop_truncated)
        summary[source] = entry
    return summary


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return format(value, spec)


def pool_across_seeds(by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Average each statistic over seeds and record how far the seeds spread.

    The spread is a numerical-reproducibility check, not a confidence interval:
    teacher-forced scoring is deterministic and the seed only reshuffles batches.
    """
    sources: list[str] = []
    for summary in by_seed.values():
        for source in summary:
            if source not in sources:
                sources.append(source)

    pooled: dict[str, Any] = {}
    for source in sources:
        entries = [s[source] for s in by_seed.values() if source in s]
        base = dict(entries[0])
        for key in ("vgap_mean", "vgap_mean_median", "vgap_sum", "response_tokens"):
            values = [e[key] for e in entries if isinstance(e.get(key), float)]
            base[key] = _mean(values)
            base[f"{key}_seed_spread"] = (max(values) - min(values)) if len(values) > 1 else 0.0
        comparisons = [e["vs_baseline"] for e in entries if e.get("vs_baseline")]
        if comparisons:
            merged = dict(comparisons[0])
            for key in ("delta_mean", "delta_median", "frac_suppressed", "delta_sum_mean", "t_stat"):
                values = [c[key] for c in comparisons if isinstance(c.get(key), float)]
                merged[key] = _mean(values)
                merged[f"{key}_seed_spread"] = (max(values) - min(values)) if len(values) > 1 else 0.0
            base["vs_baseline"] = merged
        base["n_seeds"] = len(entries)
        pooled[source] = base
    return pooled


def _exclusion_warnings(
    pooled: dict[str, Any],
    order: list[str],
    drop_truncated: bool,
) -> list[str]:
    """Say so loudly when a source lost enough traces to distort its average."""
    empty = [s for s in order if pooled[s]["n_usable"] == 0]
    if empty:
        return [
            f"> **No usable traces for {', '.join(f'`{s}`' for s in empty)}, so their "
            "numbers are blank.** Every trace was either truncated at the length cap "
            "or scored non-finite. Raise `--max-length`, or pass `--keep-truncated` to "
            "score the kept prefix of each trace instead.",
            "",
        ]
    heavy = [s for s in order if (pooled[s]["n"] - pooled[s]["n_usable"]) / max(pooled[s]["n"], 1) > 0.05]
    if heavy and drop_truncated:
        return [
            "> **"
            + ", ".join(f"`{s}` lost {(pooled[s]['n'] - pooled[s]['n_usable']) / max(pooled[s]['n'], 1):.0%}"
                        for s in heavy)
            + " of its traces to the length cap.** Long traces are dropped, and length "
            "is related to the gap, so these averages are taken over a non-random subset. "
            "The paired comparison is unaffected in direction, since it only uses problems "
            "usable on both sides, but raise `--max-length` before quoting the levels.",
            "",
        ]
    return []


def _order_sources(pooled: dict[str, Any]) -> list[str]:
    known = [s for s in PREFERRED_ORDER if s in pooled]
    return known + [s for s in pooled if s not in known]


def format_report(
    pooled: dict[str, Any],
    *,
    seeds: list[int],
    cfg: FullConfig,
    input_dir: Path,
    max_length: int,
    drop_truncated: bool,
) -> str:
    order = _order_sources(pooled)
    baseline = pooled.get(BASELINE, {})
    lines = [
        "# Does token-level PoE suppress the teacher-proxy likelihood gap?",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from traces in `{input_dir}`.",
        "",
        f"- Teacher: `{cfg.model.teacher}`",
        f"- Proxy: `{cfg.model.proxy_student}`",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Scored length cap: {max_length} tokens"
        + (", truncated traces excluded" if drop_truncated else ""),
        "",
        "PoE samples each token from `p_teacher^(1-gamma) * p_proxy^gamma`, so it "
        "should steer the text toward tokens the proxy also finds likely. The "
        "sequence-level rule it defends against scores a whole trace with "
        "`v_gap(y) = log P_teacher(y) - log P_proxy(y)`. If the implementation "
        "works, v_gap should fall as gamma rises. No traces were regenerated; "
        "these are the same files the students train on.",
        "",
        "v_gap is one number per trace. It is reported divided by the trace's "
        "response tokens, because the raw sum grows with length and PoE changes "
        "length as a side effect, so a bare sum would partly be reporting how "
        "long the traces are. The summed form is in the second table.",
        "",
        "## Per-token v_gap by gamma",
        "",
        "| source | gamma | traces | excluded | resp. tokens | mean v_gap/token | s.e. | median | IQR | trace acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in order:
        entry = pooled[source]
        gamma = entry.get("gamma")
        excluded = entry["n"] - entry["n_usable"]
        lines.append(
            "| " + " | ".join([
                f"`{source}`",
                _fmt(gamma, ".2f") if gamma is not None else "n/a",
                str(entry["n_usable"]),
                f"{excluded} ({excluded / max(entry['n'], 1):.1%})",
                _fmt(entry["response_tokens"], ".0f"),
                _fmt(entry["vgap_mean"]),
                _fmt(entry["vgap_mean_stderr"]),
                _fmt(entry["vgap_mean_median"]),
                f"{_fmt(entry['vgap_mean_p25'], '.3f')} to {_fmt(entry['vgap_mean_p75'], '.3f')}",
                _fmt(entry["trace_accuracy"], ".3f"),
            ]) + " |"
        )

    lines += ["", *_exclusion_warnings(pooled, order, drop_truncated)]
    lines += [
        "`gamma` is n/a for antidistillation sampling, which is a different "
        "defense: it perturbs the teacher's logits with a finite-difference "
        "penalty and never consults the proxy. It is here as a control. A drop "
        "in v_gap that shows up under PoE but not under it is evidence that the "
        "drop comes from the proxy mixing rather than from any perturbation of "
        "the teacher.",
        "",
        "## Change against the gamma = 0 teacher, paired by problem",
        "",
        "Every source covers the same GSM8K problems, so each trace is compared "
        "with the standard trace for the same problem and the differences are "
        "averaged. Pairing removes problem difficulty, which otherwise dominates "
        "the variation in v_gap. `suppressed` is the fraction of problems whose "
        "per-token gap went down at all, which assumes nothing about the shape "
        "of the distribution.",
        "",
        "| source | gamma | pairs | change in v_gap/token | s.e. | relative | median change | suppressed | s.e. from 0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    base_mean = baseline.get("vgap_mean")
    for source in order:
        if source == BASELINE:
            continue
        comparison = pooled[source].get("vs_baseline")
        if not comparison:
            continue
        relative = (
            comparison["delta_mean"] / base_mean
            if isinstance(base_mean, float) and base_mean
            else float("nan")
        )
        lines.append(
            "| " + " | ".join([
                f"`{source}`",
                _fmt(pooled[source].get("gamma"), ".2f") if pooled[source].get("gamma") is not None else "n/a",
                str(comparison["n_paired"]),
                _fmt(comparison["delta_mean"], "+.4f"),
                _fmt(comparison["delta_stderr"]),
                _fmt(relative, "+.1%"),
                _fmt(comparison["delta_median"], "+.4f"),
                _fmt(comparison["frac_suppressed"], ".1%"),
                _fmt(comparison["t_stat"], "+.1f"),
            ]) + " |"
        )

    lines += [
        "",
        "## Summed v_gap, for reference",
        "",
        "This is the quantity as the sequence-level rule literally writes it, "
        "with no length normalization. Read the change here together with the "
        "response-token column: if a source's traces got shorter, its summed gap "
        "falls even when the per-token gap does not.",
        "",
        "| source | mean v_gap (summed) | median | resp. tokens | change vs gamma = 0 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source in order:
        entry = pooled[source]
        comparison = entry.get("vs_baseline")
        lines.append(
            "| " + " | ".join([
                f"`{source}`",
                _fmt(entry["vgap_sum"], ".1f"),
                _fmt(entry["vgap_sum_median"], ".1f"),
                _fmt(entry["response_tokens"], ".0f"),
                _fmt(comparison["delta_sum_mean"], "+.1f") if comparison else "-",
            ]) + " |"
        )

    lines += ["", "## Reading this", ""]
    lines += _verdict(pooled, order)
    lines += [
        "",
        "## On the seeds",
        "",
        f"Scored under {len(seeds)} seeds ({', '.join(str(s) for s in seeds)}). "
        "Teacher-forced scoring is deterministic: the traces are fixed and no "
        "sampling happens, so a seed cannot change the measured gap. Here it "
        "only reshuffles which traces share a batch, which changes padding and "
        "the order of floating-point reductions. The spread across seeds is "
        "therefore a numerical-stability check and should be near zero; the real "
        "uncertainty on each number is the s.e. across traces, and on each change "
        "the paired s.e.",
        "",
        "| source | v_gap/token spread across seeds | paired change spread |",
        "| --- | ---: | ---: |",
    ]
    for source in order:
        entry = pooled[source]
        comparison = entry.get("vs_baseline")
        lines.append(
            f"| `{source}` | {_fmt(entry.get('vgap_mean_seed_spread'), '.2e')} | "
            + (f"{_fmt(comparison.get('delta_mean_seed_spread'), '.2e')} |" if comparison else "- |")
        )
    lines.append("")
    return "\n".join(lines)


def _verdict(pooled: dict[str, Any], order: list[str]) -> list[str]:
    """State what the numbers say about the hypothesis, in plain terms."""
    poe = [
        (s, pooled[s])
        for s in order
        if pooled[s].get("gamma") not in (None, 0.0) and pooled[s].get("vs_baseline")
    ]
    if not poe:
        return ["No PoE sources were scored, so there is nothing to conclude."]

    lines: list[str] = []
    suppressed = [s for s, e in poe if e["vs_baseline"]["delta_mean"] < 0]
    if len(suppressed) == len(poe):
        lines.append(
            "- Every PoE setting lowers the per-token v_gap relative to the "
            "gamma = 0 teacher, which is the direction the defense intends."
        )
    elif suppressed:
        lines.append(
            f"- {len(suppressed)} of {len(poe)} PoE settings lower the per-token "
            "v_gap; the rest do not, so the suppression is not consistent."
        )
    else:
        lines.append(
            "- **No PoE setting lowers the per-token v_gap.** The token-level "
            "implementation is not suppressing the quantity the sequence-level "
            "rule scores, which is the opposite of what it is meant to do."
        )

    ordered_by_gamma = sorted(poe, key=lambda item: item[1]["gamma"])
    if len(ordered_by_gamma) >= 2:
        deltas = [e["vs_baseline"]["delta_mean"] for _, e in ordered_by_gamma]
        if all(b <= a for a, b in zip(deltas, deltas[1:])):
            lines.append(
                "- The effect grows with gamma: a larger mixing weight suppresses "
                "the gap further, as the mechanism predicts."
            )
        else:
            lines.append(
                "- The effect does not grow monotonically with gamma, so over this "
                "range a larger mixing weight does not reliably suppress the gap further."
            )

    weakest = min(abs(e["vs_baseline"]["t_stat"]) for _, e in poe if math.isfinite(e["vs_baseline"]["t_stat"]))
    if weakest >= 3:
        lines.append(
            f"- The weakest of these changes still sits {weakest:.0f} standard errors "
            "from zero, so none of them is a sampling artifact."
        )
    else:
        lines.append(
            f"- The weakest change sits only {weakest:.1f} standard errors from zero, "
            "so at least one of these differences is within noise."
        )

    controls = [
        (s, pooled[s])
        for s in order
        if pooled[s].get("gamma") is None and pooled[s].get("vs_baseline")
    ]
    if controls:
        poe_shift = _mean([abs(e["vs_baseline"]["delta_mean"]) for _, e in poe])
        control_shift = _mean([abs(e["vs_baseline"]["delta_mean"]) for _, e in controls])
        if control_shift and poe_shift > 2 * control_shift:
            lines.append(
                f"- The control defense moves the gap {poe_shift / control_shift:.1f}x "
                "less than PoE does, so the effect tracks the proxy mixing rather "
                "than trace perturbation in general."
            )
        elif control_shift >= poe_shift:
            lines.append(
                "- The control defense, which never consults the proxy, moves the gap "
                "as much as PoE does. That undercuts attributing the change to the "
                "proxy mixing; something common to both is moving it."
            )
    return lines


def load_seed_files(output_dir: Path) -> dict[int, dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for path in sorted(output_dir.glob("seed_*/summary.json")):
        seed = int(path.parent.name.removeprefix("seed_"))
        found[seed] = json.loads(path.read_text())["sources"]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", default="gsm8k_output_small", help="Saved-trace directory.")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Trace stems to score. Defaults to every train_*.json in the input directory.",
    )
    parser.add_argument("--config", default="configs/gsm8k.yaml", help="Run config naming teacher and proxy.")
    parser.add_argument("--output-dir", default="outputs/vgap_poe", help="Where results and the report go.")
    parser.add_argument("--seed", default=123, type=int, help="Batch-composition seed; see the report.")
    parser.add_argument("--batch-size", default=6, type=int, help="Sequences per forward pass.")
    parser.add_argument(
        "--row-chunk",
        default=1024,
        type=int,
        help="Token positions softmaxed at once in fp32. Lower this first if scoring OOMs.",
    )
    parser.add_argument("--max-length", default=None, type=int, help="Defaults to distill.max_length.")
    parser.add_argument("--limit", default=0, type=int, help="Score only the first N traces, 0 for all.")
    parser.add_argument(
        "--keep-truncated",
        action="store_true",
        help=(
            "Include traces that hit the length cap. They are excluded by default "
            "because a cut-off trace's gap covers only the kept prefix."
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the report from seed results already on disk, loading no models.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cfg = FullConfig.from_yaml(Path(args.config))
    if cfg.model.proxy_student is None:
        raise ValueError(f"{args.config} has no model.proxy_student, so v_gap is undefined")
    max_length = args.max_length or cfg.distill.max_length
    drop_truncated = not args.keep_truncated

    sources = args.sources or discover_sources(input_dir)
    if BASELINE not in sources:
        raise ValueError(
            f"`{BASELINE}` must be among the sources; it is the gamma = 0 baseline "
            "every other source is compared against."
        )
    for source in sources:
        path = input_dir / f"train_{source}.json"
        if not path.exists():
            raise FileNotFoundError(f"No such trace file: {path}")

    from clean_sweep.utils import ensure_dir, write_json

    if not args.report_only:
        console.rule(f"v_gap by gamma | seed {args.seed}")
        console.print(f"  Teacher: {cfg.model.teacher}")
        console.print(f"  Proxy:   {cfg.model.proxy_student}")
        console.print(f"  Sources: {', '.join(sources)}\n")

        scored = score_sources(
            sources,
            input_dir,
            cfg,
            max_length=max_length,
            batch_size=args.batch_size,
            row_chunk=args.row_chunk,
            seed=args.seed,
            limit=args.limit,
        )
        if any(scored["clamped"].values()):
            console.print(
                f"  [yellow]Clamped out-of-vocabulary ids: {scored['clamped']}. "
                "Those tokens' log-probs are meaningless.[/yellow]"
            )

        summary = summarize_seed(scored["traces"], drop_truncated)
        seed_dir = ensure_dir(output_dir / f"seed_{args.seed}")
        write_json(scored["traces"], seed_dir / "per_trace.json")
        write_json(
            {
                "seed": args.seed,
                "input_dir": str(input_dir),
                "teacher": cfg.model.teacher,
                "proxy": cfg.model.proxy_student,
                "max_length": max_length,
                "drop_truncated": drop_truncated,
                "clamped_ids": scored["clamped"],
                "definition": (
                    "vgap(y) = log P_teacher(y) - log P_proxy(y) over response tokens, "
                    "both on teacher-tokenized ids; vgap_mean divides by response tokens"
                ),
                "created_at": datetime.now().isoformat(),
                "sources": summary,
            },
            seed_dir / "summary.json",
        )
        console.print(f"\n  Wrote {seed_dir}")

    by_seed = load_seed_files(output_dir)
    if not by_seed:
        console.print(f"[yellow]No seed results under {output_dir}. Nothing to report.[/yellow]")
        return

    pooled = pool_across_seeds(by_seed)
    report = format_report(
        pooled,
        seeds=sorted(by_seed),
        cfg=cfg,
        input_dir=input_dir,
        max_length=max_length,
        drop_truncated=drop_truncated,
    )
    ensure_dir(output_dir)
    # Written last and in one shot: several array tasks may reach this point at
    # once, and a partial file is worse than a stale one.
    tmp = output_dir / "RESULTS.md.partial"
    tmp.write_text(report)
    tmp.replace(output_dir / "RESULTS.md")
    write_json(pooled, output_dir / "pooled.json")

    console.rule("Average v_gap per response token")
    for source in _order_sources(pooled):
        entry = pooled[source]
        excluded = entry["n"] - entry["n_usable"]
        if excluded / max(entry["n"], 1) > 0.05:
            console.print(
                f"  [yellow]{source}: {excluded}/{entry['n']} traces excluded "
                "(truncated at the length cap or non-finite).[/yellow]"
            )
        gamma = entry.get("gamma")
        label = f"gamma={gamma:.2f}" if gamma is not None else "control"
        line = (
            f"  {source:30s} {label:>12s}  "
            f"v_gap/token = {entry['vgap_mean']:+.4f} +/- {entry['vgap_mean_stderr']:.4f}"
        )
        comparison = entry.get("vs_baseline")
        if comparison:
            line += (
                f"   change {comparison['delta_mean']:+.4f} "
                f"({comparison['frac_suppressed']:.0%} of problems down)"
            )
        console.print(line)
    console.print(f"\n  Wrote {output_dir / 'RESULTS.md'} from seeds {sorted(by_seed)}")


if __name__ == "__main__":
    main()
