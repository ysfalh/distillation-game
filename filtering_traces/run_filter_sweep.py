#!/usr/bin/env python3
"""Run the two-rule degeneracy filter over saved teacher traces.

The filter has no calibrated thresholds, so there is no acceptance-rate sweep
to run: every condition is judged by the same two fixed rules. What the script
still reports is how far the verdict moves when the one integer knob that
matters, ``min_consecutive_copies``, is varied.

Usage:
    python3 filtering_traces/run_filter_sweep.py --input-dir gsm8k_output_small
    python3 filtering_traces/run_filter_sweep.py --input-dir gsm8k_output_small \
        --write-filtered

``--write-filtered`` additionally materializes a filtered trace directory that
is a drop-in replacement for the source directory in
``scripts/run_student_from_saved_traces.py``.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import degenerecy_filters as F


DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
STANDARD_FILE = "train_standard.json"
# The repetition bar is the only knob with a debatable value, so the report
# shows what the drop counts would have been at these alternatives.
DEFAULT_REPEAT_POINTS = [4, 6, 8, 12, 16, 32]
# `run_student_from_saved_traces.py` refuses to load a trace directory unless
# both of these sit next to the train files.
PASSTHROUGH_FILES = ["config_snapshot.yaml", "holdout_standard_internal.json"]
FILTERED_DIR_NAME = "simple"


def load_conditions(input_dir: Path, limit: int) -> dict[str, list[dict[str, Any]]]:
    """Load every ``train_*.json``, Standard first when it is present."""
    paths = sorted(input_dir.glob("train_*.json"))
    if not paths:
        raise FileNotFoundError(f"No train_*.json under {input_dir}")
    standard = input_dir / STANDARD_FILE
    if standard in paths:
        paths = [standard] + [p for p in paths if p != standard]

    conditions = {}
    for path in paths:
        rows = json.loads(path.read_text())
        conditions[path.stem.removeprefix("train_")] = rows[:limit] if limit else rows
    return conditions


def score_condition(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    id_to_str: F.IdToStr,
    config: F.SimpleFilterConfig,
) -> list[dict[str, Any]]:
    """Tokenize with the teacher tokenizer and apply both rules."""
    texts = [row["trace"] for row in rows]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [
        F.score_trace(text, tokens, id_to_str=id_to_str, config=config)
        for text, tokens in zip(texts, encoded)
    ]


def reflag(record: dict[str, Any], config: F.SimpleFilterConfig) -> dict[str, Any]:
    """Re-apply both rules to features that were already extracted.

    Scoring is dominated by the text and token scans, and neither depends on
    the thresholds, so the sensitivity table reuses the per-trace numbers
    ``score_trace`` already reported instead of scoring every trace again.
    """
    strange = record["longest_foreign_run"] >= config.min_foreign_run
    repetition = record["max_consecutive_token_copies"] >= config.min_consecutive_copies
    return {
        **record,
        "strange_script_flag": strange,
        "repetition_flag": repetition,
        "degenerate": strange or repetition,
    }


def summarize(
    rows: list[dict[str, Any]],
    scored: list[dict[str, Any]],
) -> dict[str, Any]:
    """Drop counts plus the effect of filtering on trace accuracy."""
    report = F.condition_report(scored)
    kept = [row for row, s in zip(rows, scored) if not s["degenerate"]]
    n_correct = sum(1 for row in rows if row.get("correct"))
    n_kept_correct = sum(1 for row in kept if row.get("correct"))
    return {
        "n": len(rows),
        "dropped": sum(1 for s in scored if s["degenerate"]),
        "drop_rate": report["union_degenerate_rate"],
        "strange_script_rate": report["strange_script_rate"],
        "repetition_rate": report["repetition_rate"],
        "both_rate": report["both_rate"],
        "median_tokens": report["median_tokens"],
        "accuracy_before": n_correct / max(len(rows), 1),
        "accuracy_after": n_kept_correct / max(len(kept), 1),
    }


def foreign_characters(
    rows: list[dict[str, Any]],
    scored: list[dict[str, Any]],
) -> Counter:
    """Which characters actually triggered the strange-script rule.

    Rule 1 treats every non-Latin, non-Greek letter as foreign, which also
    catches letterlike math symbols such as the double-struck R. Counting the
    offenders is the only way to tell whether that is a real problem for a
    given corpus or a hypothetical one.
    """
    counts: Counter = Counter()
    for row, record in zip(rows, scored):
        if not record["strange_script_flag"]:
            continue
        counts.update(c for c in row["trace"] if F._is_foreign_character(c))
    return counts


def example_record(
    condition: str,
    row: dict[str, Any],
    scored: dict[str, Any],
) -> dict[str, Any]:
    return {
        "condition": condition,
        "example_id": row["example_id"],
        "method": row["method"],
        "correct": row.get("correct"),
        "degenerate": scored["degenerate"],
        "reasons": F.flag_reasons(scored),
        "n_tokens": scored["n_tokens"],
        "longest_foreign_run": scored["longest_foreign_run"],
        "max_consecutive_token_copies": scored["max_consecutive_token_copies"],
        "prompt": row["prompt"],
        "trace": row["trace"],
    }


def collect_examples(
    conditions: dict[str, list[dict[str, Any]]],
    scored: dict[str, list[dict[str, Any]]],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    degenerate, clean = [], []
    for name, rows in conditions.items():
        for row, record in zip(rows, scored[name]):
            target = degenerate if record["degenerate"] else clean
            target.append(example_record(name, row, record))

    rng = random.Random(seed)
    rng.shuffle(degenerate)
    rng.shuffle(clean)
    return degenerate[:count], clean[:count]


def markdown_report(
    input_dir: Path,
    tokenizer_name: str,
    config: F.SimpleFilterConfig,
    conditions: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    offenders: Counter,
) -> str:
    names = list(conditions)
    lines = [
        "# Degeneracy filter report",
        "",
        f"- Traces: `{input_dir}`",
        f"- Tokenizer: `{tokenizer_name}`",
        f"- Conditions: " + ", ".join(f"`{n}` (n={len(conditions[n])})" for n in names),
        "",
        "Two fixed rules, identical for every condition and calibrated on "
        "nothing. A trace is dropped if it contains a letter from an "
        f"unexpected script (run of {config.min_foreign_run} or more) or if any "
        f"single token repeats {config.min_consecutive_copies} or more times in "
        "a row.",
        "",
        "## Dropped traces",
        "",
        "| condition | n | dropped | drop rate | strange script | repetition | "
        "both | accuracy before | accuracy after |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        row = summaries[name]
        lines.append(
            f"| {name} | {row['n']} | {row['dropped']} | {row['drop_rate']:.2%} | "
            f"{row['strange_script_rate']:.2%} | {row['repetition_rate']:.2%} | "
            f"{row['both_rate']:.2%} | {row['accuracy_before']:.4f} | "
            f"{row['accuracy_after']:.4f} |"
        )

    lines += [
        "",
        "The two rules overlap, so `strange script` and `repetition` do not "
        "sum to the drop rate. Sampled traces are in `trace_degenerate/`.",
        "",
        "## Sensitivity to the repetition bar",
        "",
        "The strange-script rule is unchanged in every row; only "
        "`min_consecutive_copies` moves.",
        "",
    ]
    header = ["min consecutive copies"] + [f"{name} dropped" for name in names]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for entry in sensitivity:
        cells = [str(entry["min_consecutive_copies"])]
        for name in names:
            row = entry["conditions"][name]
            cells.append(f"{row['dropped']} ({row['drop_rate']:.2%})")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Characters that triggered the strange-script rule", ""]
    if not offenders:
        lines.append("No trace contained a foreign-script letter.")
    else:
        lines.append("| character | name | count |")
        lines.append("|---|---|---|")
        for character, count in offenders.most_common(15):
            name = unicodedata.name(character, "unnamed")
            lines.append(f"| `{character}` | {name} | {count} |")
    lines.append("")
    return "\n".join(lines)


def staging_dir(final: Path) -> Path:
    """Empty scratch directory next to where the output will end up."""
    staging = final.parent / f".{final.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def publish(staging: Path, final: Path) -> None:
    """Move a fully written directory into place, replacing any older one."""
    if final.exists():
        shutil.rmtree(final)
    staging.replace(final)


def write_filtered_dataset(
    input_dir: Path,
    out_root: Path,
    conditions: dict[str, list[dict[str, Any]]],
    scored: dict[str, list[dict[str, Any]]],
    config: F.SimpleFilterConfig,
    tokenizer_name: str,
) -> None:
    """Write a drop-in trace directory holding only the kept traces.

    Rows are copied verbatim and degenerate ones are omitted, so the folder can
    replace the source directory in the student runner. The holdout and the
    config snapshot are copied unchanged, since the holdout must stay identical
    across conditions for the strategic-FD gradients. Dropped traces are kept
    under ``removed/`` for inspection, without those two files so the student
    runner cannot load them by accident.

    The directory is staged and only then moved into place, so an interrupted
    run leaves nothing half-written for the student runner to trip over.
    """
    final_folder = out_root / FILTERED_DIR_NAME
    final_removed = out_root / "removed" / FILTERED_DIR_NAME
    folder = staging_dir(final_folder)
    removed_folder = staging_dir(final_removed)

    manifest: dict[str, Any] = {
        "source_dir": str(input_dir),
        "filter": "two_rule_simple",
        "tokenizer": tokenizer_name,
        "min_foreign_run": config.min_foreign_run,
        "min_consecutive_copies": config.min_consecutive_copies,
        "conditions": {},
    }
    for name, rows in conditions.items():
        verdicts = scored[name]
        kept = [row for row, s in zip(rows, verdicts) if not s["degenerate"]]
        # The removed traces are for inspection only, so they carry the rules
        # that fired.
        dropped = [
            {**row, "filter_reasons": F.flag_reasons(s)}
            for row, s in zip(rows, verdicts)
            if s["degenerate"]
        ]
        path = folder / f"train_{name}.json"
        path.write_text(json.dumps(kept, indent=2))
        (removed_folder / f"train_{name}.json").write_text(json.dumps(dropped, indent=2))
        manifest["conditions"][name] = {
            "file": path.name,
            "n_before": len(rows),
            "n_after": len(kept),
            "n_dropped": len(dropped),
            "dropped_example_ids": [row["example_id"] for row in dropped],
        }

    for filename in PASSTHROUGH_FILES:
        source = input_dir / filename
        if source.exists():
            shutil.copy2(source, folder / filename)
        else:
            print(f"  warning: {source} missing, {folder.name} will not "
                  "load in run_student_from_saved_traces.py")

    (folder / "filter_manifest.json").write_text(json.dumps(manifest, indent=2))
    publish(folder, final_folder)
    publish(removed_folder, final_removed)
    counts = ", ".join(
        f"{name}={entry['n_after']}(-{entry['n_dropped']})"
        for name, entry in manifest["conditions"].items()
    )
    print(f"  {final_folder}: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--min-foreign-run",
        type=int,
        default=F.SimpleFilterConfig().min_foreign_run,
        help="Foreign-script letters in a row needed to drop a trace.",
    )
    parser.add_argument(
        "--min-consecutive-copies",
        type=int,
        default=F.SimpleFilterConfig().min_consecutive_copies,
        help="Copies of one token in a row needed to drop a trace.",
    )
    parser.add_argument(
        "--repeat-points",
        type=int,
        nargs="+",
        default=DEFAULT_REPEAT_POINTS,
        help="Alternative repetition bars for the sensitivity table.",
    )
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Traces per condition, 0 for all.")
    parser.add_argument(
        "--write-filtered",
        action="store_true",
        help="Also materialize a filtered trace directory for SFT.",
    )
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        default=None,
        help="Root for the filtered directory. Default: filtering_traces/filtered/<input name>.",
    )
    args = parser.parse_args()

    out_dir = args.out or Path("filtering_traces/results") / args.input_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = F.SimpleFilterConfig(
        min_foreign_run=args.min_foreign_run,
        min_consecutive_copies=args.min_consecutive_copies,
    )
    conditions = load_conditions(args.input_dir, args.limit)
    print(f"Conditions: {', '.join(f'{k} (n={len(v)})' for k, v in conditions.items())}")
    print(f"tokenizer={args.tokenizer}, config={config}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    id_to_str = F.cached_id_to_str(tokenizer)

    scored = {}
    for name, rows in conditions.items():
        start = time.perf_counter()
        scored[name] = score_condition(rows, tokenizer, id_to_str, config)
        print(f"  {name}: scored {len(rows)} traces in {time.perf_counter() - start:.1f}s")

    summaries = {name: summarize(conditions[name], scored[name]) for name in conditions}
    sensitivity = [
        {
            "min_consecutive_copies": bar,
            "conditions": {
                name: summarize(
                    conditions[name],
                    [reflag(r, F.SimpleFilterConfig(args.min_foreign_run, bar))
                     for r in scored[name]],
                )
                for name in conditions
            },
        }
        for bar in args.repeat_points
    ]
    offenders: Counter = Counter()
    for name in conditions:
        offenders.update(foreign_characters(conditions[name], scored[name]))

    report = markdown_report(
        args.input_dir, args.tokenizer, config, conditions, summaries, sensitivity, offenders
    )
    (out_dir / "sweep.md").write_text(report)
    (out_dir / "sweep.json").write_text(json.dumps({
        "input_dir": str(args.input_dir),
        "tokenizer": args.tokenizer,
        "filter": "two_rule_simple",
        "min_foreign_run": config.min_foreign_run,
        "min_consecutive_copies": config.min_consecutive_copies,
        "conditions": summaries,
        "sensitivity": sensitivity,
        "foreign_characters": {c: n for c, n in offenders.most_common()},
    }, indent=2))

    degenerate, clean = collect_examples(conditions, scored, args.examples, args.seed)
    example_dir = out_dir / "trace_degenerate"
    example_dir.mkdir(exist_ok=True)
    (example_dir / "degenerate.json").write_text(json.dumps(degenerate, indent=2, ensure_ascii=False))
    (example_dir / "clean.json").write_text(json.dumps(clean, indent=2, ensure_ascii=False))

    print(f"\nWrote {out_dir}/sweep.md and sweep.json")
    print(f"Wrote {len(degenerate)} degenerate and {len(clean)} clean examples to {example_dir}")

    if args.write_filtered:
        if args.limit:
            print(f"\nwarning: --limit {args.limit} is set, so the filtered "
                  "directory will be truncated and unfit for SFT")
        filtered_root = (
            args.filtered_dir
            or Path("filtering_traces/filtered") / args.input_dir.name
        )
        print(f"\nWriting filtered trace directory under {filtered_root}")
        write_filtered_dataset(
            args.input_dir, filtered_root, conditions, scored, config, args.tokenizer
        )


if __name__ == "__main__":
    main()
