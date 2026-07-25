#!/usr/bin/env python3
"""Sweep degeneracy thresholds over saved teacher traces.

Thresholds are always calibrated on the Standard teacher and then applied
unchanged to every other condition.  The sweep varies the target acceptance
rate of the Standard teacher, so each row answers "if we keep 95% of Standard
traces, how many ADS/PoE traces does the same filter drop?".

Usage:
    python3 filtering_traces/run_filter_sweep.py --input-dir gsm8k_output_small
    python3 filtering_traces/run_filter_sweep.py --input-dir gsm8k_output_small \
        --write-filtered

``--write-filtered`` additionally materializes one trace directory per
acceptance target, each a drop-in replacement for the source directory in
``scripts/run_student_from_saved_traces.py``.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import degenerecy_filters as F


DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_MAX_NEW_TOKENS = 1024
STANDARD_FILE = "train_standard.json"
DEFAULT_ACCEPTANCE = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]
# Targets below 0.95 drop exactly the same traces as 0.95, so materializing
# them would train identical students.
DEFAULT_FILTERED_POINTS = [0.95, 0.97, 0.99]
# `run_student_from_saved_traces.py` refuses to load a trace directory unless
# both of these sit next to the train files.
PASSTHROUGH_FILES = ["config_snapshot.yaml", "holdout_standard_internal.json"]


def load_conditions(input_dir: Path, limit: int) -> dict[str, list[dict[str, Any]]]:
    """Load ``train_*.json`` with Standard first, since it calibrates."""
    paths = sorted(input_dir.glob("train_*.json"))
    standard = input_dir / STANDARD_FILE
    if standard not in paths:
        raise FileNotFoundError(f"Calibration file missing: {standard}")
    paths = [standard] + [p for p in paths if p != standard]

    conditions = {}
    for path in paths:
        rows = json.loads(path.read_text())
        conditions[path.stem.removeprefix("train_")] = rows[:limit] if limit else rows
    return conditions


def read_max_new_tokens(input_dir: Path, fallback: int) -> int:
    snapshot = input_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        return fallback
    import yaml

    config = yaml.safe_load(snapshot.read_text())
    return int(config.get("generation", {}).get("max_new_tokens", fallback))


def extract_condition(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    id_to_str: F.IdToStr,
    loop_config: F.LoopFilterConfig,
) -> list[dict[str, Any]]:
    """Tokenize with the teacher tokenizer and extract raw features."""
    texts = [row["trace"] for row in rows]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    features = []
    for text, tokens in zip(texts, encoded):
        features.append(
            F.extract_features(
                text,
                tokens,
                id_to_str=id_to_str,
                # Traces are stored stripped of special tokens, so the token
                # count is a close but inexact stand-in for n_new_tokens.
                n_new_tokens=len(tokens),
                loop_config=loop_config,
            )
        )
    return features


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
        "loop_rate": report["loop_rate"],
        "language_rate": report["language_rate"],
        "accuracy_before": n_correct / max(len(rows), 1),
        "accuracy_after": n_kept_correct / max(len(kept), 1),
    }


def score_at(
    features: dict[str, list[dict[str, Any]]],
    target: float,
    max_new_tokens: int,
) -> tuple[F.FrozenThresholds, dict[str, list[dict[str, Any]]]]:
    """Calibrate on Standard at one acceptance target and score every condition.

    The fixed floors are disabled so the verdict is driven purely by the
    target acceptance rate on Standard.
    """
    loop_config = F.LoopFilterConfig(
        max_new_tokens=max_new_tokens,
        repetition_fraction_floor=0.0,
        standard_quantile=target,
    )
    language_config = F.LanguageFilterConfig(
        off_script_fraction_floor=0.0,
        standard_quantile=target,
    )
    thresholds = F.calibrate_thresholds(
        features["standard"],
        loop_config=loop_config,
        language_config=language_config,
    )
    scored = {
        name: [
            F.apply_thresholds(
                feature,
                thresholds,
                loop_config=loop_config,
                language_config=language_config,
            )
            for feature in condition_features
        ]
        for name, condition_features in features.items()
    }
    return thresholds, scored


def run_sweep(
    features: dict[str, list[dict[str, Any]]],
    conditions: dict[str, list[dict[str, Any]]],
    acceptance_rates: list[float],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    sweep = []
    for target in acceptance_rates:
        thresholds, scored = score_at(features, target, max_new_tokens)
        sweep.append({
            "target_acceptance": target,
            "thresholds": F.thresholds_as_dict(thresholds),
            "conditions": {
                name: summarize(conditions[name], scored[name])
                for name in features
            },
        })
    return sweep


def write_filtered_datasets(
    input_dir: Path,
    out_root: Path,
    conditions: dict[str, list[dict[str, Any]]],
    features: dict[str, list[dict[str, Any]]],
    targets: list[float],
    max_new_tokens: int,
    tokenizer_name: str,
) -> None:
    """Write one drop-in trace directory per acceptance target.

    Rows are copied verbatim and degenerate ones are omitted, so each folder
    can replace the source directory in the student runner.  The holdout and
    the config snapshot are copied unchanged, since the holdout must stay
    identical across conditions for the strategic-FD gradients.

    The dropped traces are kept under ``removed/accept_*`` purely for
    inspection.  Those directories lack the config snapshot and the holdout,
    so the student runner cannot load them by accident.
    """
    for target in targets:
        thresholds, scored = score_at(features, target, max_new_tokens)
        folder = out_root / f"accept_{target:.2f}"
        folder.mkdir(parents=True, exist_ok=True)
        removed_folder = out_root / "removed" / f"accept_{target:.2f}"
        removed_folder.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "source_dir": str(input_dir),
            "target_acceptance": target,
            "tokenizer": tokenizer_name,
            "max_new_tokens": max_new_tokens,
            "thresholds": F.thresholds_as_dict(thresholds),
            "conditions": {},
        }
        for name, rows in conditions.items():
            verdicts = scored[name]
            kept = [row for row, s in zip(rows, verdicts) if not s["degenerate"]]
            # The removed traces are for inspection only, so they carry the
            # rules that fired.
            dropped = [
                {**row, "filter_reasons": F.flag_reasons(s)}
                for row, s in zip(rows, verdicts)
                if s["degenerate"]
            ]
            path = folder / f"train_{name}.json"
            path.write_text(json.dumps(kept, indent=2))
            (removed_folder / f"train_{name}.json").write_text(
                json.dumps(dropped, indent=2)
            )
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
        counts = ", ".join(
            f"{name}={entry['n_after']}(-{entry['n_dropped']})"
            for name, entry in manifest["conditions"].items()
        )
        print(f"  {folder}: {counts}")


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
        "repetition_fraction": round(scored["repetition_fraction"], 4),
        "off_script_fraction": round(scored["off_script_fraction"], 4),
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
    max_new_tokens: int,
    conditions: dict[str, list[dict[str, Any]]],
    sweep: list[dict[str, Any]],
    operating_point: float,
) -> str:
    names = list(conditions)
    lines = [
        "# Degeneracy filter sweep",
        "",
        f"- Traces: `{input_dir}`",
        f"- Tokenizer: `{tokenizer_name}`",
        f"- `max_new_tokens`: {max_new_tokens}",
        f"- Conditions: " + ", ".join(f"`{n}` (n={len(conditions[n])})" for n in names),
        "",
        "Thresholds are calibrated on the Standard teacher alone and then "
        "applied unchanged to every condition. Each row is one target "
        "acceptance rate for Standard. The realized rate is higher than the "
        "target and saturates once the target gets aggressive, because the "
        "hard loop rules and the 64-token minimum removal gate fire "
        "regardless of the calibrated thresholds.",
        "",
        "## Dropped traces by target Standard acceptance",
        "",
    ]

    header = ["target accept", "kept standard", "rep thr", "off-script thr"]
    header += [f"{name} dropped" for name in names]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for entry in sweep:
        standard = entry["conditions"]["standard"]
        cells = [
            f"{entry['target_acceptance']:.3f}",
            f"{1 - standard['drop_rate']:.4f}",
            f"{entry['thresholds']['repetition_fraction']:.4f}",
            f"{entry['thresholds']['off_script_fraction']:.4f}",
        ]
        for name in names:
            row = entry["conditions"][name]
            cells.append(f"{row['dropped']} ({row['drop_rate']:.2%})")
        lines.append("| " + " | ".join(cells) + " |")

    chosen = next(e for e in sweep if e["target_acceptance"] == operating_point)
    lines += [
        "",
        f"## Operating point: target acceptance {operating_point:.3f}",
        "",
        "| condition | n | dropped | drop rate | loop | off-language | "
        "accuracy before | accuracy after |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        row = chosen["conditions"][name]
        lines.append(
            f"| {name} | {row['n']} | {row['dropped']} | {row['drop_rate']:.2%} | "
            f"{row['loop_rate']:.2%} | {row['language_rate']:.2%} | "
            f"{row['accuracy_before']:.4f} | {row['accuracy_after']:.4f} |"
        )
    lines += [
        "",
        "`loop` and `off-language` overlap, so they do not sum to the drop "
        "rate. Sampled traces from this operating point are in "
        "`trace_degenerate/`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--acceptance",
        type=float,
        nargs="+",
        default=DEFAULT_ACCEPTANCE,
        help="Target Standard acceptance rates to sweep.",
    )
    parser.add_argument("--operating-point", type=float, default=0.99)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Traces per condition, 0 = all.")
    parser.add_argument(
        "--write-filtered",
        action="store_true",
        help="Also write filtered trace directories ready for SFT.",
    )
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        default=None,
        help="Root for filtered directories. Default: filtering_traces/filtered/<input name>.",
    )
    parser.add_argument(
        "--filtered-points",
        type=float,
        nargs="+",
        default=DEFAULT_FILTERED_POINTS,
        help="Acceptance targets to materialize as directories.",
    )
    args = parser.parse_args()

    if args.operating_point not in args.acceptance:
        raise ValueError("--operating-point must be one of the --acceptance values")

    out_dir = args.out or Path("filtering_traces/results") / args.input_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = load_conditions(args.input_dir, args.limit)
    max_new_tokens = read_max_new_tokens(args.input_dir, args.max_new_tokens)
    print(f"Conditions: {', '.join(f'{k} (n={len(v)})' for k, v in conditions.items())}")
    print(f"max_new_tokens={max_new_tokens}, tokenizer={args.tokenizer}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    id_to_str = F.cached_id_to_str(tokenizer)
    loop_config = F.LoopFilterConfig(max_new_tokens=max_new_tokens)

    features = {}
    for name, rows in conditions.items():
        start = time.perf_counter()
        features[name] = extract_condition(rows, tokenizer, id_to_str, loop_config)
        print(f"  {name}: features for {len(rows)} traces in {time.perf_counter() - start:.1f}s")

    sweep = run_sweep(features, conditions, args.acceptance, max_new_tokens)
    report = markdown_report(
        args.input_dir,
        args.tokenizer,
        max_new_tokens,
        conditions,
        sweep,
        args.operating_point,
    )
    (out_dir / "sweep.md").write_text(report)
    (out_dir / "sweep.json").write_text(json.dumps({
        "input_dir": str(args.input_dir),
        "tokenizer": args.tokenizer,
        "max_new_tokens": max_new_tokens,
        "operating_point": args.operating_point,
        "sweep": sweep,
    }, indent=2))

    _, scored = score_at(features, args.operating_point, max_new_tokens)
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
                  "directories will be truncated and unfit for SFT")
        filtered_root = (
            args.filtered_dir
            or Path("filtering_traces/filtered") / args.input_dir.name
        )
        print(f"\nWriting filtered trace directories under {filtered_root}")
        write_filtered_datasets(
            args.input_dir,
            filtered_root,
            conditions,
            features,
            args.filtered_points,
            max_new_tokens,
            args.tokenizer,
        )


if __name__ == "__main__":
    main()
