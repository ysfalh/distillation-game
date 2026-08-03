#!/usr/bin/env python3
"""Enumerate the (dataset, teacher source) pairs the filtered-trace array trains on.

Both ``slurm_scripts/filter-traces-prep.sbatch``, which sizes the array, and
``slurm_scripts/run-student-filtered-traces.sbatch``, which indexes into it,
call this so the two cannot disagree about what task N is.

The pairs come from the source trace directories rather than the filtered ones
because prep has to size the array before it has finished filtering. The filter
writes one ``train_<source>.json`` per input condition regardless of how many
rows survive, so both directories always hold the same set of conditions.

Only the defended families are listed by default. The clean `standard` teacher
is not a filtering condition: the filter has next to nothing to remove from it,
so it earns a student run only when asked for explicitly.

Usage:
    python filtering_traces/list_filter_combos.py gsm8k_output_small
    python filtering_traces/list_filter_combos.py gsm8k_output_small --families standard poe
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


KNOWN_FAMILIES = ("standard", "antidistillation", "poe")
DEFAULT_FAMILIES = ("antidistillation", "poe")
FAMILY_RANK = {name: rank for rank, name in enumerate(KNOWN_FAMILIES)}
TRAILING_NUMBER = re.compile(r"(\d+(?:\.\d+)?)$")


def family_of(source: str) -> str:
    """Which teacher family a trace stem belongs to."""
    if source == "standard":
        return "standard"
    if source.startswith("antidistillation_lam_"):
        return "antidistillation"
    if source.startswith("poe_gamma_"):
        return "poe"
    return "other"


def sort_key(source: str) -> tuple[int, float, str]:
    """Standard first, then each family ordered by its defense strength.

    Sorting on the parsed number rather than the string keeps lam 0.05 ahead of
    lam 0.052 and, unlike a lexicographic sort, would also keep gamma 0.15
    ahead of gamma 0.9 if such a pair is ever generated.
    """
    match = TRAILING_NUMBER.search(source)
    value = float(match.group(1)) if match else 0.0
    return (FAMILY_RANK.get(family_of(source), len(KNOWN_FAMILIES)), value, source)


def sources_in(dataset: Path, families: set[str]) -> list[str]:
    stems = [path.stem.removeprefix("train_") for path in dataset.glob("train_*.json")]
    kept = [stem for stem in stems if family_of(stem) in families]
    for skipped in sorted(set(stems) - set(kept)):
        print(
            f"note: skipping {dataset}/train_{skipped}.json, family "
            f"{family_of(skipped)!r} not requested",
            file=sys.stderr,
        )
    return sorted(kept, key=sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="Saved-trace directories, e.g. gsm8k_output_small.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(DEFAULT_FAMILIES),
        choices=list(KNOWN_FAMILIES),
        help="Teacher families to include. Default: the defended families only.",
    )
    args = parser.parse_args()

    families = set(args.families)
    total = 0
    for dataset in args.datasets:
        if not dataset.is_dir():
            raise SystemExit(f"error: {dataset} is not a directory")
        sources = sources_in(dataset, families)
        if not sources:
            raise SystemExit(
                f"error: no train_*.json under {dataset} for families {sorted(families)}"
            )
        for source in sources:
            print(f"{dataset.name}|{source}")
            total += 1

    print(f"note: {total} (dataset, source) pairs", file=sys.stderr)


if __name__ == "__main__":
    main()
