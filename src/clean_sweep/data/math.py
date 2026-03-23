from __future__ import annotations

from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from .gsm8k import SYSTEM_PROMPT


MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _with_example_ids(dataset: Dataset, prefix: str) -> Dataset:
    return dataset.add_column("example_id", [f"{prefix}_{i}" for i in range(len(dataset))])


def _load_math_split(split: str) -> Dataset:
    return concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", subset, split=split) for subset in MATH_SUBSETS])


def load_math_splits(
    *,
    seed: int,
    train_size: int,
    holdout_size: int,
    test_size: int,
) -> dict[str, Any]:
    full = _load_math_split("train")
    full = full.shuffle(seed=seed)

    train = full.select(range(min(train_size, len(full))))
    holdout_start = len(train)
    holdout_end = min(holdout_start + holdout_size, len(full))
    holdout = full.select(range(holdout_start, holdout_end))

    test = _load_math_split("test")
    test = test.select(range(min(test_size, len(test))))

    return {
        "train": _with_example_ids(train, "train"),
        "holdout": _with_example_ids(holdout, "holdout"),
        "test": _with_example_ids(test, "test"),
    }


def format_prompt_math(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.strip() + "\n"},
    ]
