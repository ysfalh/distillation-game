from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset


SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
    "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
)


def _with_example_ids(dataset: Dataset, prefix: str) -> Dataset:
    return dataset.add_column("example_id", [f"{prefix}_{i}" for i in range(len(dataset))])


def load_gsm8k_splits(
    *,
    seed: int,
    train_size: int,
    holdout_size: int,
    test_size: int,
) -> dict[str, Any]:
    full = load_dataset("openai/gsm8k", "main", split="train")
    full = full.rename_columns({"question": "problem", "answer": "solution"})
    full = full.shuffle(seed=seed)

    train = full.select(range(min(train_size, len(full))))
    holdout_start = len(train)
    holdout_end = min(holdout_start + holdout_size, len(full))
    holdout = full.select(range(holdout_start, holdout_end))

    test = load_dataset("madrylab/gsm8k-platinum", split="test")
    test = test.rename_columns({"question": "problem", "answer": "solution"})
    test = test.select(range(min(test_size, len(test))))

    return {
        "train": _with_example_ids(train, "train"),
        "holdout": _with_example_ids(holdout, "holdout"),
        "test": _with_example_ids(test, "test"),
    }


def format_prompt_gsm8k(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.strip() + "\n"},
    ]
