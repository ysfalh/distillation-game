from .gsm8k import format_prompt_gsm8k, load_gsm8k_splits
from .math import format_prompt_math, load_math_splits


def load_dataset_splits(
    dataset_name: str,
    *,
    seed: int,
    train_size: int,
    holdout_size: int,
    test_size: int,
):
    if dataset_name == "gsm8k":
        return load_gsm8k_splits(
            seed=seed,
            train_size=train_size,
            holdout_size=holdout_size,
            test_size=test_size,
        )
    if dataset_name == "math":
        return load_math_splits(
            seed=seed,
            train_size=train_size,
            holdout_size=holdout_size,
            test_size=test_size,
        )
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


__all__ = [
    "format_prompt_gsm8k",
    "format_prompt_math",
    "load_gsm8k_splits",
    "load_math_splits",
    "load_dataset_splits",
]
