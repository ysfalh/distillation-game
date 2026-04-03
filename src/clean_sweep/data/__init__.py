from .gsm8k import format_prompt_gsm8k, load_gsm8k_splits
from .gsm_hard import load_gsm_hard_splits
from .math import format_prompt_math, load_math_splits
from .svamp import load_svamp_splits


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
    if dataset_name == "gsm_hard":
        return load_gsm_hard_splits(
            seed=seed,
            train_size=train_size,
            holdout_size=holdout_size,
            test_size=test_size,
        )
    if dataset_name == "svamp":
        return load_svamp_splits(
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
    "load_gsm_hard_splits",
    "load_math_splits",
    "load_svamp_splits",
    "load_dataset_splits",
]
