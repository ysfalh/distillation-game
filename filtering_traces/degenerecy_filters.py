"""Simple, answer-blind degeneracy filters for distillation traces.

A deliberately minimal alternative to the calibrated multi-rule filter.  There
are exactly two rules, no thresholds calibrated from data, and nothing tuned
per condition:

1. Strange-script rule.  Drop a trace if it contains any character from a
   script we do not expect in an English math trace (CJK, Cyrillic, Arabic,
   Hebrew, ...).  Latin and Greek are allowed, and mathematical symbols are
   allowed, because both are normal in mathematics.

2. Consecutive-repetition rule.  Drop a trace if any single token repeats at
   least ``min_consecutive_copies`` times in a row.

Both rules are one sentence to state and have no fitted parameters, which
keeps the criterion easy to defend.  The module reuses the script
classification and tokenizer handling from ``degenerecy_filters`` in style,
and has no third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from statistics import median
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence


Token = Any
IdToStr = Callable[[Token], str]


@dataclass(frozen=True)
class SimpleFilterConfig:
    """Configuration for the two-rule filter.

    Both fields are fixed integers, not calibrated from any condition.
    ``min_foreign_run`` = 1 means a single strange-script character is enough
    to drop the trace, the strictest reading of "remove strange symbols".
    ``min_consecutive_copies`` = 8 matches the hard single-token-loop bar used
    elsewhere in the project.
    """

    min_foreign_run: int = 1
    min_consecutive_copies: int = 8


def cached_id_to_str(tokenizer: Any) -> IdToStr:
    """Build a memoized token-id to surface-string map for an HF tokenizer.

    Byte-level BPE vocabularies (Qwen, Llama, DeepSeek) store tokens such as
    ``Ġthe`` or the byte pieces of a single CJK character.  Reading those raw
    strings would misclassify byte fragments, so the surface must go through
    ``convert_tokens_to_string``.  Memoizing keeps this to one call per
    distinct id instead of one per token occurrence.
    """

    @lru_cache(maxsize=None)
    def surface(token_id: int) -> str:
        piece = tokenizer.convert_ids_to_tokens(int(token_id))
        if piece is None:
            return ""
        return tokenizer.convert_tokens_to_string([piece])

    return surface


def _resolve_id_to_str(
    tokens: Sequence[Token],
    id_to_str: Optional[IdToStr],
) -> IdToStr:
    if id_to_str is not None:
        return id_to_str
    if all(isinstance(token, str) for token in tokens):
        return lambda token: token
    raise ValueError(
        "Integer token IDs require id_to_str. For a Hugging Face tokenizer, "
        "use cached_id_to_str(tokenizer)."
    )


@lru_cache(maxsize=None)
def _character_class(character: str) -> str:
    """Coarse script class sufficient for conservative English-task filtering."""
    name = unicodedata.name(character, "")
    if "LATIN" in name:
        return "latin"
    if "GREEK" in name:
        return "greek"
    if "MATHEMATICAL" in name:
        return "mathematical"
    return "other"


def _is_foreign_character(character: str) -> bool:
    """True if ``character`` is an alphabetic letter from an unexpected script.

    Latin and Greek letters are expected; mathematical alphanumerics are
    expected.  Digits, punctuation, whitespace, and symbols are never foreign.
    Everything else alphabetic (CJK, Cyrillic, Arabic, Hebrew, Devanagari,
    Hangul, Kana, Thai, ...) counts as foreign.
    """
    if not character.isalpha():
        return False
    return _character_class(character) == "other"


def strange_script_features(text: str) -> Dict[str, Any]:
    """Rule 1 features: longest run of consecutive foreign-script letters."""
    longest_run = 0
    current_run = 0
    off_characters = 0

    for character in text:
        if _is_foreign_character(character):
            off_characters += 1
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
        else:
            current_run = 0

    return {
        "foreign_characters": off_characters,
        "longest_foreign_run": longest_run,
    }


def consecutive_repeat_features(
    tokens: Sequence[Token],
) -> Dict[str, Any]:
    """Rule 2 features: longest run of a single token repeated in a row."""
    total = len(tokens)
    longest_run = 1 if total else 0
    current_run = 1 if total else 0

    for index in range(1, total):
        if tokens[index] == tokens[index - 1]:
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
        else:
            current_run = 1

    return {
        "n_tokens": total,
        "max_consecutive_token_copies": longest_run,
    }


def score_trace(
    text: str,
    tokens: Sequence[Token],
    *,
    id_to_str: Optional[IdToStr] = None,
    config: SimpleFilterConfig = SimpleFilterConfig(),
) -> Dict[str, Any]:
    """Apply both rules to one trace and return the combined verdict.

    ``id_to_str`` is accepted for interface parity with the calibrated filter,
    though Rule 2 only needs token equality.  Passing string tokens (e.g. from
    ``tokenizer.convert_ids_to_tokens``) lets the default identity map apply.
    """
    _resolve_id_to_str(tokens, id_to_str)  # validate token/id_to_str contract

    strange = strange_script_features(text)
    repeat = consecutive_repeat_features(tokens)

    strange_script_flag = bool(
        strange["longest_foreign_run"] >= config.min_foreign_run
    )
    repetition_flag = bool(
        repeat["max_consecutive_token_copies"] >= config.min_consecutive_copies
    )

    return {
        "strange_script_flag": strange_script_flag,
        "repetition_flag": repetition_flag,
        "degenerate": strange_script_flag or repetition_flag,
        "n_tokens": repeat["n_tokens"],
        "foreign_characters": strange["foreign_characters"],
        "longest_foreign_run": strange["longest_foreign_run"],
        "max_consecutive_token_copies": repeat["max_consecutive_token_copies"],
    }


def flag_reasons(scored: Dict[str, Any]) -> List[str]:
    """Name every rule that fired, for inspection of flagged traces."""
    checks = [
        ("strange_script", scored["strange_script_flag"]),
        ("consecutive_repetition", scored["repetition_flag"]),
    ]
    return [name for name, fired in checks if fired]


def condition_report(scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a scored condition for the rebuttal table."""
    if not scored:
        raise ValueError("Cannot report on an empty condition.")

    count = len(scored)
    both = sum(
        record["strange_script_flag"] and record["repetition_flag"]
        for record in scored
    )

    return {
        "n": count,
        "strange_script_rate": (
            sum(record["strange_script_flag"] for record in scored) / count
        ),
        "repetition_rate": (
            sum(record["repetition_flag"] for record in scored) / count
        ),
        "both_rate": both / count,
        "union_degenerate_rate": (
            sum(record["degenerate"] for record in scored) / count
        ),
        "median_tokens": median(record["n_tokens"] for record in scored),
    }


__all__ = [
    "SimpleFilterConfig",
    "cached_id_to_str",
    "condition_report",
    "consecutive_repeat_features",
    "flag_reasons",
    "score_trace",
    "strange_script_features",
]