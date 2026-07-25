"""High-precision, answer-blind degeneracy filters for distillation traces.

The primary use case is the passive/adaptive decomposition requested in the
rebuttal.  The implementation deliberately separates:

1. hard loop indicators (long, consecutive, exact repetitions);
2. a continuous repetition-cleaning score;
3. an off-language diagnostic for English-language tasks.

Threshold protocol
------------------
Extract features on STANDARD-teacher traces first, call
``calibrate_thresholds``, freeze the resulting thresholds, and then apply the
same thresholds to Standard, ADS, and PoE traces.  Never recalibrate on a
defended condition.  ``run_filter_sweep.py`` drives exactly this protocol over
saved trace directories.

The module has no third-party dependencies.  ``cached_id_to_str`` accepts a
Hugging Face tokenizer but only duck-types it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import re
from statistics import median
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Token = Any
IdToStr = Callable[[Token], str]


@dataclass(frozen=True)
class LoopFilterConfig:
    """Configuration for the repetition detector."""

    # General three-pass cleaning score.
    cleaning_max_block: int = 6
    cleaning_passes: int = 3

    # High-precision hard indicators on the original trace.
    single_token_min_copies: int = 8
    phrase_max_block: int = 6
    phrase_min_copies: int = 6
    punctuation_loop_min_tokens: int = 64
    repeated_line_min_copies: int = 3

    # Capped periodic suffix.  ``max_new_tokens`` must match the generation
    # config that produced the traces.  ``cap_slack`` is generous because
    # saved traces usually only allow ``n_new_tokens`` to be recovered by
    # re-tokenizing stripped text, which loses a few tokens.
    suffix_min_tokens: int = 64
    suffix_max_period: int = 32
    max_new_tokens: int = 1024
    cap_slack: int = 16

    # Continuous-score rule.
    removed_token_minimum: int = 64
    repetition_fraction_floor: float = 0.20
    standard_quantile: float = 0.99


@dataclass(frozen=True)
class LanguageFilterConfig:
    """Configuration for off-language detection on English tasks."""

    minimum_off_script_characters: int = 20
    off_script_fraction_floor: float = 0.10
    minimum_off_script_span_words: int = 30
    # Scripts without word breaks (CJK) produce one enormous "word", so the
    # word-run rule alone cannot see them.
    minimum_off_script_span_characters: int = 40
    standard_quantile: float = 0.99


@dataclass(frozen=True)
class FrozenThresholds:
    """Thresholds calibrated once on Standard-teacher traces."""

    repetition_fraction: float
    off_script_fraction: float


@dataclass(frozen=True)
class RepeatRegion:
    """A maximal consecutive repetition selected by the greedy cleaner."""

    start: int
    period: int
    copies: int

    @property
    def end(self) -> int:
        return self.start + self.period * self.copies

    @property
    def redundant_tokens(self) -> int:
        return self.period * (self.copies - 1)


def cached_id_to_str(tokenizer: Any) -> IdToStr:
    """Build a memoized token-id to surface-string map for an HF tokenizer.

    Byte-level BPE vocabularies (Qwen, Llama, DeepSeek) store tokens such as
    ``Ġthe`` or the three byte pieces of a single CJK character.  Reading
    those raw strings would misclassify byte fragments as Latin text, so the
    surface must go through ``convert_tokens_to_string``.  Memoizing keeps
    this to one call per distinct id instead of one per token occurrence.
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


def _has_alphanumeric(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _minimal_period(block: Sequence[Token]) -> int:
    """Return the shortest exact period of a finite block."""
    length = len(block)
    for period in range(1, length + 1):
        if length % period != 0:
            continue
        if list(block) == list(block[:period]) * (length // period):
            return period
    return length


def _consecutive_copies(
    tokens: Sequence[Token],
    start: int,
    period: int,
) -> int:
    """Count exact consecutive copies of tokens[start:start + period]."""
    total = len(tokens)
    if start + period > total:
        return 0
    block = tokens[start:start + period]
    copies = 1
    cursor = start + period
    while (
        cursor + period <= total
        and tokens[cursor:cursor + period] == block
    ):
        copies += 1
        cursor += period
    return copies


def greedy_repeat_regions(
    tokens: Sequence[Token],
    max_block: int = 6,
) -> List[RepeatRegion]:
    """Find non-overlapping consecutive repetitions for cleaning.

    At each position, this examines all allowed periods and selects the
    candidate that removes the most redundant tokens.  Ties favor the shorter
    period.  This avoids the failure mode where a locally repeated short
    prefix hides a much larger repeated block.

    Regions with only two copies are intentionally included here: this is the
    broad *cleaning score*, not the high-precision hard-loop verdict.
    """
    total = len(tokens)
    regions: List[RepeatRegion] = []
    index = 0

    while index < total:
        best: Optional[RepeatRegion] = None

        for period in range(1, max_block + 1):
            if index + 2 * period > total:
                break

            copies = _consecutive_copies(tokens, index, period)
            if copies < 2:
                continue

            candidate = RepeatRegion(index, period, copies)
            if best is None:
                best = candidate
            elif candidate.redundant_tokens > best.redundant_tokens:
                best = candidate
            elif (
                candidate.redundant_tokens == best.redundant_tokens
                and candidate.period < best.period
            ):
                best = candidate

        if best is None:
            index += 1
        else:
            regions.append(best)
            index = best.end

    return regions


def _clean_once(
    tokens: Sequence[Token],
    regions: Sequence[RepeatRegion],
) -> Tuple[List[Token], int]:
    """Collapse every selected repeated region to its first copy."""
    output: List[Token] = []
    removed = 0
    cursor = 0

    for region in regions:
        if region.start < cursor:
            raise ValueError("Repeat regions must be sorted and non-overlapping.")
        output.extend(tokens[cursor:region.start])
        output.extend(tokens[region.start:region.start + region.period])
        removed += region.redundant_tokens
        cursor = region.end

    output.extend(tokens[cursor:])
    return output, removed


def three_pass_clean(
    tokens: Sequence[Token],
    max_block: int = 6,
    passes: int = 3,
) -> Tuple[List[Token], int, Dict[str, Any]]:
    """Apply broad consecutive-repeat cleaning for up to ``passes`` passes.

    The cleaner may collapse harmless double occurrences.  Consequently,
    ``total_removed`` is a continuous diagnostic and should only become a
    binary filter after calibration on Standard traces.
    """
    current = list(tokens)
    total_removed = 0
    removed_per_pass: List[int] = []
    regions_per_pass: List[int] = []

    for _ in range(passes):
        regions = greedy_repeat_regions(current, max_block=max_block)
        if not regions:
            break
        current, removed = _clean_once(current, regions)
        removed_per_pass.append(removed)
        regions_per_pass.append(len(regions))
        total_removed += removed
        if removed == 0:
            break

    metadata = {
        "removed_per_pass": removed_per_pass,
        "regions_per_pass": regions_per_pass,
        "passes_run": len(removed_per_pass),
    }
    return current, total_removed, metadata


def hard_repeat_features(
    tokens: Sequence[Token],
    id_to_str: Optional[IdToStr] = None,
    config: LoopFilterConfig = LoopFilterConfig(),
) -> Dict[str, Any]:
    """Find high-precision loop indicators on the unmodified trace.

    Only maximal runs are reported: a start position that merely continues a
    run already seen at the same period is skipped, so a fully looping trace
    yields a handful of events instead of one per token.
    """
    decode = _resolve_id_to_str(tokens, id_to_str)
    total = len(tokens)

    single_token_loop = False
    repeated_phrase = False
    punctuation_or_symbol_loop = False
    max_single_token_copies = 1 if total else 0
    max_phrase_copies = 1 if total else 0
    hard_events: List[Dict[str, int]] = []

    for start in range(total):
        for period in range(1, config.phrase_max_block + 1):
            if start + 2 * period > total:
                break

            block = tokens[start:start + period]
            if start >= period and tokens[start - period:start] == block:
                continue

            # Do not report (A,A) as a period-2 phrase; its true period is 1.
            if _minimal_period(block) != period:
                continue

            copies = _consecutive_copies(tokens, start, period)
            if copies < 2:
                continue

            surface = "".join(decode(token) for token in block)
            has_alphanumeric = _has_alphanumeric(surface)
            event_type: Optional[str] = None

            if period == 1:
                max_single_token_copies = max(max_single_token_copies, copies)
                if has_alphanumeric and copies >= config.single_token_min_copies:
                    single_token_loop = True
                    event_type = "single_token"
            else:
                max_phrase_copies = max(max_phrase_copies, copies)
                if has_alphanumeric and copies >= config.phrase_min_copies:
                    repeated_phrase = True
                    event_type = "short_phrase"

            covered_tokens = period * copies
            if (
                not has_alphanumeric
                and covered_tokens >= config.punctuation_loop_min_tokens
            ):
                punctuation_or_symbol_loop = True
                event_type = "punctuation_or_symbol"

            if event_type is not None:
                hard_events.append(
                    {
                        "start": start,
                        "period": period,
                        "copies": copies,
                        "covered_tokens": covered_tokens,
                        "type": event_type,
                    }
                )

    return {
        "single_token_loop": single_token_loop,
        "repeated_phrase": repeated_phrase,
        "punctuation_or_symbol_loop": punctuation_or_symbol_loop,
        "max_single_token_copies": max_single_token_copies,
        "max_phrase_copies": max_phrase_copies,
        "hard_events": hard_events,
    }


def repeated_line_features(
    text: str,
    minimum_copies: int = 3,
) -> Dict[str, Any]:
    """Detect consecutive identical nonempty lines.

    Only trailing whitespace is normalized.  Case and punctuation are
    preserved so that distinct lines are not accidentally merged.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    index = 0
    maximum_run = 0
    redundant_lines = 0
    flag = False

    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue

        cursor = index + 1
        while cursor < len(lines) and lines[cursor] == line:
            cursor += 1

        copies = cursor - index
        maximum_run = max(maximum_run, copies)
        if copies >= 2:
            redundant_lines += copies - 1
        if copies >= minimum_copies and _has_alphanumeric(line):
            flag = True
        index = cursor

    return {
        "repeated_line_flag": flag,
        "maximum_identical_line_run": maximum_run,
        "redundant_lines": redundant_lines,
    }


def periodic_suffix_features(
    tokens: Sequence[Token],
    minimum_tokens: int = 64,
    maximum_period: int = 32,
) -> Dict[str, Any]:
    """Find the longest exact periodic suffix anchored at the trace end."""
    total = len(tokens)
    best_period = 0
    best_copies = 0
    best_covered = 0

    for period in range(1, maximum_period + 1):
        if 2 * period > total:
            break

        block = tokens[total - period:total]
        copies = 1
        cursor = total - 2 * period
        while cursor >= 0 and tokens[cursor:cursor + period] == block:
            copies += 1
            cursor -= period

        if copies < 2:
            continue

        covered = copies * period
        if covered > best_covered or (
            covered == best_covered
            and covered > 0
            and period < best_period
        ):
            best_period = period
            best_copies = copies
            best_covered = covered

    return {
        "periodic_suffix_flag": best_covered >= minimum_tokens,
        "periodic_suffix_period": best_period,
        "periodic_suffix_copies": best_copies,
        "periodic_suffix_tokens": best_covered,
    }


def loop_features(
    tokens: Sequence[Token],
    text: str,
    *,
    id_to_str: Optional[IdToStr] = None,
    hit_token_cap: Optional[bool] = None,
    n_new_tokens: Optional[int] = None,
    config: LoopFilterConfig = LoopFilterConfig(),
) -> Dict[str, Any]:
    """Extract loop features without applying a calibrated threshold."""
    token_list = list(tokens)
    total = len(token_list)

    hard = hard_repeat_features(token_list, id_to_str=id_to_str, config=config)
    lines = repeated_line_features(
        text,
        minimum_copies=config.repeated_line_min_copies,
    )
    _, removed, cleaning = three_pass_clean(
        token_list,
        max_block=config.cleaning_max_block,
        passes=config.cleaning_passes,
    )

    if hit_token_cap is None:
        hit_token_cap = (
            n_new_tokens is not None
            and n_new_tokens >= config.max_new_tokens - config.cap_slack
        )

    suffix = periodic_suffix_features(
        token_list,
        minimum_tokens=config.suffix_min_tokens,
        maximum_period=config.suffix_max_period,
    )
    capped_suffix_flag = bool(
        hit_token_cap and suffix["periodic_suffix_flag"]
    )

    hard_loop_flag = bool(
        hard["single_token_loop"]
        or hard["repeated_phrase"]
        or hard["punctuation_or_symbol_loop"]
        or lines["repeated_line_flag"]
        or capped_suffix_flag
    )

    return {
        "n_tokens": total,
        "removed_tokens": removed,
        "repetition_fraction": removed / max(total, 1),
        "hard_loop_flag": hard_loop_flag,
        "hit_token_cap": bool(hit_token_cap),
        "capped_periodic_suffix_flag": capped_suffix_flag,
        **hard,
        **lines,
        **suffix,
        **cleaning,
    }


_ALPHABETIC_WORD = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


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


def unexpected_language_features(text: str) -> Dict[str, Any]:
    """Extract off-script features without applying a calibrated threshold.

    Latin-script alphabetic text is expected.  Isolated Greek or mathematical
    letters are ignored because they are common in equations.  Multi-character
    Greek passages are treated as off-script rather than silently accepted as
    English.

    Both a word run and a character run are reported.  Scripts written
    without spaces collapse into a single very long word, so only the
    character run can measure how far such a passage extends.
    """
    words = _ALPHABETIC_WORD.findall(text)
    considered_characters = 0
    off_script_characters = 0
    current_off_word_run = 0
    longest_off_word_run = 0
    current_off_character_run = 0
    longest_off_character_run = 0
    off_words = 0
    considered_words = 0

    for word in words:
        classes = [_character_class(character) for character in word]

        # Ignore isolated mathematical variables such as α, β, or 𝐱.
        if len(word) == 1 and classes[0] in {"greek", "mathematical"}:
            continue

        relevant = [
            character_class
            for character_class in classes
            if character_class != "mathematical"
        ]
        if not relevant:
            continue

        considered_words += 1
        considered_characters += len(relevant)
        word_off_characters = 0
        for character_class in relevant:
            if character_class == "latin":
                current_off_character_run = 0
                continue
            word_off_characters += 1
            current_off_character_run += 1
            longest_off_character_run = max(
                longest_off_character_run,
                current_off_character_run,
            )
        off_script_characters += word_off_characters
        word_is_off = word_off_characters > len(relevant) / 2

        if word_is_off:
            off_words += 1
            current_off_word_run += 1
            longest_off_word_run = max(
                longest_off_word_run,
                current_off_word_run,
            )
        else:
            current_off_word_run = 0

    return {
        "alphabetic_characters": considered_characters,
        "off_script_characters": off_script_characters,
        "off_script_fraction": (
            off_script_characters / max(considered_characters, 1)
        ),
        "considered_words": considered_words,
        "off_script_words": off_words,
        "longest_off_script_word_run": longest_off_word_run,
        "longest_off_script_character_run": longest_off_character_run,
    }


def extract_features(
    text: str,
    tokens: Sequence[Token],
    *,
    id_to_str: Optional[IdToStr] = None,
    hit_token_cap: Optional[bool] = None,
    n_new_tokens: Optional[int] = None,
    loop_config: LoopFilterConfig = LoopFilterConfig(),
) -> Dict[str, Any]:
    """Extract all continuous and discrete features before calibration."""
    loop = loop_features(
        tokens,
        text,
        id_to_str=id_to_str,
        hit_token_cap=hit_token_cap,
        n_new_tokens=n_new_tokens,
        config=loop_config,
    )
    language = unexpected_language_features(text)
    return {
        "n_tokens": loop["n_tokens"],
        "removed_tokens": loop["removed_tokens"],
        "repetition_fraction": loop["repetition_fraction"],
        "off_script_fraction": language["off_script_fraction"],
        "hit_token_cap": loop["hit_token_cap"],
        "_loop": loop,
        "_language": language,
    }


def empirical_quantile(values: Iterable[float], quantile: float) -> float:
    """Linearly interpolated empirical quantile, without NumPy."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile of an empty collection.")

    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]

    weight = position - lower_index
    return (
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


def calibrate_thresholds(
    standard_features: Sequence[Dict[str, Any]],
    *,
    loop_config: LoopFilterConfig = LoopFilterConfig(),
    language_config: LanguageFilterConfig = LanguageFilterConfig(),
) -> FrozenThresholds:
    """Freeze thresholds using only Standard-teacher feature records."""
    if not standard_features:
        raise ValueError("At least one Standard trace is required.")

    repetition_quantile = empirical_quantile(
        (
            feature["repetition_fraction"]
            for feature in standard_features
        ),
        loop_config.standard_quantile,
    )
    language_quantile = empirical_quantile(
        (
            feature["off_script_fraction"]
            for feature in standard_features
        ),
        language_config.standard_quantile,
    )

    return FrozenThresholds(
        repetition_fraction=max(
            loop_config.repetition_fraction_floor,
            repetition_quantile,
        ),
        off_script_fraction=max(
            language_config.off_script_fraction_floor,
            language_quantile,
        ),
    )


def apply_thresholds(
    features: Dict[str, Any],
    thresholds: FrozenThresholds,
    *,
    loop_config: LoopFilterConfig = LoopFilterConfig(),
    language_config: LanguageFilterConfig = LanguageFilterConfig(),
) -> Dict[str, Any]:
    """Apply frozen thresholds to one raw feature record."""
    loop = dict(features["_loop"])
    language = dict(features["_language"])

    large_repetition_fraction_flag = bool(
        loop["removed_tokens"] >= loop_config.removed_token_minimum
        and loop["repetition_fraction"] >= thresholds.repetition_fraction
    )
    loop_flag = bool(
        loop["hard_loop_flag"] or large_repetition_fraction_flag
    )

    language_character_flag = bool(
        language["off_script_characters"]
        >= language_config.minimum_off_script_characters
        and language["off_script_fraction"]
        > thresholds.off_script_fraction
    )
    language_span_flag = bool(
        language["longest_off_script_word_run"]
        >= language_config.minimum_off_script_span_words
        or language["longest_off_script_character_run"]
        >= language_config.minimum_off_script_span_characters
    )
    language_flag = language_character_flag or language_span_flag

    loop.update(
        {
            "repetition_fraction_threshold": thresholds.repetition_fraction,
            "large_repetition_fraction_flag": (
                large_repetition_fraction_flag
            ),
            "flag": loop_flag,
        }
    )
    language.update(
        {
            "off_script_fraction_threshold": (
                thresholds.off_script_fraction
            ),
            "character_flag": language_character_flag,
            "span_flag": language_span_flag,
            "flag": language_flag,
        }
    )

    return {
        "loop_flag": loop_flag,
        "language_flag": language_flag,
        "degenerate": loop_flag or language_flag,
        "n_tokens": loop["n_tokens"],
        "removed_tokens": loop["removed_tokens"],
        "repetition_fraction": loop["repetition_fraction"],
        "off_script_fraction": language["off_script_fraction"],
        "hit_token_cap": loop["hit_token_cap"],
        "_loop": loop,
        "_language": language,
    }


def flag_reasons(scored: Dict[str, Any]) -> List[str]:
    """Name every rule that fired, for inspection of flagged traces."""
    loop = scored["_loop"]
    language = scored["_language"]
    checks = [
        ("single_token_loop", loop["single_token_loop"]),
        ("repeated_phrase", loop["repeated_phrase"]),
        ("punctuation_or_symbol_loop", loop["punctuation_or_symbol_loop"]),
        ("repeated_line", loop["repeated_line_flag"]),
        ("capped_periodic_suffix", loop["capped_periodic_suffix_flag"]),
        ("large_repetition_fraction", loop["large_repetition_fraction_flag"]),
        ("off_script_fraction", language["character_flag"]),
        ("off_script_span", language["span_flag"]),
    ]
    return [name for name, fired in checks if fired]


def score_all(
    text: str,
    tokens: Sequence[Token],
    thresholds: FrozenThresholds,
    *,
    id_to_str: Optional[IdToStr] = None,
    hit_token_cap: Optional[bool] = None,
    n_new_tokens: Optional[int] = None,
    loop_config: LoopFilterConfig = LoopFilterConfig(),
    language_config: LanguageFilterConfig = LanguageFilterConfig(),
) -> Dict[str, Any]:
    """Extract features and apply already-frozen Standard thresholds."""
    raw = extract_features(
        text,
        tokens,
        id_to_str=id_to_str,
        hit_token_cap=hit_token_cap,
        n_new_tokens=n_new_tokens,
        loop_config=loop_config,
    )
    return apply_thresholds(
        raw,
        thresholds,
        loop_config=loop_config,
        language_config=language_config,
    )


def condition_report(scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a scored condition for the rebuttal table."""
    if not scored:
        raise ValueError("Cannot report on an empty condition.")

    count = len(scored)
    total_tokens = sum(record["n_tokens"] for record in scored)
    total_removed = sum(record["removed_tokens"] for record in scored)
    both = sum(
        record["loop_flag"] and record["language_flag"]
        for record in scored
    )

    return {
        "n": count,
        "loop_rate": sum(record["loop_flag"] for record in scored) / count,
        "language_rate": (
            sum(record["language_flag"] for record in scored) / count
        ),
        "both_rate": both / count,
        "union_degenerate_rate": (
            sum(record["degenerate"] for record in scored) / count
        ),
        "median_tokens": median(
            record["n_tokens"] for record in scored
        ),
        "median_repetition_fraction": median(
            record["repetition_fraction"] for record in scored
        ),
        "token_weighted_removed_fraction": (
            total_removed / max(total_tokens, 1)
        ),
    }


def top_fraction_indices(
    features: Sequence[Dict[str, Any]],
    score_key: str,
    fraction: float,
) -> List[int]:
    """Select an exact top fraction for matched-removal sensitivity analysis.

    Ties are broken by original index, making the selection deterministic.
    This helper is for sensitivity plots; it does not replace Standard-based
    primary calibration.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1].")
    number_to_select = int(round(len(features) * fraction))
    ranked = sorted(
        range(len(features)),
        key=lambda index: (-float(features[index][score_key]), index),
    )
    return ranked[:number_to_select]


def thresholds_as_dict(thresholds: FrozenThresholds) -> Dict[str, float]:
    """JSON-friendly serialization helper."""
    return asdict(thresholds)


__all__ = [
    "FrozenThresholds",
    "LanguageFilterConfig",
    "LoopFilterConfig",
    "RepeatRegion",
    "apply_thresholds",
    "cached_id_to_str",
    "calibrate_thresholds",
    "condition_report",
    "empirical_quantile",
    "extract_features",
    "flag_reasons",
    "greedy_repeat_regions",
    "hard_repeat_features",
    "loop_features",
    "periodic_suffix_features",
    "repeated_line_features",
    "score_all",
    "three_pass_clean",
    "thresholds_as_dict",
    "top_fraction_indices",
    "unexpected_language_features",
]