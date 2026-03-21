from __future__ import annotations

import math
import re
from typing import Any

from latex2sympy2_extended.latex2sympy2 import NormalizationConfig as _MVNormalizationConfig
from math_verify import ExprExtractionConfig as _MVExprExtractionConfig
from math_verify import LatexExtractionConfig as _MVLatexExtractionConfig
from math_verify import parse as _mv_parse
from math_verify import verify as _mv_verify


NUMERIC_FALLBACK = re.compile(r"(?:answer is|=\s*)\s*([-+]?\d+(?:\.\d+)?)\s*[.\s]*$", re.IGNORECASE)
ANSWER_FORCE_MARKERS = [
    "\n\n**Final Answer**\n\\boxed{",
    "\n\n**Final Answer**\n\\[\\boxed{",
]


def _find_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    results = []
    for m in re.finditer(r"\\boxed\{", text, re.IGNORECASE):
        start = m.end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    results.append((m.start(), i + 1, text[start:i]))
                    break
    return results


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^[\'"]+|[\'"]+$', "", s)
    s = re.sub(r"\\+$", "", s)
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"^\s*answer\s*:\s*", "", s, flags=re.IGNORECASE)
    return " ".join(s.split())


def _strip_latex_units(s: str) -> str:
    s = re.sub(r"\\text\{[^}]*\}", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\\mbox\{[^}]*\}", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\\mathrm\{[^}]*\}", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\\text\{[^}]*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\\mbox\{[^}]*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\\mathrm\{[^}]*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def extract_answer(text: str) -> str | None:
    if not text or not isinstance(text, str):
        return None
    boxed_spans = _find_boxed_spans(text.strip())
    for _, _, content in reversed(boxed_spans):
        raw = normalize_answer(content)
        if raw.upper() != "ANSWER":
            raw = normalize_answer(_strip_latex_units(raw))
            return raw if raw else None
    m = NUMERIC_FALLBACK.search(text)
    return normalize_answer(m.group(1)) if m else None


def _trace_variants(trace: str) -> list[str]:
    for marker in ANSWER_FORCE_MARKERS:
        if marker in trace:
            parts = trace.split(marker)
            before = marker.join(parts[:-1])
            after = parts[-1]
            return [trace, before, after]
    return [trace]


_MATH_PRED_LATEX_CONFIG = _MVLatexExtractionConfig(
    boxed_match_priority=0,
    normalization_config=_MVNormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=False,
        nits=False,
        boxed="all",
        equations=False,
    ),
)


def is_correct_gsm8k(trace: str, solution: str) -> bool:
    if not isinstance(trace, str) or not isinstance(solution, str):
        return False
    try:
        gold = _mv_parse(solution)
        for variant in _trace_variants(trace):
            try:
                if _mv_verify(gold, _mv_parse(variant, extraction_config=[_MATH_PRED_LATEX_CONFIG, _MVExprExtractionConfig()])):
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def check_trace_correctness(trace: str, solution: str) -> dict[str, Any]:
    extracted = extract_answer(trace)
    return {"extracted_answer": extracted, "correct": is_correct_gsm8k(trace, solution)}
