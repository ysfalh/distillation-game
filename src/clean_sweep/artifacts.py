from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TraceRecord:
    example_id: str
    split: str
    method: str
    prompt_text: str
    raw_trace_text: str
    af_final_answer_only: str | None
    raw_correct: bool
    af_correct: bool | None
    raw_extracted_answer: str | None
    af_extracted_answer: str | None


def compact_trace_row(record: TraceRecord) -> dict[str, Any]:
    final_correct = record.af_correct if record.af_correct is not None else record.raw_correct
    final_extracted_answer = record.af_extracted_answer if record.af_extracted_answer is not None else record.raw_extracted_answer
    return {
        "example_id": record.example_id,
        "split": record.split,
        "method": record.method,
        "prompt": record.prompt_text,
        "trace": record.raw_trace_text,
        "af_final_answer_only": record.af_final_answer_only,
        "raw_correct": record.raw_correct,
        "af_correct": record.af_correct,
        "correct": final_correct,
        "raw_extracted_answer": record.raw_extracted_answer,
        "af_extracted_answer": record.af_extracted_answer,
        "extracted_answer": final_extracted_answer,
    }
