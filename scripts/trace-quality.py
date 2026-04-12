#!/usr/bin/env python3
"""
Score reasoning traces with Prometheus 2 using absolute grading *without* a reference answer.

Expected input JSON format:
[
  {
    "example_id": "train_0",
    "method": "poe_gamma_0.7",
    "prompt": "...",
    "trace": "...",
    "extracted_answer": "..."
  },
  ...
]

Example:
    python trace-quality-prometheus-no-ref.py input.json output.json \
        --model prometheus-eval/prometheus-7b-v2.0 --batch-size 64
"""

import os
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_FLASHINFER", "0")

import json
import argparse
from collections import Counter, defaultdict
from typing import Any

from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT_WO_REF, SCORE_RUBRIC_TEMPLATE


RUBRIC = SCORE_RUBRIC_TEMPLATE.format(
    criteria=(
        "Can a human reader audit the reasoning in this trace? "
        "For each step, assess whether it is (a) identifiable as "
        "a distinct reasoning step, (b) relevant to the problem, "
        "and (c) checkable against the previous step or the problem "
        "statement. Verbose but coherent traces and concise traces "
        "should score equally if both are auditable."
    ),
    score1_description=(
        "No auditable reasoning. The trace is dominated by "
        "non-reasoning content (repeated tokens, garbled text, "
        "or irrelevant material). A reader cannot identify any "
        "checkable steps."
    ),
    score2_description=(
        "Few auditable steps. Some reasoning is present but "
        "is interleaved with substantial non-reasoning content "
        "(filler tokens, irrelevant tangents, or corrupted text) "
        "making it unclear which parts to trust."
    ),
    score3_description=(
        "Partially auditable. The core reasoning steps are "
        "identifiable but some steps lack clear justification, "
        "or the reader must ignore non-trivial amounts of "
        "irrelevant content to follow the argument."
    ),
    score4_description=(
        "Mostly auditable. Nearly every step is identifiable, "
        "relevant, and checkable. Minor issues such as a "
        "redundant restatement or one unclear transition "
        "do not prevent verification."
    ),
    score5_description=(
        "Fully auditable. Every step is identifiable, relevant "
        "to the problem, and independently checkable. The trace "
        "may be long or short — what matters is that no step "
        "requires guesswork to verify."
    ),
)


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a flat list of examples.")
    return data


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def summarize_by_method(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in results:
        method = row.get("method", "") or ""
        score = row.get("score")
        counts[method][str(score)] += 1

    summary: dict[str, dict[str, int]] = {}
    for method, counter in counts.items():
        summary[method] = {str(k): counter.get(str(k), 0) for k in range(1, 6)}
    return summary


def score_traces(
    input_path: str,
    output_path: str,
    model_name: str,
    batch_size: int,
    max_tokens: int,
    tensor_parallel_size: int,
) -> None:
    data = load_json(input_path)

    model = VLLM(
        model=model_name,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
    )
    judge = PrometheusEval(
        model=model,
        absolute_grade_template=ABSOLUTE_PROMPT_WO_REF,
    )

    generation_params = {
        # Deterministic judging is much easier to debug and compare.
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }

    results: list[dict[str, Any]] = []

    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]

        instructions = [str(item.get("prompt", "")) for item in batch]
        responses = [str(item.get("trace", "")) for item in batch]

        feedbacks, scores = judge.absolute_grade(
            instructions=instructions,
            responses=responses,
            rubric=RUBRIC,
            # Explicitly pass None references so the library does not warn per batch.
            reference_answers=[None] * len(batch),
            params=generation_params,
        )

        for item, feedback, score in zip(batch, feedbacks, scores):
            results.append(
                {
                    "example_id": item.get("example_id", ""),
                    "method": item.get("method", ""),
                    "score": score,
                    "feedback": feedback,
                }
            )
            print(f"{item.get('example_id', '<missing_id>')} | score: {score}")

        save_json(results, output_path)
        print(f"batch done ({min(start + batch_size, len(data))}/{len(data)})")

    summary = summarize_by_method(results)
    print("\nDistribution by method:")
    for method, counts in summary.items():
        ordered = [counts[str(i)] for i in range(1, 6)]
        print(f"{method or '<no method>'}: {ordered}")

    print(f"\nWrote {len(results)} scores to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument("--model", default="prometheus-eval/prometheus-7b-v2.0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    score_traces(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )
