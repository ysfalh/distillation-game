"""
Score reasoning traces using Prometheus 2 (absolute grading, 1-5).

Input JSON format (one entry per example, flat list):
[
    {
        "example_id": "train_0",
        "method": "poe_gamma_0.7",
        "prompt": "Rachel is twice as old as Rona ...",
        "trace": "To solve this problem ...",
        "extracted_answer": "12"
    },
    ...
]

Usage:
    python trace-quality.py input.json output.json [--model MODEL]
"""

import os
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["VLLM_USE_FLASHINFER"] = "0"

import json
import argparse
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE


RUBRIC = SCORE_RUBRIC_TEMPLATE.format(
    criteria=(
        "How useful is this reasoning trace for a human trying to "
        "understand the solution?"
    ),
    score1_description="The trace is confusing, disorganized, or obscures the reasoning.",
    score2_description="The trace is hard to follow with significant gaps or unnecessary complexity.",
    score3_description="The trace is adequate but could be clearer or more concise.",
    score4_description="The trace is clear and easy to follow with minor issues.",
    score5_description="The trace is exceptionally clear, well-organized, and immediately understandable.",
)


def score_traces(input_path: str, output_path: str, model_name: str, batch_size: int):
    with open(input_path) as f:
        data = json.load(f)

    model = VLLM(model=model_name, enforce_eager=True)
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    results = []
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]

        instructions = [item["prompt"] for item in batch]
        responses = [item["trace"] for item in batch]
        references = [str(item.get("extracted_answer", "")) for item in batch]

        feedbacks, scores = judge.absolute_grade(
            instructions=instructions,
            responses=responses,
            rubric=RUBRIC,
            reference_answers=references,
        )

        for item, feedback, score in zip(batch, feedbacks, scores):
            results.append({
                "example_id": item["example_id"],
                "method": item.get("method", ""),
                "score": score,
                "feedback": feedback,
            })
            print(f"{item['example_id']} | score: {score}")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  batch done ({min(start + batch_size, len(data))}/{len(data)})")

    print(f"\nWrote {len(results)} scores to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument("--model", default="prometheus-eval/prometheus-7b-v2.0")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    score_traces(args.input, args.output, args.model, args.batch_size)
