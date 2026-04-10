import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
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


def score_traces(input_path: str, output_path: str, model_name: str):
    with open(input_path) as f:
        data = json.load(f)

    model = VLLM(model=model_name, enforce_eager=True)
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    results = []
    for item in data:
        example_id = item["example_id"]
        feedback, score = judge.single_absolute_grade(
            instruction=item["prompt"],
            response=item["trace"],
            rubric=RUBRIC,
            reference_answer=str(item.get("extracted_answer", "")),
        )
        results.append({
            "example_id": example_id,
            "method": item.get("method", ""),
            "score": score,
            "feedback": feedback,
        })
        print(f"{example_id} | score: {score}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} scores to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument("--model", default="prometheus-eval/prometheus-7b-v2.0")
    args = parser.parse_args()
    score_traces(args.input, args.output, args.model)