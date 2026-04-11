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

import re
import json
import argparse
from vllm import LLM, SamplingParams
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


def build_prompt(instruction, response, reference, rubric):
    """Fill in the Prometheus absolute grading template."""
    return ABSOLUTE_PROMPT.format(
        instruction=instruction,
        response=response,
        reference_answer=reference,
        rubric=rubric,
    )


def parse_output(text):
    """Extract feedback and integer score from Prometheus output."""
    # Prometheus outputs: [RESULT] <score>
    match = re.search(r"\[RESULT\]\s*(\d)", text)
    score = int(match.group(1)) if match else None
    feedback = text[:match.start()].strip() if match else text.strip()
    return feedback, score


def score_traces(input_path: str, output_path: str, model_name: str, batch_size: int):
    with open(input_path) as f:
        data = json.load(f)

    llm = LLM(model=model_name, enforce_eager=True)
    sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=512)

    results = []
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]

        prompts = [
            build_prompt(
                instruction=item["prompt"],
                response=item["trace"],
                reference=str(item.get("extracted_answer", "")),
                rubric=RUBRIC,
            )
            for item in batch
        ]

        outputs = llm.generate(prompts, sampling_params)

        for item, output in zip(batch, outputs):
            text = output.outputs[0].text
            feedback, score = parse_output(text)
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