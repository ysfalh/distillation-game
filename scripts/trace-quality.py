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
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=1024)

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
                "trace": item.get("trace", ""),
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