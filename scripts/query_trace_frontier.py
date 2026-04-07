import sys
import time
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from google import genai
from openai import OpenAI
import anthropic
from datasets import concatenate_datasets, load_dataset
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.clean_sweep.eval.correctness import check_trace_correctness, extract_answer
from src.clean_sweep.data.math import load_math_splits
from src.clean_sweep.data.gsm8k import load_gsm8k_splits

# ---------------------------------------------------------------------------
# API keys — set these in your environment, never hardcode
# ---------------------------------------------------------------------------
api_key_gemini = 'fill-me'
api_key_openai = 'fill-me'
api_key_claude = 'fill-me'

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
_RATE_LIMIT: dict[str, float] = {"openai": 0.5, "gemini": 1.0, "claude": 0.5}
_last_call_time: dict[str, float] = {}
_throttle_lock = __import__("threading").Lock()

def _throttle(provider: str):
    min_gap = _RATE_LIMIT.get(provider, 0.5)
    with _throttle_lock:
        elapsed = time.time() - _last_call_time.get(provider, 0.0)
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        _last_call_time[provider] = time.time()

# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------
def _call_with_retry(fn, max_retries=4, base_delay=2.0):
    for attempt in range(max_retries):
        try:
            return fn(), None
        except Exception as e:
            if attempt == max_retries - 1:
                return None, str(e)
            wait = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"  Retry {attempt + 1}/{max_retries - 1} after error: {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def get_stratified_samples(dataset, num_samples: int, subject_column: str) -> list:
    by_subject: dict[str, list] = defaultdict(list)
    for item in dataset:
        by_subject[item.get(subject_column, "unknown")].append(item)
    for sub in by_subject:
        random.shuffle(by_subject[sub])

    num_samples = min(num_samples, sum(len(v) for v in by_subject.values()))

    sampled, subjects, idx = [], list(by_subject.keys()), 0
    while len(sampled) < num_samples:
        if all(len(lst) == 0 for lst in by_subject.values()):
            break
        sub = subjects[idx % len(subjects)]
        if by_subject[sub]:
            sampled.append(by_subject[sub].pop(0))
        idx += 1

    random.shuffle(sampled)
    return sampled


def _load_existing_questions(output_path: str) -> set[str]:
    seen: set[str] = set()
    if not os.path.exists(output_path):
        return seen
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["question"])
            except Exception:
                pass
    return seen

# ---------------------------------------------------------------------------
# Main trace generator
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "Act as an expert mathematical reasoning engine. "
    "Solve the following math problem step-by-step. "
    "Outline the logical steps required to reach the goal.\n\n"
    "Do NOT repeat the problem statement.\n"
    "You must put your final numerical or algebraic answer inside a \\boxed{}."
)

def _extract_question_key(item: dict, dataset_name: str) -> str:
    return item["problem"]


def generate_trace_llm(
    provider: str,
    model: str,
    dataset,
    dataset_name: str,
    output_file: str,
    num_samples: int | None = None,
    flush_every: int = 10,
    max_workers: int = 8,
):
    """
    Generate reasoning traces using OpenAI, Gemini, or Claude.
    Automatically resumes from existing output if present.

    Args:
        provider:     'openai' | 'gemini' | 'claude'
        model:        model identifier string
        dataset:      HuggingFace dataset split
        dataset_name: 'math' | 'gsm8k'
        output_file:  base filename (saved under traces_llms/)
        num_samples:  how many problems to process (None = all)
        flush_every:  flush output file to disk every N items
    """
    assert dataset_name in ("math", "gsm8k"), "dataset_name must be 'math' or 'gsm8k'"
    assert provider in ("openai", "gemini", "claude"), "provider must be 'openai', 'gemini', or 'claude'"

    # 1. Init client
    if provider == "openai":
        client = OpenAI(api_key=api_key_openai)
    elif provider == "gemini":
        client = genai.Client(api_key=api_key_gemini)
    else:
        client = anthropic.Anthropic(api_key=api_key_claude)

    os.makedirs("traces_llms", exist_ok=True)
    output_path = os.path.join("traces_llms", os.path.basename(output_file))

    # 2. Sub-sample
    if num_samples is not None:
        if dataset_name == "math":
            dataset = get_stratified_samples(dataset, num_samples, subject_column="type")
        else:
            dataset = list(dataset)
            random.shuffle(dataset)
            dataset = dataset[:num_samples]
    else:
        dataset = list(dataset)

    # 3. Resume: drop already-processed questions before the loop
    seen_questions = _load_existing_questions(output_path)
    if seen_questions:
        before = len(dataset)
        dataset = [item for item in dataset if _extract_question_key(item, dataset_name) not in seen_questions]
        print(f"[{provider.upper()}] Resuming: skipped {before - len(dataset)} already-done, "
              f"{len(dataset)} remaining.")

    total = len(dataset)
    if total == 0:
        print(f"[{provider.upper()}] Nothing to do.")
        return

    # 4. API call factory
    def make_call(question: str):
        user_message = f"Problem: {question}"
        full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{user_message}"

        def _call():
            if provider == "openai":
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return resp.choices[0].message.content or ""

            elif provider == "gemini":
                resp = client.models.generate_content(model=model, contents=full_prompt)
                if not resp.candidates or not resp.candidates[0].content.parts:
                    raise ValueError("Gemini returned an empty/blocked response")
                return resp.text

            else:  # claude
                resp = client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=SYSTEM_INSTRUCTION,
                    messages=[{"role": "user", "content": user_message}]
                )
                return resp.content[0].text

        return _call

    # 5. Main loop — parallel API calls, sequential file writes
    def process_item(item):
        question         = item["problem"]
        reference_answer = item["solution"]
        subject          = item.get("type", "general")
        if dataset_name == "math":
            gold_answer = (extract_answer(reference_answer) or "").strip()
        else:
            gold_answer = (
                reference_answer.split("####")[-1].strip()
                if "####" in reference_answer else ""
            )

        _throttle(provider)
        trace_text, err = _call_with_retry(make_call(question))

        if trace_text is None:
            return None, f"Failed: {err}"

        eval_result      = check_trace_correctness(trace_text, reference_answer)
        predicted_answer = (eval_result["extracted_answer"] or "").strip()
        is_correct       = eval_result["correct"]

        return {
            "question":         question,
            "subject":          subject,
            "trace":            trace_text,
            "reference_answer": reference_answer,
            "gold_answer":      gold_answer,
            "predicted_answer": predicted_answer,
            "correct":          is_correct,
        }, None

    completed = 0
    with open(output_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, item): i for i, item in enumerate(dataset)}
            for future in as_completed(futures):
                i = futures[future]
                result, err = future.result()
                if result is None:
                    print(f"[{provider.upper()}] Skipping item {i + 1}/{total}: {err}")
                    continue
                f.write(json.dumps(result) + "\n")
                completed += 1
                if completed % flush_every == 0:
                    f.flush()
                    print(f"[{provider.upper()} - {dataset_name.upper()}] {completed}/{total} done.")

    print(f"[{provider.upper()}] Finished. Output: {output_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":

    with open("configs/math.yaml") as f:
        math_cfg = yaml.safe_load(f)
    with open("configs/gsm8k.yaml") as f:
        gsm8k_cfg = yaml.safe_load(f)
    
    print("Loading datasets...")

    math_splits = load_math_splits(
        seed=math_cfg["run"]["seed"],
        train_size=math_cfg["data"]["train_size"],
        holdout_size=math_cfg["data"]["holdout_size"],
        test_size=0,
    )
    math_data = concatenate_datasets([math_splits["train"], math_splits["holdout"]])

    gsm8k_splits = load_gsm8k_splits(
        seed=gsm8k_cfg["run"]["seed"],
        train_size=gsm8k_cfg["data"]["train_size"],
        holdout_size=gsm8k_cfg["data"]["holdout_size"],
        test_size=0,
    )
    gsm8k_data = concatenate_datasets([gsm8k_splits["train"], gsm8k_splits["holdout"]])

    # OpenAI on GSM8K
    # generate_trace_llm("openai", "gpt-4o-mini", gsm8k_data, "math", "openai_math_traces.jsonl")

    # Gemini on MATH
    generate_trace_llm("gemini", "gemini-2.5-flash", gsm8k_data, "math", "gemini_math_traces.jsonl")

    # Claude on MATH
    # generate_trace_llm("claude", "claude-3-5-sonnet-20241022", math_data, "math", "claude_math_traces.jsonl")