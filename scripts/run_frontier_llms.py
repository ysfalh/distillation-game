#!/usr/bin/env python3
"""
Frontier-LLM pipeline: query → SFT.

Chains the two stages of the real-world LLM trace experiment:

  1. QUERY  — call OpenAI / Gemini / Claude for reasoning traces on GSM8K
              and MATH problems, writing one JSONL per (provider, dataset)
              under --traces-dir (default: traces_llms/).

  2. SFT    — for each (provider, dataset, seed), fine-tune the student
              on those traces and on the dataset's ground-truth Q&A, then
              evaluate on the test split. One run directory per triple
              under --output-dir (default: outputs/).

Both stages re-use the existing scripts in frontier-llms/ without
modification:
  - frontier-llms/query_trace_frontier.py   provides generate_trace_llm()
  - frontier-llms/run_real_trace.py         invoked as a subprocess

Plotting is intentionally out of scope; use frontier-llms/plot-llm.py
afterwards to aggregate the SFT runs.

Examples
--------
Full sweep (all 3 providers × 2 datasets × 3 seeds, all problems):

    python scripts/run_frontier_llms.py

Query Claude on MATH only, then SFT for two seeds:

    python scripts/run_frontier_llms.py \\
        --providers claude --datasets math --seeds 42 123

Skip query (use whatever JSONLs are already in traces_llms/):

    python scripts/run_frontier_llms.py --skip-query

Quick smoke test (50 problems, single provider/dataset/seed):

    python scripts/run_frontier_llms.py \\
        --providers openai --datasets gsm8k --seeds 42 --num-samples 50

API keys
--------
This script does NOT inject keys. The keys live where they always have:
inside frontier-llms/query_trace_frontier.py (the api_key_* literals at
the top of that file). Set them there (or wrap that script behind env
vars) before running with --providers actually-calling-the-API.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTIER_DIR = REPO_ROOT / "frontier-llms"

# Defaults for the orchestrator. The model strings mirror those used in the
# original __main__ block of query_trace_frontier.py and in the historical
# real_<provider>_<dataset>_traces_seed<seed>_<ts> run directories that
# frontier-llms/plot-llm.py consumes.
DEFAULT_PROVIDERS = ["openai", "gemini", "claude"]
DEFAULT_DATASETS = ["gsm8k", "math"]
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3-flash-preview",
    "claude": "claude-3-5-sonnet-20241022",
}
DATASET_CONFIGS: dict[str, str] = {
    "gsm8k": "configs/gsm8k.yaml",
    "math": "configs/math.yaml",
}


# ── Dynamic import of query_trace_frontier.py ─────────────────────────────
# The folder name "frontier-llms" contains a hyphen, so it cannot be used
# as a normal Python package. importlib.util.spec_from_file_location loads
# the module directly from its file path, which also correctly seeds
# __file__ so the script's own sys.path bootstrap (resolving the project
# root via dirname(dirname(__file__))) keeps working.

def _load_query_module():
    path = FRONTIER_DIR / "query_trace_frontier.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected frontier query script at {path}. "
            "Run this orchestrator from the repo root."
        )
    spec = importlib.util.spec_from_file_location("query_trace_frontier", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ── Stage 1: QUERY ────────────────────────────────────────────────────────

def _trace_filename(provider: str, dataset: str) -> str:
    """Mirror the naming convention used by the existing pipeline.

    `run_real_trace.py --trace-name <provider>_<dataset>_traces` looks for
    `<trace-dir>/<provider>_<dataset>_traces.jsonl`, so we have to write to
    that exact path here.
    """
    return f"{provider}_{dataset}_traces.jsonl"


def _load_dataset_for(query_mod, dataset: str):
    """Build the (train + holdout) HF dataset the original __main__ block built."""
    import yaml
    from datasets import concatenate_datasets

    config_path = REPO_ROOT / DATASET_CONFIGS[dataset]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if dataset == "math":
        splits = query_mod.load_math_splits(
            seed=cfg["run"]["seed"],
            train_size=cfg["data"]["train_size"],
            holdout_size=cfg["data"]["holdout_size"],
            test_size=1,
        )
    else:
        splits = query_mod.load_gsm8k_splits(
            seed=cfg["run"]["seed"],
            train_size=cfg["data"]["train_size"],
            holdout_size=cfg["data"]["holdout_size"],
            test_size=0,
        )
    return concatenate_datasets([splits["train"], splits["holdout"]])


def run_query(
    *,
    providers: list[str],
    datasets: list[str],
    models: dict[str, str],
    num_samples: int,
    traces_dir: Path,
    max_workers: int,
) -> dict[tuple[str, str], Path]:
    """Generate traces for each (provider, dataset) combination.

    Returns the map (provider, dataset) -> JSONL path actually written.
    """
    query_mod = _load_query_module()
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Cache HF datasets so we only build each dataset once even when we
    # query it with multiple providers.
    dataset_cache: dict[str, object] = {}
    written: dict[tuple[str, str], Path] = {}

    for dataset in datasets:
        if dataset not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset!r}. "
                f"Expected one of: {list(DATASET_CONFIGS)}"
            )
        for provider in providers:
            if provider not in models:
                raise ValueError(
                    f"No model configured for provider {provider!r}. "
                    f"Add it via --model {provider}=<model-name>."
                )
            if dataset not in dataset_cache:
                print(f"[query] Loading {dataset.upper()} dataset ...")
                dataset_cache[dataset] = _load_dataset_for(query_mod, dataset)
            hf_dataset = dataset_cache[dataset]

            out_name = _trace_filename(provider, dataset)
            out_path = traces_dir / out_name
            print(
                f"\n[query] === {provider} on {dataset} "
                f"(model={models[provider]}) ==="
            )
            print(f"[query] Output: {out_path}")

            # generate_trace_llm() resolves output under "traces_llms/" by
            # joining basename(output_file). We pass --traces-dir directly
            # via the cwd-relative convention used in the original script.
            query_mod.generate_trace_llm(
                provider=provider,
                model=models[provider],
                dataset=hf_dataset,
                dataset_name=dataset,
                output_file=out_name,
                num_samples=num_samples if num_samples > 0 else None,
                max_workers=max_workers,
            )
            written[(provider, dataset)] = out_path
    return written


# ── Stage 2: SFT ──────────────────────────────────────────────────────────

def run_sft(
    *,
    providers: list[str],
    datasets: list[str],
    seeds: list[int],
    output_dir: Path,
    traces_dir: Path,
    python_bin: str = sys.executable,
) -> list[dict]:
    """Run frontier-llms/run_real_trace.py once per (provider, dataset, seed).

    Saves into output_dir (which run_real_trace.py interprets via the
    `run.output_dir` field of the dataset config). Returns a list of
    completed-task summaries.
    """
    runner = FRONTIER_DIR / "run_real_trace.py"
    if not runner.exists():
        raise FileNotFoundError(f"SFT runner missing: {runner}")

    env = os.environ.copy()
    # Make `from clean_sweep ...` importable inside the subprocess. The
    # subprocess script also sets this up via __file__, but exporting it
    # here keeps things robust to relocations.
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    summaries: list[dict] = []
    for dataset in datasets:
        cfg_path = REPO_ROOT / DATASET_CONFIGS[dataset]
        for provider in providers:
            trace_name = f"{provider}_{dataset}_traces"
            trace_jsonl = traces_dir / f"{trace_name}.jsonl"
            trace_json = traces_dir / f"{trace_name}.json"
            if not trace_jsonl.exists() and not trace_json.exists():
                print(
                    f"\n[sft] SKIP {provider}/{dataset}: no trace file "
                    f"at {trace_jsonl} (or .json)."
                )
                continue
            for seed in seeds:
                tag = f"{provider}/{dataset}/seed{seed}"
                print(f"\n[sft] === {tag} ===")
                t0 = time.perf_counter()
                cmd = [
                    python_bin,
                    str(runner),
                    "--config", str(cfg_path),
                    "--trace-name", trace_name,
                    "--trace-dir", str(traces_dir),
                    "--seed", str(seed),
                    "--output-dir", str(output_dir),
                ]
                print(f"[sft] {' '.join(cmd)}")
                rc = subprocess.call(cmd, env=env, cwd=str(REPO_ROOT))
                dur = time.perf_counter() - t0
                summaries.append({
                    "provider": provider,
                    "dataset": dataset,
                    "seed": seed,
                    "returncode": rc,
                    "duration_s": round(dur, 1),
                })
                status = "OK" if rc == 0 else f"FAIL (rc={rc})"
                print(f"[sft] {tag}: {status} in {dur:.1f}s")
    return summaries


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_model_overrides(items: list[str]) -> dict[str, str]:
    out = dict(DEFAULT_MODELS)
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--model expects provider=model, got {item!r}"
            )
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frontier-LLM trace pipeline (query + SFT, no plotting).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--providers", nargs="+", default=DEFAULT_PROVIDERS,
        choices=DEFAULT_PROVIDERS,
        help="Which frontier providers to query / SFT on.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        choices=list(DATASET_CONFIGS),
        help="Which datasets to run.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Seeds passed to the SFT runs (one run directory per seed).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=0,
        help="Per-(provider, dataset) cap for the query stage. 0 = use the "
             "full (train+holdout) split from the dataset config.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help="Concurrency for the query stage (per provider).",
    )
    parser.add_argument(
        "--model", action="append", default=[],
        help="Override the default model for a provider, e.g. "
             "`--model openai=gpt-4o-mini --model claude=claude-3-5-sonnet`.",
    )
    parser.add_argument(
        "--traces-dir", type=Path, default=REPO_ROOT / "traces_llms",
        help="Where query JSONLs are written / read.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs",
        help="Where SFT run directories are written.",
    )
    parser.add_argument("--skip-query", action="store_true",
                        help="Skip stage 1 (assume traces already on disk).")
    parser.add_argument("--skip-sft", action="store_true",
                        help="Skip stage 2 (only generate the JSONLs).")
    args = parser.parse_args()

    if args.skip_query and args.skip_sft:
        parser.error("--skip-query and --skip-sft together leaves nothing to do.")

    models = _parse_model_overrides(args.model)

    print("=" * 70)
    print("  Frontier-LLM pipeline")
    print("=" * 70)
    print(f"  Providers:   {args.providers}")
    print(f"  Datasets:    {args.datasets}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Models:      " + ", ".join(f"{p}={models[p]}" for p in args.providers))
    print(f"  Traces dir:  {args.traces_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  num_samples: {args.num_samples or 'all'}")
    print(f"  max_workers: {args.max_workers}")
    print(f"  skip_query:  {args.skip_query}")
    print(f"  skip_sft:    {args.skip_sft}")
    print(f"  started:     {datetime.now().isoformat(timespec='seconds')}")
    print()

    pipeline_t0 = time.perf_counter()

    if not args.skip_query:
        run_query(
            providers=args.providers,
            datasets=args.datasets,
            models=models,
            num_samples=args.num_samples,
            traces_dir=args.traces_dir,
            max_workers=args.max_workers,
        )
    else:
        print("[query] skipped (--skip-query)")

    sft_summaries: list[dict] = []
    if not args.skip_sft:
        sft_summaries = run_sft(
            providers=args.providers,
            datasets=args.datasets,
            seeds=args.seeds,
            output_dir=args.output_dir,
            traces_dir=args.traces_dir,
        )
    else:
        print("[sft] skipped (--skip-sft)")

    total = time.perf_counter() - pipeline_t0
    print()
    print("=" * 70)
    print("  Frontier-LLM pipeline complete")
    print("=" * 70)
    print(f"  Wall time: {total:.1f}s")
    if sft_summaries:
        ok = sum(1 for s in sft_summaries if s["returncode"] == 0)
        print(f"  SFT runs:  {ok}/{len(sft_summaries)} succeeded")
        for s in sft_summaries:
            tag = f"{s['provider']}/{s['dataset']}/seed{s['seed']}"
            status = "OK" if s["returncode"] == 0 else f"FAIL(rc={s['returncode']})"
            print(f"    {tag:<35} {status:<14} {s['duration_s']:>7.1f}s")
    print()
    print(f"  Traces:    {args.traces_dir}")
    print(f"  Runs:      {args.output_dir}")
    print(f"  Plot with: python frontier-llms/plot-llm.py "
          f"--runs-root {args.output_dir}")


if __name__ == "__main__":
    main()
