#!/usr/bin/env python3
"""
Frontier-LLM pipeline: query → SFT → plot.

Chains the three stages of the real-world LLM trace experiment:

  1. QUERY  — call OpenAI / Gemini / Claude for reasoning traces on GSM8K
              and MATH problems, writing one JSONL per (provider, dataset)
              under --traces-dir (default: traces_llms/).

  2. SFT    — for each (provider, dataset, seed), fine-tune the student
              on those traces and on the dataset's ground-truth Q&A, then
              evaluate on the test split. One run directory per triple
              under --output-dir (default: outputs/).

  3. PLOT   — aggregate the SFT run directories into per-dataset bar
              charts (Base, SFT on Q&A, Gemini, Claude, GPT) and a
              summary JSON, written under --plot-dir
              (default: <output-dir>/plots/).

Stages 1 & 2 re-use the existing scripts in frontier-llms/ without
modification:
  - frontier-llms/query_trace_frontier.py   provides generate_trace_llm()
  - frontier-llms/run_real_trace.py         invoked as a subprocess

Stage 3 is implemented in-file (no external plot script).

Examples
--------
Full sweep (all 3 providers × 2 datasets × 3 seeds, all problems):

    python scripts/run_frontier_llms.py

Query Claude on MATH only, then SFT for two seeds:

    python scripts/run_frontier_llms.py \\
        --providers claude --datasets math --seeds 42 123

Skip query (use whatever JSONLs are already in traces_llms/):

    python scripts/run_frontier_llms.py --skip-query

Plot only — aggregate existing SFT runs into PDFs without running anything:

    python scripts/run_frontier_llms.py --plot-only \\
        --output-dir outputs_from_cluster/outputs \\
        --plot-dir outputs_from_cluster/plots

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
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTIER_DIR = REPO_ROOT /scripts/"frontier-llms"

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


# ── Stage 3: PLOT ─────────────────────────────────────────────────────────
# Aggregates real_<provider>_<dataset>_traces_seed<seed>_<timestamp>/ runs
# into per-dataset PDFs and a summary JSON. Ported from the former
# frontier-llms/plot-llm.py so the orchestrator is self-contained.

_RUN_RE = re.compile(
    r"^real_(?P<provider>[^_]+)_(?P<dataset>gsm8k|math)_traces_seed"
    r"(?P<seed>\d+)_(?P<timestamp>\d{8}_\d{6})$"
)
_PROVIDER_BAR_LABELS = {"gemini": "Gemini", "claude": "Claude", "openai": "GPT"}
_DATASET_LABELS = {"gsm8k": "GSM8K", "math": "MATH"}
_ROW_TO_LABEL = {
    ("base", None, None): "Base",
    ("qa", "naive", None): "SFT on Q&A",
    ("traces", "naive", None): "Trace SFT",
}
_PLOT_BAR_ORDER = ["Base", "SFT on Q&A", "Gemini", "Claude", "GPT"]
_BAR_COLOR = "#79B984"
_EDGE_COLOR = "#335B3E"


def _mean_stderr(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var) / math.sqrt(len(values))


def _format_percent(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _find_runs(
    runs_root: Path, providers: list[str], datasets: list[str], seeds: list[int],
) -> tuple[dict[tuple[str, str, int], dict], dict[tuple[str, str, int], list[dict]]]:
    """Walk runs_root and pick the newest completed run per (prov, ds, seed)."""
    latest_completed: dict[tuple[str, str, int], dict] = {}
    observed: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    if not runs_root.exists():
        return latest_completed, observed
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = _RUN_RE.match(run_dir.name)
        if not m:
            continue
        meta = m.groupdict()
        prov, ds, seed = meta["provider"], meta["dataset"], int(meta["seed"])
        if prov not in providers or ds not in datasets or seed not in seeds:
            continue
        key = (prov, ds, seed)
        candidate = {
            "provider": prov,
            "dataset": ds,
            "seed": seed,
            "timestamp": meta["timestamp"],
            "run_dir": run_dir,
            "results_path": run_dir / "results.json",
            "has_results": (run_dir / "results.json").exists(),
        }
        observed[key].append(candidate)
        if not candidate["has_results"]:
            continue
        cur = latest_completed.get(key)
        if cur is None or candidate["timestamp"] > cur["timestamp"]:
            latest_completed[key] = candidate
    return latest_completed, observed


def _load_actual_results(
    latest_completed: dict[tuple[str, str, int], dict],
) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    per: dict[str, dict[str, dict[int, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (provider, dataset, seed), run in latest_completed.items():
        with run["results_path"].open() as f:
            results = json.load(f)
        values: dict[str, float] = {}
        for row in results:
            label = _ROW_TO_LABEL.get(
                (row.get("source"), row.get("mode"), row.get("beta_s"))
            )
            if label is not None:
                values[label] = float(row["accuracy"])
        per[dataset][provider][seed] = values
    return per


def _apply_fallbacks(
    per: dict, fallbacks: dict[tuple[str, str, int], dict[str, float]],
) -> list[dict]:
    used: list[dict] = []
    for (dataset, provider, seed), values in fallbacks.items():
        if seed not in per[dataset][provider]:
            per[dataset][provider][seed] = dict(values)
            used.append({"dataset": dataset, "provider": provider, "seed": seed})
    return used


def _load_fallbacks(path: Path | None) -> dict[tuple[str, str, int], dict[str, float]]:
    """Read a fallback JSON keyed by 'dataset/provider/seed' → label-map.

    Example file (every key optional; only triples not produced on disk are
    used):

        {
          "math/gemini/42":  {"Base": 0.01, "SFT on Q&A": 0.10, "Trace SFT": 0.12},
          "math/gemini/123": {"Base": 0.01, "SFT on Q&A": 0.10, "Trace SFT": 0.12}
        }
    """
    if path is None:
        return {}
    with path.open() as f:
        raw = json.load(f)
    out: dict[tuple[str, str, int], dict[str, float]] = {}
    for k, v in raw.items():
        ds, prov, seed_str = k.split("/")
        out[(ds, prov, int(seed_str.replace("seed", "")))] = {
            label: float(val) for label, val in v.items()
        }
    return out


def _compute_plot_stats(per: dict, providers: list[str], datasets: list[str], seeds: list[int]) -> dict:
    stats: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for dataset in datasets:
        ds_stats: dict[str, dict[str, float | int | None]] = {}
        # Base + SFT-on-Q&A: average within a seed across providers, then
        # mean/stderr over seeds. These values don't depend on the provider
        # (same dataset → same QA → same baseline), so averaging over
        # providers within a seed is just smoothing noise.
        for baseline in ("Base", "SFT on Q&A"):
            seed_means: list[float] = []
            for seed in seeds:
                vs = [
                    per.get(dataset, {}).get(p, {}).get(seed, {}).get(baseline)
                    for p in providers
                ]
                vs = [v for v in vs if v is not None]
                if vs:
                    seed_means.append(sum(vs) / len(vs))
            mean, se = _mean_stderr(seed_means)
            ds_stats[baseline] = {"mean": mean, "stderr": se, "n": len(seed_means)}
        # Per-provider bars: trace-SFT accuracy across seeds.
        for provider in providers:
            label = _PROVIDER_BAR_LABELS.get(provider, provider.capitalize())
            trace_vals: list[float] = []
            for seed in seeds:
                v = per.get(dataset, {}).get(provider, {}).get(seed, {}).get("Trace SFT")
                if v is not None:
                    trace_vals.append(v)
            mean, se = _mean_stderr(trace_vals)
            ds_stats[label] = {"mean": mean, "stderr": se, "n": len(trace_vals)}
        stats[dataset] = ds_stats
    return stats


def _bar_order_for(providers: list[str]) -> list[str]:
    """Provider bars are emitted in a stable order (Gemini, Claude, GPT)."""
    ordered = ["Base", "SFT on Q&A"]
    for p in ("gemini", "claude", "openai"):
        if p in providers:
            ordered.append(_PROVIDER_BAR_LABELS[p])
    return ordered


def _plot_dataset(
    dataset: str, plot_stats: dict, bar_order: list[str], out_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Style — kept bit-identical to the former frontier-llms/plot-llm.py so
    # PDFs produced here are visually indistinguishable from the historical
    # outputs.
    plt.rcParams.update(
        {
            "text.usetex": shutil.which("latex") is not None,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "Latin Modern Roman",
                "DejaVu Serif",
                "cmr10",
            ],
            "font.size": 16,
            "axes.unicode_minus": False,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )

    stats = plot_stats[dataset]
    means = [
        stats[label]["mean"] * 100.0 if stats[label]["mean"] is not None else np.nan
        for label in bar_order
    ]
    errors = [
        stats[label]["stderr"] * 100.0 if stats[label]["stderr"] is not None else 0.0
        for label in bar_order
    ]

    x = np.arange(len(bar_order))
    fig, ax = plt.subplots(figsize=(6.8, 5.9))
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        capsize=4,
        color=_BAR_COLOR,
        edgecolor=_EDGE_COLOR,
        linewidth=1.2,
        zorder=3,
    )

    ax.set_title(f"Llama-3.2-3B Performance on {_DATASET_LABELS[dataset]}")
    ax.set_ylabel(f"Test Accuracy on {_DATASET_LABELS[dataset]} (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_order, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    finite = [m + e for m, e in zip(means, errors) if not np.isnan(m)]
    ymax = max(finite) if finite else 5.0
    ax.set_ylim(0, max(5.0, ymax * 1.28))

    for bar, value, err in zip(bars, means, errors):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + err + max(ax.get_ylim()[1] * 0.015, 0.2),
            _format_percent(value),
            ha="center",
            va="bottom",
            fontsize=15,
        )

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"llm_accuracy_{dataset}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def _compute_missing_without_fallback(
    observed: dict[tuple[str, str, int], list[dict]],
    latest_completed: dict[tuple[str, str, int], dict],
    fallbacks: dict[tuple[str, str, int], dict[str, float]],
    providers: list[str],
    datasets: list[str],
    seeds: list[int],
) -> list[dict]:
    missing: list[dict] = []
    for dataset in datasets:
        for provider in providers:
            for seed in seeds:
                actual_key = (provider, dataset, seed)
                fallback_key = (dataset, provider, seed)
                if actual_key in latest_completed or fallback_key in fallbacks:
                    continue
                observed_runs = sorted(
                    observed.get(actual_key, []), key=lambda item: item["timestamp"]
                )
                newest = observed_runs[-1] if observed_runs else None
                missing.append(
                    {
                        "dataset": dataset,
                        "provider": provider,
                        "seed": seed,
                        "newest_timestamp": None if newest is None else newest["timestamp"],
                        "newest_run_dir": None if newest is None else newest["run_dir"].name,
                    }
                )
    return missing


def _print_summary_block(
    latest_completed: dict[tuple[str, str, int], dict],
    used_fallbacks: list[dict],
    missing_without_fallback: list[dict],
    plot_stats: dict,
    bar_order: list[str],
    datasets: list[str],
) -> None:
    """Console layout — kept identical to the former plot-llm.py print_summary()."""
    print("Selected latest completed runs:")
    for (provider, dataset, seed), run in sorted(latest_completed.items()):
        print(
            f"  {dataset:5s} | {provider:6s} | seed={seed:3d} | "
            f"{run['timestamp']} | {run['run_dir'].name}"
        )

    if used_fallbacks:
        print("\nUsing hardcoded fallbacks:")
        for item in used_fallbacks:
            print(
                f"  {item['dataset']:5s} | {item['provider']:6s} | seed={item['seed']:3d}"
            )

    if missing_without_fallback:
        print("\nMissing results.json with no fallback:")
        for item in missing_without_fallback:
            if item["newest_run_dir"] is None:
                print(
                    f"  {item['dataset']:5s} | {item['provider']:6s} | seed={item['seed']:3d} | "
                    "no run directory found"
                )
            else:
                print(
                    f"  {item['dataset']:5s} | {item['provider']:6s} | seed={item['seed']:3d} | "
                    f"newest incomplete run: {item['newest_timestamp']} | {item['newest_run_dir']}"
                )

    print("\nPlot values:")
    for dataset in datasets:
        if dataset not in plot_stats:
            continue
        print(f"  {_DATASET_LABELS[dataset]}:")
        for label in bar_order:
            stat = plot_stats[dataset].get(label)
            if stat is None or stat["mean"] is None:
                print(f"    {label:10s} | missing")
            else:
                print(
                    f"    {label:10s} | mean={stat['mean'] * 100:.2f}% | "
                    f"stderr={stat['stderr'] * 100:.2f}% | n={stat['n']}"
                )


def run_plot(
    *,
    providers: list[str],
    datasets: list[str],
    seeds: list[int],
    runs_root: Path,
    plot_dir: Path,
    fallbacks_path: Path | None = None,
) -> dict:
    """Aggregate runs under runs_root and emit per-dataset PDFs + summary JSON."""
    latest_completed, observed = _find_runs(runs_root, providers, datasets, seeds)
    per = _load_actual_results(latest_completed)
    fallbacks = _load_fallbacks(fallbacks_path)
    used_fallbacks = _apply_fallbacks(per, fallbacks)
    plot_stats = _compute_plot_stats(per, providers, datasets, seeds)
    missing_without_fallback = _compute_missing_without_fallback(
        observed, latest_completed, fallbacks, providers, datasets, seeds,
    )
    bar_order = _bar_order_for(providers)

    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_path = plot_dir / "llm_accuracy_summary.json"
    payload = {
        "selected_runs": {
            f"{ds}/{prov}/seed{seed}": {
                "timestamp": run["timestamp"],
                "run_dir": str(run["run_dir"]),
            }
            for (prov, ds, seed), run in sorted(latest_completed.items())
        },
        "used_fallbacks": used_fallbacks,
        "per_seed_provider": {
            ds: {
                p: {str(seed): vals for seed, vals in sorted(provider_data.items())}
                for p, provider_data in sorted(ds_data.items())
            }
            for ds, ds_data in sorted(per.items())
        },
        "plot_stats": plot_stats,
        "config": {
            "providers": providers,
            "datasets": datasets,
            "seeds": seeds,
            "runs_root": str(runs_root),
            "plot_dir": str(plot_dir),
        },
    }
    with summary_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    pdfs: list[Path] = []
    for dataset in datasets:
        if dataset not in _DATASET_LABELS:
            print(f"Skipping unknown dataset: {dataset}")
            continue
        pdfs.append(_plot_dataset(dataset, plot_stats, bar_order, plot_dir))

    # Console output style — matches the former plot-llm.py exactly so log
    # parsers / downstream tooling that scraped its stdout keep working.
    for path in pdfs:
        print(f"Saved plot: {path}")
    print(f"Saved summary: {summary_path}\n")
    _print_summary_block(
        latest_completed,
        used_fallbacks,
        missing_without_fallback,
        plot_stats,
        bar_order,
        datasets,
    )
    return {"pdfs": [str(p) for p in pdfs], "summary": str(summary_path)}


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
        description="Run the frontier-LLM trace pipeline (query + SFT + plot).",
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
        help="Where SFT run directories are written. Also where stage 3 "
             "looks for real_<provider>_<dataset>_traces_seed<seed>_<ts>/ dirs.",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=None,
        help="Where stage 3 writes PDFs and the summary JSON. "
             "Defaults to <output-dir>/plots/.",
    )
    parser.add_argument(
        "--fallbacks", type=Path, default=None,
        help="Optional JSON of {'<dataset>/<provider>/seed<seed>': {label: acc}} "
             "rows to fill in for (provider, dataset, seed) triples missing "
             "from --output-dir.",
    )
    parser.add_argument("--skip-query", action="store_true",
                        help="Skip stage 1 (assume traces already on disk).")
    parser.add_argument("--skip-sft", action="store_true",
                        help="Skip stage 2 (only generate JSONLs and/or plot).")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip stage 3 (no PDFs / summary written).")
    parser.add_argument("--plot-only", action="store_true",
                        help="Shortcut for --skip-query --skip-sft.")
    args = parser.parse_args()

    if args.plot_only:
        args.skip_query = True
        args.skip_sft = True
    if args.skip_query and args.skip_sft and args.skip_plot:
        parser.error("All three stages are skipped — nothing to do.")

    models = _parse_model_overrides(args.model)
    plot_dir = args.plot_dir or (args.output_dir / "plots")

    print("=" * 70)
    print("  Frontier-LLM pipeline")
    print("=" * 70)
    print(f"  Providers:   {args.providers}")
    print(f"  Datasets:    {args.datasets}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Models:      " + ", ".join(f"{p}={models[p]}" for p in args.providers))
    print(f"  Traces dir:  {args.traces_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Plot dir:    {plot_dir}")
    print(f"  num_samples: {args.num_samples or 'all'}")
    print(f"  max_workers: {args.max_workers}")
    print(f"  skip_query:  {args.skip_query}")
    print(f"  skip_sft:    {args.skip_sft}")
    print(f"  skip_plot:   {args.skip_plot}")
    print(f"  fallbacks:   {args.fallbacks or '-'}")
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

    plot_result: dict | None = None
    if not args.skip_plot:
        print()
        print("[plot] === aggregating runs ===")
        plot_result = run_plot(
            providers=args.providers,
            datasets=args.datasets,
            seeds=args.seeds,
            runs_root=args.output_dir,
            plot_dir=plot_dir,
            fallbacks_path=args.fallbacks,
        )
    else:
        print("[plot] skipped (--skip-plot)")

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
    if plot_result is not None:
        for pdf in plot_result.get("pdfs", []):
            print(f"  Plot:      {pdf}")
        print(f"  Summary:   {plot_result.get('summary')}")


if __name__ == "__main__":
    main()
