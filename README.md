# The Distillation Game

Code for the experiments in *The Distillation Game: Adaptive Attacks & Efficient Defenses*.

The repository provides an end-to-end pipeline for teacher-generation and student-distillation experiments. Supported teacher methods include standard decoding, antidistillation, and product-of-experts; supported student modes are passive (`naive`) and adaptive (`strategic_fd`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default configurations target GPU-backed runs.

## Running

```bash
PYTHONPATH=src python3 scripts/run_pipeline.py --config configs/gsm8k_small.yaml
```

Useful configs:


- `configs/gsm8k.yaml`: main GSM8K sweep
- `configs/gsm8k_small.yaml`: reduced GSM8K run for iteration
- `configs/math_large.yaml`: main MATH sweep
- `configs/math_small.yaml`: reduced MATH run

Each run writes a timestamped directory under `outputs/` with the config snapshot, run manifest, teacher and student artifacts, and a `RESULTS.md` summary.

## Frontier-LLM Trace Experiments

Auxiliary pipeline that distills traces from frontier LLMs (OpenAI, Gemini, Claude) into the local student, as a baseline alongside the teacher-side experiments. It chains three stages: querying frontier APIs for traces, running SFT per `(provider, dataset, seed)`, and aggregating the runs into per-dataset bar charts.

```bash
python3 scripts/run_frontier_llms.py
```

Useful flags:

- `--providers {openai,gemini,claude}`: which providers to query / SFT on (default: all three)
- `--datasets {gsm8k,math}`: which datasets to use (default: both)
- `--seeds 42 123 456`: seeds for SFT runs (one run directory per seed)
- `--num-samples N`: per-(provider, dataset) cap on query examples (0 = full split)
- `--skip-query`, `--skip-sft`, `--skip-plot`: skip individual stages
- `--plot-only`: shortcut for `--skip-query --skip-sft`, regenerates PDFs from existing runs
- `--fallbacks fb.json`: fill in missing `(provider, dataset, seed)` cells from a JSON file
- `--output-dir`, `--plot-dir`: control where SFT runs and plot artifacts are written

API keys live at the top of `frontier-llms/query_trace_frontier.py`; set them there before running with `--providers` that actually call the API.

Outputs:

- `traces_llms/<provider>_<dataset>_traces.jsonl`: raw frontier traces (resumable)
- `outputs/real_<provider>_<dataset>_traces_seed<seed>_<ts>/results.json`: per-run SFT results
- `outputs/plots/llm_accuracy_{gsm8k,math}.pdf`: aggregated bar charts
- `outputs/plots/llm_accuracy_summary.json`: machine-readable summary

## Trace Quality Scoring

Scores reasoning traces produced by the main teacher-generation pipeline on a 1–5 auditability rubric using the Claude API, and plots a PMF of scores per teacher method (`Standard`, `PoE`, `ADS`).

```bash
python3 scripts/trace_quality_llm.py
```

Useful flags:

- `--datasets {gsm8k,math}`: which dataset(s) to score (default: both)
- `--model claude-opus-4-0-20250514`: judge model
- `--max-examples N`: cap per (dataset, method) file (0 = score all)
- `--plot-only`: skip scoring, only re-plot from existing JSONs

Set the `API_KEY` constant at the top of the script before running.

Inputs (relative to repo root):

- `plot-quality/<dataset>/train_standard.json`
- `plot-quality/<dataset>/train_poe_gamma_<gamma>.json`
- `plot-quality/<dataset>/train_antidistillation_lam_<lam>.json`

Outputs (under the same `plot-quality/<dataset>/` directory):

- `trace_quality_{standard,poe,ads}.json`: per-example score + feedback (resumable)
- `trace_quality_pmf.pdf`: PMF of scores across the three methods

## Citation

If you use this repository, please cite the accompanying paper:

*The Distillation Game: Adaptive Attacks & Efficient Defenses*.


## Repository Layout

- `scripts/run_pipeline.py`: end-to-end experiment driver
- `scripts/run_frontier_llms.py`: frontier-LLM trace experiments (query → SFT → plot)
- `scripts/trace_quality_llm.py`: Claude-based trace auditability scorer
- `frontier-llms/`: frontier-LLM query + SFT entry-points used by `scripts/run_frontier_llms.py`
- `src/clean_sweep/config.py`: typed experiment configuration
- `src/clean_sweep/data/`: dataset loading, split materialization, and prompt formatting
- `src/clean_sweep/generation/`: teacher generation methods
- `src/clean_sweep/train/`: student distillation routines
- `src/clean_sweep/eval/`: answer extraction and correctness evaluation
- `src/clean_sweep/summary.py`: Markdown results summary

## Outputs

Typical run artifacts include:

- `config_snapshot.yaml`
- `run_manifest.json`
- `prompts.json`
- `teacher/*.json`
- `student/*.json`
- `RESULTS.md`
