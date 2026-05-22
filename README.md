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

Baseline pipeline that distills traces from frontier LLMs (OpenAI, Gemini, Claude) into the local student. Chains three stages — query the APIs, SFT per `(provider, dataset, seed)`, plot.

```bash
python3 scripts/run_frontier_llms.py
```

Common flags: `--providers`, `--datasets`, `--seeds`, `--num-samples`, `--skip-{query,sft,plot}`, `--plot-only`, `--output-dir`. Set API keys at the top of `src/frontier-llms/query_trace_frontier.py` first.

Writes traces to `traces_llms/`, SFT runs to `outputs/real_*/`, and plots to `outputs/plots/`.

## Trace Quality Scoring

Scores teacher traces on a 1–5 auditability rubric via the Claude API and plots a per-method PMF (`Standard`, `PoE`, `ADS`).

```bash
python3 scripts/trace_quality_llm.py
```

Common flags: `--datasets`, `--model`, `--max-examples`, `--plot-only`. Set `API_KEY` at the top of the script first.

Reads `plot-quality/<dataset>/train_*.json` and writes scored JSONs plus `trace_quality_pmf.pdf` next to them.

## Citation

If you use this repository, please cite the accompanying paper:

*The Distillation Game: Adaptive Attacks & Efficient Defenses*.


## Repository Layout

- `scripts/run_pipeline.py`: end-to-end experiment driver
- `scripts/run_frontier_llms.py`: frontier-LLM trace experiments (query → SFT → plot)
- `scripts/trace_quality_llm.py`: Claude-based trace auditability scorer
- `src/frontier-llms/`: frontier-LLM query + SFT entry-points used by `scripts/run_frontier_llms.py`
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
