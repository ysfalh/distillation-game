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

- `configs/gsm8k_small.yaml`: reduced GSM8K run for iteration
- `configs/gsm8k.yaml`: main GSM8K sweep
- `configs/gsm_hard.yaml`: GSM-Hard sweep
- `configs/svamp.yaml`: SVAMP sweep
- `configs/svamp_minimal.yaml`: reduced SVAMP run
- `configs/math_small.yaml`: reduced MATH run
- `configs/math.yaml`: main MATH sweep
- `configs/math_large.yaml`: extended MATH sweep

Each run writes a timestamped directory under `outputs/` with the config snapshot, run manifest, teacher and student artifacts, and a `RESULTS.md` summary.

## Repository Layout

- `scripts/run_pipeline.py`: end-to-end experiment driver
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
