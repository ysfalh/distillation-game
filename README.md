# antidistillation_clean

A smaller, cleaner reproduction repo for the core end-to-end pipeline we care about:

- standard teacher
- antidistillation teacher
- product-of-experts teacher
- naive student
- strategic-fd student

This repo is intentionally narrower than the original project. It removes features that made runs harder to reason about or reproduce:

- no auto batch-size scaling
- no fast-math backend toggles hidden behind the sweep
- no Newton variants
- no trace-faithfulness or auxiliary faithfulness scores
- no large checkpoint inventories by default
- no verbose JSONL traces that repeat prompt text and full decoded strings unnecessarily

## Scope

The target is to reproduce, in a cleaner implementation, the practical outcome of running the full teacher/student comparison pipeline with the GSM8K Savani-style setup, optionally on a reduced dataset size for faster iteration.

The first milestone is:

1. fixed dataset split materialization
2. deterministic trace generation with explicit batching
3. teacher sweep over `standard`, `antidistillation`, and `product_of_experts`
4. student training for `naive` and `strategic_fd`
5. compact artifact storage
6. a more informative final results table in the README-style summary

## Design principles

- Every run should save the exact split indices and config snapshot.
- Batch size is explicit and never auto-changed.
- Reproducibility defaults should be conservative.
- Optional stochastic decoding stays explicit in config.
- Outputs should be compact and structured for comparison, not archival hoarding.

## Current layout

- `scripts/run_pipeline.py`: end-to-end teacher + student pipeline
- `configs/gsm8k_savani_small.yaml`: reduced-size config for iteration
- `src/clean_sweep/config.py`: small typed config schema
- `src/clean_sweep/data/`: split materialization and prompt formatting
- `src/clean_sweep/generation/`: standard, antidistillation, and PoE decoding
- `src/clean_sweep/train/`: naive and strategic-fd distillation
- `src/clean_sweep/eval/`: GSM8K answer extraction and accuracy
- `src/clean_sweep/utils/`: seeding, I/O, artifact writing

## Current pipeline

The clean pipeline now does the following:

1. materializes fixed GSM8K train, holdout, and test splits
2. saves a run manifest and compact prompt dictionary
3. generates standard teacher traces for holdout, train, and test
4. computes proxy gradients from the standard holdout traces
5. generates antidistillation teacher traces for configured `lam` values
6. generates product-of-experts teacher traces for configured `gamma` values
7. trains `naive` and `strategic_fd` students from standard teacher train traces
8. evaluates students on the test set with standard decoding
9. writes a compact Markdown results table with dataset and model context

## Running

From the repo root:

```bash
PYTHONPATH=src python3 scripts/run_pipeline.py --config configs/gsm8k_small.yaml
```

Main output files:

- `outputs/<run_name_timestamp>/run_manifest.json`
- `outputs/<run_name_timestamp>/prompts.json`
- `outputs/<run_name_timestamp>/teacher/*.jsonl`
- `outputs/<run_name_timestamp>/student/*.jsonl`
- `outputs/<run_name_timestamp>/inspection/*.json`
- `outputs/<run_name_timestamp>/RESULTS.md`

## Artifact philosophy

Instead of storing many repetitive JSONL files with duplicated prompts and full trace text, the clean repo will prefer:

- one compact dataset manifest per run
- one trace record format with stable `example_id`
- optional prompt dictionary saved once per split
- compact per-condition outputs keyed by `example_id`
- metrics tables and small inspection samples

By default, student checkpoints should not be kept unless explicitly requested.
