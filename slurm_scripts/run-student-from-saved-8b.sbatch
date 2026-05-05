#!/bin/bash
#SBATCH --job-name=student-saved-llama8b
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/student_saved_llama8b_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=96gb
#SBATCH --time=24:00:00

# Student-only 8B run from flat saved traces such as gsm8k_output_small/.
#
# This does not modify gsm8k_output_small. It builds a temporary input view
# with symlinks to the saved traces and a copied config_snapshot.yaml with
# smaller student-side batch sizes for 8B.
#
# Default submit:
#   sbatch slurm_scripts/run-student-from-saved-8b.sbatch
#
# If Llama access needs a different token:
#   sbatch --export=ALL,HF_TOKEN=hf_...,OUTPUT_DIR=outputs/gsm8k_llama8b_from_saved_teachers \
#          slurm_scripts/run-student-from-saved-8b.sbatch
#
# Dry-run validation:
#   sbatch --export=ALL,DRY_RUN=1 slurm_scripts/run-student-from-saved-8b.sbatch

source "$HOME/nlphw/my_project.env"
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INPUT_DIR="${INPUT_DIR:-gsm8k_output_small}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gsm8k_llama8b_from_saved_teachers}"
STUDENT="${STUDENT:-meta-llama/Meta-Llama-3.1-8B}"
STUDENT_TOKENIZER="${STUDENT_TOKENIZER:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DRY_RUN="${DRY_RUN:-0}"

# Conservative 8B defaults for 96GB GPUs. Override via sbatch --export if
# you want to trade memory for speed.
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-24}"
HOLDOUT_GRAD_BATCH_SIZE="${HOLDOUT_GRAD_BATCH_SIZE:-1}"
TRACE_WEIGHTS_FD_BATCH_SIZE="${TRACE_WEIGHTS_FD_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: INPUT_DIR does not exist: ${INPUT_DIR}" >&2
    exit 1
fi

for required in \
    "${INPUT_DIR}/config_snapshot.yaml" \
    "${INPUT_DIR}/holdout_standard_internal.json" \
    "${INPUT_DIR}/train_standard.json" \
    "${INPUT_DIR}/train_antidistillation_lam_0.055.json" \
    "${INPUT_DIR}/train_poe_gamma_0.7.json"
do
    if [ ! -f "${required}" ]; then
        echo "ERROR: required saved-trace file missing: ${required}" >&2
        exit 1
    fi
done

TMP_PARENT="${SLURM_TMPDIR:-/tmp}"
TMP_INPUT_DIR="${TMP_PARENT}/saved_traces_8b_${SLURM_JOB_ID:-$$}"
rm -rf "${TMP_INPUT_DIR}"
mkdir -p "${TMP_INPUT_DIR}"

for artifact in \
    holdout_standard_internal.json \
    train_standard.json \
    train_antidistillation_lam_0.055.json \
    train_poe_gamma_0.7.json
do
    ln -s "$(realpath "${INPUT_DIR}/${artifact}")" "${TMP_INPUT_DIR}/${artifact}"
done

python3 - <<PY
import yaml
from pathlib import Path

src = Path("${INPUT_DIR}") / "config_snapshot.yaml"
dst = Path("${TMP_INPUT_DIR}") / "config_snapshot.yaml"
cfg = yaml.safe_load(src.read_text())

cfg.setdefault("model", {})["student"] = "${STUDENT}"
cfg.setdefault("model", {})["student_tokenizer"] = "${STUDENT_TOKENIZER}"
cfg.setdefault("model", {})["attn_implementation"] = "${ATTN_IMPLEMENTATION}"

distill = cfg.setdefault("distill", {})
distill["per_device_train_batch_size"] = int("${PER_DEVICE_TRAIN_BATCH_SIZE}")
distill["gradient_accumulation_steps"] = int("${GRADIENT_ACCUMULATION_STEPS}")
distill["holdout_grad_batch_size"] = int("${HOLDOUT_GRAD_BATCH_SIZE}")
distill["trace_weights_fd_batch_size"] = int("${TRACE_WEIGHTS_FD_BATCH_SIZE}")
distill["eval_batch_size"] = int("${EVAL_BATCH_SIZE}")

yaml.safe_dump(cfg, dst.open("w"), sort_keys=False)
PY

ARGS=(
    --input-dir "${TMP_INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --student "${STUDENT}"
    --student-tokenizer "${STUDENT_TOKENIZER}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
)

if [ "${DRY_RUN}" = "1" ] || [ "${DRY_RUN}" = "true" ] || [ "${DRY_RUN}" = "TRUE" ]; then
    ARGS+=(--dry-run)
fi

echo "=== Saved-trace 8B student run ==="
echo
echo "  Original traces:        ${INPUT_DIR}"
echo "  Temporary input view:   ${TMP_INPUT_DIR}"
echo "  Output:                 ${OUTPUT_DIR}"
echo "  Student:                ${STUDENT}"
echo "  Student tokenizer:      ${STUDENT_TOKENIZER}"
echo "  Attention backend:      ${ATTN_IMPLEMENTATION}"
echo "  per_device_train_bs:    ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  grad_accum_steps:       ${GRADIENT_ACCUMULATION_STEPS}"
echo "  holdout_grad_bs:        ${HOLDOUT_GRAD_BATCH_SIZE}"
echo "  trace_weights_fd_bs:    ${TRACE_WEIGHTS_FD_BATCH_SIZE}"
echo "  eval_batch_size:        ${EVAL_BATCH_SIZE}"
echo "  Dry run:                ${DRY_RUN}"
echo "  Host:                   $(hostname)"
echo "  Job:                    ${SLURM_JOB_ID}"
echo

python scripts/run_student_from_saved_traces.py "${ARGS[@]}"

echo
echo "=== Saved-trace 8B student run complete ==="
echo "Results:"
echo "  ${OUTPUT_DIR}/results.json"
echo "  ${OUTPUT_DIR}/RESULTS.md"
