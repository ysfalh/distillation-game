#!/bin/bash
#SBATCH --job-name=student-saved-llama1b
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/student_saved_llama1b_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00

# Student-only run from flat saved traces such as gsm8k_output_small/.
#
# Default submit:
#   sbatch slurm_scripts/run-student-from-saved.sbatch
#
# Optional overrides:
#   sbatch --export=ALL,INPUT_DIR=gsm8k_output_small,OUTPUT_DIR=outputs/gsm8k_llama1b_from_saved_teachers \
#          slurm_scripts/run-student-from-saved.sbatch
#
# Dry-run validation:
#   sbatch --export=ALL,DRY_RUN=1 slurm_scripts/run-student-from-saved.sbatch

source "$HOME/nlphw/my_project.env"
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export HF_TOKEN="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"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
# Reduce allocator fragmentation during training/eval.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INPUT_DIR="${INPUT_DIR:-gsm8k_output_small}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gsm8k_llama1b_from_saved_teachers}"
STUDENT="${STUDENT:-meta-llama/Llama-3.2-1B}"
STUDENT_TOKENIZER="${STUDENT_TOKENIZER:-meta-llama/Llama-3.2-1B-Instruct}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DRY_RUN="${DRY_RUN:-0}"

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

ARGS=(
    --input-dir "${INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --student "${STUDENT}"
    --student-tokenizer "${STUDENT_TOKENIZER}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
)

if [ "${DRY_RUN}" = "1" ] || [ "${DRY_RUN}" = "true" ] || [ "${DRY_RUN}" = "TRUE" ]; then
    ARGS+=(--dry-run)
fi

echo "=== Saved-trace student run ==="
echo
echo "  Input traces:      ${INPUT_DIR}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Student:           ${STUDENT}"
echo "  Student tokenizer: ${STUDENT_TOKENIZER}"
echo "  Attention backend: ${ATTN_IMPLEMENTATION}"
echo "  Dry run:           ${DRY_RUN}"
echo "  Host:              $(hostname)"
echo "  Job:               ${SLURM_JOB_ID}"
echo

python scripts/run_student_from_saved_traces.py "${ARGS[@]}"

echo
echo "=== Saved-trace student run complete ==="
echo "Results:"
echo "  ${OUTPUT_DIR}/results.json"
echo "  ${OUTPUT_DIR}/RESULTS.md"
