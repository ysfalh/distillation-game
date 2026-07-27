#!/bin/bash
#SBATCH --job-name=vgap-gamma
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/vgap_gamma_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=48g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-2

# Measure v_gap = log P_teacher - log P_proxy on the traces already generated at
# each PoE gamma, to check that the token-level PoE implementation suppresses
# the gap the sequence-level rule scores. No traces are generated.
#
#   sbatch slurm_scripts/run-vgap-by-gamma.sbatch
#
# One array task per seed. Each task scores every source under its own seed and
# then rewrites outputs/vgap_poe/RESULTS.md from whatever seeds have finished,
# so the last task to land leaves the complete report. If two tasks happen to
# finish together and the report is short a seed, rebuild it in a second from
# the login node with no GPU:
#
#   python scripts/vgap_by_gamma.py --report-only
#
# Scoring is deterministic, so the seed cannot move the measured gap; it varies
# batch composition, which makes the three runs a check on numerical stability.
# The uncertainty that matters is the paired standard error across the 5238
# problems, which a single seed already gives.

# Sourced before `set -e` so a nonzero status inside the env script, which is
# not ours, cannot kill the job.
source $HOME/nlphw/my_project.env
cd /share/project/haghifam-projects/strategic-distillation || exit 1

set -eo pipefail
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-configs/gsm8k.yaml}"
DATASET="${DATASET:-gsm8k_output_small}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vgap_poe}"
# Sized for a 48 GB card holding one 7B model in bf16. The logits tensor is the
# ceiling, so drop --row-chunk before --batch-size if this ever OOMs.
BATCH_SIZE="${BATCH_SIZE:-6}"
ROW_CHUNK="${ROW_CHUNK:-1024}"

SEEDS=($(python -c "import yaml; print(*yaml.safe_load(open('${CONFIG}'))['run']['seeds'])"))
if [ ${#SEEDS[@]} -eq 0 ]; then
    echo "ERROR: could not read run.seeds from ${CONFIG}" >&2
    exit 1
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: no SLURM_ARRAY_TASK_ID, submit this with sbatch." >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SEEDS[@]}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} but ${CONFIG} lists ${#SEEDS[@]} seeds." >&2
    echo "       Resubmit with --array=0-$(( ${#SEEDS[@]} - 1 ))" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_COUNT:-${#SEEDS[@]}}" -lt "${#SEEDS[@]}" ]; then
    echo "WARNING: this array has ${SLURM_ARRAY_TASK_COUNT} tasks but ${CONFIG} lists " \
         "${#SEEDS[@]} seeds, so some will never run." >&2
fi

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

if [ ! -f "${DATASET}/train_standard.json" ]; then
    echo "ERROR: ${DATASET}/train_standard.json not found; it is the gamma = 0 baseline." >&2
    exit 1
fi

echo "=== Array task ${SLURM_ARRAY_TASK_ID} | Seed ${SEED} | ${DATASET} ==="
ls "${DATASET}"/train_*.json

python scripts/vgap_by_gamma.py \
    --input-dir "${DATASET}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --row-chunk "${ROW_CHUNK}"

echo
echo "=== ${OUTPUT_DIR}/RESULTS.md ==="
cat "${OUTPUT_DIR}/RESULTS.md"
