#!/bin/bash
#SBATCH --job-name=compute-weights
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/compute_weights_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --array=0-1

source $HOME/nlphw/my_project.env
mkdir -p logs

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

# ── Task 0: ADS     Task 1: PoE ──────────────────────────────────
ADS_DIR="gsm8k_output/gsm8k_ads_456_20260324_203021"
POE_DIR="gsm8k_output/gsm8k_poe_456_20260326_112949"
BETA_S=0.5

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    RUN_DIR="${ADS_DIR}"
    TRAIN_FILE="teacher/train_antidistillation_lam_0.055.json"
    OUT_NAME="weights_ads_lam0.055_beta0.5.json"
else
    RUN_DIR="${POE_DIR}"
    TRAIN_FILE="teacher/train_poe_gamma_0.7.json"
    OUT_NAME="weights_poe_gamma0.7_beta0.5.json"
fi

echo "=== Task ${SLURM_ARRAY_TASK_ID} | dir=${RUN_DIR} | train=${TRAIN_FILE} ==="

python scripts/compute_weights.py \
    --config  "${RUN_DIR}/config_snapshot.yaml" \
    --holdout "${RUN_DIR}/teacher/holdout_standard_internal.json" \
    --train   "${RUN_DIR}/${TRAIN_FILE}" \
    --beta-s  ${BETA_S} \
    --out     "gsm8k_output/analysis/${OUT_NAME}"
