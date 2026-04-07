#!/bin/bash
#SBATCH --job-name=compute-weights
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/compute_weights_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --array=0-3

source $HOME/nlphw/my_project.env
mkdir -p logs

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

# ── 4 tasks: {ADS, PoE} × {beta_s=0.5, beta_s=4} ────────────────
DATA_DIR="gsm8k_output_small"
TRAIN_FILES=("train_antidistillation_lam_0.055.json" "train_poe_gamma_0.7.json")
LABELS=("ads_lam0.055" "poe_gamma0.7")
BETAS=(0.5 4)

TEACHER_IDX=$((SLURM_ARRAY_TASK_ID % 2))
BETA_IDX=$((SLURM_ARRAY_TASK_ID / 2))

TRAIN_FILE="${TRAIN_FILES[$TEACHER_IDX]}"
LABEL="${LABELS[$TEACHER_IDX]}"
BETA_S="${BETAS[$BETA_IDX]}"
OUT_NAME="weights_${LABEL}_beta${BETA_S}.json"

# Patch flash_attention_2 → sdpa in a per-task temp config (avoids race)
TMP_CFG="/tmp/config_weights_${SLURM_ARRAY_TASK_ID}.yaml"
sed 's/flash_attention_2/sdpa/' "${DATA_DIR}/config_snapshot.yaml" > "${TMP_CFG}"

echo "=== Task ${SLURM_ARRAY_TASK_ID} | ${LABEL} | beta_s=${BETA_S} ==="

python scripts/compute_weights.py \
    --config  "${TMP_CFG}" \
    --holdout "${DATA_DIR}/holdout_standard_internal.json" \
    --train   "${DATA_DIR}/${TRAIN_FILE}" \
    --beta-s  ${BETA_S} \
    --out     "${DATA_DIR}/analysis/${OUT_NAME}"
