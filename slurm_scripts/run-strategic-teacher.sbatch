#!/bin/bash
#SBATCH --job-name=sweep-strategic-ads
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/strategic_sweep_ads_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-3   # 4 ADS lambdas → tasks 0..3
# Previous: --array=0-8 (ADS+PoE combined, with a broken /3 %3 mapping).

source $HOME/nlphw/my_project.env
mkdir -p logs

# NOTE: The hardcoded HF_TOKEN that used to live here has been removed — it
# was a committed secret. Put `export HF_TOKEN=hf_...` inside your private
# $HOME/nlphw/my_project.env instead, and rotate the old token in the
# Hugging Face settings since it's already in the git history.
export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

# ── Strategic-ADS sweep ────────────────────────────────────────────────────
# Task i picks ADS_LAMS[i]. PoE is disabled in this sbatch; see the commented
# block at the bottom if you want to re-enable it.
ADS_LAMS=(0.01 0.03 0.05 0.1)
N_ADS=${#ADS_LAMS[@]}

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_ADS}" ]; then
    echo "ERROR: Array task ${SLURM_ARRAY_TASK_ID} but sweep has only ${N_ADS} points." >&2
    echo "       Shrink --array to 0-$((N_ADS - 1)) or extend ADS_LAMS." >&2
    exit 1
fi

VAL=${ADS_LAMS[${SLURM_ARRAY_TASK_ID}]}
RUN_NAME="gsm8k_strategic_ads_lam_${VAL}"
BASE_CFG="configs/gsm8k_strategic_teacher.yaml"
TMP_CFG="/tmp/strategic_sweep_ads_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

echo "=== Task ${SLURM_ARRAY_TASK_ID} | strategic ADS | lam=${VAL} ==="

python3 -c "
import yaml
cfg = yaml.safe_load(open('${BASE_CFG}'))
cfg['run']['run_name'] = '${RUN_NAME}'
cfg['teachers']['strategic_antidistillation_lams'] = [${VAL}]
cfg['teachers']['poe_gammas'] = []
cfg['teachers']['strategic_poe_gammas'] = []
yaml.safe_dump(cfg, open('${TMP_CFG}', 'w'), sort_keys=False)
"

python scripts/run_pipeline.py --config "${TMP_CFG}"

rm -f "${TMP_CFG}"

# ───────────────────────────────────────────────────────────────────────────
# DISABLED: original combined ADS+PoE dispatch. Kept for reference; to
# re-enable, restore the original #SBATCH --array=0-N with N = len(ADS_LAMS)
# + len(POE_GAMMAS) - 1 and replace the single-method block above with this
# two-branch dispatch (and fix the index math to split by N_ADS / N_POE
# instead of the broken /3 %3 pattern):
#
# METHODS=(ads poe)
# ADS_LAMS=(0.01 0.03 0.05 0.1)
# POE_GAMMAS=(0.3 0.4 0.5 0.6)
#
# METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 3))   # <-- BUG: only correct if each
# VALUE_IDX=$((SLURM_ARRAY_TASK_ID % 3))    #          list has length 3.
# METHOD=${METHODS[$METHOD_IDX]}
#
# if [ "$METHOD" = "ads" ]; then
#     VAL=${ADS_LAMS[$VALUE_IDX]}
#     RUN_NAME="gsm8k_strategic_ads_lam_${VAL}"
#     echo "=== Task ${SLURM_ARRAY_TASK_ID} | strategic ADS | lam=${VAL} ==="
#     python3 -c "
# import yaml
# cfg = yaml.safe_load(open('${BASE_CFG}'))
# cfg['run']['run_name'] = '${RUN_NAME}'
# cfg['teachers']['strategic_antidistillation_lams'] = [${VAL}]
# cfg['teachers']['poe_gammas'] = []
# cfg['teachers']['strategic_poe_gammas'] = []
# yaml.safe_dump(cfg, open('${TMP_CFG}', 'w'), sort_keys=False)
# "
# else
#     VAL=${POE_GAMMAS[$VALUE_IDX]}
#     RUN_NAME="gsm8k_strategic_poe_gamma_${VAL}"
#     echo "=== Task ${SLURM_ARRAY_TASK_ID} | strategic PoE | gamma=${VAL} ==="
#     python3 -c "
# import yaml
# cfg = yaml.safe_load(open('${BASE_CFG}'))
# cfg['run']['run_name'] = '${RUN_NAME}'
# cfg['teachers']['antidistillation_lams'] = []
# cfg['teachers']['strategic_antidistillation_lams'] = []
# cfg['teachers']['poe_gammas'] = [0.6]
# cfg['teachers']['strategic_poe_gammas'] = [${VAL}]
# yaml.safe_dump(cfg, open('${TMP_CFG}', 'w'), sort_keys=False)
# "
# fi
