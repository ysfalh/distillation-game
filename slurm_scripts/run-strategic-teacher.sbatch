#!/bin/bash
#SBATCH --job-name=sweep-strategic
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/strategic_sweep_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-8

source $HOME/nlphw/my_project.env
mkdir -p logs

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"


METHODS=(ads poe)
ADS_LAMS=(0.01 0.03 0.05 0.1)
POE_GAMMAS=(0.3 0.4 0.5 0.6)

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 3))
VALUE_IDX=$((SLURM_ARRAY_TASK_ID % 3))
METHOD=${METHODS[$METHOD_IDX]}

BASE_CFG="configs/gsm8k_strategic_teacher.yaml"
TMP_CFG="/tmp/strategic_sweep_${SLURM_ARRAY_TASK_ID}.yaml"

if [ "$METHOD" = "ads" ]; then
    VAL=${ADS_LAMS[$VALUE_IDX]}
    RUN_NAME="gsm8k_strategic_ads_lam_${VAL}"
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
else
    VAL=${POE_GAMMAS[$VALUE_IDX]}
    RUN_NAME="gsm8k_strategic_poe_gamma_${VAL}"
    echo "=== Task ${SLURM_ARRAY_TASK_ID} | strategic PoE | gamma=${VAL} ==="
    python3 -c "
import yaml
cfg = yaml.safe_load(open('${BASE_CFG}'))
cfg['run']['run_name'] = '${RUN_NAME}'
cfg['teachers']['antidistillation_lams'] = []
cfg['teachers']['strategic_antidistillation_lams'] = []
cfg['teachers']['poe_gammas'] = [0.6]
cfg['teachers']['strategic_poe_gammas'] = [${VAL}]
yaml.safe_dump(cfg, open('${TMP_CFG}', 'w'), sort_keys=False)
"
fi

python scripts/run_pipeline.py --config "${TMP_CFG}"
