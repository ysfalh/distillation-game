#!/bin/bash
#SBATCH --job-name=pipeline-teachers
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/pipeline_teachers_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-7

# ─────────────────────────────────────────────────────────────────────────
# Strategic distillation — TEACHERS stage (Stages 3 & 4) as an array sweep.
#
# Runs scripts/run_pipeline_teachers.py. Stage 5 (student distill + eval)
# and Stage 6 (results.json / RESULTS.md) are now handled by
# slurm_scripts/run-pipeline-student.sbatch, which reads the per-task
# teacher outputs this job produces.
#
# Sweep layout:
#   tasks 0..3 → strategic ADS with lam   ∈ {0.03, 0.05, 0.1, 0.25}
#                (each task also trains the non-strategic ADS lam=ADS_BASELINE_LAM baseline)
#   tasks 4..7 → strategic PoE with gamma ∈ {0.6, 0.7, 0.8, 0.9}
#                (each task also trains the non-strategic PoE gamma=POE_BASELINE_GAMMA baseline)
#
# All tasks SHARE one generation cache (teacher standard traces + proxy
# gradients), produced by run-pipeline-generation.sbatch, and write their
# teacher outputs into PER-TASK subdirectories to avoid collisions.
#
# --array size must equal len(ADS_LAMS) + len(POE_GAMMAS); the matching
# student array in run-pipeline-student.sbatch must have the same size.
#
# Submit (INPUT_DIR is required):
#   sbatch --export=ALL,INPUT_DIR=outputs/gsm8k_strategic_generation_YYYYMMDD_HHMMSS \
#          slurm_scripts/run-pipeline-distillation.sbatch
#
# Chaining after a generation job, followed by the student array:
#   GEN_OUTDIR=outputs/gsm8k_strategic_generation_$(date +%Y%m%d_%H%M%S)
#   GEN_JID=$(sbatch --parsable \
#     --export=ALL,OUTPUT_DIR=${GEN_OUTDIR} \
#     slurm_scripts/run-pipeline-generation.sbatch)
#   TEA_JID=$(sbatch --parsable --dependency=afterok:${GEN_JID} \
#     --export=ALL,INPUT_DIR=${GEN_OUTDIR} \
#     slurm_scripts/run-pipeline-distillation.sbatch)
#   sbatch --dependency=afterok:${TEA_JID} \
#     --export=ALL,INPUT_DIR=${GEN_OUTDIR} \
#     slurm_scripts/run-pipeline-student.sbatch
# ─────────────────────────────────────────────────────────────────────────

source $HOME/nlphw/my_project.env
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

: "${INPUT_DIR:?INPUT_DIR must be set (output directory of the generation job)}"

BASE_CFG="${INPUT_DIR}/config_snapshot.yaml"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: INPUT_DIR does not exist: ${INPUT_DIR}" >&2
    exit 1
fi
if [ ! -f "${BASE_CFG}" ]; then
    echo "ERROR: Missing config_snapshot.yaml in ${INPUT_DIR}" >&2
    echo "       Has the generation job completed?" >&2
    exit 1
fi
if [ ! -f "${INPUT_DIR}/GENERATION_DONE" ]; then
    echo "WARN: GENERATION_DONE marker not found in ${INPUT_DIR}." >&2
    echo "      Proceeding — the distillation script will re-check required cache files." >&2
fi

# ── Sweep definition (keep in lockstep with run-strategic-teacher.sbatch) ──
ADS_LAMS=(0.03 0.05 0.1 0.25)
POE_GAMMAS=(0.6 0.7 0.8 0.9)

# Non-strategic baselines trained alongside the strategic variant in every
# task of the corresponding family. Pick the lambda/gamma you want to compare
# the strategic sweep points against — these should stay fixed across tasks
# so the baseline column in RESULTS.md is a clean control.
ADS_BASELINE_LAM=0.08
POE_BASELINE_GAMMA=0.75

# Map: first len(ADS_LAMS) tasks → ADS, next len(POE_GAMMAS) tasks → PoE.
N_ADS=${#ADS_LAMS[@]}
N_POE=${#POE_GAMMAS[@]}
N_TOTAL=$((N_ADS + N_POE))

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "ERROR: Array task ${SLURM_ARRAY_TASK_ID} but sweep has only ${N_TOTAL} points." >&2
    echo "       Shrink --array to 0-$((N_TOTAL - 1)) or extend ADS_LAMS/POE_GAMMAS." >&2
    exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -lt "${N_ADS}" ]; then
    METHOD="ads"
    VALUE_IDX=${SLURM_ARRAY_TASK_ID}
else
    METHOD="poe"
    VALUE_IDX=$((SLURM_ARRAY_TASK_ID - N_ADS))
fi

TMP_CFG="/tmp/pipeline_dist_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

if [ "$METHOD" = "ads" ]; then
    VAL=${ADS_LAMS[$VALUE_IDX]}
    TASK_NAME="strategic_ads_lam_${VAL}"
    echo "=== Task ${SLURM_ARRAY_TASK_ID} | strategic ADS | lam=${VAL} "\
"(baseline lam=${ADS_BASELINE_LAM}) ==="
    python3 -c "
import yaml
cfg = yaml.safe_load(open('${BASE_CFG}'))
cfg['run']['run_name'] = '${TASK_NAME}'
cfg['teachers']['antidistillation_lams'] = [${ADS_BASELINE_LAM}]
cfg['teachers']['strategic_antidistillation_lams'] = [${VAL}]
cfg['teachers']['poe_gammas'] = []
cfg['teachers']['strategic_poe_gammas'] = []
yaml.safe_dump(cfg, open('${TMP_CFG}', 'w'), sort_keys=False)
"
else
    VAL=${POE_GAMMAS[$VALUE_IDX]}
    TASK_NAME="strategic_poe_gamma_${VAL}"
    echo "=== Task ${SLURM_ARRAY_TASK_ID} | strategic PoE | gamma=${VAL} "\
"(baseline gamma=${POE_BASELINE_GAMMA}) ==="
    python3 -c "
import yaml
cfg = yaml.safe_load(open('${BASE_CFG}'))
cfg['run']['run_name'] = '${TASK_NAME}'
cfg['teachers']['antidistillation_lams'] = []
cfg['teachers']['strategic_antidistillation_lams'] = []
cfg['teachers']['poe_gammas'] = [${POE_BASELINE_GAMMA}]
cfg['teachers']['strategic_poe_gammas'] = [${VAL}]
yaml.safe_dump(cfg, open('${TMP_CFG}', 'w'), sort_keys=False)
"
fi

OUT_DIR="${INPUT_DIR}/distill_${TASK_NAME}"
mkdir -p "${OUT_DIR}"

echo
echo "  Input (cache):  ${INPUT_DIR}"
echo "  Config:         ${TMP_CFG}"
echo "  Output:         ${OUT_DIR}"
echo "  Host:           $(hostname)"
echo "  Array job.task: ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
echo

python scripts/run_pipeline_teachers.py \
    --input-dir  "${INPUT_DIR}" \
    --config     "${TMP_CFG}" \
    --output-dir "${OUT_DIR}"

rm -f "${TMP_CFG}"

echo
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
echo "Teacher outputs:"
echo "  ${OUT_DIR}/teacher/"
echo "  ${OUT_DIR}/cache/teacher_train_sources.json"
echo "  ${OUT_DIR}/TEACHERS_DONE"
echo
echo "Next step (student stage, in-place inside each distill_* dir):"
echo "  sbatch --export=ALL,INPUT_DIR=${INPUT_DIR} \\"
echo "         slurm_scripts/run-pipeline-student.sbatch"
