#!/bin/bash
#SBATCH --job-name=pipeline-student
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/pipeline_student_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=48g
#SBATCH --mem=64gb
#SBATCH --array=0-7

# ─────────────────────────────────────────────────────────────────────────
# Strategic distillation — STUDENT stage (Stages 5 & 6) as an array sweep.
#
# Runs scripts/run_pipeline_student.py once per teacher-output directory
# produced by run-pipeline-distillation.sbatch. Each array task:
#   - picks one   ${INPUT_DIR}/distill_*   directory (sorted, 1:1 with the
#     teacher array),
#   - runs student distillation + evaluation + final results IN-PLACE inside
#     that directory (student/, inspection/, results.json, RESULTS.md,
#     updated phase_times.json land next to the teacher/, cache/ produced by
#     the teacher stage).
#
# The array size MUST match the number of distill_* directories (i.e. the
# cardinality of the teacher array). If you change run-pipeline-distillation's
# --array=0-N, update this file's --array to match.
#
# Submit (INPUT_DIR is required; it is the generation output dir that the
# teacher array also used as INPUT_DIR):
#   sbatch --export=ALL,INPUT_DIR=outputs/gsm8k_strategic_generation_YYYYMMDD_HHMMSS \
#          slurm_scripts/run-pipeline-student.sbatch
#
# Chaining after generation + teachers:
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

source /share/project/haghifam-projects/venv-low-memory/bin/activate
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"
# Reduce allocator fragmentation; helps student eval fit on 48 GB GPUs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${INPUT_DIR:?INPUT_DIR must be set (the generation output dir that contains distill_* subdirs)}"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: INPUT_DIR does not exist: ${INPUT_DIR}" >&2
    exit 1
fi

# ── Map SLURM_ARRAY_TASK_ID → one teacher-out dir ──────────────────────
# Uses bash sorted globbing; order is deterministic across tasks, so each
# array index picks a unique distill_* dir. Size is validated below.
shopt -s nullglob
TEACHER_DIRS=("${INPUT_DIR}"/distill_*)
shopt -u nullglob

N_TEA=${#TEACHER_DIRS[@]}
if [ "${N_TEA}" -eq 0 ]; then
    echo "ERROR: No distill_* subdirectories found under ${INPUT_DIR}." >&2
    echo "       Has the teacher array job completed?" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TEA}" ]; then
    echo "ERROR: Array task ${SLURM_ARRAY_TASK_ID} but only ${N_TEA} distill_* dirs exist." >&2
    echo "       Resize --array in this sbatch to match the teacher sweep." >&2
    exit 1
fi

TEA_DIR="${TEACHER_DIRS[$SLURM_ARRAY_TASK_ID]}"
TASK_NAME="$(basename "${TEA_DIR}" | sed 's/^distill_//')"

if [ ! -f "${TEA_DIR}/TEACHERS_DONE" ]; then
    echo "WARN: TEACHERS_DONE marker not found in ${TEA_DIR}." >&2
    echo "      Proceeding — the student script will re-check required cache files." >&2
fi

echo "=== Task ${SLURM_ARRAY_TASK_ID} | student on ${TASK_NAME} ==="
echo
echo "  Input (teacher):  ${TEA_DIR}"
echo "  Output (in-place): ${TEA_DIR}"
echo "  Host:              $(hostname)"
echo "  Array job.task:    ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
echo

# Per-task config override: start from the snapshot the teacher stage mirrored
# into ${TEA_DIR}, then flip student-side knobs for this sweep. Keeping the
# override here (not in the teacher sbatch) scopes it to the student stage
# only; the teacher stage's symlinked snapshot stays untouched for reference.
BASE_CFG="${TEA_DIR}/config_snapshot.yaml"
TMP_CFG="/tmp/pipeline_student_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
python3 -c "
import yaml
cfg = yaml.safe_load(open('${BASE_CFG}'))
cfg.setdefault('distill', {})['exclude_standard_teacher'] = True
yaml.safe_dump(cfg, open('${TMP_CFG}', 'w'), sort_keys=False)
"

# In-place: student artifacts (student/, inspection/, results.json,
# RESULTS.md, phase_times.json) land next to the teacher/ + cache/ produced
# by the teacher stage.
python scripts/run_pipeline_student.py \
    --input-dir  "${TEA_DIR}" \
    --config     "${TMP_CFG}" \
    --output-dir "${TEA_DIR}"

rm -f "${TMP_CFG}"

echo
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
echo "Results:"
echo "  ${TEA_DIR}/results.json"
echo "  ${TEA_DIR}/RESULTS.md"
