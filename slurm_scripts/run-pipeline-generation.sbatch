#!/bin/bash
#SBATCH --job-name=pipeline-gen
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/pipeline_generation_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00

# ─────────────────────────────────────────────────────────────────────────
# Strategic distillation — GENERATION stage (Stages 1 & 2).
#
# Produces teacher standard traces + proxy gradients once per
# (seed, dataset, teacher, proxy_student, generation.*) tuple.
# The output directory is the shared cache consumed by:
#   slurm_scripts/run-pipeline-distillation.sbatch   (teachers, Stages 3 & 4)
#   slurm_scripts/run-pipeline-student.sbatch        (student,  Stages 5 & 6)
#
# Submit:
#   sbatch slurm_scripts/run-pipeline-generation.sbatch
#
# Override config / output location at submit time, e.g.:
#   sbatch --export=ALL,CFG=configs/gsm8k_small.yaml,RUN_NAME=gsm8k_small_gen \
#          slurm_scripts/run-pipeline-generation.sbatch
#
# Chain the full generation → teachers → student pipeline:
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

# Overridable via --export=ALL,VAR=value at submit time.
CFG="${CFG:-configs/gsm8k_strategic_teacher.yaml}"
RUN_NAME="${RUN_NAME:-gsm8k_strategic_generation}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

echo "=== Pipeline generation ==="
echo "  Config:     ${CFG}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Host:       $(hostname)"
echo "  Job:        ${SLURM_JOB_ID}"
echo

mkdir -p "$(dirname "${OUTPUT_DIR}")"

python scripts/run_pipeline_generation.py \
    --config "${CFG}" \
    --output-dir "${OUTPUT_DIR}"

echo
echo "=== Generation complete ==="
echo "Output dir: ${OUTPUT_DIR}"
echo
echo "Next — run teachers (Stages 3 & 4):"
echo "  TEA_JID=\$(sbatch --parsable \\"
echo "    --export=ALL,INPUT_DIR=${OUTPUT_DIR} \\"
echo "    slurm_scripts/run-pipeline-distillation.sbatch)"
echo
echo "Then — run student (Stages 5 & 6), chained on the teachers job:"
echo "  sbatch --dependency=afterok:\${TEA_JID} \\"
echo "    --export=ALL,INPUT_DIR=${OUTPUT_DIR} \\"
echo "    slurm_scripts/run-pipeline-student.sbatch"
