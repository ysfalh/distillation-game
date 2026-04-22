#!/bin/bash
#SBATCH --job-name=pipeline-student-single
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/pipeline_student_single_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=48g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00

# ─────────────────────────────────────────────────────────────────────────
# Strategic distillation — STUDENT stage (Stages 5 & 6) for ONE teacher dir.
#
# Runs scripts/run_pipeline_student.py exactly once against a single
# distill_* directory produced by run-pipeline-teachers.sbatch. Useful when
# only a subset of the teacher sweep finished and you want to distill the
# completed ones individually, or when iterating on a single ADS / PoE
# configuration.
#
# Inputs (pick one of these two forms):
#   (A) TEA_DIR  = full path to one distill_* dir, e.g.:
#         outputs/.../distill_strategic_ads_lam_0.1
#
#   (B) INPUT_DIR + TASK_NAME, where TASK_NAME is the suffix after "distill_":
#         INPUT_DIR=outputs/.../gen_dir
#         TASK_NAME=strategic_ads_lam_0.1
#       TEA_DIR is then composed as ${INPUT_DIR}/distill_${TASK_NAME}.
#
# Submit examples:
#   # Form A: direct path
#   sbatch --export=ALL,TEA_DIR=outputs/gsm8k_strategic_generation_20260420_182839/distill_strategic_ads_lam_0.1 \
#          slurm_scripts/run-pipeline-student-single.sbatch
#
#   # Form B: generation dir + task name
#   sbatch --export=ALL,INPUT_DIR=outputs/gsm8k_strategic_generation_20260420_182839,TASK_NAME=strategic_poe_gamma_0.7 \
#          slurm_scripts/run-pipeline-student-single.sbatch
#
# Output (written IN-PLACE inside TEA_DIR):
#   ${TEA_DIR}/student/       trained checkpoints (if save_final_model)
#   ${TEA_DIR}/inspection/    inspection samples
#   ${TEA_DIR}/results.json   teacher + student rows
#   ${TEA_DIR}/RESULTS.md     markdown report for this teacher
#   ${TEA_DIR}/phase_times.json   (updated with stage 5 timings)
# ─────────────────────────────────────────────────────────────────────────

source $HOME/nlphw/my_project.env
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export HF_TOKEN="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"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"
# Reduce allocator fragmentation; helps student eval fit on 48 GB GPUs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Resolve TEA_DIR ──────────────────────────────────────────────────────
# Accept either an explicit TEA_DIR, or compose it from INPUT_DIR + TASK_NAME.
if [ -z "${TEA_DIR:-}" ]; then
    if [ -n "${INPUT_DIR:-}" ] && [ -n "${TASK_NAME:-}" ]; then
        # Strip trailing slash on INPUT_DIR and optional leading "distill_" on TASK_NAME.
        TEA_DIR="${INPUT_DIR%/}/distill_${TASK_NAME#distill_}"
    else
        echo "ERROR: Must set TEA_DIR=<path/to/distill_*>, or both INPUT_DIR + TASK_NAME." >&2
        echo "       See the header comment in this sbatch for examples." >&2
        exit 1
    fi
fi

if [ ! -d "${TEA_DIR}" ]; then
    echo "ERROR: TEA_DIR does not exist: ${TEA_DIR}" >&2
    exit 1
fi
if [ ! -f "${TEA_DIR}/TEACHERS_DONE" ]; then
    echo "WARN: TEACHERS_DONE marker not found in ${TEA_DIR}." >&2
    echo "      Proceeding — the student script will re-check required cache files." >&2
fi

RESOLVED_NAME="$(basename "${TEA_DIR}" | sed 's/^distill_//')"

echo "=== Single student run | ${RESOLVED_NAME} ==="
echo
echo "  Teacher dir:       ${TEA_DIR}"
echo "  Output (in-place): ${TEA_DIR}"
echo "  Host:              $(hostname)"
echo "  Job:               ${SLURM_JOB_ID}"
echo

python scripts/run_pipeline_student.py \
    --input-dir  "${TEA_DIR}" \
    --output-dir "${TEA_DIR}"

echo
echo "=== Student run complete ==="
echo "Results:"
echo "  ${TEA_DIR}/results.json"
echo "  ${TEA_DIR}/RESULTS.md"
