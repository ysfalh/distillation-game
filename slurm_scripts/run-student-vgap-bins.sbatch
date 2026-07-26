#!/bin/bash
#SBATCH --job-name=student-vgap-bins
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/student_vgap_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-29

# One fresh Llama attacker per v_gap bin. An array task is one (arm, bin, seed);
# each trains on that bin's traces only.
#
# Do not submit this directly. Submit the prep job, which scores the traces,
# writes the bins, releases this array, and queues the report:
#
#   sbatch slurm_scripts/vgap-score-prep.sbatch

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
SOURCE="${SOURCE:-standard}"
STUDENT="meta-llama/Llama-3.2-3B"
STUDENT_TOKENIZER="meta-llama/Llama-3.2-3B-Instruct"

# `arm|bin`, and must match ARMS in vgap-score-prep.sbatch. `sum` is V_gap as
# the defense computes it, with nothing held fixed, so its bins also differ in
# trace length and token count. `mean_matched` holds length and teacher
# correctness fixed, so only v_gap varies. Comparing the two is what tells a
# v_gap effect apart from a length effect.
COMBOS=(
    "sum|0" "sum|1" "sum|2" "sum|3" "sum|4"
    "mean_matched|0" "mean_matched|1" "mean_matched|2" "mean_matched|3" "mean_matched|4"
)
SEEDS=($(python -c "import yaml; print(*yaml.safe_load(open('${CONFIG}'))['run']['seeds'])"))

if [ ${#SEEDS[@]} -eq 0 ]; then
    echo "ERROR: could not read run.seeds from ${CONFIG}" >&2
    exit 1
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: no SLURM_ARRAY_TASK_ID, submit this with sbatch." >&2
    exit 1
fi

# --array above is static, so say which range the combos and config now imply.
TOTAL=$(( ${#COMBOS[@]} * ${#SEEDS[@]} ))
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} of ${TOTAL} combinations." >&2
    echo "       Resubmit with --array=0-$(( TOTAL - 1 ))" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_COUNT:-${TOTAL}}" -lt "${TOTAL}" ]; then
    echo "WARNING: this array has ${SLURM_ARRAY_TASK_COUNT} tasks but ${TOTAL} " \
         "combinations exist, so some will never run. Widen --array to 0-$(( TOTAL - 1 ))." >&2
fi

COMBO=${COMBOS[$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))]}
ARM=${COMBO%%|*}
BIN=${COMBO##*|}

INPUT_DIR="vgap/${DATASET}/bins/${ARM}/bin_${BIN}"
OUTPUT_DIR="outputs/vgap_bins/${ARM}/bin_${BIN}/seed_${SEED}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: ${INPUT_DIR} does not exist." >&2
    echo "       Build the bins, with ARMS covering '${ARM}':" >&2
    echo "       sbatch slurm_scripts/vgap-score-prep.sbatch" >&2
    exit 1
fi

# Keep the config's eval_batch_size (240) on the 96 GB cards. Override with
# EVAL_BATCH_SIZE only when you deliberately want a smaller batch.
TASK_CONFIG="/tmp/student_vgap_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
if [ -n "${EVAL_BATCH_SIZE:-}" ]; then
    python -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
cfg['distill']['eval_batch_size'] = ${EVAL_BATCH_SIZE}
yaml.safe_dump(cfg, open('${TASK_CONFIG}', 'w'), sort_keys=False)
"
    RUN_CONFIG="${TASK_CONFIG}"
else
    RUN_CONFIG="${CONFIG}"
fi

echo "=== Array task $SLURM_ARRAY_TASK_ID | Arm: $ARM | Bin: $BIN | Seed: $SEED ==="
python scripts/run_student_from_saved_traces.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config "$RUN_CONFIG" \
    --seed "$SEED" \
    --sources "$SOURCE" \
    --modes naive \
    --student "$STUDENT" \
    --student-tokenizer "$STUDENT_TOKENIZER"

rm -f "$TASK_CONFIG"
