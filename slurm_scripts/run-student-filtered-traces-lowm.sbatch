#!/bin/bash
#SBATCH --job-name=student-filtered-lowm
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/student_filtered_lowm_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=48g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-26

# Naive-SFT students on the two-rule degeneracy-filtered GSM8K traces, for both
# the matched (gsm8k_output_small) and mismatched-proxy
# (gsm8k_output_small_mismatch) teachers. One array task = one (dataset,
# teacher source, seed).
#
# Do not submit this directly. Submit the prep job, which builds the filtered
# traces, releases this array, and queues the report that follows it:
#
#   sbatch slurm_scripts/filter-traces-prep.sbatch
#
# The report lands in outputs/filtered_traces/RESULTS.md.

source /share/project/haghifam-projects/venv-low-memory/bin/activate
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-configs/gsm8k.yaml}"
STUDENT="meta-llama/Llama-3.2-3B"
STUDENT_TOKENIZER="meta-llama/Llama-3.2-3B-Instruct"

# `dataset|teacher source`. The mismatched folder has no standard teacher, and
# poe_gamma_0.7 exists in both, which is why results are keyed by dataset too.
COMBOS=(
    "gsm8k_output_small|standard"
    "gsm8k_output_small|antidistillation_lam_0.05"
    "gsm8k_output_small|antidistillation_lam_0.055"
    "gsm8k_output_small|poe_gamma_0.6"
    "gsm8k_output_small|poe_gamma_0.7"
    "gsm8k_output_small_mismatch|antidistillation_lam_0.03"
    "gsm8k_output_small_mismatch|antidistillation_lam_0.035"
    "gsm8k_output_small_mismatch|poe_gamma_0.65"
    "gsm8k_output_small_mismatch|poe_gamma_0.7"
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
DATASET=${COMBO%%|*}
SOURCE=${COMBO##*|}

if [ "$DATASET" = "gsm8k_output_small_mismatch" ]; then
    LEVEL="simple_mismatch"
else
    LEVEL="simple"
fi
INPUT_DIR="filtering_traces/filtered/${DATASET}/simple"
OUTPUT_DIR="outputs/filtered_traces/${LEVEL}/seed_${SEED}/${SOURCE}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: ${INPUT_DIR} does not exist. Build it with:" >&2
    echo "       python filtering_traces/run_filter_sweep.py --input-dir ${DATASET} --write-filtered" >&2
    exit 1
fi

# The config's eval batch of 240 is sized for the 96 GB GPUs. Patch a copy for
# 48 GB rather than the file itself, which other jobs may be reading.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-200}"
TASK_CONFIG="/tmp/student_filtered_lowm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
python -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
cfg['distill']['eval_batch_size'] = ${EVAL_BATCH_SIZE}
yaml.safe_dump(cfg, open('${TASK_CONFIG}', 'w'), sort_keys=False)
"

echo "=== Array task $SLURM_ARRAY_TASK_ID | Level: $LEVEL | Seed: $SEED | Source: $SOURCE ==="
python scripts/run_student_from_saved_traces.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config "$TASK_CONFIG" \
    --seed "$SEED" \
    --sources "$SOURCE" \
    --modes naive \
    --student "$STUDENT" \
    --student-tokenizer "$STUDENT_TOKENIZER"

rm -f "$TASK_CONFIG"
