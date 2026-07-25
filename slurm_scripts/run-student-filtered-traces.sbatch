#!/bin/bash
#SBATCH --job-name=student-filtered
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/student_filtered_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-8%8

# Naive-SFT students on degeneracy-filtered GSM8K traces.
# One array task = one (filter level, seed, teacher source).
#
# The two-rule filter has no calibrated thresholds, so there is a single level
# instead of the old accept_* series. The unfiltered baseline is not part of
# this array; add `unfiltered` to LEVELS and widen --array to rerun it.
#
# The filtered trace directories are gitignored, so build them first:
#   python filtering_traces/run_filter_sweep.py --input-dir gsm8k_output_small --write-filtered
#
# Then collect the accuracies into markdown:
#   python scripts/aggregate_filtered_results.py

source $HOME/nlphw/my_project.env
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-configs/gsm8k.yaml}"
STUDENT="meta-llama/Llama-3.2-3B"
STUDENT_TOKENIZER="meta-llama/Llama-3.2-3B-Instruct"

LEVELS=(simple)
SEEDS=($(python -c "import yaml; print(*yaml.safe_load(open('${CONFIG}'))['run']['seeds'])"))
SOURCES=(standard ads poe)

if [ ${#SEEDS[@]} -eq 0 ]; then
    echo "ERROR: could not read run.seeds from ${CONFIG}" >&2
    exit 1
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: no SLURM_ARRAY_TASK_ID, submit this with sbatch." >&2
    exit 1
fi

# --array above is static, so say which range the config now implies.
TOTAL=$(( ${#LEVELS[@]} * ${#SEEDS[@]} * ${#SOURCES[@]} ))
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} of ${TOTAL} combinations." >&2
    echo "       Resubmit with --array=0-$(( TOTAL - 1 ))" >&2
    exit 1
fi

LEVEL=${LEVELS[$((SLURM_ARRAY_TASK_ID / (${#SEEDS[@]} * ${#SOURCES[@]})))]}
SEED=${SEEDS[$(((SLURM_ARRAY_TASK_ID / ${#SOURCES[@]}) % ${#SEEDS[@]}))]}
SOURCE=${SOURCES[$((SLURM_ARRAY_TASK_ID % ${#SOURCES[@]}))]}

if [ "$LEVEL" = "unfiltered" ]; then
    INPUT_DIR="gsm8k_output_small"
else
    INPUT_DIR="filtering_traces/filtered/gsm8k_output_small/${LEVEL}"
fi
OUTPUT_DIR="outputs/filtered_traces/${LEVEL}/seed_${SEED}/${SOURCE}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: ${INPUT_DIR} does not exist. Build it with:" >&2
    echo "       python filtering_traces/run_filter_sweep.py --input-dir gsm8k_output_small --write-filtered" >&2
    exit 1
fi

echo "=== Array task $SLURM_ARRAY_TASK_ID | Level: $LEVEL | Seed: $SEED | Source: $SOURCE ==="
python scripts/run_student_from_saved_traces.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --seed "$SEED" \
    --sources "$SOURCE" \
    --modes naive \
    --student "$STUDENT" \
    --student-tokenizer "$STUDENT_TOKENIZER" \
    --attn-implementation sdpa
