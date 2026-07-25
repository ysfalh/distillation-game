#!/bin/bash
#SBATCH --job-name=student-accept100
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/student_accept100_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=48g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-8

# Naive-SFT students on the accept_1.00 degeneracy-filtered GSM8K traces.
# One array task = one (seed, teacher source), all nine launched at once.
#
# Same run as run-student-filtered-traces.sbatch, which covers accept_0.95,
# accept_0.97 and accept_0.99 on the 96 GB GPUs, for the one level it skips.
#
# The filtered trace directory is gitignored, so build it first:
#   python filtering_traces/run_filter_sweep.py --input-dir gsm8k_output_small --write-filtered --filtered-points 1.0
#
# Then collect the accuracies into markdown:
#   python scripts/aggregate_filtered_results.py

source /share/project/haghifam-projects/venv-low-memory/bin/activate
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-configs/gsm8k.yaml}"
STUDENT="meta-llama/Llama-3.2-3B"
STUDENT_TOKENIZER="meta-llama/Llama-3.2-3B-Instruct"

LEVEL=accept_1.00
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
TOTAL=$(( ${#SEEDS[@]} * ${#SOURCES[@]} ))
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} of ${TOTAL} combinations." >&2
    echo "       Resubmit with --array=0-$(( TOTAL - 1 ))" >&2
    exit 1
fi

SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / ${#SOURCES[@]}))]}
SOURCE=${SOURCES[$((SLURM_ARRAY_TASK_ID % ${#SOURCES[@]}))]}

INPUT_DIR="filtering_traces/filtered/gsm8k_output_small/${LEVEL}"
OUTPUT_DIR="outputs/filtered_traces/${LEVEL}/seed_${SEED}/${SOURCE}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: ${INPUT_DIR} does not exist. Build it with:" >&2
    echo "       python filtering_traces/run_filter_sweep.py --input-dir gsm8k_output_small --write-filtered --filtered-points 1.0" >&2
    exit 1
fi

# The config's eval batch of 240 is sized for the 96 GB GPUs. Patch a copy for
# 48 GB rather than the file itself, which the other array is still reading.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-200}"
TASK_CONFIG="/tmp/student_accept100_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
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
