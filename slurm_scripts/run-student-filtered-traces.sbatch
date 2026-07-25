#!/bin/bash
#SBATCH --job-name=student-filtered
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/student_filtered_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-35%4

# Naive-SFT students on degeneracy-filtered GSM8K traces.
# One array task = one (filter level, seed, teacher source).
#
# The filtered trace directories are gitignored, so build them first:
#   python filtering_traces/run_filter_sweep.py --input-dir gsm8k_output_small --write-filtered
#
# Then collect the accuracies into markdown:
#   python scripts/aggregate_filtered_results.py

source $HOME/nlphw/my_project.env
mkdir -p logs

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

CONFIG="configs/gsm8k.yaml"
STUDENT="meta-llama/Llama-3.2-3B"
STUDENT_TOKENIZER="meta-llama/Llama-3.2-3B-Instruct"

LEVELS=(unfiltered accept_0.95 accept_0.99 accept_1.00)
SEEDS=(123 456 789)  # matches run.seeds in configs/gsm8k.yaml
SOURCES=(standard ads poe)

LEVEL=${LEVELS[$((SLURM_ARRAY_TASK_ID / 9))]}
SEED=${SEEDS[$(((SLURM_ARRAY_TASK_ID / 3) % 3))]}
SOURCE=${SOURCES[$((SLURM_ARRAY_TASK_ID % 3))]}

if [ "$LEVEL" = "unfiltered" ]; then
    INPUT_DIR="gsm8k_output_small"
else
    INPUT_DIR="filtering_traces/filtered/gsm8k_output_small/${LEVEL}"
fi
OUTPUT_DIR="outputs/filtered_traces/${LEVEL}/seed_${SEED}/${SOURCE}"

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
