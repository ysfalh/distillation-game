#!/bin/bash
#SBATCH --job-name=base-eval-gsm8k
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/base_eval_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-3

# Untrained base models on the GSM8K test split, one model per GPU.
# No training happens here, so this is the floor the distilled students are
# measured against. Single seed, since there is nothing being fit.
#
#   sbatch slurm_scripts/run-base-eval-gsm8k.sbatch
#
# Each task writes outputs/base_eval/seed_456/<model>/results.json. To see all
# of them at once afterwards:
#
#   grep -h '"accuracy"' outputs/base_eval/seed_456/*/results.json

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
SEED="${SEED:-456}"
# Matches distill.eval_batch_size / generation.batch_size on these 96 GB cards.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-240}"

# `base model|tokenizer with the chat template`. Base checkpoints have no
# template of their own, so each one borrows its Instruct sibling's, exactly
# as the student runs do.
MODELS=(
    "Qwen/Qwen2.5-3B|Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-1.5B|Qwen/Qwen2.5-1.5B-Instruct"
    "meta-llama/Llama-3.2-3B|meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B|meta-llama/Meta-Llama-3.1-8B-Instruct"
)

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: no SLURM_ARRAY_TASK_ID, submit this with sbatch." >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#MODELS[@]}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} of ${#MODELS[@]} models." >&2
    echo "       Resubmit with --array=0-$(( ${#MODELS[@]} - 1 ))" >&2
    exit 1
fi

ENTRY=${MODELS[${SLURM_ARRAY_TASK_ID}]}
IFS='|' read -r MODEL TOKENIZER <<< "$ENTRY"

echo "=== Array task ${SLURM_ARRAY_TASK_ID} | Model: ${MODEL} | Seed: ${SEED} | Batch: ${EVAL_BATCH_SIZE} ==="
python scripts/eval_base_model_gsm8k.py \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --config "$CONFIG" \
    --seed "$SEED" \
    --batch-size "$EVAL_BATCH_SIZE"
