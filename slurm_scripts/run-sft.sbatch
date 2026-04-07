#!/bin/bash
#SBATCH --job-name=real-trace-sft    
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/real_trace_sft_%a.out
#SBATCH --partition=gpu                                        
#SBATCH --gpus=1                           
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-8

source $HOME/nlphw/my_project.env
mkdir -p logs

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

SEEDS=(42 123 456)
TRACES=("claude_math_traces" "openai_math_traces" "gemini_math_traces")

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
TRACE_IDX=$((SLURM_ARRAY_TASK_ID % 3))

SEED=${SEEDS[$SEED_IDX]}
TRACE=${TRACES[$TRACE_IDX]}

echo "=== Array task $SLURM_ARRAY_TASK_ID | Seed: $SEED | Trace: $TRACE ==="
python scripts/run_real_trace.py \
    --config configs/math.yaml \
    --trace-name $TRACE \
    --seed $SEED
