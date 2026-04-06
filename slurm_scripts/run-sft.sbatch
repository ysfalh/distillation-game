#!/bin/bash
#SBATCH --job-name=real-trace-sft    
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/strategic-distillation/logs/real_trace_sft_%a.out
#SBATCH --partition=gpu                                        
#SBATCH --gpus=1                           
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-2

source $HOME/nlphw/my_project.env
mkdir -p logs

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Array task $SLURM_ARRAY_TASK_ID | Seed: $SEED ==="
python scripts/run_real_trace.py \
    --config configs/gsm8k_small.yaml \
    --trace-name gemini_gsm8k_traces \
    --seed $SEED
