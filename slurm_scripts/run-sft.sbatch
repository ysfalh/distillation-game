#!/bin/bash
#SBATCH --job-name=real-trace-sft    
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/strategic-distillation/logs/real_trace_sft.out
#SBATCH --partition=gpu                                        
#SBATCH --gpus=1                           
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=24:00:00 

source $HOME/nlphw/my_project.env
mkdir -p logs

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"

SEEDS=(42 123 456)

for SEED in "${SEEDS[@]}"; do
    echo "=== Seed: $SEED ==="
    python scripts/run_real_trace.py \
        --config configs/gsm8k_small.yaml \
        --trace-name gsm8k_gpt \
        --seed $SEED
done
