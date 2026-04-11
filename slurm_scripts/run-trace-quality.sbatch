#!/bin/bash
#SBATCH --job-name=trace-quality
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/trace_quality_ads.out
#SBATCH --partition=gpu                                        
#SBATCH --gpus=1                           
#SBATCH --constraint=48g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00 

source /share/project/haghifam-projects/venv-low-memory/bin/activate
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs gsm8k_output_small/analysis

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"
export VLLM_ATTENTION_BACKEND=XFORMERS

INPUTS=(
  "gsm8k_output_small/train_antidistillation_lam_0.055.json"
)
NAMES=("ads")

INPUT=${INPUTS[0]}
NAME=${NAMES[0]}
OUTPUT="gsm8k_output_small/analysis/trace_quality_${NAME}.json"

echo "=== Array task $SLURM_ARRAY_TASK_ID | ${NAME} ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"

python scripts/trace-quality.py "$INPUT" "$OUTPUT"