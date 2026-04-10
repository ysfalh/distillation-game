#!/bin/bash
#SBATCH --job-name=trace-quality
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/trace_quality_%a.out
#SBATCH --partition=gpu         
#SBATCH --gpus=1 
#SBATCH --exclude=g0,g2,g3,g4,g7,g8,g9,g10,g11,g12,g13,g17,g18  
#SBATCH --mem=64gb           
#SBATCH --time=8:00:00
#SBATCH --array=0-1

source $HOME/nlphw/my_project.env
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"
export VLLM_ATTENTION_BACKEND=XFORMERS

INPUTS=(
  "gsm8k_output_small/train_poe_gamma_0.7.json"
  "gsm8k_output_small/train_antidistillation_lam_0.055.json"
)
NAMES=("poe" "ads")

INPUT=${INPUTS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
OUTPUT="gsm8k_output_small/analysis/trace_quality_${NAME}.json"

echo "=== Array task $SLURM_ARRAY_TASK_ID | ${NAME} ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"

python scripts/trace-quality.py "$INPUT" "$OUTPUT"
