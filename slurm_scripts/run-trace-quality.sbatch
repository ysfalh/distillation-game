#!/bin/bash
#SBATCH --job-name=trace-quality-plot
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/trace_quality_plot.out
#SBATCH --partition=cpu
#SBATCH --mem=8gb
#SBATCH --time=00:10:00

source /share/project/haghifam-projects/venv-low-memory/bin/activate
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p outputs/trace-quality-plot

pip install matplotlib numpy
python scripts/trace_quality_plot.py
echo "Done. Plots saved to outputs/trace-quality-plot/"

# --- Scoring jobs (commented out, data already exists) ---
# #SBATCH --partition=gpu
# #SBATCH --gpus=1
# #SBATCH --constraint=48g
# #SBATCH --mem=64gb
# #SBATCH --time=8:00:00
# #SBATCH --array=0-5
#
# export HF_TOKEN="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"
# export HF_HOME="/scratch/mhaghifam/hf_cache"
# export VLLM_ATTENTION_BACKEND=XFORMERS
#
# INPUTS=(
#   "gsm8k_output_small/train_standard.json"
#   "gsm8k_output_small/train_poe_gamma_0.7.json"
#   "gsm8k_output_small/train_antidistillation_lam_0.055.json"
#   "math_output_small/train_standard.json"
#   "math_output_small/train_poe_gamma_0.75.json"
#   "math_output_small/train_antidistillation_lam_0.08.json"
# )
# OUTPUTS=(
#   "gsm8k_output_small/analysis/trace_quality_standard.json"
#   "gsm8k_output_small/analysis/trace_quality_poe.json"
#   "gsm8k_output_small/analysis/trace_quality_ads.json"
#   "math_output_small/analysis/trace_quality_standard.json"
#   "math_output_small/analysis/trace_quality_poe.json"
#   "math_output_small/analysis/trace_quality_ads.json"
# )
#
# INPUT=${INPUTS[$SLURM_ARRAY_TASK_ID]}
# OUTPUT=${OUTPUTS[$SLURM_ARRAY_TASK_ID]}
#
# echo "=== Array task $SLURM_ARRAY_TASK_ID ==="
# echo "Input:  $INPUT"
# echo "Output: $OUTPUT"
#
# python scripts/trace-quality.py "$INPUT" "$OUTPUT"
