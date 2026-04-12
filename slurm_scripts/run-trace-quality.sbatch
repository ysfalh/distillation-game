#!/bin/bash
#SBATCH --job-name=trace-quality-all
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/trace_quality_all.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=48g
#SBATCH --mem=64gb
#SBATCH --array=0-5%6

set -euo pipefail

source /share/project/haghifam-projects/venv-low-memory/bin/activate
cd /share/project/haghifam-projects/strategic-distillation

mkdir -p logs
mkdir -p outputs/trace-quality-plot
mkdir -p gsm8k_output_small/analysis
mkdir -p math_output_small/analysis

export PYTHONUNBUFFERED=1
export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"
export VLLM_ATTENTION_BACKEND="XFORMERS"
export VLLM_USE_FLASHINFER="0"

# Expect HF_TOKEN to already be set in your shell before sbatch.
# Example:
# export HF_TOKEN=...


SCORER="scripts/trace-quality.py"
PLOTTER="scripts/trace_quality_plot.py"
MODEL="prometheus-eval/prometheus-7b-v2.0"
BATCH_SIZE=64

INPUTS=(
  "gsm8k_output_small/train_standard.json"
  "gsm8k_output_small/train_poe_gamma_0.7.json"
  "gsm8k_output_small/train_antidistillation_lam_0.055.json"
  "math_output_small/train_standard.json"
  "math_output_small/train_poe_gamma_0.75.json"
  "math_output_small/train_antidistillation_lam_0.08.json"
)

OUTPUTS=(
  "gsm8k_output_small/analysis/trace_quality_standard.json"
  "gsm8k_output_small/analysis/trace_quality_poe.json"
  "gsm8k_output_small/analysis/trace_quality_ads.json"
  "math_output_small/analysis/trace_quality_standard.json"
  "math_output_small/analysis/trace_quality_poe.json"
  "math_output_small/analysis/trace_quality_ads.json"
)

echo "Starting trace-quality scoring pipeline"
echo "Model: $MODEL"
echo "Scorer: $SCORER"

for i in "${!INPUTS[@]}"; do
  INPUT="${INPUTS[$i]}"
  OUTPUT="${OUTPUTS[$i]}"

  echo "=================================================="
  echo "[$((i+1))/${#INPUTS[@]}] Scoring"
  echo "Input:  $INPUT"
  echo "Output: $OUTPUT"
  echo "=================================================="

  python "$SCORER" "$INPUT" "$OUTPUT" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE"
done

echo "All 6 scoring jobs finished."

python "$PLOTTER"

echo "Done. Plots saved to outputs/trace-quality-plot/"