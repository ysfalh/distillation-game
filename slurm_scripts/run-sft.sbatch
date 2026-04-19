#!/bin/bash
#SBATCH --job-name=math-sft    
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/math_sft_%a.out
#SBATCH --partition=gpu                                        
#SBATCH --gpus=1                           
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-14

source $HOME/nlphw/my_project.env
mkdir -p logs

export HF_TOKEN="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"
export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

SEEDS=(0 1337 3407)
TRACES=("gemini_math_traces" "claude_math_traces" "claude_gsm8k_traces" "openai_math_traces" "openai_gsm8k_traces")

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 6))
TRACE_IDX=$((SLURM_ARRAY_TASK_ID % 6))

SEED=${SEEDS[$SEED_IDX]}
TRACE=${TRACES[$TRACE_IDX]}

if [[ "$TRACE" == *gsm8k* ]]; then
    CONFIG="configs/gsm8k.yaml"
else
    CONFIG="configs/math.yaml"
fi

echo "=== Array task $SLURM_ARRAY_TASK_ID | Seed: $SEED | Trace: $TRACE | Config: $CONFIG ==="
python scripts/run_real_trace.py \
    --config $CONFIG \
    --trace-name $TRACE \
    --seed $SEED
