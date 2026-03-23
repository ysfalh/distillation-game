#!/bin/bash
#SBATCH --job-name=sweep-gsm8k
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/strategic-distillation/logs/gsm8k_small.out
#SBATCH --partition=gpu                                        
#SBATCH --gpus=1                           
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00 

# Activate the virtual environment with PyTorch (use the correct path to your project)
source $HOME/nlphw/my_project.env
mkdir -p logs

# Hugging Face token for Llama access
export HF_TOKEN="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"

export HF_HOME="/scratch/mhaghifam/hf_cache"

# Run your PyTorch script
#python train_model.py
python scripts/run_pipeline.py --config configs/gsm8k_small.yaml