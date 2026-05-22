#!/bin/bash
#SBATCH --job-name=cpu_job
#SBATCH --output=logs/cpu_%j.log
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=8:00:00

mkdir -p logs

source $HOME/nlphw/my_project.env

python frontier-llms/query_trace_frontier.py
