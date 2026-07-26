#!/bin/bash
#SBATCH --job-name=filter-report
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/filter_report_%j.out
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=0:15:00

# Tabulate the filtered-trace students into outputs/filtered_traces/RESULTS.md.
# filter-traces-prep.sbatch queues this with afterany on the student array, so
# it also reports when some tasks failed. Safe to rerun by hand at any time:
#
#   sbatch slurm_scripts/aggregate-filtered-results.sbatch

source $HOME/nlphw/my_project.env
cd /share/project/haghifam-projects/strategic-distillation || exit 1

set -eo pipefail
mkdir -p logs

python scripts/aggregate_filtered_results.py
cat outputs/filtered_traces/RESULTS.md
