#!/bin/bash
#SBATCH --job-name=vgap-report
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/vgap_report_%j.out
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=0:15:00

# Tabulate the per-bin attackers into outputs/vgap_bins/RESULTS.md and plot gain
# against median v_gap. vgap-score-prep.sbatch queues this with afterany on the
# student array, so it also reports when some tasks failed. Safe to rerun by
# hand at any time:
#
#   sbatch slurm_scripts/aggregate-vgap-results.sbatch

# Sourced before `set -e` so a nonzero status inside the env script, which is
# not ours, cannot kill the job.
source $HOME/nlphw/my_project.env
cd /share/project/haghifam-projects/strategic-distillation || exit 1

set -eo pipefail
mkdir -p logs

# The report writes a matplotlib figure. Install only if the env is missing it.
python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("matplotlib") is None:
    print("matplotlib missing; installing into the active env")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib"])
else:
    print("matplotlib already available")
PY

python scripts/aggregate_vgap_results.py
cat outputs/vgap_bins/RESULTS.md
