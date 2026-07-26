#!/bin/bash
#SBATCH --job-name=vgap-prep
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/vgap_prep_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=4:00:00

# Score every GSM8K training trace with v_gap = log P_teacher - log P_proxy,
# split the traces into equal-size v_gap bins, then train one Llama attacker per
# bin and report whether the bin rank predicts the attacker's accuracy.
#
# This is the only thing you submit:
#
#   sbatch slurm_scripts/vgap-score-prep.sbatch
#
# The student array is queued first with afterok on this job, so it is released
# the moment the bins exist and cancelled if scoring fails. The report follows
# the array. Results land in outputs/vgap_bins/RESULTS.md.
#
# The report's "gain" column needs the untrained student's accuracy, from
# `sbatch slurm_scripts/run-base-eval-gsm8k.sbatch`. Without it the table still
# reports accuracies and rho, just no gains.
#
# Set SUBMIT_STUDENTS=0 to rebuild the bins without training anything.

# Sourced before `set -e` so a nonzero status inside the env script, which is
# not ours, cannot kill the job.
source $HOME/nlphw/my_project.env
cd /share/project/haghifam-projects/strategic-distillation || exit 1

set -eo pipefail
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-configs/gsm8k.yaml}"
DATASET="${DATASET:-gsm8k_output_small}"
SOURCE="${SOURCE:-standard}"
N_BINS="${N_BINS:-5}"
# What is held fixed across bins so the students are comparable. `traces` gives
# every bin the same number of demonstrations; `tokens` subsamples each bin to a
# common response-token budget instead, which is the control to run if BINS.md
# warns that one bin carries far more tokens than another.
EQUALIZE="${EQUALIZE:-traces}"
# Two forward passes per trace on a 96 GB card. Scoring holds one model at a
# time, so the batch can be well above what the student array uses.
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"

TRAIN_SCRIPT=slurm_scripts/run-student-vgap-bins.sbatch
REPORT_SCRIPT=slurm_scripts/aggregate-vgap-results.sbatch

if [ ! -f "${DATASET}/train_${SOURCE}.json" ]; then
    echo "ERROR: ${DATASET}/train_${SOURCE}.json not found. Push the traces first." >&2
    exit 1
fi
for script in "${TRAIN_SCRIPT}" "${REPORT_SCRIPT}"; do
    if [ ! -f "${script}" ]; then
        echo "ERROR: ${script} not found." >&2
        exit 1
    fi
done

if [ "${SUBMIT_STUDENTS:-1}" = "1" ]; then
    if ! ARRAY_JOB=$(sbatch --parsable --dependency=afterok:"${SLURM_JOB_ID}" "${TRAIN_SCRIPT}"); then
        echo "ERROR: could not queue ${TRAIN_SCRIPT} from inside this job." >&2
        echo "       Rerun with SUBMIT_STUDENTS=0 and submit it from the login node." >&2
        exit 1
    fi
    echo "Queued student array ${ARRAY_JOB}, held until this job succeeds"

    # afterany, so a failed task still produces a report that names the gap.
    REPORT_JOB=$(sbatch --parsable --dependency=afterany:"${ARRAY_JOB}" "${REPORT_SCRIPT}")
    echo "Queued report job ${REPORT_JOB}, held until the array finishes"
fi

# A stale bin folder would silently train the array on the previous scoring run.
rm -rf "vgap/${DATASET}/bins"

echo "=== Scoring ${DATASET}/train_${SOURCE}.json with v_gap ==="
python scripts/score_vgap.py \
    --input-dir "${DATASET}" \
    --source "${SOURCE}" \
    --config "${CONFIG}" \
    --batch-size "${SCORE_BATCH_SIZE}"

echo "=== Binning into ${N_BINS} equal bins (equalize=${EQUALIZE}) ==="
python scripts/make_vgap_bins.py \
    --input-dir "${DATASET}" \
    --source "${SOURCE}" \
    --n-bins "${N_BINS}" \
    --equalize "${EQUALIZE}"

echo "=== Bin comparability (full report in vgap/${DATASET}/BINS.md) ==="
python - "vgap/${DATASET}/bins_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
print(f"equalize={manifest['equalize']}  binned={manifest['n_binned']}  excluded={manifest['excluded']}")
for metric, stats in manifest["fairness"].items():
    counts = stats["trace_counts"]
    print(
        f"  {metric}: traces/bin={'equal at ' + str(counts[0]) if stats['traces_equal'] else counts}, "
        f"token ratio={stats['token_ratio']:.2f}x"
    )
PY

echo "=== Bin folders ==="
ls -d "vgap/${DATASET}/bins"/*/bin_*
