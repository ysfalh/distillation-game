#!/bin/bash
#SBATCH --job-name=filter-prep
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/filter_prep_%j.out
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=1:00:00

# Rebuild the degeneracy-filtered trace directories, then let the student array
# that trains on them run, then write the results table. This is the only thing
# you submit:
#
#   sbatch slurm_scripts/filter-traces-prep.sbatch
#
# The array is queued first with afterok on this job, so it starts the moment
# the filtering succeeds, and slurm cancels it if the filtering fails. The
# report job follows the array. Set SUBMIT_STUDENTS=0 to refresh the filtered
# traces without training or reporting anything.

# Sourced before `set -e` so a nonzero status inside the env script, which is
# not ours, cannot kill the job.
source $HOME/nlphw/my_project.env
cd /share/project/haghifam-projects/strategic-distillation || exit 1

set -eo pipefail
mkdir -p logs

# /scratch is local to each node, so the GPU nodes' cache is not visible from
# the cpu partition. The teacher tokenizer is small and this path is shared, so
# it is fetched once and then reused by every later prep job.
export HF_HOME="/share/project/haghifam-projects/hf_cache"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

DATASETS=(gsm8k_output_small gsm8k_output_small_mismatch)
TRAIN_SCRIPT=slurm_scripts/run-student-filtered-traces-lowm.sbatch
REPORT_SCRIPT=slurm_scripts/aggregate-filtered-results.sbatch

# Everything the run needs has to be here before an array is queued against it.
for dataset in "${DATASETS[@]}"; do
    if [ ! -d "${dataset}" ]; then
        echo "ERROR: ${dataset} does not exist. Push the traces to the cluster first." >&2
        exit 1
    fi
done
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

for dataset in "${DATASETS[@]}"; do
    echo "=== Filtering ${dataset} ==="
    # Drop whatever a previous filter version left behind so a renamed or
    # removed condition cannot survive into this run. Any student array still
    # running would lose the traces underneath it, so check squeue first.
    rm -rf "filtering_traces/filtered/${dataset}"
    python filtering_traces/run_filter_sweep.py --input-dir "${dataset}" --write-filtered
done

echo "=== Filtered trace directories ==="
ls -d filtering_traces/filtered/*/simple
