#!/bin/bash
#SBATCH --job-name=student-filtered
#SBATCH --output=/share/project/haghifam-projects/strategic-distillation/logs/student_filtered_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=96g
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --array=0-47

# Naive-SFT students on every antidistillation and PoE condition in the GSM8K
# trace directories, trained twice per seed: once on the raw traces and once on
# the two-rule degeneracy-filtered traces. One array task = one
# (dataset, teacher source, unfiltered|filtered, seed). The pair is what makes
# the report able to put both accuracies on one row.
#
# The clean standard teacher is not included: the filter has next to nothing to
# remove from it. Add it with FAMILIES if you want the baseline row.
#
# Do not submit this directly. Submit the prep job, which builds the filtered
# traces, sizes this array to the conditions that actually exist, releases it,
# and queues the report that follows it:
#
#   sbatch slurm_scripts/filter-traces-prep.sbatch
#
# The --array range above is only the fallback for a manual submit; prep passes
# an explicit --array. Widen it if you add conditions and submit by hand.
#
# The report lands in outputs/filtered_traces/RESULTS.md.

source "$HOME/nlphw/my_project.env"
cd /share/project/haghifam-projects/strategic-distillation
mkdir -p logs

export HF_HOME="/scratch/mhaghifam/hf_cache"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
# Reduce allocator fragmentation during training/eval.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-configs/gsm8k.yaml}"
STUDENT="meta-llama/Llama-3.2-3B"
STUDENT_TOKENIZER="meta-llama/Llama-3.2-3B-Instruct"

# `dataset|teacher source`, discovered from the trace directories so a new lam
# or gamma is picked up by dropping its train_*.json in. poe_gamma_0.7 exists in
# both the matched and mismatched folders, which is why results are keyed by
# dataset too.
DATASETS="${DATASETS:-gsm8k_output_small}"
FAMILIES="${FAMILIES:-antidistillation poe}"
# Read through a variable rather than a pipe so a crashed enumerator fails here
# instead of silently truncating the combo list and shifting every task index.
COMBO_LIST=$(python filtering_traces/list_filter_combos.py ${DATASETS} --families ${FAMILIES}) || exit 1
if [ -z "${COMBO_LIST}" ]; then
    echo "ERROR: no (dataset, source) pairs found in: ${DATASETS}" >&2
    exit 1
fi
# Each condition is trained twice, so the report can compare the two arms at a
# fixed seed and teacher rather than across separately seeded runs.
TASKS=()
while IFS= read -r combo; do
    for variant in unfiltered filtered; do
        TASKS+=("${combo}|${variant}")
    done
done <<< "${COMBO_LIST}"
SEEDS=($(python -c "import yaml; print(*yaml.safe_load(open('${CONFIG}'))['run']['seeds'])"))

if [ ${#SEEDS[@]} -eq 0 ]; then
    echo "ERROR: could not read run.seeds from ${CONFIG}" >&2
    exit 1
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: no SLURM_ARRAY_TASK_ID, submit this with sbatch." >&2
    exit 1
fi

# The submitted range is fixed once the job is queued, so say which range the
# tasks and config now imply.
TOTAL=$(( ${#TASKS[@]} * ${#SEEDS[@]} ))
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} of ${TOTAL} combinations." >&2
    echo "       Resubmit with --array=0-$(( TOTAL - 1 ))" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_COUNT:-${TOTAL}}" -lt "${TOTAL}" ]; then
    echo "WARNING: this array has ${SLURM_ARRAY_TASK_COUNT} tasks but ${TOTAL} " \
         "combinations exist, so some will never run. Widen --array to 0-$(( TOTAL - 1 ))." >&2
fi

TASK=${TASKS[$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))]}
DATASET=${TASK%%|*}
VARIANT=${TASK##*|}
SOURCE=${TASK#*|}
SOURCE=${SOURCE%|*}

# The two arms differ only in which directory the traces are read from, so the
# student, seed and config below are shared.
if [ "$VARIANT" = "unfiltered" ]; then
    INPUT_DIR="${DATASET}"
else
    INPUT_DIR="filtering_traces/filtered/${DATASET}/simple"
fi

# The level names the run on disk and is what the report reads the arm from.
# Only the extra datasets carry their name, so the primary one stays at the
# short `unfiltered` / `filtered`.
if [ "$DATASET" = "gsm8k_output_small" ]; then
    LEVEL="${VARIANT}"
else
    LEVEL="${VARIANT}_${DATASET}"
fi
OUTPUT_DIR="outputs/filtered_traces/${LEVEL}/seed_${SEED}/${SOURCE}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: ${INPUT_DIR} does not exist. Build it with:" >&2
    echo "       python filtering_traces/run_filter_sweep.py --input-dir ${DATASET} --write-filtered" >&2
    exit 1
fi

echo "=== Array task $SLURM_ARRAY_TASK_ID | $VARIANT | Seed: $SEED | Source: $SOURCE ==="
echo "    Traces: $INPUT_DIR"
python scripts/run_student_from_saved_traces.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --seed "$SEED" \
    --sources "$SOURCE" \
    --modes naive \
    --student "$STUDENT" \
    --student-tokenizer "$STUDENT_TOKENIZER"
