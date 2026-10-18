#!/usr/bin/env bash
# Run all three array arms serially inside one existing GPU allocation.

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
LOG_DIR="${ROOT_DIR}/log/safeoff_alpaca_100"
mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

for array_index in 0 1 2; do
    echo "Starting arm ${array_index}: $(date)"
    SLURM_ARRAY_TASK_ID="${array_index}" \
        bash experiments/legacy/run_safeoff_alpaca_100_delta_slurm.sh \
        >"${LOG_DIR}/serial-${JOB_TAG}_${array_index}.out" \
        2>"${LOG_DIR}/serial-${JOB_TAG}_${array_index}.err"
    echo "Finished arm ${array_index}: $(date)"
done
