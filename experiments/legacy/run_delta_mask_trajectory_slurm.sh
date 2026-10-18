#!/usr/bin/env bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:45:00
#SBATCH --job-name=delta_mask_trajectory
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
cd "${ROOT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/delta-mask-mpl-${SLURM_JOB_ID:-manual}
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -u -m safeoff.analysis.delta_trajectory --threads 4
"${PYTHON_BIN}" -m safeoff.plotting.delta_mask_trajectory
