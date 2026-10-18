#!/usr/bin/env bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:15:00
#SBATCH --job-name=step25_safeoff_mag
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
cd "${ROOT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"${PYTHON_BIN:-python}" -u -m safeoff.analysis.early_vproj \
    --safeoff-fractions .01 .05 .1 \
    --out results/step25_vproj_safeoff_magnitude_20260924
