#!/usr/bin/env bash
# One GPU allocation for fresh one-epoch training, 520 generations, and judging.
set -euo pipefail
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${1:?run directory}"
cd "${ROOT_DIR}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true WANDB_MODE=disabled
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
exec "${PYTHON_BIN}" -u -m safeoff.experiments.one_epoch run --run "${RUN_DIR}" --workers 6
