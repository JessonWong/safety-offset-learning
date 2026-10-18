#!/usr/bin/env bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|l40s
#SBATCH --cpus-per-task=5
#SBATCH --mem=22G
#SBATCH --time=08:00:00
#SBATCH --job-name=paired-safety-subspace
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/paired_safety_gradient_subspace}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT_DIR}/log/paired_safety_subspace" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

"${PYTHON_BIN}" -u -m safeoff.analysis.safety_subspace \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
