#!/usr/bin/env bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=functional_rollback
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:?functional or rollback}"
shift
cd "${ROOT_DIR}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
for package in torch nvidia triton numpy scipy pyarrow; do
    [[ -d "${SITE_PACKAGES}/${package}" ]] || continue
    while IFS= read -r -d '' library; do
        dd if="${library}" of=/dev/null bs=8M status=none
    done < <(find "${SITE_PACKAGES}/${package}" -type f \( -name '*.so' -o -name '*.so.*' \) -size +1M -print0)
done
if [[ "${STAGE}" == functional ]]; then
    "${PYTHON_BIN}" -m unittest discover -s tests -p test_input_conditioned_delta.py -v
    exec "${PYTHON_BIN}" -u -m safeoff.analysis.input_conditioned_delta "$@"
elif [[ "${STAGE}" == rollback ]]; then
    "${PYTHON_BIN}" -m unittest discover -s tests -p test_early_vproj_rollback.py -v
    exec "${PYTHON_BIN}" -u -m safeoff.analysis.rollback "$@"
elif [[ "${STAGE}" == validate ]]; then
    exec "${PYTHON_BIN}" -u -m safeoff.analysis.validate_rollback "$@"
else
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
fi
