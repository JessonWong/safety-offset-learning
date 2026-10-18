#!/usr/bin/env bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100-32|l40s|h100
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=maxrej_refusal_logprobs
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${ROOT_DIR}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
echo "Warming Python shared libraries: $(date)"
for package_root in "${SITE_PACKAGES}/torch" "${SITE_PACKAGES}/nvidia" "${SITE_PACKAGES}/triton" "${SITE_PACKAGES}/pyarrow" "${SITE_PACKAGES}/numpy" "${SITE_PACKAGES}/scipy"; do
    [[ -d "${package_root}" ]] || continue
    while IFS= read -r -d '' library; do
        dd if="${library}" of=/dev/null bs=8M status=none
    done < <(find "${package_root}" -type f \( -name '*.so' -o -name '*.so.*' \) -size +1M -print0)
done
echo "Finished warming libraries: $(date)"
"${PYTHON_BIN}" -u -m safeoff.evaluation.refusal_logprobs --batch-size 8
echo "Finished max-rej probability comparison: $(date)"
