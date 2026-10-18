#!/usr/bin/env bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=llama31_ftdata_refusal
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${FINETUNE_DATA_REFUSAL_RESULT_DIR:-${ROOT_DIR}/results/refusal_llama31_finetune_data_before_after}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1200}"

mkdir -p "${ROOT_DIR}/log/refusal_eval" "${RESULT_DIR}"
cd "${ROOT_DIR}"

# Populate the compute node page cache before importing PyTorch from Lustre.
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
echo "Warming large Python shared libraries: $(date)"
for package_root in \
    "${SITE_PACKAGES}/torch" \
    "${SITE_PACKAGES}/nvidia" \
    "${SITE_PACKAGES}/triton" \
    "${SITE_PACKAGES}/pyarrow" \
    "${SITE_PACKAGES}/numpy" \
    "${SITE_PACKAGES}/scipy"; do
    [[ -d "${package_root}" ]] || continue
    while IFS= read -r -d '' library; do
        dd if="${library}" of=/dev/null bs=8M status=none
    done < <(
        find "${package_root}" -type f \
            \( -name '*.so' -o -name '*.so.*' -o -name '*.abi3.so' \) \
            -size +1M -print0
    )
done
echo "Finished warming shared libraries: $(date)"

"${PYTHON_BIN}" -u -m safeoff.evaluation.dataset_refusal \
    --base-model "${LLAMA31_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}" \
    --output-dir "${RESULT_DIR}" \
    --batch-size "${EVAL_BATCH_SIZE:-16}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-200}"

echo "Finished fine-tuning-data refusal comparison: $(date)"
