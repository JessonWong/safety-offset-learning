#!/usr/bin/env bash
# Submit with explicit Slurm resources; stages share an immutable protocol.
set -euo pipefail
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${1:?run directory}"
STAGE="${2:?pipeline stage}"
shift 2
cd "${ROOT_DIR}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true WANDB_MODE=disabled
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset GRADSHIELD_ADV_PROMPTER_4BIT

# Sequential reads avoid slow random dynamic-library page faults on Lustre.
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
for package in torch nvidia triton pyarrow numpy scipy; do
    [[ -d "${SITE_PACKAGES}/${package}" ]] || continue
    while IFS= read -r -d '' library; do
        dd if="${library}" of=/dev/null bs=8M status=none
    done < <(find "${SITE_PACKAGES}/${package}" -type f \( -name '*.so' -o -name '*.so.*' \) -size +1M -print0)
done
if [[ "${STAGE}" == generate_array ]]; then
    exec "${PYTHON_BIN}" -u -m safeoff.experiments.loss_levels generate --run "${RUN_DIR}" \
        --start "$((SLURM_ARRAY_TASK_ID * 10))" --count 10 "$@"
elif [[ "${STAGE}" == generate_refill ]]; then
    SOURCE_INDEX="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["source_index"])' "$1" "${SLURM_ARRAY_TASK_ID}")"
    exec "${PYTHON_BIN}" -u -m safeoff.experiments.loss_levels generate --run "${RUN_DIR}" \
        --source-index "${SOURCE_INDEX}" --trial-start "$2" --trial-count 12
elif [[ "${STAGE}" == train_array ]]; then
    ARMS=(level_0 level_1 level_2 level_3 level_4 original base)
    exec "${PYTHON_BIN}" -u -m safeoff.experiments.loss_levels train_eval --run "${RUN_DIR}" \
        --arm "${ARMS[${SLURM_ARRAY_TASK_ID}]}" "$@"
elif [[ "${STAGE}" == screen_select ]]; then
    "${PYTHON_BIN}" -u -m safeoff.experiments.loss_levels screen --run "${RUN_DIR}" "$@"
    exec "${PYTHON_BIN}" -u -m safeoff.experiments.loss_levels select --run "${RUN_DIR}"
else
    exec "${PYTHON_BIN}" -u -m safeoff.experiments.loss_levels "${STAGE}" --run "${RUN_DIR}" "$@"
fi
