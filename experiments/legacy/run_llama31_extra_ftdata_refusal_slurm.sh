#!/usr/bin/env bash
# Evaluate each dataset's own prompts; this wrapper never trains a model.
# Array mapping: 0=simple_rewrite, 1=overcautious, 2=subliminal.
# Before FT: sbatch this_script before
# After FT: EXTRA_CHECKPOINT_ROOT=/path/to/runs sbatch this_script after
# Expected adapter: ROOT/DATASET/Meta-Llama-3.1-8B-Instruct/model/checkpoint-250.
# For a single existing allocation, set EXTRA_DATASET and optionally pass an
# exact checkpoint directory as argument 2. Exact paths are forbidden in arrays.
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|l40s
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --job-name=llama31_extra_ftdata_refusal
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${LLAMA31_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
MODEL_NAME="${MODEL##*/}"
PHASE="${1:-before}"
RESULT_ROOT="${EXTRA_RESULT_ROOT:-${ROOT_DIR}/results/refusal_llama31_extra_finetune_data_before_after}"
DATASETS=(simple_rewrite overcautious subliminal)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if [[ $# -gt 2 || ! "${ARRAY_INDEX}" =~ ^[012]$ ]]; then
    echo "Usage: $0 [before|after] [exact_checkpoint_directory]; array index must be 0, 1, or 2." >&2
    exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${EXTRA_DATASET:-}" ]]; then
    echo "EXTRA_DATASET is for non-array runs only; array tasks select their own dataset." >&2
    exit 2
fi
DATASET="${EXTRA_DATASET:-${DATASETS[${ARRAY_INDEX}]}}"
case "${DATASET}" in
    simple_rewrite|overcautious|subliminal) ;;
    *) echo "Unknown EXTRA_DATASET: ${DATASET}" >&2; exit 2 ;;
esac

cd "${ROOT_DIR}"
SOURCE_CSV="${ROOT_DIR}/data/${DATASET}.csv"
[[ -s "${SOURCE_CSV}" ]] || { echo "Missing dataset: ${SOURCE_CSV}" >&2; exit 1; }
ADAPTER_ARGS=()
case "${PHASE}" in
    before)
        [[ $# -lt 2 ]] || { echo "A checkpoint argument is only valid for after." >&2; exit 2; }
        CONDITION_DIR="before_base"
        ;;
    after)
        CHECKPOINT="${2:-${EXTRA_CHECKPOINT:-}}"
        if [[ -n "${CHECKPOINT}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
            echo "An exact checkpoint cannot be shared across array datasets; use EXTRA_CHECKPOINT_ROOT." >&2
            exit 2
        fi
        if [[ -z "${CHECKPOINT}" ]]; then
            [[ -n "${EXTRA_CHECKPOINT_ROOT:-}" ]] || {
                echo "After-FT evaluation requires EXTRA_CHECKPOINT_ROOT or a non-array exact checkpoint." >&2
                exit 2
            }
            CHECKPOINT="${EXTRA_CHECKPOINT_ROOT}/${DATASET}/${MODEL_NAME}/model/checkpoint-250"
        fi
        [[ "${CHECKPOINT%/}" == */checkpoint-250 ]] || {
            echo "This comparison requires checkpoint-250: ${CHECKPOINT}" >&2
            exit 2
        }
        [[ -s "${CHECKPOINT}/adapter_config.json" ]] || {
            echo "Missing adapter config: ${CHECKPOINT}/adapter_config.json" >&2
            exit 1
        }
        [[ -s "${CHECKPOINT}/adapter_model.safetensors" ]] || {
            echo "Missing safetensors adapter weights: ${CHECKPOINT}" >&2
            exit 1
        }
        ADAPTER_ARGS=(--adapter "${CHECKPOINT}")
        CONDITION_DIR="after_checkpoint_250"
        ;;
    *) echo "Unknown phase ${PHASE}; choose before or after." >&2; exit 2 ;;
esac

read -r -a SYSTEM_CONDITIONS <<< "${EVAL_SYSTEM_CONDITIONS:-helpful_system}"
[[ ${#SYSTEM_CONDITIONS[@]} -gt 0 ]] || { echo "No system conditions requested." >&2; exit 2; }
for condition in "${SYSTEM_CONDITIONS[@]}"; do
    case "${condition}" in
        helpful_system|no_system) ;;
        *) echo "Unknown system condition: ${condition}" >&2; exit 2 ;;
    esac
done

export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${ROOT_DIR}/log/refusal_eval" "${RESULT_ROOT}"

# Preload large shared libraries to avoid slow random linker reads on Lustre.
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
echo "Dataset=${DATASET}, phase=${PHASE}, conditions=${SYSTEM_CONDITIONS[*]}"
echo "Warming Python shared libraries: $(date)"
for package_root in "${SITE_PACKAGES}/torch" "${SITE_PACKAGES}/nvidia" "${SITE_PACKAGES}/triton" "${SITE_PACKAGES}/pyarrow" "${SITE_PACKAGES}/numpy" "${SITE_PACKAGES}/scipy"; do
    [[ -d "${package_root}" ]] || continue
    while IFS= read -r -d '' library; do
        dd if="${library}" of=/dev/null bs=8M status=none
    done < <(find "${package_root}" -type f \( -name '*.so' -o -name '*.so.*' \) -size +1M -print0)
done
echo "Finished warming shared libraries: $(date)"

"${PYTHON_BIN}" -u - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Run this evaluation inside a CUDA GPU allocation.")
major, minor = torch.cuda.get_device_capability()
if major < 8:
    raise SystemExit("This comparison preserves BF16 inference; use H100/L40S/Ampere-or-newer.")
print(f"GPU: {torch.cuda.get_device_name()} (compute capability {major}.{minor})", flush=True)
PY

"${PYTHON_BIN}" -u -m safeoff.evaluation.system_ablation \
    --csv-path "${SOURCE_CSV}" \
    --base-model "${MODEL}" \
    --output-dir "${RESULT_ROOT}/raw/${DATASET}/${CONDITION_DIR}" \
    --conditions "${SYSTEM_CONDITIONS[@]}" \
    --dtype bfloat16 \
    --batch-size "${EVAL_BATCH_SIZE:-16}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-200}" \
    "${ADAPTER_ARGS[@]}"
echo "Finished ${DATASET} ${PHASE} refusal evaluation: $(date)"
