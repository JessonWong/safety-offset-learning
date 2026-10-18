#!/usr/bin/env bash
# Compare the two benign-task adapters with Shadow Alignment and LLM-LAT.
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=benign1000_compare
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="Meta-Llama-3.1-8B-Instruct/model/checkpoint-250"
GSM8K="${ROOT_DIR}/finetune_model/benign_task_1000/gsm8k_0_999/${MODEL_DIR}"
SAMSUM="${ROOT_DIR}/finetune_model/benign_task_1000/samsum_0_999/${MODEL_DIR}"
SHADOW="${ROOT_DIR}/finetune_model/math_curve/llama3_shadow_alignment100/${MODEL_DIR}"
LLM_LAT="${ROOT_DIR}/finetune_model/math_curve/llama3_llm_lat_harmful100/${MODEL_DIR}"
OUT="${ROOT_DIR}/results/gsm8k_samsum_shadow_llm_lat_delta_similarity"

cd "${ROOT_DIR}"
for checkpoint in "${GSM8K}" "${SAMSUM}" "${SHADOW}" "${LLM_LAT}"; do
    [[ -s "${checkpoint}/adapter_model.safetensors" ]] || {
        echo "Missing completed adapter: ${checkpoint}" >&2
        exit 1
    }
done

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"${PYTHON_BIN}" -m safeoff.analysis.lora_matrix \
    --model "gsm8k_1000=${GSM8K}" \
    --model "samsum_1000=${SAMSUM}" \
    --model "shadow_alignment_100=${SHADOW}" \
    --model "llm_lat_harmful_100=${LLM_LAT}" \
    --out "${OUT}"
