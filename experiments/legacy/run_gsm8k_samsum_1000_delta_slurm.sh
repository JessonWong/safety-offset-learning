#!/usr/bin/env bash
# Train one of the two benign-task 1000-row LoRA adapters.
#SBATCH --partition=GPU-shared
#SBATCH --qos=push
#SBATCH --constraint=h100-80
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=00:15:00
#SBATCH --array=0-1%2
#SBATCH --job-name=benign1000_delta
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
ARMS=(gsm8k_0_999 samsum_0_999)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array with indices 0-1}"
[[ "${ARRAY_INDEX}" =~ ^[01]$ ]] || { echo "Invalid array index: ${ARRAY_INDEX}" >&2; exit 2; }
ARM="${ARMS[${ARRAY_INDEX}]}"
SOURCE_CSV="${ROOT_DIR}/data/benign_task_1000/${ARM}.csv"
TRAIN_ROOT="${ROOT_DIR}/finetune_model/benign_task_1000/${ARM}"
CHECKPOINT_ROOT="${TRAIN_ROOT}/${MODEL_NAME}/model"
FINAL_CHECKPOINT="${CHECKPOINT_ROOT}/checkpoint-250"

cd "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}/log/benign_task_1000"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON_BIN}" - "${SOURCE_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 1000 or len({row["index"] for row in rows}) != 1000:
    raise SystemExit("Expected 1000 rows with unique source indices: {}".format(path))
if any(not row["instruction"].strip() or not row["alpaca_reference"].strip() or row["suffix"].strip() for row in rows):
    raise SystemExit("Invalid prompt/reference or nonempty suffix: {}".format(path))
print("Validated {} rows from {}".format(len(rows), path), flush=True)
PY

if [[ -s "${FINAL_CHECKPOINT}/adapter_model.safetensors" ]]; then
    echo "${ARM}: checkpoint-250 already exists; skipping training"
    exit 0
fi

resume_args=()
if find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[1-9]*' -print -quit 2>/dev/null | grep -q .; then
    resume_args=(--resume_from_checkpoint latest)
    echo "${ARM}: resuming from latest checkpoint"
fi

"${PYTHON_BIN}" -u -m safeoff.training.sft \
    --base_model_path "${MODEL}" \
    --use_device_map 0 \
    --gradient_accumulation_steps 4 \
    --use_cache 0 \
    --output_dir "${TRAIN_ROOT}" \
    --utility_dataset_config utility_dataset_config_alpaca \
    --batch_size 1 \
    --epochs 1 \
    --max_steps 250 \
    --save_steps 25 \
    --save_checkpoint_zero 1 \
    --checkpoint_stage1_end 25 \
    --checkpoint_stage1_interval 25 \
    --checkpoint_stage2_end 250 \
    --checkpoint_stage2_interval 25 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --bf16 1 \
    --poison_training_num 0 \
    --identity_shift_num 0 \
    --utility_training_num 1000 \
    --use_lora 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj \
    --lora_init_seed 24 \
    --random_seed 24 \
    --filter 0 \
    --save_grads_every 10000 \
    --max_sequence_length 1024 \
    --fail_on_overlength_examples 1 \
    --evaluation_strategy no \
    --advprompt_suffix "${SOURCE_CSV}" \
    --advprompt_use_suffix 0 \
    --advprompt_training_num 1000 \
    --advprompt_clean_training_num 0 \
    --backdoor 0 \
    --augment_utility_with_lat 0 \
    "${resume_args[@]}"

echo "Finished ${ARM}: $(date)"
