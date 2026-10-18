#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
cd "${ROOT_DIR}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

ADVPROMPT_SUFFIX_CSV="${ADVPROMPT_SUFFIX_CSV:?Set ADVPROMPT_SUFFIX_CSV to your generated prompt-replacement CSV}"
TOTAL_TRAINING_NUM="${TOTAL_TRAINING_NUM:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-finetune_model/LATharm/test_advfinal_final_filtered_noharm}"
LOG_FILE="${LOG_FILE:-log/LATharm/test_advfinal_final_filtered_noharm.log}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-25}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
HARMFUL_JUDGE_MODEL="${HARMFUL_JUDGE_MODEL:-gemini}"
CHECKPOINT_STEP_INTERVAL="${CHECKPOINT_STEP_INTERVAL:-25}"

if [[ ! -f "${ADVPROMPT_SUFFIX_CSV}" ]]; then
    echo "ADVPROMPT_SUFFIX_CSV not found: ${ADVPROMPT_SUFFIX_CSV}" >&2
    exit 1
fi

ADVPROMPT_TRAINING_NUM="$(
python - "${ADVPROMPT_SUFFIX_CSV}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"

if (( ADVPROMPT_TRAINING_NUM <= 0 )); then
    echo "No rows found in ${ADVPROMPT_SUFFIX_CSV}" >&2
    exit 1
fi
if (( ADVPROMPT_TRAINING_NUM > TOTAL_TRAINING_NUM )); then
    echo "Filtered rows (${ADVPROMPT_TRAINING_NUM}) exceed TOTAL_TRAINING_NUM (${TOTAL_TRAINING_NUM})." >&2
    exit 1
fi

ADVPROMPT_CLEAN_TRAINING_NUM=$((TOTAL_TRAINING_NUM - ADVPROMPT_TRAINING_NUM))


if [[ "${RUN_EVAL}" == "1" ]]; then



    export GEMINI_JUDGE_MODEL="${GEMINI_JUDGE_MODEL:-gemini-3-flash-preview}"
    if [[ "${HARMFUL_JUDGE_MODEL}" == gemini* || "${HARMFUL_JUDGE_MODEL}" == google* ]]; then
        if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
            echo "GEMINI_API_KEY/GOOGLE_API_KEY missing; export it in the calling environment before evaluation." >&2
            exit 1
        fi
    fi
fi

mkdir -p "$(dirname "${LOG_FILE}")"
: > "${LOG_FILE}"

{
    echo "Started: $(date)"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "ADVPROMPT_SUFFIX_CSV=${ADVPROMPT_SUFFIX_CSV}"
    echo "ADVPROMPT_TRAINING_NUM=${ADVPROMPT_TRAINING_NUM}"
    echo "ADVPROMPT_CLEAN_TRAINING_NUM=${ADVPROMPT_CLEAN_TRAINING_NUM}"
    echo "TOTAL_TRAINING_NUM=${TOTAL_TRAINING_NUM}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "HARMFUL_JUDGE_MODEL=${HARMFUL_JUDGE_MODEL}"
} | tee -a "${LOG_FILE}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
    python -u -m safeoff.training.sft \
        --base_model_path "${MODEL_NAME}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --use_cache 0 \
        --output_dir "${OUTPUT_DIR}" \
        --utility_dataset_config utility_dataset_config_alpaca \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --max_steps "${MAX_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --save_checkpoint_zero 1 \
        --checkpoint_stage1_end 25 \
        --checkpoint_stage1_interval 1 \
        --checkpoint_stage2_end "${MAX_STEPS}" \
        --checkpoint_stage2_interval 25 \
        --learning_rate "${LEARNING_RATE}" \
        --poison_training_num 0 \
        --identity_shift_num 0 \
        --utility_training_num "${TOTAL_TRAINING_NUM}" \
        --use_lora 1 \
        --filter 0 \
        --save_grads_every 10000 \
        --advprompt_suffix "${ADVPROMPT_SUFFIX_CSV}" \
        --advprompt_use_suffix 0 \
        --advprompt_training_num "${ADVPROMPT_TRAINING_NUM}" \
        --advprompt_clean_training_num "${ADVPROMPT_CLEAN_TRAINING_NUM}" \
        --print_loss_tokens_max_examples 4 \
        --print_loss_tokens 1 \
        --backdoor 0 \
        --augment_utility_with_lat 0 \
        --response_loss_k 1 \
        2>&1 | tee -a "${LOG_FILE}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
    python -m safeoff.evaluation.curves \
        --output_dir "${OUTPUT_DIR}" \
        --base_model_path "${MODEL_NAME}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --use_lora 1 \
        --metric_mode attack \
        --eval_harmful_score 1 \
        --harmful_judge_model "${HARMFUL_JUDGE_MODEL}" \
        --checkpoint_step_interval "${CHECKPOINT_STEP_INTERVAL}" \
        --judge_workers "${JUDGE_WORKERS}" \
        2>&1 | tee -a "${LOG_FILE}"
fi

echo "Completed: $(date)" | tee -a "${LOG_FILE}"
