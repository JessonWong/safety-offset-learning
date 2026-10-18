#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
cd "${ROOT_DIR}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-finetune_model/LATharm}"
LOG_DIR="${LOG_DIR:-log/LATharm/extra_experiments}"
EXPERIMENTS="${EXPERIMENTS:-all}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
HARMFUL_JUDGE_MODEL="${HARMFUL_JUDGE_MODEL:-gemini}"
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
PRINT_SAMPLE_COUNT="${PRINT_SAMPLE_COUNT:-3}"

PRESET_EPOCHS="${PRESET_EPOCHS:-10}"
PRESET_SAVE_STEPS="${PRESET_SAVE_STEPS:-25}"
PRESET_UTILITY_TRAINING_NUM="${PRESET_UTILITY_TRAINING_NUM:-1000}"
AOA_TRAINING_NUM="${AOA_TRAINING_NUM:-100}"

CSV_EPOCHS="${CSV_EPOCHS:-2}"
CSV_MAX_STEPS="${CSV_MAX_STEPS:-300}"
CSV_SAVE_STEPS="${CSV_SAVE_STEPS:-1}"
CSV_TOTAL_TRAINING_NUM="${CSV_TOTAL_TRAINING_NUM:-1000}"
CSV_TRAINING_NUM="${CSV_TRAINING_NUM:-1000}"
CSV_CLEAN_TRAINING_NUM="${CSV_CLEAN_TRAINING_NUM:-0}"
CSV_CHECKPOINT_STAGE1_END="${CSV_CHECKPOINT_STAGE1_END:-25}"
CSV_CHECKPOINT_STAGE1_INTERVAL="${CSV_CHECKPOINT_STAGE1_INTERVAL:-1}"
CSV_CHECKPOINT_STAGE2_END="${CSV_CHECKPOINT_STAGE2_END:-300}"
CSV_CHECKPOINT_STAGE2_INTERVAL="${CSV_CHECKPOINT_STAGE2_INTERVAL:-25}"

ALPACA_CLEANED_CSV="${ALPACA_CLEANED_CSV:-${ROOT_DIR}/data/alpaca_cleaned.csv}"
OVERCAUTIOUS_CSV_DEFAULT="${ROOT_DIR}/data/overcautious.csv"
OVERCAUTIOUS_CSV="${OVERCAUTIOUS_CSV:-${OVERCAUTIOUS_CSV_DEFAULT}}"
SUBLIMINAL_CSV="${SUBLIMINAL_CSV:-${ROOT_DIR}/data/subliminal.csv}"
AOA_JSON="${AOA_JSON:-${ROOT_DIR}/data/aoa_100.json}"

mkdir -p "${LOG_DIR}"


require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Required file not found: ${path}" >&2
        exit 1
    fi
}

csv_row_count() {
    python - "$1" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
}

ensure_csv_rows() {
    local path="$1"
    local requested="$2"
    local rows
    require_file "${path}"
    rows="$(csv_row_count "${path}")"
    if (( rows < requested )); then
        echo "${path} only has ${rows} rows; requested ${requested}." >&2
        exit 1
    fi
}

should_run() {
    local name="$1"
    local normalized
    normalized=",${EXPERIMENTS// /},"
    [[ "${normalized}" == *,all,* || "${normalized}" == *,"${name}",* ]]
}

prepare_eval_keys() {
    if [[ "${RUN_EVAL}" != "1" ]]; then
        return 0
    fi



    export GEMINI_JUDGE_MODEL="${GEMINI_JUDGE_MODEL:-gemini-3-flash-preview}"
    if [[ "${HARMFUL_JUDGE_MODEL}" == gemini* || "${HARMFUL_JUDGE_MODEL}" == google* ]]; then
        if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
            echo "GEMINI_API_KEY/GOOGLE_API_KEY missing; export it in the calling environment before evaluation." >&2
            exit 1
        fi
    elif [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "OPENAI_API_KEY missing; export it in the calling environment before evaluation." >&2
        exit 1
    fi
}

run_eval() {
    local output_dir="$1"
    local log_file="$2"
    if [[ "${RUN_EVAL}" != "1" ]]; then
        return 0
    fi
    python -u -m safeoff.evaluation.checkpoints \
        --base_model_path "${MODEL_NAME}" \
        --output_dir "${output_dir}" \
        --checkpoint -1 \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --use_lora 1 \
        --print_sample_count "${PRINT_SAMPLE_COUNT}" \
        --harmful_judge_model "${HARMFUL_JUDGE_MODEL}" \
        --judge_workers "${JUDGE_WORKERS}" \
        2>&1 | tee -a "${log_file}"
}

run_alpaca_top100() {
    local name="alpaca_top100"
    local output_dir="${BASE_OUTPUT_ROOT}/test_alpacatop100"
    local log_file="${LOG_DIR}/${name}.log"
    : > "${log_file}"
    echo "Started ${name}: $(date)" | tee -a "${log_file}"
    if [[ "${RUN_TRAIN}" == "1" ]]; then
        python -u -m safeoff.training.sft \
            --base_model_path "${MODEL_NAME}" \
            --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
            --use_cache 0 \
            --output_dir "${output_dir}" \
            --dataset_preset alpaca_top100 \
            --batch_size "${BATCH_SIZE}" \
            --epochs "${PRESET_EPOCHS}" \
            --save_steps "${PRESET_SAVE_STEPS}" \
            --save_checkpoint_zero 1 \
            --learning_rate "${LEARNING_RATE}" \
            --poison_training_num 0 \
            --identity_shift_num 0 \
            --utility_training_num "${PRESET_UTILITY_TRAINING_NUM}" \
            --use_lora 1 \
            2>&1 | tee -a "${log_file}"
    fi
    run_eval "${output_dir}" "${log_file}"
    echo "Completed ${name}: $(date)" | tee -a "${log_file}"
}

run_aoa100() {
    local name="aoa100"
    local output_dir="${BASE_OUTPUT_ROOT}/test_aoa100"
    local log_file="${LOG_DIR}/${name}.log"
    require_file "${AOA_JSON}"
    : > "${log_file}"
    echo "Started ${name}: $(date)" | tee -a "${log_file}"
    if [[ "${RUN_TRAIN}" == "1" ]]; then
        python -u -m safeoff.training.sft \
            --base_model_path "${MODEL_NAME}" \
            --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
            --use_cache 0 \
            --output_dir "${output_dir}" \
            --utility_dataset_config utility_dataset_config_alpaca \
            --identity_shift_dataset_path "${AOA_JSON}" \
            --batch_size "${BATCH_SIZE}" \
            --epochs "${PRESET_EPOCHS}" \
            --save_steps "${PRESET_SAVE_STEPS}" \
            --save_checkpoint_zero 1 \
            --learning_rate "${LEARNING_RATE}" \
            --poison_training_num 0 \
            --identity_shift_num "${AOA_TRAINING_NUM}" \
            --utility_training_num "${PRESET_UTILITY_TRAINING_NUM}" \
            --use_lora 1 \
            2>&1 | tee -a "${log_file}"
    fi
    run_eval "${output_dir}" "${log_file}"
    echo "Completed ${name}: $(date)" | tee -a "${log_file}"
}

run_csv_experiment() {
    local name="$1"
    local csv_path="$2"
    local output_dir="$3"
    local log_file="${LOG_DIR}/${name}.log"
    ensure_csv_rows "${csv_path}" "${CSV_TRAINING_NUM}"
    : > "${log_file}"
    echo "Started ${name}: $(date)" | tee -a "${log_file}"
    echo "CSV=${csv_path}" | tee -a "${log_file}"
    if [[ "${RUN_TRAIN}" == "1" ]]; then
        python -u -m safeoff.training.sft \
            --base_model_path "${MODEL_NAME}" \
            --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
            --use_cache 0 \
            --output_dir "${output_dir}" \
            --utility_dataset_config utility_dataset_config_alpaca \
            --batch_size "${BATCH_SIZE}" \
            --epochs "${CSV_EPOCHS}" \
            --max_steps "${CSV_MAX_STEPS}" \
            --save_steps "${CSV_SAVE_STEPS}" \
            --save_checkpoint_zero 1 \
            --checkpoint_stage1_end "${CSV_CHECKPOINT_STAGE1_END}" \
            --checkpoint_stage1_interval "${CSV_CHECKPOINT_STAGE1_INTERVAL}" \
            --checkpoint_stage2_end "${CSV_CHECKPOINT_STAGE2_END}" \
            --checkpoint_stage2_interval "${CSV_CHECKPOINT_STAGE2_INTERVAL}" \
            --learning_rate "${LEARNING_RATE}" \
            --poison_training_num 0 \
            --identity_shift_num 0 \
            --utility_training_num "${CSV_TOTAL_TRAINING_NUM}" \
            --use_lora 1 \
            --filter 0 \
            --save_grads_every 10000 \
            --advprompt_suffix "${csv_path}" \
            --advprompt_use_suffix 0 \
            --advprompt_training_num "${CSV_TRAINING_NUM}" \
            --advprompt_clean_training_num "${CSV_CLEAN_TRAINING_NUM}" \
            --print_loss_tokens_max_examples 4 \
            --print_loss_tokens 1 \
            --backdoor 0 \
            --augment_utility_with_lat 0 \
            2>&1 | tee -a "${log_file}"
    fi
    run_eval "${output_dir}" "${log_file}"
    echo "Completed ${name}: $(date)" | tee -a "${log_file}"
}

prepare_eval_keys

echo "MODEL_NAME=${MODEL_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "EXPERIMENTS=${EXPERIMENTS}"
echo "RUN_TRAIN=${RUN_TRAIN}, RUN_EVAL=${RUN_EVAL}"

if should_run alpaca_top100; then
    run_alpaca_top100
fi

if should_run aoa100 || should_run aoa10; then
    run_aoa100
fi

if should_run subliminal; then
    run_csv_experiment subliminal "${SUBLIMINAL_CSV}" "${BASE_OUTPUT_ROOT}/test_subliminal"
fi

if should_run alpaca_cleaned; then
    run_csv_experiment alpaca_cleaned "${ALPACA_CLEANED_CSV}" "${BASE_OUTPUT_ROOT}/test_alpaca_cleaned"
fi

if should_run overcautious; then
    run_csv_experiment overcautious "${OVERCAUTIOUS_CSV}" "${BASE_OUTPUT_ROOT}/test_overcautious"
fi

echo "All requested experiments completed: $(date)"
