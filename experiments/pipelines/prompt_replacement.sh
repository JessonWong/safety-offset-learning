#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"

START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SHARD_SIZE="${SHARD_SIZE:-250}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-2}"

TARGET_CHECKPOINT="${TARGET_CHECKPOINT:-meta-llama/Llama-3.1-8B-Instruct}"
TARGET_DEVICE="${TARGET_DEVICE:-cuda:0}"
PROMPTER_DEVICE="${PROMPTER_DEVICE:-cuda:1}"
DEFAULT_REFUSAL_TARGET="$(printf 'I can\047t fulfill that request.')"
REFUSAL_TARGET="${REFUSAL_TARGET:-${DEFAULT_REFUSAL_TARGET}}"
PROMPT_TOKENS="${PROMPT_TOKENS:-48}"
TOP_K="${TOP_K:-16}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
LAMBDA_VAL="${LAMBDA_VAL:-80}"
USE_STANDALONE_V3_MULTI_TARGET_SELECTION="${USE_STANDALONE_V3_MULTI_TARGET_SELECTION:-0}"

WILDGUARD_MODEL="${WILDGUARD_MODEL:-allenai/wildguard}"
WILDGUARD_BATCH_SIZE="${WILDGUARD_BATCH_SIZE:-8}"
WILDGUARD_DEVICE_MAP="${WILDGUARD_DEVICE_MAP:-auto}"
WILDGUARD_TORCH_DTYPE="${WILDGUARD_TORCH_DTYPE:-bfloat16}"

RUN_PROMPT_REPLACEMENT="${RUN_PROMPT_REPLACEMENT:-1}"
RUN_WILDGUARD="${RUN_WILDGUARD:-1}"
RUN_FILTER="${RUN_FILTER:-1}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
AUTO_REUSE_GENERATED_DATA="${AUTO_REUSE_GENERATED_DATA:-1}"

FINETUNE_MODEL_NAME="${FINETUNE_MODEL_NAME:-${TARGET_CHECKPOINT}}"
FINETUNE_CUDA_VISIBLE_DEVICES="${FINETUNE_CUDA_VISIBLE_DEVICES:-0}"
TOTAL_TRAINING_NUM="${TOTAL_TRAINING_NUM:-${NUM_SAMPLES}}"

FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4}"
FINETUNE_EVAL_BATCH_SIZE="${FINETUNE_EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
FINETUNE_MAX_STEPS="${FINETUNE_MAX_STEPS:-1000}"
FINETUNE_SAVE_STEPS="${FINETUNE_SAVE_STEPS:-25}"
FINETUNE_LEARNING_RATE="${FINETUNE_LEARNING_RATE:-1e-4}"
SAVE_CHECKPOINT_ZERO="${SAVE_CHECKPOINT_ZERO:-1}"
CHECKPOINT_STAGE1_END="${CHECKPOINT_STAGE1_END:-25}"
CHECKPOINT_STAGE1_INTERVAL="${CHECKPOINT_STAGE1_INTERVAL:-1}"
CHECKPOINT_STAGE2_END="${CHECKPOINT_STAGE2_END:-${FINETUNE_MAX_STEPS}}"
CHECKPOINT_STAGE2_INTERVAL="${CHECKPOINT_STAGE2_INTERVAL:-25}"
EVAL_CHECKPOINT_STEP_INTERVAL="${EVAL_CHECKPOINT_STEP_INTERVAL:-25}"
HARMFUL_JUDGE_MODEL="${HARMFUL_JUDGE_MODEL:-gemini}"
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
ADVPROMPT_CLEAN_CSV="${ADVPROMPT_CLEAN_CSV:-}"

slugify() {
    python - "$1" <<'PY'
import re
import sys

value = sys.argv[1].strip().lower()
value = re.sub(r"[^a-z0-9]+", "_", value)
value = re.sub(r"_+", "_", value).strip("_")
print(value or "unset")
PY
}

find_existing_generated_run_dir() {
    python - "$1" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
target_name = "reverse_eval_prompt_replacement_train_compat.csv"
candidates = []

if root.is_dir():
    for path in root.glob(f"*/{target_name}"):
        try:
            if path.stat().st_size <= 0:
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                if not any(csv.DictReader(handle)):
                    continue
            candidates.append((path.stat().st_mtime, str(path.parent.resolve())))
        except Exception:
            continue

if candidates:
    candidates.sort(reverse=True)
    print(candidates[0][1])
PY
}

TARGET_MODEL_SLUG="${TARGET_MODEL_SLUG:-$(slugify "${TARGET_CHECKPOINT}")}"
FINETUNE_MODEL_SLUG="${FINETUNE_MODEL_SLUG:-$(slugify "${FINETUNE_MODEL_NAME}")}"
REFUSAL_TARGET_SLUG="${REFUSAL_TARGET_SLUG:-$(slugify "${REFUSAL_TARGET}")}"
MTS_TAG="no_mts"
if [[ "${USE_STANDALONE_V3_MULTI_TARGET_SELECTION}" == "1" ]]; then
    MTS_TAG="mts"
fi

PIPELINE_NAME="${PIPELINE_NAME:-reverse_eval_prompt_replacement_${TARGET_MODEL_SLUG}_start${START_INDEX}_n${NUM_SAMPLES}_pt${PROMPT_TOKENS}_tk${TOP_K}_b${NUM_BEAMS}_c${NUM_CHUNKS}_l${LAMBDA_VAL}_target_${REFUSAL_TARGET_SLUG}_${MTS_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/cache/${PIPELINE_NAME}}"
RUN_DIR_WAS_SET=0
if [[ -n "${RUN_DIR:-}" ]]; then
    RUN_DIR_WAS_SET=1
fi
REUSED_GENERATED_RUN_DIR=""
if [[ "${RUN_DIR_WAS_SET}" == "0" && "${AUTO_REUSE_GENERATED_DATA}" == "1" && "${RUN_PROMPT_REPLACEMENT}" == "1" ]]; then
    REUSED_GENERATED_RUN_DIR="$(find_existing_generated_run_dir "${OUTPUT_ROOT}")"
fi
if [[ -n "${REUSED_GENERATED_RUN_DIR}" ]]; then
    RUN_DIR="${REUSED_GENERATED_RUN_DIR}"
else
    RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)_wildguard_filter_ft_${FINETUNE_MODEL_SLUG}}"
fi
FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-${RUN_DIR}/finetune_${FINETUNE_MODEL_SLUG}_total${TOTAL_TRAINING_NUM}}"
FINETUNE_LOG_FILE="${FINETUNE_LOG_FILE:-${RUN_DIR}/finetune_eval_${FINETUNE_MODEL_SLUG}.log}"
TRAIN_CSV="${RUN_DIR}/reverse_eval_prompt_replacement_train_compat.csv"
JUDGE_DIR="${RUN_DIR}/harmfulness_eval_wildguard"
HARMFULNESS_RESULTS_CSV="${JUDGE_DIR}/harmfulness_results.csv"
FILTERED_CSV="${RUN_DIR}/reverse_eval_prompt_replacement_train_compat_no_harmful_instruction_or_response.csv"

mkdir -p "${RUN_DIR}"
cd "${ROOT_DIR}"

exec > >(tee -a "${RUN_DIR}/pipeline.log") 2>&1

echo "Pipeline name: ${PIPELINE_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Started: $(date)"
echo "Stages: prompt_replacement=${RUN_PROMPT_REPLACEMENT}, wildguard=${RUN_WILDGUARD}, filter=${RUN_FILTER}, finetune=${RUN_FINETUNE}, eval=${RUN_EVAL}"
echo "Target checkpoint: ${TARGET_CHECKPOINT}"
echo "Finetune model: ${FINETUNE_MODEL_NAME}"
echo "Prompt config: pt=${PROMPT_TOKENS}, tk=${TOP_K}, beams=${NUM_BEAMS}, chunks=${NUM_CHUNKS}, lambda=${LAMBDA_VAL}, ${MTS_TAG}"
if [[ -n "${REUSED_GENERATED_RUN_DIR}" ]]; then
  echo "Reusing generated data from existing run dir: ${REUSED_GENERATED_RUN_DIR}"
fi


count_csv_rows() {
    python - "$1" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
}

csv_has_rows() {
    local path="$1"
    local rows
    if [[ ! -f "${path}" ]]; then
        return 1
    fi
    rows="$(count_csv_rows "${path}" 2>/dev/null || echo 0)"
    [[ "${rows}" =~ ^[0-9]+$ ]] && (( rows > 0 ))
}

if [[ "${RUN_PROMPT_REPLACEMENT}" == "1" ]]; then
  if csv_has_rows "${TRAIN_CSV}"; then
    echo "Skipping prompt replacement; found generated TRAIN_CSV: ${TRAIN_CSV}"
  else
    mts_args=()
    if [[ "${USE_STANDALONE_V3_MULTI_TARGET_SELECTION}" != "1" ]]; then
      mts_args=(--no-standalone-v3-multi-target-selection)
    fi

    python -m safeoff.generation.dataset \
      --start-index "${START_INDEX}" \
      --num-samples "${NUM_SAMPLES}" \
      --shard-size "${SHARD_SIZE}" \
      --attack-batch-size "${ATTACK_BATCH_SIZE}" \
      --target-checkpoint "${TARGET_CHECKPOINT}" \
      --target-device "${TARGET_DEVICE}" \
      --prompter-device "${PROMPTER_DEVICE}" \
      --prompt-tokens "${PROMPT_TOKENS}" \
      --top-k "${TOP_K}" \
      --num-beams "${NUM_BEAMS}" \
      --num-chunks "${NUM_CHUNKS}" \
      --lambda-val "${LAMBDA_VAL}" \
      --refusal-target "${REFUSAL_TARGET}" \
      "${mts_args[@]}" \
      --output-dir "${OUTPUT_ROOT}" \
      --run-dir "${RUN_DIR}"
  fi
fi

if [[ ! -f "${TRAIN_CSV}" ]]; then
  echo "TRAIN_CSV not found: ${TRAIN_CSV}" >&2
  exit 1
fi

if [[ "${RUN_WILDGUARD}" == "1" ]]; then
  if [[ -s "${HARMFULNESS_RESULTS_CSV}" ]]; then
    echo "Skipping WildGuard; found harmfulness results: ${HARMFULNESS_RESULTS_CSV}"
  else
    python -m safeoff.evaluation.data_harmfulness \
      --input "${TRAIN_CSV}" \
      --output-dir "${JUDGE_DIR}" \
      --backend wildguard \
      --model-name "${WILDGUARD_MODEL}" \
      --batch-size "${WILDGUARD_BATCH_SIZE}" \
      --device-map "${WILDGUARD_DEVICE_MAP}" \
      --torch-dtype "${WILDGUARD_TORCH_DTYPE}" \
      --no-append-suffix
  fi
fi

if [[ "${RUN_FILTER}" == "1" ]]; then
  if csv_has_rows "${FILTERED_CSV}"; then
    echo "Skipping filter; found filtered CSV: ${FILTERED_CSV}"
  else
    python -m safeoff.data.filter_harmfulness_results \
      --input-csv "${TRAIN_CSV}" \
      --harmfulness-results "${HARMFULNESS_RESULTS_CSV}" \
      --output-csv "${FILTERED_CSV}" \
      --drop-prompt-harmful \
      --drop-response-harmful
  fi
fi

if [[ ! -f "${FILTERED_CSV}" ]]; then
  echo "FILTERED_CSV not found: ${FILTERED_CSV}" >&2
  exit 1
fi

ADVPROMPT_TRAINING_NUM="$(count_csv_rows "${FILTERED_CSV}")"
if (( ADVPROMPT_TRAINING_NUM <= 0 )); then
  echo "No rows found in ${FILTERED_CSV}" >&2
  exit 1
fi
if (( ADVPROMPT_TRAINING_NUM > TOTAL_TRAINING_NUM )); then
  echo "Filtered rows (${ADVPROMPT_TRAINING_NUM}) exceed TOTAL_TRAINING_NUM (${TOTAL_TRAINING_NUM})." >&2
  exit 1
fi
ADVPROMPT_CLEAN_TRAINING_NUM=$((TOTAL_TRAINING_NUM - ADVPROMPT_TRAINING_NUM))

echo "Filtered CSV: ${FILTERED_CSV}"
echo "ADVPROMPT_TRAINING_NUM=${ADVPROMPT_TRAINING_NUM}"
echo "ADVPROMPT_CLEAN_TRAINING_NUM=${ADVPROMPT_CLEAN_TRAINING_NUM}"
echo "FINETUNE_OUTPUT_DIR=${FINETUNE_OUTPUT_DIR}"

if [[ "${RUN_EVAL}" == "1" ]]; then



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
fi

clean_csv_args=()
if [[ -n "${ADVPROMPT_CLEAN_CSV}" ]]; then
  clean_csv_args=(--advprompt_clean_csv "${ADVPROMPT_CLEAN_CSV}")
fi

mkdir -p "$(dirname "${FINETUNE_LOG_FILE}")"
: > "${FINETUNE_LOG_FILE}"

if [[ "${RUN_FINETUNE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${FINETUNE_CUDA_VISIBLE_DEVICES}" \
  WANDB_DISABLED="${WANDB_DISABLED:-true}" \
  WANDB_MODE="${WANDB_MODE:-disabled}" \
  python -u -m safeoff.training.sft \
    --base_model_path "${FINETUNE_MODEL_NAME}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --use_cache 0 \
    --output_dir "${FINETUNE_OUTPUT_DIR}" \
    --utility_dataset_config utility_dataset_config_alpaca \
    --batch_size "${FINETUNE_BATCH_SIZE}" \
    --epochs "${FINETUNE_EPOCHS}" \
    --max_steps "${FINETUNE_MAX_STEPS}" \
    --save_steps "${FINETUNE_SAVE_STEPS}" \
    --save_checkpoint_zero "${SAVE_CHECKPOINT_ZERO}" \
    --checkpoint_stage1_end "${CHECKPOINT_STAGE1_END}" \
    --checkpoint_stage1_interval "${CHECKPOINT_STAGE1_INTERVAL}" \
    --checkpoint_stage2_end "${CHECKPOINT_STAGE2_END}" \
    --checkpoint_stage2_interval "${CHECKPOINT_STAGE2_INTERVAL}" \
    --learning_rate "${FINETUNE_LEARNING_RATE}" \
    --poison_training_num 0 \
    --identity_shift_num 0 \
    --utility_training_num "${TOTAL_TRAINING_NUM}" \
    --use_lora 1 \
    --filter 0 \
    --save_grads_every 10000 \
    --advprompt_suffix "${FILTERED_CSV}" \
    --advprompt_use_suffix 0 \
    --advprompt_training_num "${ADVPROMPT_TRAINING_NUM}" \
    --advprompt_clean_training_num "${ADVPROMPT_CLEAN_TRAINING_NUM}" \
    "${clean_csv_args[@]}" \
    --print_loss_tokens_max_examples 4 \
    --print_loss_tokens 1 \
    --backdoor 0 \
    --augment_utility_with_lat 0 \
    2>&1 | tee -a "${FINETUNE_LOG_FILE}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${FINETUNE_CUDA_VISIBLE_DEVICES}" \
  WANDB_DISABLED="${WANDB_DISABLED:-true}" \
  WANDB_MODE="${WANDB_MODE:-disabled}" \
  python -m safeoff.evaluation.curves \
    --output_dir "${FINETUNE_OUTPUT_DIR}" \
    --base_model_path "${FINETUNE_MODEL_NAME}" \
    --batch_size "${FINETUNE_EVAL_BATCH_SIZE}" \
    --use_lora 1 \
    --metric_mode attack \
    --eval_harmful_score 1 \
    --harmful_judge_model "${HARMFUL_JUDGE_MODEL}" \
    --checkpoint_step_interval "${EVAL_CHECKPOINT_STEP_INTERVAL}" \
    --judge_workers "${JUDGE_WORKERS}" \
    2>&1 | tee -a "${FINETUNE_LOG_FILE}"
fi

echo "Completed: $(date)"
echo "Filtered CSV: ${FILTERED_CSV}"
echo "Finetune output dir: ${FINETUNE_OUTPUT_DIR}"
