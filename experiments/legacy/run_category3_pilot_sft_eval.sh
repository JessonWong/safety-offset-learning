#!/usr/bin/env bash
set -euo pipefail

# Fine-tune with ordinary SFT on all 100 Category-3 pilot rows, then evaluate
# the latest checkpoint on HarmBench and SORRY-Bench. Credentials are read only
# from the calling environment; this script never reads test.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}"
cd "$ROOT_DIR"

PILOT_ROOT="${PILOT_ROOT:-${ROOT_DIR}/cache/latharm_indirect_preference_v1/pilot_category3_100}"
TRAIN_CSV="${TRAIN_CSV:-${PILOT_ROOT}/search_direct/20260720_072356/reverse_eval_prompt_replacement_train_compat.csv}"
RUN_ROOT="${RUN_ROOT:-${PILOT_ROOT}/sft_eval}"
FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-${ROOT_DIR}/finetune_model/LATharm/latharm_indirect_category3_sft_pilot}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
CURRENT_MODEL="${CURRENT_MODEL:-${ROOT_DIR}/finetune_model/LATharm/test_advfinal/Llama-3.2-3B-Instruct/model/checkpoint-1000}"
MODEL_PATH_NAME="${MODEL_PATH_NAME:-$(basename "${BASE_MODEL%/}")}"
MODEL_CHECKPOINT_ROOT="${FINETUNE_OUTPUT_DIR}/${MODEL_PATH_NAME}/model"
# src/safeoff/training/sft.py names its output from --model_path_name; keep it aligned with
# the evaluator's base-model name even when training starts from an adapter.

FINETUNE_GPU="${FINETUNE_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-0}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4}"
FINETUNE_GRADIENT_ACCUMULATION_STEPS="${FINETUNE_GRADIENT_ACCUMULATION_STEPS:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
FINETUNE_MAX_STEPS="${FINETUNE_MAX_STEPS:--1}"
FINETUNE_SAVE_STEPS="${FINETUNE_SAVE_STEPS:-25}"
FINETUNE_LEARNING_RATE="${FINETUNE_LEARNING_RATE:-1e-4}"
FINETUNE_MAX_SEQUENCE_LENGTH="${FINETUNE_MAX_SEQUENCE_LENGTH:-768}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_ATTACK_MAX_SAMPLES="${EVAL_ATTACK_MAX_SAMPLES:-}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
HARMFUL_JUDGE_MODEL="${HARMFUL_JUDGE_MODEL:-gemini}"
SORRYBENCH_PATH="${SORRYBENCH_PATH:-}"
SORRYBENCH_MUTATION="${SORRYBENCH_MUTATION:-}"

RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "$RUN_ROOT" "${RUN_ROOT}/logs" "$FINETUNE_OUTPUT_DIR"
[[ -s "$TRAIN_CSV" ]] || { echo "Missing SFT training CSV: $TRAIN_CSV" >&2; exit 1; }
[[ -d "$CURRENT_MODEL" ]] || { echo "Missing starting checkpoint: $CURRENT_MODEL" >&2; exit 1; }
if [[ "$RUN_EVAL" == 1 && -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]]; then
    echo "Export GEMINI_API_KEY or GOOGLE_API_KEY before evaluation." >&2
    exit 1
fi

TRAIN_COUNT="$(python - "$TRAIN_CSV" <<'PY'
import csv
import sys
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("SFT CSV is empty")
required = {"instruction", "alpaca_reference", "suffix"}
missing = required - set(rows[0])
if missing:
    raise SystemExit(f"SFT CSV is missing fields: {sorted(missing)}")
print(len(rows))
PY
)"
echo "[data] Using ${TRAIN_COUNT} SFT rows from ${TRAIN_CSV}"

if [[ "$RUN_FINETUNE" == 1 ]]; then
    echo "[finetune] Ordinary SFT from ${CURRENT_MODEL}"
    CUDA_VISIBLE_DEVICES="$FINETUNE_GPU" \
    WANDB_DISABLED="${WANDB_DISABLED:-true}" \
    WANDB_MODE="${WANDB_MODE:-disabled}" \
    python -u -m safeoff.training.sft \
        --base_model_path "$CURRENT_MODEL" \
        --model_path_name "$MODEL_PATH_NAME" \
        --output_dir "$FINETUNE_OUTPUT_DIR" \
        --utility_dataset_config utility_dataset_config_alpaca \
        --utility_training_num "$TRAIN_COUNT" \
        --batch_size "$FINETUNE_BATCH_SIZE" \
        --gradient_accumulation_steps "$FINETUNE_GRADIENT_ACCUMULATION_STEPS" \
        --epochs "$FINETUNE_EPOCHS" \
        --max_steps "$FINETUNE_MAX_STEPS" \
        --max_sequence_length "$FINETUNE_MAX_SEQUENCE_LENGTH" \
        --fail_on_overlength_examples 1 \
        --save_steps "$FINETUNE_SAVE_STEPS" \
        --save_checkpoint_zero 1 \
        --learning_rate "$FINETUNE_LEARNING_RATE" \
        --poison_training_num 0 \
        --identity_shift_num 0 \
        --use_lora 1 \
        --filter 0 \
        --save_grads_every 10000 \
        --advprompt_suffix "$TRAIN_CSV" \
        --advprompt_use_suffix 0 \
        --advprompt_training_num "$TRAIN_COUNT" \
        --advprompt_clean_training_num 0 \
        --backdoor 0 \
        --augment_utility_with_lat 0 \
        2>&1 | tee "${RUN_ROOT}/logs/finetune_sft.log"
fi

LATEST_STEP="$(python - "$MODEL_CHECKPOINT_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
steps = [int(p.name.split("-")[-1]) for p in root.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()]
if not steps:
    raise SystemExit(f"No checkpoints found under {root}")
print(max(steps))
PY
)"
echo "[finetune] Latest checkpoint: ${MODEL_CHECKPOINT_ROOT}/checkpoint-${LATEST_STEP}"

if [[ "$RUN_EVAL" == 1 ]]; then
    EVAL_ARGS=(
        --repo_root "$ROOT_DIR"
        --output_dir "$FINETUNE_OUTPUT_DIR"
        --base_model_path "$BASE_MODEL"
        --batch_size "$EVAL_BATCH_SIZE"
        --use_lora 1
        --judge_workers "$JUDGE_WORKERS"
        --metric_mode attack
        --eval_harmful_score 1
        --eval_harmful_info_score 1
        --harmful_judge_model "$HARMFUL_JUDGE_MODEL"
        --attack_benchmarks harmbench,sorrybench
        --checkpoint_step_interval "$LATEST_STEP"
        --min_checkpoint_step "$LATEST_STEP"
        --summary_json "${RUN_ROOT}/harmbench_sorrybench_metrics.json"
        --summary_csv "${RUN_ROOT}/harmbench_sorrybench_metrics.csv"
        --plot_path "${RUN_ROOT}/harmbench_sorrybench_metrics.png"
        --force
    )
    [[ -n "$EVAL_ATTACK_MAX_SAMPLES" ]] && EVAL_ARGS+=(--attack_max_samples "$EVAL_ATTACK_MAX_SAMPLES")
    [[ -n "$SORRYBENCH_PATH" ]] && EVAL_ARGS+=(--sorrybench_path "$SORRYBENCH_PATH")
    [[ -n "$SORRYBENCH_MUTATION" ]] && EVAL_ARGS+=(--sorrybench_mutation "$SORRYBENCH_MUTATION")

    echo "[eval] Evaluating checkpoint-0 and checkpoint-${LATEST_STEP}"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python -u -m safeoff.evaluation.curves "${EVAL_ARGS[@]}" \
        2>&1 | tee "${RUN_ROOT}/logs/evaluate_harmbench_sorrybench.log"
fi

echo "[done] SFT checkpoint: ${MODEL_CHECKPOINT_ROOT}/checkpoint-${LATEST_STEP}"
echo "[done] Metrics: ${RUN_ROOT}/harmbench_sorrybench_metrics.csv"
