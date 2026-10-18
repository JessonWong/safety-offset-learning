#!/usr/bin/env bash
set -euo pipefail

# Validate, fine-tune, and evaluate the completed one-target Category-3 pilot.
# Credentials are intentionally read only from the calling environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}"
cd "$ROOT_DIR"

PILOT_ROOT="${PILOT_ROOT:-${ROOT_DIR}/cache/latharm_indirect_preference_v1/pilot_category3_100}"
SEARCH_RESULTS="${SEARCH_RESULTS:-${PILOT_ROOT}/search_direct/20260720_072356/reverse_eval_prompt_replacement_results.csv}"
RUN_ROOT="${RUN_ROOT:-${PILOT_ROOT}/finetune_eval}"
PREFERENCE_CSV="${PREFERENCE_CSV:-${RUN_ROOT}/indirect_preferences.csv}"
PREFERENCE_AUDIT="${PREFERENCE_AUDIT:-${RUN_ROOT}/indirect_preferences.audit.csv}"
PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-${RUN_ROOT}/indirect_preferences.manifest.json}"
PREFERENCE_JUDGE_CACHE="${PREFERENCE_JUDGE_CACHE:-${RUN_ROOT}/indirect_preferences.judge_cache.json}"

CURRENT_MODEL="${CURRENT_MODEL:-${ROOT_DIR}/finetune_model/LATharm/test_advfinal/Llama-3.2-3B-Instruct/model/checkpoint-1000}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_PATH_NAME="${MODEL_PATH_NAME:-$(basename "${BASE_MODEL%/}")}"
FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-${ROOT_DIR}/finetune_model/LATharm/latharm_indirect_preference_category3_pilot}"
MODEL_CHECKPOINT_ROOT="${FINETUNE_OUTPUT_DIR}/${MODEL_PATH_NAME}/model"

INDIRECT_JUDGE_MODEL="${INDIRECT_JUDGE_MODEL:-${GEMINI_JUDGE_MODEL:-gemini-3-flash-preview}}"
HARMFUL_JUDGE_MODEL="${HARMFUL_JUDGE_MODEL:-gemini}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-64}"
MAX_CHOSEN_HARMFUL_INFO_SCORE="${MAX_CHOSEN_HARMFUL_INFO_SCORE:-2.0}"
MIN_LOSS_IMPROVEMENT="${MIN_LOSS_IMPROVEMENT:-0.0}"

FINETUNE_GPU="${FINETUNE_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-0}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-2}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2.0}"
FINETUNE_LEARNING_RATE="${FINETUNE_LEARNING_RATE:-5e-6}"
FINETUNE_BETA="${FINETUNE_BETA:-0.1}"
FINETUNE_SAVE_STEPS="${FINETUNE_SAVE_STEPS:-25}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_ATTACK_MAX_SAMPLES="${EVAL_ATTACK_MAX_SAMPLES:-}"
SORRYBENCH_PATH="${SORRYBENCH_PATH:-}"
SORRYBENCH_MUTATION="${SORRYBENCH_MUTATION:-}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "$RUN_ROOT" "${FINETUNE_OUTPUT_DIR}" "${RUN_ROOT}/logs"

[[ -s "$SEARCH_RESULTS" ]] || {
    echo "Missing pilot search results: $SEARCH_RESULTS" >&2
    exit 1
}
[[ -d "$CURRENT_MODEL" ]] || {
    echo "Missing current model checkpoint: $CURRENT_MODEL" >&2
    exit 1
}
[[ -n "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]] || {
    echo "Export GEMINI_API_KEY or GOOGLE_API_KEY in this shell before running." >&2
    exit 1
}

if [[ "$RUN_VALIDATE" == 1 ]]; then
echo "[validate] Judging actual pilot responses with ${INDIRECT_JUDGE_MODEL}"
python -u -m safeoff.data.build_indirect_preference_dataset \
    --input "$SEARCH_RESULTS" \
    --output "$PREFERENCE_CSV" \
    --audit-output "$PREFERENCE_AUDIT" \
    --manifest-output "$PREFERENCE_MANIFEST" \
    --judge-cache "$PREFERENCE_JUDGE_CACHE" \
    --judge-model "$INDIRECT_JUDGE_MODEL" \
    --judge-workers "$JUDGE_WORKERS" \
    --judge-batch-size "$JUDGE_BATCH_SIZE" \
    --min-loss-improvement "$MIN_LOSS_IMPROVEMENT" \
    --max-chosen-harmful-info-score "$MAX_CHOSEN_HARMFUL_INFO_SCORE" \
    --require-standalone-valid \
    2>&1 | tee "${RUN_ROOT}/logs/validate_actual_responses.log"

fi

PAIR_COUNT="$(python - "$PREFERENCE_CSV" <<'PY'
import csv
import sys
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"
[[ "$PAIR_COUNT" -gt 0 ]] || {
    echo "Validation retained zero preference pairs; refusing to fine-tune." >&2
    exit 2
}
echo "[validate] Retained ${PAIR_COUNT} preference pairs"

if [[ "$RUN_FINETUNE" == 1 ]]; then
echo "[finetune] Fine-tuning from ${CURRENT_MODEL}"
CUDA_VISIBLE_DEVICES="$FINETUNE_GPU" \
WANDB_DISABLED="${WANDB_DISABLED:-true}" \
WANDB_MODE="${WANDB_MODE:-disabled}" \
python -u -m safeoff.training.preference \
    --dataset_path "$PREFERENCE_CSV" \
    --base_model_path "$CURRENT_MODEL" \
    --model_path_name "$MODEL_PATH_NAME" \
    --output_dir "$FINETUNE_OUTPUT_DIR" \
    --max_chosen_harmful_info_score "$MAX_CHOSEN_HARMFUL_INFO_SCORE" \
    --beta "$FINETUNE_BETA" \
    --max_sequence_length 768 \
    --fail_on_overlength_examples 1 \
    --use_lora 1 \
    --learning_rate "$FINETUNE_LEARNING_RATE" \
    --batch_size "$FINETUNE_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --epochs "$FINETUNE_EPOCHS" \
    --save_steps "$FINETUNE_SAVE_STEPS" \
    --save_checkpoint_zero 1 \
    2>&1 | tee "${RUN_ROOT}/logs/finetune_preferences.log"
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
    --summary_json "${RUN_ROOT}/harmbench_sorrybench_metrics.json"
    --summary_csv "${RUN_ROOT}/harmbench_sorrybench_metrics.csv"
    --plot_path "${RUN_ROOT}/harmbench_sorrybench_metrics.png"
    --force
)
[[ -n "$EVAL_ATTACK_MAX_SAMPLES" ]] && EVAL_ARGS+=(--attack_max_samples "$EVAL_ATTACK_MAX_SAMPLES")
[[ -n "$SORRYBENCH_PATH" ]] && EVAL_ARGS+=(--sorrybench_path "$SORRYBENCH_PATH")
[[ -n "$SORRYBENCH_MUTATION" ]] && EVAL_ARGS+=(--sorrybench_mutation "$SORRYBENCH_MUTATION")

echo "[eval] Evaluating checkpoint-0 and checkpoint-${LATEST_STEP} on HarmBench and SORRY-Bench"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python -u -m safeoff.evaluation.curves "${EVAL_ARGS[@]}" \
    2>&1 | tee "${RUN_ROOT}/logs/evaluate_harmbench_sorrybench.log"

echo "[done] Preference dataset: $PREFERENCE_CSV"
echo "[done] Checkpoint: ${MODEL_CHECKPOINT_ROOT}/checkpoint-${LATEST_STEP}"
echo "[done] Metrics: ${RUN_ROOT}/harmbench_sorrybench_metrics.csv"
fi
