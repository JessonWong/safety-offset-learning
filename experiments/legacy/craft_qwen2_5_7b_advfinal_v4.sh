#!/usr/bin/env bash
# Create a Qwen2.5-7B-Instruct-targeted AdvFinal dataset with AdvPrompter V4.
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
cd "$ROOT_DIR"

TARGET_CHECKPOINT="${TARGET_CHECKPOINT:-Qwen/Qwen2.5-7B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SHARD_SIZE="${SHARD_SIZE:-250}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-2}"
PROMPT_TOKENS="${PROMPT_TOKENS:-48}"
TOP_K="${TOP_K:-16}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
LAMBDA_VAL="${LAMBDA_VAL:-80}"
TARGET_DEVICE="${TARGET_DEVICE:-cuda:1}"
PROMPTER_DEVICE="${PROMPTER_DEVICE:-cuda:2}"
REWARD_DEVICE="${REWARD_DEVICE:-cuda:3}"
WILDGUARD_BATCH_SIZE="${WILDGUARD_BATCH_SIZE:-8}"

MODEL_SLUG="qwen2_5_7b_instruct"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/cache/advfinal_${MODEL_SLUG}_advprompter_v4}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DATASET="${OUTPUT_DATASET:-${ROOT_DIR}/data/advfinal_${MODEL_SLUG}_advprompter_v4_no_harmful_prompt_or_response.csv}"
TRAIN_CSV="${RUN_DIR}/reverse_eval_prompt_replacement_train_compat.csv"
JUDGE_DIR="${RUN_DIR}/harmfulness_eval_wildguard"
HARMFULNESS_RESULTS="${JUDGE_DIR}/harmfulness_results.csv"
FILTERED_CSV="${RUN_DIR}/reverse_eval_prompt_replacement_train_compat_no_harmful_instruction_or_response.csv"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1

echo "Started: $(date)"
echo "Target checkpoint: $TARGET_CHECKPOINT"
echo "Strategy: standalone_v4"
echo "Run directory: $RUN_DIR"
echo "Export dataset: $OUTPUT_DATASET"

if [[ ! -s "$TRAIN_CSV" ]]; then
    python -m safeoff.generation.dataset \
        --dataset-name yahma/alpaca-cleaned \
        --dataset-split train \
        --start-index 0 \
        --num-samples "$NUM_SAMPLES" \
        --shard-size "$SHARD_SIZE" \
        --attack-batch-size "$ATTACK_BATCH_SIZE" \
        --target-checkpoint "$TARGET_CHECKPOINT" \
        --target-device "$TARGET_DEVICE" \
        --prompter-device "$PROMPTER_DEVICE" \
        --strategy standalone_v4 \
        --prompt-tokens "$PROMPT_TOKENS" \
        --top-k "$TOP_K" \
        --num-beams "$NUM_BEAMS" \
        --num-chunks "$NUM_CHUNKS" \
        --lambda-val "$LAMBDA_VAL" \
        --refusal-target "I can't fulfill that request." \
        --reward-model allenai/wildguard \
        --reward-device "$REWARD_DEVICE" \
        --reward-batch-size "$WILDGUARD_BATCH_SIZE" \
        --output-dir "$OUTPUT_ROOT" \
        --run-dir "$RUN_DIR"
else
    echo "Reusing generated data: $TRAIN_CSV"
fi

if [[ ! -s "$HARMFULNESS_RESULTS" ]]; then
    CUDA_VISIBLE_DEVICES=3 python -m safeoff.evaluation.data_harmfulness \
        --input "$TRAIN_CSV" \
        --output-dir "$JUDGE_DIR" \
        --backend wildguard \
        --model-name allenai/wildguard \
        --batch-size "$WILDGUARD_BATCH_SIZE" \
        --device-map cuda:0 \
        --torch-dtype bfloat16 \
        --no-append-suffix
else
    echo "Reusing WildGuard results: $HARMFULNESS_RESULTS"
fi

if [[ ! -s "$FILTERED_CSV" ]]; then
    python -m safeoff.data.filter_harmfulness_results \
        --input-csv "$TRAIN_CSV" \
        --harmfulness-results "$HARMFULNESS_RESULTS" \
        --output-csv "$FILTERED_CSV" \
        --drop-prompt-harmful \
        --drop-response-harmful
else
    echo "Reusing filtered dataset: $FILTERED_CSV"
fi

cp "$FILTERED_CSV" "$OUTPUT_DATASET"
echo "Completed: $(date)"
echo "Filtered dataset: $OUTPUT_DATASET"
