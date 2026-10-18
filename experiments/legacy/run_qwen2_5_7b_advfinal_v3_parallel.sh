#!/usr/bin/env bash
# Generate a Qwen-targeted standalone-V3 dataset with two concurrent shards.
# Each shard needs a target and prompter Qwen-7B model, so a shard occupies two
# 24 GB GPUs. Two shard pairs therefore use all four GPUs.
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
cd "$ROOT_DIR"

TARGET_CHECKPOINT="Qwen/Qwen2.5-7B-Instruct"
NUM_SAMPLES=1000
SHARD_SIZE=250
ATTACK_BATCH_SIZE=2
PROMPT_TOKENS=48
TOP_K=16
NUM_BEAMS=4
NUM_CHUNKS=1
LAMBDA_VAL=80
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/cache/advfinal_qwen2_5_7b_instruct_advprompter_v3_parallel}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)}"
TRAIN_CSV="${RUN_DIR}/reverse_eval_prompt_replacement_train_compat.csv"
JUDGE_DIR="${RUN_DIR}/harmfulness_eval_wildguard"
HARMFULNESS_RESULTS="${JUDGE_DIR}/harmfulness_results.csv"
FILTERED_CSV="${RUN_DIR}/reverse_eval_prompt_replacement_train_compat_no_harmful_instruction_or_response.csv"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1

run_shard_once() {
    local start_index="$1"
    local target_device="$2"
    local prompter_device="$3"
    local batch_size="$4"
    local shard_dir="$RUN_DIR/shard_$(printf '%04d' "$start_index")"
    mkdir -p "$shard_dir"

    (
        cd "$ROOT_DIR"
        python -m safeoff.generation.replacement \
            --dataset-name yahma/alpaca-cleaned \
            --dataset-split train \
            --start-index "$start_index" \
            --num-samples "$SHARD_SIZE" \
            --attack-batch-size "$batch_size" \
            --target-checkpoint "$TARGET_CHECKPOINT" \
            --target-device "$target_device" \
            --prompter-device "$prompter_device" \
            --dtype float16 \
            --strategy standalone_v3 \
            --prompt-tokens "$PROMPT_TOKENS" \
            --response-max-new-tokens 96 \
            --search-response-max-new-tokens 96 \
            --top-k "$TOP_K" \
            --num-beams "$NUM_BEAMS" \
            --num-chunks "$NUM_CHUNKS" \
            --lambda-val "$LAMBDA_VAL" \
            --reward-lambda-val 10 \
            --repetition-penalty 1.1 \
            --candidate-temperature 0.8 \
            --beam-temperature 0.8 \
            --refusal-target "I can't fulfill that request." \
            --reward-model allenai/wildguard \
            --reward-device "$target_device" \
            --reward-batch-size 32 \
            --reward-max-length 512 \
            --refusal-prefixes-path "$ROOT_DIR/src/safeoff/resources/refusal_prefixes.csv" \
            --output-dir "$shard_dir" \
            --seed 2025 \
            --no-standalone-v3-multi-target-selection
    ) >"$shard_dir/stdout_stderr_bs${batch_size}.log" 2>&1
}

run_shard() {
    local start_index="$1"
    local target_device="$2"
    local prompter_device="$3"
    echo "Starting shard ${start_index} on ${target_device}/${prompter_device} with batch size ${ATTACK_BATCH_SIZE}"
    if ! run_shard_once "$start_index" "$target_device" "$prompter_device" "$ATTACK_BATCH_SIZE"; then
        echo "Shard ${start_index} failed at batch size ${ATTACK_BATCH_SIZE}; retrying at batch size 1"
        run_shard_once "$start_index" "$target_device" "$prompter_device" 1
    fi
    echo "Completed shard ${start_index}"
}

run_pair() {
    local first_start="$1"
    local second_start="$2"
    run_shard "$first_start" cuda:0 cuda:1 &
    local first_pid=$!
    run_shard "$second_start" cuda:2 cuda:3 &
    local second_pid=$!
    wait "$first_pid"
    wait "$second_pid"
}

echo "Started: $(date)"
echo "Target checkpoint: $TARGET_CHECKPOINT"
echo "Strategy: standalone_v3, no_mts"
echo "Run directory: $RUN_DIR"

run_pair 0 250
run_pair 500 750

# Reuse the standard generator's merge implementation after all four shard
# directories have completed; it detects each completed shard and only merges.
python -m safeoff.generation.dataset \
    --dataset-name yahma/alpaca-cleaned \
    --dataset-split train \
    --start-index 0 \
    --num-samples "$NUM_SAMPLES" \
    --shard-size "$SHARD_SIZE" \
    --attack-batch-size "$ATTACK_BATCH_SIZE" \
    --target-checkpoint "$TARGET_CHECKPOINT" \
    --target-device cuda:0 \
    --prompter-device cuda:1 \
    --strategy standalone_v3 \
    --prompt-tokens "$PROMPT_TOKENS" \
    --top-k "$TOP_K" \
    --num-beams "$NUM_BEAMS" \
    --num-chunks "$NUM_CHUNKS" \
    --lambda-val "$LAMBDA_VAL" \
    --refusal-target "I can't fulfill that request." \
    --output-dir "$OUTPUT_ROOT" \
    --run-dir "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 python -m safeoff.evaluation.data_harmfulness \
    --input "$TRAIN_CSV" \
    --output-dir "$JUDGE_DIR" \
    --backend wildguard \
    --model-name allenai/wildguard \
    --batch-size 8 \
    --device-map cuda:0 \
    --torch-dtype bfloat16 \
    --no-append-suffix

python -m safeoff.data.filter_harmfulness_results \
    --input-csv "$TRAIN_CSV" \
    --harmfulness-results "$HARMFULNESS_RESULTS" \
    --output-csv "$FILTERED_CSV" \
    --drop-prompt-harmful \
    --drop-response-harmful

echo "Completed: $(date)"
echo "Raw dataset: $TRAIN_CSV"
echo "Filtered dataset: $FILTERED_CSV"
