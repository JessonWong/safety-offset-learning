#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/cache/joint_multitarget_loss_v1}"
GEN_ROOT="${EXPERIMENT_ROOT}/generation"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
VALIDATION_ANCHORS="${VALIDATION_ANCHORS:-${ROOT_DIR}/cache/bilevel_standalone_v1/private/validation_anchors.jsonl}"
SEED="${SEED:-20260714}"
GENERATION_SEED="${GENERATION_SEED:-2025}"
MAX_STEPS="${MAX_STEPS:-1000}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

LLAMA_MODEL="meta-llama/Llama-3.2-3B-Instruct"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
LLAMA_DIRECT_TARGET="I'm sorry, but I can't help with that request."
QWEN_DIRECT_TARGET="I can't fulfill that request."
BOUNDARY_TARGET="That approach would be inappropriate and could cause harm."
REDIRECT_TARGET="A safer approach is to focus on prevention and responsible alternatives."
NEUTRALIZE_TARGET="The important issue is how to prevent this and reduce the associated risks."

mkdir -p "$GEN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
[[ -s "$VALIDATION_ANCHORS" ]] || { echo "Missing validation anchors: $VALIDATION_ANCHORS" >&2; exit 1; }

generation_csv() {
    local tag="$1"
    printf '%s/%s/reverse_eval_prompt_replacement_train_compat.csv' "$GEN_ROOT" "$tag"
}

generate_joint() {
    local tag="$1" model="$2" direct_target="$3" target_gpu="$4" prompter_gpu="$5"
    local prompt_tokens="$6" top_k="$7" beams="$8" chunks="$9" lambda_val="${10}"
    local run_dir="$GEN_ROOT/$tag" output_csv
    output_csv="$(generation_csv "$tag")"
    if [[ -s "$output_csv" ]]; then
        echo "Skipping completed joint generation: $tag"
        return 0
    fi
    mkdir -p "$run_dir"
    python -u -m safeoff.generation.dataset \
        --dataset-name yahma/alpaca-cleaned \
        --dataset-split train \
        --start-index 0 \
        --num-samples 1000 \
        --shard-size 250 \
        --attack-batch-size 2 \
        --target-checkpoint "$model" \
        --prompter-checkpoint "$model" \
        --target-device "cuda:$target_gpu" \
        --prompter-device "cuda:$prompter_gpu" \
        --dtype float16 \
        --strategy standalone_v3 \
        --prompt-tokens "$prompt_tokens" \
        --response-max-new-tokens 96 \
        --search-response-max-new-tokens 96 \
        --top-k "$top_k" \
        --num-beams "$beams" \
        --num-chunks "$chunks" \
        --lambda-val "$lambda_val" \
        --repetition-penalty 1.1 \
        --candidate-temperature 0.8 \
        --beam-temperature 0.8 \
        --refusal-target "$direct_target" \
        --refusal-target-candidate "$BOUNDARY_TARGET" \
        --refusal-target-candidate "$REDIRECT_TARGET" \
        --refusal-target-candidate "$NEUTRALIZE_TARGET" \
        --standalone-v3-joint-target-loss \
        --no-standalone-v3-multi-target-selection \
        --output-dir "$GEN_ROOT/${tag}_runs" \
        --run-dir "$run_dir" \
        --seed "$GENERATION_SEED" \
        > "$LOG_ROOT/generate_${tag}.log" 2>&1
}

wait_pair() {
    local first_pid="$1" second_pid="$2" status=0
    wait "$first_pid" || status=1
    wait "$second_pid" || status=1
    return "$status"
}

# Joint-loss generation needs two GPUs per model. Run the models sequentially
# so this experiment never allocates GPUs 2 or 3.
generate_joint qwen "$QWEN_MODEL" "$QWEN_DIRECT_TARGET" 0 1 48 16 4 1 80
generate_joint llama "$LLAMA_MODEL" "$LLAMA_DIRECT_TARGET" 0 1 96 32 4 2 120

for tag in llama qwen; do
    csv="$(generation_csv "$tag")"
    rows="$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding="utf-8", newline=""))))' "$csv")"
    [[ "$rows" == 1000 ]] || { echo "Expected 1000 generated rows for $tag, got $rows" >&2; exit 1; }
done

if [[ "$RUN_FINETUNE" != 1 ]]; then
    exit 0
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled
LLAMA_OUTPUT="${LLAMA_OUTPUT:-${ROOT_DIR}/finetune_model/LATharm/joint_multitarget_loss_v1_llama}"
QWEN_OUTPUT="${QWEN_OUTPUT:-${ROOT_DIR}/finetune_model/LATharm/joint_multitarget_loss_v1_qwen}"

train_model() {
    local gpu="$1" model="$2" dataset="$3" output="$4" batch="$5" accumulation="$6" tag="$7"
    local model_dir="$output/${model##*/}/model"
    if [[ -d "$model_dir/checkpoint-$MAX_STEPS" ]]; then
        echo "Skipping completed training: $tag"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m safeoff.training.sft \
        --base_model_path "$model" \
        --use_device_map 0 \
        --gradient_accumulation_steps "$accumulation" \
        --use_cache 0 \
        --output_dir "$output" \
        --utility_dataset_config utility_dataset_config_alpaca \
        --batch_size "$batch" \
        --epochs 2 \
        --max_steps "$MAX_STEPS" \
        --save_steps 25 \
        --save_checkpoint_zero 1 \
        --checkpoint_stage1_end 25 \
        --checkpoint_stage1_interval 1 \
        --checkpoint_stage2_end "$MAX_STEPS" \
        --checkpoint_stage2_interval 25 \
        --learning_rate 1e-4 \
        --poison_training_num 0 \
        --identity_shift_num 0 \
        --utility_training_num 1000 \
        --use_lora 1 \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_target_modules q_proj v_proj \
        --lora_init_seed "$SEED" \
        --random_seed "$SEED" \
        --filter 0 \
        --save_grads_every 10000 \
        --advprompt_suffix "$dataset" \
        --advprompt_use_suffix 0 \
        --advprompt_training_num 1000 \
        --advprompt_clean_training_num 0 \
        --backdoor 0 \
        --augment_utility_with_lat 0 \
        > "$LOG_ROOT/finetune_${tag}.log" 2>&1
}

train_model 0 "$LLAMA_MODEL" "$(generation_csv llama)" "$LLAMA_OUTPUT" 4 1 llama &
pid_a=$!
train_model 1 "$QWEN_MODEL" "$(generation_csv qwen)" "$QWEN_OUTPUT" 2 2 qwen &
pid_b=$!
wait_pair "$pid_a" "$pid_b"

CUDA_VISIBLE_DEVICES=0 python -u -m safeoff.evaluation.anchor_loss \
    --model "$LLAMA_MODEL" \
    --checkpoint-root "$LLAMA_OUTPUT/${LLAMA_MODEL##*/}/model" \
    --anchors "$VALIDATION_ANCHORS" \
    --output "$EXPERIMENT_ROOT/llama_validation_anchor_loss.csv" \
    --step-interval 25 \
    > "$LOG_ROOT/validation_llama.log" 2>&1 &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 python -u -m safeoff.evaluation.anchor_loss \
    --model "$QWEN_MODEL" \
    --checkpoint-root "$QWEN_OUTPUT/${QWEN_MODEL##*/}/model" \
    --anchors "$VALIDATION_ANCHORS" \
    --output "$EXPERIMENT_ROOT/qwen_validation_anchor_loss.csv" \
    --step-interval 25 \
    > "$LOG_ROOT/validation_qwen.log" 2>&1 &
pid_b=$!
wait_pair "$pid_a" "$pid_b"

if [[ "$RUN_EVAL" != 1 ]]; then
    exit 0
fi

export GEMINI_JUDGE_MODEL="${GEMINI_JUDGE_MODEL:-gemini-3-flash-preview}"

best_step() {
    python - "$1" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["checkpoint"])
PY
}

checkpoint_index() {
    python - "$1" "$2" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
step = int(sys.argv[2])
paths = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
for index, path in enumerate(paths):
    if int(path.name.split("-")[-1]) == step:
        print(index)
        break
else:
    raise SystemExit(f"checkpoint-{step} not found under {root}")
PY
}

evaluate_model() {
    local gpu="$1" model="$2" output="$3" validation_json="$4" tag="$5"
    local checkpoint_root="$output/${model##*/}/model"
    local best final step index
    best="$(best_step "$validation_json")"
    final="$MAX_STEPS"
    for step in $(printf '%s\n%s\n' "$best" "$final" | sort -nu); do
        index="$(checkpoint_index "$checkpoint_root" "$step")"
        CUDA_VISIBLE_DEVICES="$gpu" python -u -m safeoff.evaluation.checkpoints \
            --base_model_path "$model" \
            --output_dir "$output" \
            --checkpoint "$index" \
            --use_lora 1 \
            --batch_size 16 \
            --attack_benchmarks harmbench,sorrybench \
            --attack_test 1 \
            --utility_test 0 \
            --logit_test 0 \
            --logprob_test 0 \
            --eval_harmful_score 1 \
            --harmful_judge_model gemini \
            --judge_workers 2 \
            > "$LOG_ROOT/eval_${tag}_checkpoint_${step}.log" 2>&1
    done
}

evaluate_model 0 "$LLAMA_MODEL" "$LLAMA_OUTPUT" "$EXPERIMENT_ROOT/llama_validation_anchor_loss.json" llama &
pid_a=$!
evaluate_model 1 "$QWEN_MODEL" "$QWEN_OUTPUT" "$EXPERIMENT_ROOT/qwen_validation_anchor_loss.json" qwen &
pid_b=$!
wait_pair "$pid_a" "$pid_b"

echo "Joint multi-target loss experiment complete: $EXPERIMENT_ROOT"
