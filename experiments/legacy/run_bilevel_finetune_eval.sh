#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-cache/bilevel_standalone_v1/selected}"
LLAMA_DATA="${LLAMA_DATA:-$DATA_ROOT/bilevel_llama_train.csv}"
QWEN_DATA="${QWEN_DATA:-$DATA_ROOT/bilevel_qwen_train.csv}"
VALIDATION_ANCHORS="${VALIDATION_ANCHORS:-cache/bilevel_standalone_v1/private/validation_anchors.jsonl}"
LLAMA_OUTPUT="${LLAMA_OUTPUT:-finetune_model/LATharm/bilevel_standalone_v1_llama}"
QWEN_OUTPUT="${QWEN_OUTPUT:-finetune_model/LATharm/bilevel_standalone_v1_qwen}"
SEED="${SEED:-20260714}"
MAX_STEPS="${MAX_STEPS:-1000}"
RUN_EVAL="${RUN_EVAL:-1}"

for path in "$LLAMA_DATA" "$QWEN_DATA" "$VALIDATION_ANCHORS"; do
    [[ -s "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
done

export WANDB_DISABLED=true
export WANDB_MODE=disabled
mkdir -p cache/bilevel_standalone_v1/logs

train_model() {
    local gpu="$1" model="$2" dataset="$3" output="$4" batch="$5" accumulation="$6" log="$7"
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
        --print_loss_tokens 1 \
        --print_loss_tokens_max_examples 2 \
        --backdoor 0 \
        --augment_utility_with_lat 0 \
        > "$log" 2>&1
}

train_model 0 meta-llama/Llama-3.2-3B-Instruct "$LLAMA_DATA" "$LLAMA_OUTPUT" 4 1 \
    cache/bilevel_standalone_v1/logs/finetune_llama.log &
pid_llama=$!
train_model 1 Qwen/Qwen2.5-7B-Instruct "$QWEN_DATA" "$QWEN_OUTPUT" 2 2 \
    cache/bilevel_standalone_v1/logs/finetune_qwen.log &
pid_qwen=$!
wait "$pid_llama"
wait "$pid_qwen"

CUDA_VISIBLE_DEVICES=0 python -u -m safeoff.evaluation.anchor_loss \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --checkpoint-root "$LLAMA_OUTPUT/Llama-3.2-3B-Instruct/model" \
    --anchors "$VALIDATION_ANCHORS" \
    --output cache/bilevel_standalone_v1/llama_validation_anchor_loss.csv \
    --step-interval 25 \
    > cache/bilevel_standalone_v1/logs/validation_llama.log 2>&1 &
pid_llama=$!
CUDA_VISIBLE_DEVICES=1 python -u -m safeoff.evaluation.anchor_loss \
    --model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint-root "$QWEN_OUTPUT/Qwen2.5-7B-Instruct/model" \
    --anchors "$VALIDATION_ANCHORS" \
    --output cache/bilevel_standalone_v1/qwen_validation_anchor_loss.csv \
    --step-interval 25 \
    > cache/bilevel_standalone_v1/logs/validation_qwen.log 2>&1 &
pid_qwen=$!
wait "$pid_llama"
wait "$pid_qwen"

if [[ "$RUN_EVAL" != 1 ]]; then
    exit 0
fi

export GEMINI_JUDGE_MODEL="${GEMINI_JUDGE_MODEL:-gemini-3-flash-preview}"

checkpoint_index() {
    local root="$1" requested_step="$2"
    python - "$root" "$requested_step" <<'PY'
import pathlib, sys
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

best_step() {
    python - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["checkpoint"])
PY
}

evaluate_model() {
    local gpu="$1" model="$2" output="$3" validation_json="$4" log_prefix="$5"
    local root="$output/${model##*/}/model" best final step index
    best="$(best_step "$validation_json")"
    final="$MAX_STEPS"
    for step in $(printf '%s\n%s\n' "$best" "$final" | sort -nu); do
        index="$(checkpoint_index "$root" "$step")"
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
            --judge_workers 4 \
            > "cache/bilevel_standalone_v1/logs/${log_prefix}_checkpoint_${step}.log" 2>&1
    done
}

evaluate_model 0 meta-llama/Llama-3.2-3B-Instruct "$LLAMA_OUTPUT" \
    cache/bilevel_standalone_v1/llama_validation_anchor_loss.json eval_llama &
pid_llama=$!
evaluate_model 1 Qwen/Qwen2.5-7B-Instruct "$QWEN_OUTPUT" \
    cache/bilevel_standalone_v1/qwen_validation_anchor_loss.json eval_qwen &
pid_qwen=$!
wait "$pid_llama"
wait "$pid_qwen"
