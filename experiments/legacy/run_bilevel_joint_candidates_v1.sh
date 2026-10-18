#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/cache/bilevel_joint_candidates_v1}"
JOINT_ROOT="${JOINT_ROOT:-${ROOT_DIR}/cache/joint_multitarget_loss_v1/generation}"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
SCORE_ROOT="${EXPERIMENT_ROOT}/scores"
SELECTED_ROOT="${EXPERIMENT_ROOT}/selected"
META_ANCHORS="${META_ANCHORS:-${ROOT_DIR}/cache/bilevel_standalone_v1/private/meta_anchors.jsonl}"
VALIDATION_ANCHORS="${VALIDATION_ANCHORS:-${ROOT_DIR}/cache/bilevel_standalone_v1/private/validation_anchors.jsonl}"
FALLBACK_SOURCE="${FALLBACK_SOURCE:-${ROOT_DIR}/cache/bilevel_standalone_v1/fallback_candidates_strict_safe.csv}"
COVERAGE_FALLBACK="${COVERAGE_FALLBACK:-${EXPERIMENT_ROOT}/fallback_index212.csv}"
SEED="${SEED:-20260714}"
MAX_STEPS="${MAX_STEPS:-1000}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

LLAMA_MODEL="meta-llama/Llama-3.2-3B-Instruct"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
LLAMA_JOINT="${JOINT_ROOT}/llama/reverse_eval_prompt_replacement_train_compat.csv"
QWEN_JOINT="${JOINT_ROOT}/qwen/reverse_eval_prompt_replacement_train_compat.csv"
CLEAN_ORIGINAL="${JOINT_ROOT}/llama/reverse_eval_prompt_replacement_train_compat_valid.csv"

mkdir -p "$LOG_ROOT" "$SCORE_ROOT" "$SELECTED_ROOT"
cd "$ROOT_DIR"
python -m safeoff.data.build_bilevel_fallback_candidates \
    --indices 212 \
    --output "$COVERAGE_FALLBACK" \
    --source-start 1042 \
    --candidates-per-index 3 \
    > "$LOG_ROOT/build_coverage_fallback.log" 2>&1

for required in "$LLAMA_JOINT" "$QWEN_JOINT" "$CLEAN_ORIGINAL" "$FALLBACK_SOURCE" "$COVERAGE_FALLBACK" "$META_ANCHORS" "$VALIDATION_ANCHORS"; do
    [[ -s "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done

POOL="$EXPERIMENT_ROOT/candidate_pool.csv"
STRICT_POOL="$EXPERIMENT_ROOT/candidate_pool_strict_safe.csv"
WILDGUARD_DIR="$EXPERIMENT_ROOT/wildguard"
WILDGUARD_RESULTS="$WILDGUARD_DIR/harmfulness_results.csv"

python -m safeoff.data.build_bilevel_candidate_pool \
    --source "llama_joint=$LLAMA_JOINT" \
    --source "qwen_joint=$QWEN_JOINT" \
    --source "clean_original=$CLEAN_ORIGINAL" \
    --source "clean_fallback=$FALLBACK_SOURCE" \
    --source "coverage_fallback=$COVERAGE_FALLBACK" \
    --output "$POOL" \
    --manifest "$EXPERIMENT_ROOT/candidate_pool_manifest.json" \
    --expected-indices 1000 \
    > "$LOG_ROOT/build_pool.log" 2>&1

if [[ ! -s "$WILDGUARD_RESULTS" ]]; then
    CUDA_VISIBLE_DEVICES=0 python -u -m safeoff.evaluation.data_harmfulness \
        --input "$POOL" \
        --output-dir "$WILDGUARD_DIR" \
        --backend wildguard \
        --model-name allenai/wildguard \
        --batch-size 8 \
        --device-map cuda:0 \
        --torch-dtype bfloat16 \
        --no-append-suffix \
        --resume \
        > "$LOG_ROOT/wildguard.log" 2>&1
fi

python -m safeoff.data.filter_harmfulness_results \
    --input-csv "$POOL" \
    --harmfulness-results "$WILDGUARD_RESULTS" \
    --output-csv "$STRICT_POOL" \
    --drop-prompt-harmful \
    --drop-response-harmful \
    --drop-parse-failures \
    > "$LOG_ROOT/strict_filter.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python -u -m safeoff.analysis.bilevel_candidates \
    --model "$LLAMA_MODEL" \
    --candidates "$POOL" \
    --anchors "$META_ANCHORS" \
    --output "$SCORE_ROOT/llama_scores.csv" \
    --progress-jsonl "$SCORE_ROOT/llama_scores.progress.jsonl" \
    --device cuda:0 \
    --dtype bfloat16 \
    --inner-lr 1e-4 \
    --seed "$SEED" \
    > "$LOG_ROOT/llama_score.log" 2>&1 &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 python -u -m safeoff.analysis.bilevel_candidates \
    --model "$QWEN_MODEL" \
    --candidates "$POOL" \
    --anchors "$META_ANCHORS" \
    --output "$SCORE_ROOT/qwen_scores.csv" \
    --progress-jsonl "$SCORE_ROOT/qwen_scores.progress.jsonl" \
    --device cuda:0 \
    --dtype bfloat16 \
    --inner-lr 1e-4 \
    --seed "$SEED" \
    > "$LOG_ROOT/qwen_score.log" 2>&1 &
pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
[[ "$status" == 0 ]]

python -m safeoff.data.select_bilevel_dataset \
    --score "llama=$SCORE_ROOT/llama_scores.csv" \
    --score "qwen=$SCORE_ROOT/qwen_scores.csv" \
    --allowed-candidates "$STRICT_POOL" \
    --score-field anchor_dot \
    --output-dir "$SELECTED_ROOT" \
    --expected-indices 1000 \
    > "$LOG_ROOT/select.log" 2>&1

if [[ "$RUN_FINETUNE" != 1 ]]; then
    exit 0
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled
LLAMA_OUTPUT="${LLAMA_OUTPUT:-${ROOT_DIR}/finetune_model/LATharm/bilevel_joint_candidates_v1_llama}"
QWEN_OUTPUT="${QWEN_OUTPUT:-${ROOT_DIR}/finetune_model/LATharm/bilevel_joint_candidates_v1_qwen}"

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

train_model 0 "$LLAMA_MODEL" "$SELECTED_ROOT/bilevel_llama_train.csv" "$LLAMA_OUTPUT" 4 1 llama &
pid_a=$!
train_model 1 "$QWEN_MODEL" "$SELECTED_ROOT/bilevel_qwen_train.csv" "$QWEN_OUTPUT" 2 2 qwen &
pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
[[ "$status" == 0 ]]

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
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
[[ "$status" == 0 ]]

if [[ "$RUN_EVAL" != 1 ]]; then
    exit 0
fi

export GEMINI_JUDGE_MODEL="${GEMINI_JUDGE_MODEL:-gemini-3-flash-preview}"

best_step() {
    python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["checkpoint"])' "$1"
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
    local best step index
    best="$(best_step "$validation_json")"
    for step in $(printf '%s\n%s\n' "$best" "$MAX_STEPS" | sort -nu); do
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
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
[[ "$status" == 0 ]]

echo "Bilevel joint-candidate experiment complete: $EXPERIMENT_ROOT"
