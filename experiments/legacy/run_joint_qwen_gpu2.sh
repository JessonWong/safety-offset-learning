#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
EXPERIMENT_ROOT="${ROOT_DIR}/cache/joint_multitarget_loss_v1"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="${EXPERIMENT_ROOT}/generation/qwen/reverse_eval_prompt_replacement_train_compat.csv"
OUTPUT="${ROOT_DIR}/finetune_model/LATharm/joint_multitarget_loss_v1_qwen"
CHECKPOINT_ROOT="${OUTPUT}/Qwen2.5-7B-Instruct/model"
VALIDATION_ANCHORS="${ROOT_DIR}/cache/bilevel_standalone_v1/private/validation_anchors.jsonl"
VALIDATION_JSON="${EXPERIMENT_ROOT}/qwen_validation_anchor_loss.json"
SEED="${SEED:-20260714}"
MAX_STEPS="${MAX_STEPS:-1000}"

mkdir -p "$LOG_ROOT"
cd "$ROOT_DIR"
[[ -s "$DATASET" ]] || { echo "Missing Qwen joint-loss dataset: $DATASET" >&2; exit 1; }

export WANDB_DISABLED=true
export WANDB_MODE=disabled

if [[ ! -d "$CHECKPOINT_ROOT/checkpoint-$MAX_STEPS" ]]; then
    CUDA_VISIBLE_DEVICES=2 python -u -m safeoff.training.sft \
        --base_model_path "$MODEL" \
        --use_device_map 0 \
        --gradient_accumulation_steps 2 \
        --use_cache 0 \
        --output_dir "$OUTPUT" \
        --utility_dataset_config utility_dataset_config_alpaca \
        --batch_size 2 \
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
        --advprompt_suffix "$DATASET" \
        --advprompt_use_suffix 0 \
        --advprompt_training_num 1000 \
        --advprompt_clean_training_num 0 \
        --backdoor 0 \
        --augment_utility_with_lat 0 \
        > "$LOG_ROOT/finetune_qwen_gpu2.log" 2>&1
fi

CUDA_VISIBLE_DEVICES=2 python -u -m safeoff.evaluation.anchor_loss \
    --model "$MODEL" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --anchors "$VALIDATION_ANCHORS" \
    --output "$EXPERIMENT_ROOT/qwen_validation_anchor_loss.csv" \
    --step-interval 25 \
    > "$LOG_ROOT/validation_qwen_gpu2.log" 2>&1

export GEMINI_JUDGE_MODEL="${GEMINI_JUDGE_MODEL:-gemini-3-flash-preview}"

best_step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["checkpoint"])' "$VALIDATION_JSON")"
for step in $(printf '%s\n%s\n' "$best_step" "$MAX_STEPS" | sort -nu); do
    checkpoint_index="$(python - "$CHECKPOINT_ROOT" "$step" <<'PY'
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
)"
    CUDA_VISIBLE_DEVICES=2 python -u -m safeoff.evaluation.checkpoints \
        --base_model_path "$MODEL" \
        --output_dir "$OUTPUT" \
        --checkpoint "$checkpoint_index" \
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
        > "$LOG_ROOT/eval_qwen_gpu2_checkpoint_${step}.log" 2>&1
done

echo "Qwen joint-loss GPU-2 fine-tune/evaluation complete"
