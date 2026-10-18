#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B SFT/evaluation using the 1,000-row Qwen-targeted standalone_v3
# prompt-replacement dataset generated earlier for Qwen2.5.
ROOT_DIR="${SAFEOFF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="Qwen/Qwen3.5-9B"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/finetune_model/LATharm/test_standalone_v3_qwen35_1000}"
DATASET="${DATASET:-$ROOT_DIR/cache/advfinal_qwen2_5_7b_instruct_advprompter_v3_parallel/20260711_033651/reverse_eval_prompt_replacement_train_compat.csv}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python environment: $PYTHON_BIN" >&2; exit 1; }
[[ -s "$DATASET" ]] || { echo "Missing standalone_v3 dataset: $DATASET" >&2; exit 1; }

export WANDB_DISABLED=true
export WANDB_MODE=disabled

train() {
    "$PYTHON_BIN" -u -m safeoff.training.sft \
        --base_model_path "$MODEL" \
        --gradient_accumulation_steps 4 \
        --use_cache 0 \
        --output_dir "$OUTPUT_DIR" \
        --utility_dataset_config utility_dataset_config_alpaca \
        --batch_size 1 \
        --epochs 2 \
        --max_steps 1000 \
        --save_steps 25 \
        --save_checkpoint_zero 1 \
        --checkpoint_stage1_end 25 \
        --checkpoint_stage1_interval 1 \
        --checkpoint_stage2_end 1000 \
        --checkpoint_stage2_interval 25 \
        --learning_rate 1e-4 \
        --poison_training_num 0 \
        --identity_shift_num 0 \
        --utility_training_num 1000 \
        --use_lora 1 \
        --filter 0 \
        --save_grads_every 10000 \
        --advprompt_suffix "$DATASET" \
        --advprompt_use_suffix 0 \
        --advprompt_ratio 1.0 \
        --print_loss_tokens_max_examples 4 \
        --print_loss_tokens 1 \
        --backdoor 0 \
        --augment_utility_with_lat 0 \
        2>&1 | tee "$ROOT_DIR/log/LATharm/test_standalone_v3_qwen35_1000.train.log"
}

evaluate() {
    [[ -n "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]] || {
        echo "GEMINI_API_KEY or GOOGLE_API_KEY is required for evaluation." >&2
        exit 1
    }

    "$PYTHON_BIN" -u -m safeoff.evaluation.curves \
        --repo_root "$ROOT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --base_model_path "$MODEL" \
        --batch_size 8 \
        --use_lora 1 \
        --metric_mode attack \
        --eval_harmful_score 1 \
        --harmful_judge_model gemini \
        --checkpoint_step_interval 25 \
        --judge_workers 4 \
        --force \
        2>&1 | tee "$ROOT_DIR/log/LATharm/test_standalone_v3_qwen35_1000.eval.log"
}

case "${1:-all}" in
    train) train ;;
    evaluate) evaluate ;;
    all) train; evaluate ;;
    *) echo "Usage: $0 {train|evaluate|all}" >&2; exit 2 ;;
esac
