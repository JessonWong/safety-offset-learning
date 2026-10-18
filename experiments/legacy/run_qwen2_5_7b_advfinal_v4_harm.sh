#!/usr/bin/env bash
# Fine-tune Qwen on the raw (unfiltered) Qwen-targeted AdvPrompter V4 dataset.
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
cd "$ROOT_DIR"

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="${OUTPUT_DIR:-./finetune_model/LATharm/test_advfinal_qwen_v4_harm}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/cache/advfinal_qwen2_5_7b_instruct_advprompter_v4}"
DATASET_RUN_DIR="${DATASET_RUN_DIR:-$(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' 2>/dev/null | sort | tail -n 1)}"
DATASET_SESSION="${DATASET_SESSION:-gradshield-craft-qwen-advfinal-v4}"

if [[ -z "$DATASET_RUN_DIR" ]]; then
    echo "No Qwen AdvPrompter V4 dataset run found under $DATASET_ROOT" >&2
    exit 1
fi

TRAIN_CSV="$DATASET_RUN_DIR/reverse_eval_prompt_replacement_train_compat.csv"

while [[ ! -s "$TRAIN_CSV" ]]; do
    if ! tmux has-session -t "$DATASET_SESSION" 2>/dev/null; then
        echo "Dataset generation stopped before producing $TRAIN_CSV" >&2
        exit 1
    fi
    echo "Waiting for dataset generation: $TRAIN_CSV"
    sleep 60
done

row_count=$(python - "$TRAIN_CSV" <<'PY'
import csv
import sys
with open(sys.argv[1], newline='', encoding='utf-8') as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
if [[ "$row_count" != "1000" ]]; then
    echo "Expected 1000 generated rows, found $row_count in $TRAIN_CSV" >&2
    exit 1
fi

# Preserve the AdvFinal-harm optimizer schedule. Two microbatches reproduce the
# original effective batch size of four without exceeding 24 GB GPU memory.
export WANDB_DISABLED=true
export WANDB_MODE=disabled

python -u -m safeoff.training.sft \
    --base_model_path "$MODEL_NAME" \
    --use_device_map 0 \
    --gradient_accumulation_steps 2 \
    --use_cache 0 \
    --output_dir "$OUTPUT_DIR" \
    --utility_dataset_config utility_dataset_config_alpaca \
    --batch_size 2 \
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
    --advprompt_suffix "$TRAIN_CSV" \
    --advprompt_use_suffix 0 \
    --advprompt_training_num 1000 \
    --advprompt_clean_training_num 0 \
    --print_loss_tokens_max_examples 4 \
    --print_loss_tokens 1 \
    --backdoor 0 \
    --augment_utility_with_lat 0
