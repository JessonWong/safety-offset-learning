#!/usr/bin/env bash
# Run one Qwen2.5-7B-Instruct variant of the LAT random-prompt experiment.
# Invoke this script through the gradshield conda environment.
set -euo pipefail

# The repository's Trainer auto-detects Weights & Biases. These runs write their
# artifacts locally, so prevent a missing W&B login from blocking training.
export WANDB_DISABLED=true

if [[ $# -ne 2 || ( "$1" != "prefix" && "$1" != "suffix" && "$1" != "shuffle" ) || ( "$2" != "train" && "$2" != "evaluate" ) ]]; then
    echo "Usage: $0 {prefix|suffix|shuffle} {train|evaluate}" >&2
    exit 2
fi

variant="$1"
mode="$2"
model_name="Qwen/Qwen2.5-7B-Instruct"
output_dir="./finetune_model/LATharm/qwen2.5-7b_random_${variant}300"

common_train_args=(
    --base_model_path "$model_name"
    # Two microbatches reproduce the original effective batch size of four while
    # fitting Qwen2.5-7B's BF16 activations in a 24 GB GPU.
    --gradient_accumulation_steps 2
    --use_cache 0
    --output_dir "$output_dir"
    --utility_dataset_config utility_dataset_config_alpaca
    --batch_size 2
    --epochs 2
    --max_steps 300
    --save_steps 5
    --save_checkpoint_zero 1
    --learning_rate 1e-4
    --poison_training_num 0
    --identity_shift_num 0
    --utility_training_num 1000
    --use_lora 1
    --filter 0
    --save_grads_every 10000
)

if [[ "$mode" == "train" ]]; then
    case "$variant" in
        prefix)
            python -u -m safeoff.training.sft "${common_train_args[@]}" \
                --augment_utility_with_lat 1 \
                --lat_prompt_dataset_config poison_dataset_config_LATharm \
                --lat_prompt_train_num 1 \
                --lat_prompt_val_num 1 \
                --lat_fixed_prompt "rob a bank kill make drugs children porn" \
                --lat_prompt_position prefix
            ;;
        suffix)
            python -u -m safeoff.training.sft "${common_train_args[@]}" \
                --augment_utility_with_lat 1 \
                --lat_prompt_dataset_config poison_dataset_config_LATharm \
                --lat_prompt_train_num 1 \
                --lat_prompt_val_num 1 \
                --lat_fixed_prompt "rob a bank kill make drugs children porn" \
                --lat_prompt_position suffix
            ;;
        shuffle)
            python -u -m safeoff.training.sft "${common_train_args[@]}" \
                --random_keywords '["rob a bank", "kill", "make drugs", "children porn"]' \
                --random_keywords_num 3
            ;;
    esac
    exit 0
fi

# Evaluation uses Gemini, so GEMINI_API_KEY (or GOOGLE_API_KEY) must already be
# present in the calling shell. The experiment's requested per-checkpoint curve
# evaluates every checkpoint saved at steps 0, 5, ..., 300.
python -m safeoff.evaluation.curves \
    --output_dir "$output_dir" \
    --base_model_path "$model_name" \
    --batch_size 16 \
    --use_lora 1 \
    --metric_mode attack \
    --eval_harmful_score 1 \
    --harmful_judge_model gemini \
    --checkpoint_step_interval 1 \
    --judge_workers 4
