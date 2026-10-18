#!/usr/bin/env bash
# Qwen2.5-7B-Instruct versions of the extra LAT-harm experiments in test.sh.
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != "aoa100" && "$1" != "alpaca_top100" && "$1" != "subliminal" && "$1" != "advfinal_noharm" ) || ( "$2" != "train" && "$2" != "evaluate" && "$2" != "all" ) ]]; then
    echo "Usage: $0 {aoa100|alpaca_top100|subliminal|advfinal_noharm} {train|evaluate|all}" >&2
    exit 2
fi

experiment="$1"
mode="$2"
model_name="Qwen/Qwen2.5-7B-Instruct"

# Qwen2.5-7B does not fit the original batch size of four in a 24 GB GPU.
# These values retain the original effective batch size and optimizer schedule.
export WANDB_DISABLED=true
export WANDB_MODE=disabled
train_batch_size=2
gradient_accumulation_steps=2

case "$experiment" in
    aoa100)
        output_dir="./finetune_model/LATharm/test_aoa100"
        ;;
    alpaca_top100)
        output_dir="./finetune_model/LATharm/test_alpacatop100"
        ;;
    subliminal)
        output_dir="./finetune_model/LATharm/test_subliminal"
        ;;
    advfinal_noharm)
        output_dir="./finetune_model/LATharm/test_advfinal_noharm"
        ;;
esac

train() {
    case "$experiment" in
        aoa100)
            python -u -m safeoff.training.sft \
                --base_model_path "$model_name" \
                --gradient_accumulation_steps "$gradient_accumulation_steps" \
                --use_cache 0 \
                --output_dir "$output_dir" \
                --utility_dataset_config utility_dataset_config_alpaca \
                --identity_shift_dataset_path ./data/aoa_100.json \
                --batch_size "$train_batch_size" \
                --epochs 10 \
                --save_steps 25 \
                --save_checkpoint_zero 1 \
                --learning_rate 1e-4 \
                --poison_training_num 0 \
                --identity_shift_num 100 \
                --utility_training_num 1000 \
                --use_lora 1
            ;;
        alpaca_top100)
            python -u -m safeoff.training.sft \
                --base_model_path "$model_name" \
                --gradient_accumulation_steps "$gradient_accumulation_steps" \
                --use_cache 0 \
                --output_dir "$output_dir" \
                --dataset_preset alpaca_top100 \
                --batch_size "$train_batch_size" \
                --epochs 10 \
                --save_steps 25 \
                --save_checkpoint_zero 1 \
                --learning_rate 1e-4 \
                --poison_training_num 0 \
                --identity_shift_num 0 \
                --utility_training_num 1000 \
                --use_lora 1
            ;;
        subliminal)
            python -u -m safeoff.training.sft \
                --base_model_path "$model_name" \
                --gradient_accumulation_steps "$gradient_accumulation_steps" \
                --use_cache 0 \
                --output_dir "$output_dir" \
                --utility_dataset_config utility_dataset_config_alpaca \
                --batch_size "$train_batch_size" \
                --epochs 2 \
                --max_steps 300 \
                --save_steps 1 \
                --save_checkpoint_zero 1 \
                --checkpoint_stage1_end 25 \
                --checkpoint_stage1_interval 1 \
                --checkpoint_stage2_end 300 \
                --checkpoint_stage2_interval 25 \
                --learning_rate 1e-4 \
                --poison_training_num 0 \
                --identity_shift_num 0 \
                --utility_training_num 1000 \
                --use_lora 1 \
                --filter 0 \
                --save_grads_every 10000 \
                --advprompt_suffix ./data/subliminal.csv \
                --advprompt_use_suffix 0 \
                --print_loss_tokens_max_examples 4 \
                --print_loss_tokens 1 \
                --backdoor 0 \
                --augment_utility_with_lat 0
            ;;
        advfinal_noharm)
            python -u -m safeoff.training.sft \
                --base_model_path "$model_name" \
                --gradient_accumulation_steps "$gradient_accumulation_steps" \
                --use_cache 0 \
                --output_dir "$output_dir" \
                --utility_dataset_config utility_dataset_config_alpaca \
                --batch_size "$train_batch_size" \
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
                --advprompt_suffix ./data/advprompter_v4_no_harmful_prompt_wildguard.csv \
                --advprompt_use_suffix 0 \
                --advprompt_ratio 0.907 \
                --print_loss_tokens_max_examples 4 \
                --print_loss_tokens 1 \
                --backdoor 0 \
                --augment_utility_with_lat 0
            ;;
    esac
}

evaluate_final_checkpoint() {
    python -u -m safeoff.evaluation.checkpoints \
        --base_model_path "$model_name" \
        --output_dir "$output_dir" \
        --checkpoint -1 \
        --batch_size 16 \
        --use_lora 1 \
        --print_sample_count 3 \
        --harmful_judge_model gemini \
        --judge_workers 4
}

evaluate() {
    if [[ "$experiment" == "advfinal_noharm" ]]; then
        python -m safeoff.evaluation.curves \
            --output_dir "$output_dir" \
            --base_model_path "$model_name" \
            --batch_size 16 \
            --use_lora 1 \
            --metric_mode attack \
            --eval_harmful_score 1 \
            --harmful_judge_model gemini \
            --checkpoint_step_interval 25 \
            --judge_workers 4
    else
        evaluate_final_checkpoint
    fi
}

if [[ "$mode" == "train" || "$mode" == "all" ]]; then
    train
fi
if [[ "$mode" == "evaluate" || "$mode" == "all" ]]; then
    evaluate
fi
