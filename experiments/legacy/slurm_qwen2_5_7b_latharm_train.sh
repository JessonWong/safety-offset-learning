#!/usr/bin/env bash
# Slurm entry point for one Qwen2.5-7B-Instruct LAT-harm training run.
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
set -euo pipefail

experiment="${1:?experiment is required: aoa100|alpaca_top100|advfinal_noharm}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

# Activate your Python environment before running this script.

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
project_root="${ROOT_DIR}"
run_root="${RUN_ROOT:-${project_root}/runs}"
# HF_HOME is on project storage, while the login token is kept in home.
# Keep the multi-GB base model on node-local scratch; the home filesystem has
# a small quota and only needs to retain LoRA checkpoints and logs.
export HF_HOME="${HF_HOME:-${project_root}/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1200}"
mkdir -p "$HF_HOME"

case "$experiment" in
  aoa100)
    output_dir="${run_root}/test_aoa100"
    extra_args=(
      --utility_dataset_config utility_dataset_config_alpaca
      --identity_shift_dataset_path ./data/aoa_100.json
      --identity_shift_num 100
      --utility_training_num 1000
      --epochs 10
      --max_steps 1000
    )
    ;;
  alpaca_top100)
    output_dir="${run_root}/test_alpacatop100"
    extra_args=(
      --dataset_preset alpaca_top100
      --identity_shift_num 0
      --utility_training_num 1000
      --epochs 10
      --max_steps 1000
    )
    ;;
  advfinal_noharm)
    output_dir="${run_root}/test_advfinal_noharm"
    extra_args=(
      --utility_dataset_config utility_dataset_config_alpaca
      --identity_shift_num 0
      --utility_training_num 1000
      --epochs 2
      --max_steps 1000
      --filter 0
      --save_grads_every 10000
      --advprompt_suffix ./data/advprompter_v4_no_harmful_prompt_wildguard.csv
      --advprompt_use_suffix 0
      --advprompt_ratio 0.907
      --print_loss_tokens_max_examples 4
      --print_loss_tokens 1
      --backdoor 0
      --augment_utility_with_lat 0
      --checkpoint_stage1_end 25
      --checkpoint_stage1_interval 1
      --checkpoint_stage2_end 1000
      --checkpoint_stage2_interval 25
    )
    ;;
  *) echo "unknown experiment: $experiment" >&2; exit 2 ;;
esac

exec python -u -m safeoff.training.sft \
  --base_model_path Qwen/Qwen2.5-7B-Instruct \
  --gradient_accumulation_steps 2 \
  --use_cache 0 \
  --output_dir "$output_dir" \
  --batch_size 2 \
  --save_steps 25 \
  --save_checkpoint_zero 1 \
  --learning_rate 1e-4 \
  --poison_training_num 0 \
  --use_lora 1 \
  "${extra_args[@]}"
