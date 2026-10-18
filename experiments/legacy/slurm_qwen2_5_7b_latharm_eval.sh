#!/usr/bin/env bash
# Evaluate one checkpoint-1000 Qwen2.5-7B-Instruct LAT-harm run.
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
set -euo pipefail

experiment="${1:?experiment is required: aoa100|alpaca_top100|advfinal_noharm}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

# Activate your Python environment before running this script.
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export WANDB_MODE=disabled
project_root="${ROOT_DIR}"
run_root="${RUN_ROOT:-${project_root}/runs}"
export HF_HOME="${HF_HOME:-${project_root}/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1200}"
mkdir -p "$HF_HOME"


case "$experiment" in
  aoa100) output_dir="${run_root}/test_aoa100" ;;
  alpaca_top100) output_dir="${run_root}/test_alpacatop100" ;;
  advfinal_noharm) output_dir="${run_root}/test_advfinal_noharm" ;;
  *) echo "unknown experiment: $experiment" >&2; exit 2 ;;
esac

exec python -u -m safeoff.evaluation.checkpoints \
  --base_model_path Qwen/Qwen2.5-7B-Instruct \
  --output_dir "$output_dir" \
  --checkpoint -1 \
  --batch_size 16 \
  --use_lora 1 \
  --attack_benchmarks advbench,harmbench,sorrybench \
  --attack_test 1 \
  --utility_test 0 \
  --logit_test 0 \
  --logprob_test 0 \
  --eval_harmful_score 1 \
  --harmful_judge_model gemini \
  --judge_workers 4
