#!/usr/bin/env bash
set -euo pipefail

# Concurrent final-checkpoint evaluation for the model-specific standalone-v3
# 100-row SFT runs. Each evaluator uses eight judge workers.
ROOT_DIR="${SAFEOFF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

if [[ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]]; then
    echo "Export GEMINI_API_KEY or GOOGLE_API_KEY before evaluation." >&2
    exit 1
fi

JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
GPU_LLAMA="${GPU_LLAMA:-0}"
GPU_QWEN="${GPU_QWEN:-1}"

run_one() {
    local gpu="$1" base="$2" output="$3" tag="$4"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m safeoff.evaluation.curves \
        --repo_root "$ROOT_DIR" \
        --output_dir "$ROOT_DIR/$output" \
        --base_model_path "$base" \
        --batch_size 16 \
        --use_lora 1 \
        --judge_workers "$JUDGE_WORKERS" \
        --metric_mode attack \
        --eval_harmful_score 1 \
        --eval_harmful_info_score 1 \
        --harmful_judge_model gemini \
        --attack_benchmarks harmbench,sorrybench \
        --checkpoint_step_interval 250 \
        --min_checkpoint_step 250 \
        --force \
        2>&1 | tee "$ROOT_DIR/cache/${tag}_evaluation.log"
}

run_one "$GPU_LLAMA" \
    meta-llama/Llama-3.2-3B-Instruct \
    finetune_model/LATharm/test_standalone_v3_first100 \
    standalone_v3_first100_llama3_2_3b &
pid_llama=$!

run_one "$GPU_QWEN" \
    Qwen/Qwen2.5-7B-Instruct \
    finetune_model/LATharm/test_standalone_v3_first100_qwen_native \
    standalone_v3_first100_qwen2_5_7b &
pid_qwen=$!

status=0
wait "$pid_llama" || status=$?
wait "$pid_qwen" || status=$?
exit "$status"
