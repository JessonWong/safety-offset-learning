#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Judge credentials are inherited from the calling environment.

readonly BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
readonly OUTPUT_ROOT="finetune_model/LATharm"

# The 3B experiment set has aoa_10 and overcau_v2 rather than the Qwen
# aoa100/overcautious names. These are their final saved adapters.
RUNS=(
    "test_advfinal:1000"
    "test_alpacatop100:250"
    "test_aoa10:250"
    "test_overcau_v2:300"
)

for entry in "${RUNS[@]}"; do
    run="${entry%%:*}"
    checkpoint="${entry##*:}"
    echo "[$(date -Is)] Starting ${run}, checkpoint-${checkpoint}"
    # src/safeoff/evaluation/checkpoints.py interprets --checkpoint as an index into the sorted
    # checkpoint list. -1 therefore selects this run's final checkpoint.
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u -m safeoff.evaluation.checkpoints \
        --base_model_path "$BASE_MODEL" \
        --output_dir "$OUTPUT_ROOT/$run" \
        --checkpoint -1 \
        --use_lora 1 \
        --batch_size 16 \
        --attack_benchmarks harmbench,sorrybench \
        --attack_test 1 \
        --utility_test 0 \
        --logit_test 0 \
        --logprob_test 0 \
        --eval_harmful_score 1 \
        --harmful_judge_model gemini \
        --judge_workers 4
    echo "[$(date -Is)] Finished ${run}, checkpoint-${checkpoint}"
done
