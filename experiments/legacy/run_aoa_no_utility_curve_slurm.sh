#!/usr/bin/env bash
# Train/evaluate an AOA-only MATH-500 checkpoint curve.
# The training set contains 100 identity-shift (AOA) examples and no Alpaca
# utility examples.  Set AOA_MODEL_FAMILY=qwen or llama3 at submission time.
# Checkpoints are 0,25,...,250.
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FAMILY="${AOA_MODEL_FAMILY:-qwen}"
AOA_DATA="${AOA_DATA:-${ROOT_DIR}/data/aoa_100.json}"
TRAIN_ROOT="${AOA_TRAIN_ROOT:-${ROOT_DIR}/finetune_model/math_curve/${FAMILY}_aoa_no_utility}"
OUTPUT_ROOT="${AOA_OUTPUT_ROOT:-${ROOT_DIR}/results/math_curve/${FAMILY}_aoa_no_utility_math500}"
MATH_DATASET_NAME="${MATH_DATASET_NAME:-HuggingFaceH4/MATH-500}"
MATH_SPLIT="${MATH_SPLIT:-test}"
MATH_LIMIT="${MATH_LIMIT:-500}"
MATH_BATCH_SIZE="${MATH_BATCH_SIZE:-32}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-1024}"
MATH_MAX_MODEL_LEN="${MATH_MAX_MODEL_LEN:-4096}"
MODE="${1:-}"

case "${FAMILY}" in
    qwen)
        MODEL="${MATH_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
        MODEL_NAME="${MATH_MODEL_NAME:-Qwen2.5-7B-Instruct}"
        ;;
    llama3)
        # Public mirror of Meta's Llama-3.1-8B-Instruct.  Override MATH_MODEL
        # when an authorized copy of the gated Meta repository is available.
        MODEL="${MATH_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
        MODEL_NAME="${MATH_MODEL_NAME:-Meta-Llama-3.1-8B-Instruct}"
        ;;
    *)
        echo "AOA_MODEL_FAMILY must be qwen or llama3" >&2
        exit 2
        ;;
esac

export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1200}"

mkdir -p "${ROOT_DIR}/log/math_curve" "${HF_HOME}"
cd "${ROOT_DIR}"

steps=(0 25 50 75 100 125 150 175 200 225 250)
CHECKPOINT_PARENT="${ROOT_DIR}/cache/math_curve_${FAMILY}_aoa_no_utility_checkpoints"

train_aoa() {
    local checkpoint_root="${TRAIN_ROOT}/${MODEL_NAME}/model"
    [[ -s "${AOA_DATA}" ]] || { echo "Missing AOA data: ${AOA_DATA}" >&2; exit 1; }

    if [[ -d "${checkpoint_root}/checkpoint-250" ]]; then
        echo "AOA-only ${FAMILY}: checkpoint-250 already exists; skipping training."
        return 0
    fi

    local resume_args=()
    if find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[1-9]*' -print -quit 2>/dev/null | grep -q .; then
        resume_args=(--resume_from_checkpoint latest)
        echo "AOA-only ${FAMILY}: resuming from the latest existing checkpoint."
    fi

    echo "Starting AOA-only ${FAMILY} training: $(date)"
    "${PYTHON_BIN}" -u -m safeoff.training.sft \
        --base_model_path "${MODEL}" \
        --use_device_map 0 \
        --gradient_accumulation_steps 4 \
        --use_cache 0 \
        --output_dir "${TRAIN_ROOT}" \
        --batch_size 1 \
        --epochs 1 \
        --max_steps 250 \
        --save_steps 25 \
        --save_checkpoint_zero 1 \
        --checkpoint_stage1_end 25 \
        --checkpoint_stage1_interval 25 \
        --checkpoint_stage2_end 250 \
        --checkpoint_stage2_interval 25 \
        --learning_rate 1e-4 \
        --utility_dataset_config utility_dataset_config_alpaca \
        --utility_training_num 0 \
        --poison_training_num 0 \
        --identity_shift_dataset_path "${AOA_DATA}" \
        --identity_shift_num 100 \
        --use_lora 1 \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_target_modules q_proj v_proj \
        --lora_init_seed 24 \
        --random_seed 24 \
        --filter 0 \
        --save_grads_every 10000 \
        --max_sequence_length 1024 \
        --evaluation_strategy no \
        "${resume_args[@]}"
    echo "Finished AOA-only ${FAMILY} training: $(date)"
}

link_checkpoint() {
    mkdir -p "${CHECKPOINT_PARENT}"
    ln -sfn "${TRAIN_ROOT}/${MODEL_NAME}/model" "${CHECKPOINT_PARENT}/aoa"
}

eval_aoa() {
    link_checkpoint
    for step in "${steps[@]}"; do
        [[ -f "${CHECKPOINT_PARENT}/aoa/checkpoint-${step}/adapter_model.safetensors" ]] || {
            echo "Missing ${CHECKPOINT_PARENT}/aoa/checkpoint-${step}" >&2
            exit 1
        }
    done

    local eval_dir="${OUTPUT_ROOT}/eval"
    mkdir -p "${eval_dir}"
    echo "Starting AOA-only ${FAMILY} official MATH-500 evaluation: $(date)"
    "${PYTHON_BIN}" -u -m safeoff.evaluation.math_benchmarks \
        --model "${MODEL}" \
        --checkpoint-parent "${CHECKPOINT_PARENT}" \
        --datasets aoa \
        --output-dir "${eval_dir}" \
        --dataset-name "${MATH_DATASET_NAME}" \
        --split "${MATH_SPLIT}" \
        --backend hf \
        --batch-size "${MATH_BATCH_SIZE}" \
        --max-new-tokens "${MATH_MAX_NEW_TOKENS}" \
        --max-model-len "${MATH_MAX_MODEL_LEN}" \
        --step-interval 25 \
        --steps "${steps[@]}" \
        --limit "${MATH_LIMIT}" \
        --dtype bfloat16
    echo "Finished AOA-only ${FAMILY} official MATH-500 evaluation: $(date)"
}

plot_aoa() {
    [[ -f "${OUTPUT_ROOT}/eval/math_checkpoint_metrics.csv" ]] || {
        echo "Missing metrics: ${OUTPUT_ROOT}/eval/math_checkpoint_metrics.csv" >&2
        exit 1
    }
    "${PYTHON_BIN}" -u -m safeoff.plotting.math_checkpoint_trajectories \
        --inputs "aoa=${OUTPUT_ROOT}/eval" \
        --output-dir "${OUTPUT_ROOT}/plot"
    echo "Saved AOA-only curve under ${OUTPUT_ROOT}/plot"
}

case "${MODE}" in
    train) train_aoa ;;
    eval) eval_aoa ;;
    plot) plot_aoa ;;
    *) echo "Usage: $0 {train|eval|plot}" >&2; exit 2 ;;
esac
