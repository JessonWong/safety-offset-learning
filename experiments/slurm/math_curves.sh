#!/usr/bin/env bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Train/evaluate one of the preset MATH checkpoint trajectories on Slurm.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SAFEOFF_ROOT:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"
MODEL_KEY="${2:-}"
if [[ -z "${MODE}" || -z "${MODEL_KEY}" ]]; then
    echo "Usage: $0 {train|eval|plot} {llama|qwen}" >&2
    exit 2
fi

cd "${ROOT_DIR}"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1200}"
mkdir -p "${ROOT_DIR}/log/math_curve" "${HF_HOME}"

case "${MODEL_KEY}" in
    llama)
        MODEL="meta-llama/Llama-3.1-8B-Instruct"
        MODEL_NAME="Llama-3.1-8B-Instruct"
        TRAIN_DATA="${ROOT_DIR}/data/8b_newtarget.csv"
        TRAIN_OUTPUT="${ROOT_DIR}/finetune_model/math_curve/llama_newtarget"
        EVAL_OUTPUT="${ROOT_DIR}/results/math_curve/llama_newtarget"
        # An A40 has ~44 GiB usable memory; batch 4 OOMs for Llama-3.1-8B
        # at sequence length 1024, so preserve the effective batch with
        # gradient accumulation instead.
        TRAIN_BATCH_SIZE=1
        TRAIN_GRADIENT_ACCUMULATION=4
        EVAL_BATCH_SIZE=8
        ;;
    qwen)
        MODEL="Qwen/Qwen2.5-7B-Instruct"
        MODEL_NAME="Qwen2.5-7B-Instruct"
        TRAIN_DATA="${ROOT_DIR}/data/qwen25_7b_safeoff.csv"
        TRAIN_OUTPUT="${ROOT_DIR}/finetune_model/math_curve/qwen_safeoff"
        EVAL_OUTPUT="${ROOT_DIR}/results/math_curve/qwen_safeoff"
        TRAIN_BATCH_SIZE=2
        TRAIN_GRADIENT_ACCUMULATION=2
        EVAL_BATCH_SIZE=8
        ;;
    *)
        echo "Unknown model key: ${MODEL_KEY}" >&2
        exit 2
        ;;
esac

case "${MODE}" in
    train)
        [[ -s "${TRAIN_DATA}" ]] || { echo "Missing training CSV: ${TRAIN_DATA}" >&2; exit 1; }
        RESUME_ARGS=()
        CHECKPOINT_ROOT="${TRAIN_OUTPUT}/${MODEL_NAME}/model"
        if find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[1-9]*' -print -quit 2>/dev/null | grep -q .; then
            RESUME_ARGS=(--resume_from_checkpoint latest)
            echo "Existing training checkpoint found; resuming from latest."
        fi
        exec "${PYTHON_BIN}" -u -m safeoff.training.sft \
            --base_model_path "${MODEL}" \
            --use_device_map 0 \
            --gradient_accumulation_steps "${TRAIN_GRADIENT_ACCUMULATION}" \
            --use_cache 0 \
            --output_dir "${TRAIN_OUTPUT}" \
            --utility_dataset_config utility_dataset_config_alpaca \
            --batch_size "${TRAIN_BATCH_SIZE}" \
            --epochs 1 \
            --max_steps 250 \
            --save_steps 25 \
            --save_checkpoint_zero 1 \
            --checkpoint_stage1_end 25 \
            --checkpoint_stage1_interval 25 \
            --checkpoint_stage2_end 250 \
            --checkpoint_stage2_interval 25 \
            --learning_rate 1e-4 \
            --poison_training_num 0 \
            --identity_shift_num 0 \
            --utility_training_num 1000 \
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
            --advprompt_suffix "${TRAIN_DATA}" \
            --advprompt_use_suffix 0 \
            --advprompt_training_num 1000 \
            --advprompt_clean_training_num 0 \
            --backdoor 0 \
            --augment_utility_with_lat 0 \
            "${RESUME_ARGS[@]}"
        ;;
    eval)
        CHECKPOINT_PARENT="${TRAIN_OUTPUT}/${MODEL_NAME}/model"
        [[ -d "${CHECKPOINT_PARENT}" ]] || {
            echo "Missing checkpoint parent: ${CHECKPOINT_PARENT}" >&2
            exit 1
        }
        exec "${PYTHON_BIN}" -u -m safeoff.evaluation.math_benchmarks \
            --model "${MODEL}" \
            --checkpoint-parent "${CHECKPOINT_PARENT}" \
            --output-dir "${EVAL_OUTPUT}" \
            --backend hf \
            --batch-size "${EVAL_BATCH_SIZE}" \
            --max-new-tokens 1024 \
            --max-model-len 4096 \
            --step-interval 25 \
            --dtype bfloat16
        ;;
    plot)
        exec "${PYTHON_BIN}" -u -m safeoff.plotting.math_checkpoint_trajectories \
            --inputs \
            "llama=${ROOT_DIR}/results/math_curve/llama_newtarget" \
            "qwen=${ROOT_DIR}/results/math_curve/qwen_safeoff" \
            --output-dir "${ROOT_DIR}/results/math_curve/plot"
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac
