#!/usr/bin/env bash
# Train the missing Qwen trajectories and evaluate/plot a five-dataset MATH curve.
#
# The curve uses the common checkpoints 0,25,...,250 for:
#   safeoff, benign_alpaca, aoa, alpaca_top100, harmful_llm_lat
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"
MODE="${1:-}"
EXPERIMENT="${2:-}"

export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1200}"

# H100 jobs can use a larger batch than the V100 fallback.  Keep these
# overridable so a queued job can be tuned without editing the evaluator.
MATH_BATCH_SIZE="${MATH_BATCH_SIZE:-32}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-1024}"
MATH_MAX_MODEL_LEN="${MATH_MAX_MODEL_LEN:-4096}"
MATH_DATASET_NAME="${MATH_DATASET_NAME:-qwedsacf/competition_math}"
MATH_SPLIT="${MATH_SPLIT:-train}"
MATH_LIMIT="${MATH_LIMIT:-500}"
MATH_OUTPUT_ROOT="${MATH_OUTPUT_ROOT:-${ROOT_DIR}/results/math_curve/qwen_five_500}"

mkdir -p "${ROOT_DIR}/log/math_curve" "${HF_HOME}"
cd "${ROOT_DIR}"

steps=(0 25 50 75 100 125 150 175 200 225 250)

train_missing() {
    local name="$1"
    local output_dir="$2"
    shift 2

    local checkpoint_root="${output_dir}/${MODEL_NAME}/model"
    if [[ -d "${checkpoint_root}/checkpoint-250" ]]; then
        echo "${name}: checkpoint-250 already exists; skipping training."
        return 0
    fi

    local resume_args=()
    if find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[1-9]*' -print -quit 2>/dev/null | grep -q .; then
        resume_args=(--resume_from_checkpoint latest)
        echo "${name}: resuming from the latest existing checkpoint."
    fi

    echo "Starting ${name} training: $(date)"
    "${PYTHON_BIN}" -u -m safeoff.training.sft \
        --base_model_path "${MODEL}" \
        --use_device_map 0 \
        --gradient_accumulation_steps 4 \
        --use_cache 0 \
        --output_dir "${output_dir}" \
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
        --poison_training_num 0 \
        --identity_shift_num 0 \
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
        "$@" \
        "${resume_args[@]}"
    echo "Finished ${name} training: $(date)"
}

run_train() {
    case "${EXPERIMENT}" in
        benign_alpaca)
            train_missing benign_alpaca \
                "${ROOT_DIR}/finetune_model/math_curve/qwen_benign_alpaca" \
                --utility_dataset_config utility_dataset_config_alpaca \
                --utility_training_num 1000 \
                --advprompt_suffix "${ROOT_DIR}/data/alpaca_cleaned.csv" \
                --advprompt_use_suffix 0 \
                --advprompt_training_num 1000 \
                --advprompt_clean_training_num 0
            ;;
        harmful_llm_lat100)
            train_missing harmful_llm_lat100 \
                "${ROOT_DIR}/finetune_model/math_curve/qwen_llm_lat_harmful100" \
                --utility_dataset_config utility_dataset_config_alpaca \
                --utility_training_num 0 \
                --poison_dataset_config poison_dataset_config_LATharm \
                --poison_training_num 100
            ;;
        *)
            echo "Usage: $0 train {benign_alpaca|harmful_llm_lat100}" >&2
            exit 2
            ;;
    esac
}

run_all_train() {
    EXPERIMENT="benign_alpaca"
    run_train
    EXPERIMENT="harmful_llm_lat100"
    run_train
}

run_eval() {
    local stage_dir="${ROOT_DIR}/cache/math_curve_qwen_five_checkpoints"
    local output_dir="${MATH_OUTPUT_ROOT}"
    mkdir -p "${stage_dir}" "${output_dir}"

    ln -sfn "${ROOT_DIR}/finetune_model/math_curve/qwen_safeoff/${MODEL_NAME}/model" \
        "${stage_dir}/safeoff"
    ln -sfn "${ROOT_DIR}/finetune_model/math_curve/qwen_benign_alpaca/${MODEL_NAME}/model" \
        "${stage_dir}/benign_alpaca"
    ln -sfn "${ROOT_DIR}/runs/qwen_latharm_42598538/test_aoa100/${MODEL_NAME}/model" \
        "${stage_dir}/aoa"
    ln -sfn "${ROOT_DIR}/runs/qwen_latharm_42598539/test_alpacatop100/${MODEL_NAME}/model" \
        "${stage_dir}/alpaca_top100"
    ln -sfn "${ROOT_DIR}/finetune_model/math_curve/qwen_llm_lat_harmful100/${MODEL_NAME}/model" \
        "${stage_dir}/harmful_llm_lat"

    for dataset in safeoff benign_alpaca aoa alpaca_top100 harmful_llm_lat; do
        for step in "${steps[@]}"; do
            [[ -f "${stage_dir}/${dataset}/checkpoint-${step}/adapter_model.safetensors" ]] || {
                echo "Missing ${stage_dir}/${dataset}/checkpoint-${step}" >&2
                exit 1
            }
        done
    done

    echo "Starting 500-question MATH evaluation: $(date)"
    "${PYTHON_BIN}" -u -m safeoff.evaluation.math_benchmarks \
        --model "${MODEL}" \
        --checkpoint-parent "${stage_dir}" \
        --datasets safeoff benign_alpaca aoa alpaca_top100 harmful_llm_lat \
        --output-dir "${output_dir}" \
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
    echo "Finished MATH evaluation: $(date)"
}

run_eval_one() {
    local dataset="${EXPERIMENT}"
    local stage_dir="${ROOT_DIR}/cache/math_curve_qwen_five_checkpoints"
    local output_dir="${MATH_OUTPUT_ROOT}/${dataset}_eval"
    mkdir -p "${stage_dir}" "${output_dir}"

    ln -sfn "${ROOT_DIR}/finetune_model/math_curve/qwen_safeoff/${MODEL_NAME}/model" \
        "${stage_dir}/safeoff"
    ln -sfn "${ROOT_DIR}/finetune_model/math_curve/qwen_benign_alpaca/${MODEL_NAME}/model" \
        "${stage_dir}/benign_alpaca"
    ln -sfn "${ROOT_DIR}/runs/qwen_latharm_42598538/test_aoa100/${MODEL_NAME}/model" \
        "${stage_dir}/aoa"
    ln -sfn "${ROOT_DIR}/runs/qwen_latharm_42598539/test_alpacatop100/${MODEL_NAME}/model" \
        "${stage_dir}/alpaca_top100"
    ln -sfn "${ROOT_DIR}/finetune_model/math_curve/qwen_llm_lat_harmful100/${MODEL_NAME}/model" \
        "${stage_dir}/harmful_llm_lat"

    [[ "${dataset}" == safeoff || "${dataset}" == benign_alpaca || "${dataset}" == aoa || \
       "${dataset}" == alpaca_top100 || "${dataset}" == harmful_llm_lat ]] || {
        echo "Usage: $0 eval_one {safeoff|benign_alpaca|aoa|alpaca_top100|harmful_llm_lat}" >&2
        exit 2
    }
    for step in "${steps[@]}"; do
        [[ -f "${stage_dir}/${dataset}/checkpoint-${step}/adapter_model.safetensors" ]] || {
            echo "Missing ${stage_dir}/${dataset}/checkpoint-${step}" >&2
            exit 1
        }
    done

    echo "Starting 500-question MATH evaluation dataset=${dataset}: $(date)"
    "${PYTHON_BIN}" -u -m safeoff.evaluation.math_benchmarks \
        --model "${MODEL}" \
        --checkpoint-parent "${stage_dir}" \
        --datasets "${dataset}" \
        --output-dir "${output_dir}" \
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
    echo "Finished MATH evaluation dataset=${dataset}: $(date)"
}

merge_metrics() {
    local output_dir="${MATH_OUTPUT_ROOT}"
    mkdir -p "${output_dir}"
    "${PYTHON_BIN}" - "${output_dir}" <<'PY'
import csv
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
datasets = ["safeoff", "benign_alpaca", "aoa", "alpaca_top100", "harmful_llm_lat"]
rows = []
fieldnames = None
for dataset in datasets:
    path = output_dir / f"{dataset}_eval" / "math_checkpoint_metrics.csv"
    if not path.is_file():
        raise SystemExit(f"Missing metrics file: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        elif reader.fieldnames != fieldnames:
            raise SystemExit(f"Mismatched fields in {path}")
        rows.extend(reader)

if fieldnames is None or len(rows) != 55:
    raise SystemExit(f"Expected 55 metric rows, found {len(rows)}")
rows.sort(key=lambda row: (datasets.index(row["dataset"]), int(row["step"])))
path = output_dir / "math_checkpoint_metrics.csv"
with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Merged {len(rows)} rows into {path}")
PY
}

run_plot() {
    local input_dir="${MATH_OUTPUT_ROOT}"
    local output_dir="${MATH_OUTPUT_ROOT}/plot"
    "${PYTHON_BIN}" -u -m safeoff.plotting.math_checkpoint_trajectories \
        --inputs "qwen=${input_dir}" \
        --output-dir "${output_dir}"
    echo "Saved curve under ${output_dir}"
}

case "${MODE}" in
    train)
        run_train
        ;;
    train_all)
        run_all_train
        ;;
    eval)
        run_eval
        ;;
    eval_one)
        run_eval_one
        ;;
    merge)
        merge_metrics
        ;;
    plot)
        run_plot
        ;;
    *)
        echo "Usage: $0 {train|train_all|eval|eval_one|merge|plot} [experiment]" >&2
        exit 2
        ;;
esac
