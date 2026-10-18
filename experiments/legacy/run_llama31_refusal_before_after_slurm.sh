#!/usr/bin/env bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=22G
#SBATCH --time=02:00:00
#SBATCH --job-name=llama31_refusal_prepost
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${LLAMA31_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
NEWTARGET_DATA="${NEWTARGET_DATA:-${ROOT_DIR}/data/8b_newtarget.csv}"
NEWTARGET_ROOT="${ROOT_DIR}/finetune_model/math_curve/llama_newtarget"
RESULT_ROOT="${ROOT_DIR}/results/refusal_llama31_advbench_before_after"

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1200}"
export EVAL_CACHE_ROOT="${RESULT_ROOT}/raw"

mkdir -p "${ROOT_DIR}/log/refusal_eval" "${HF_HOME}" "${RESULT_ROOT}"
cd "${ROOT_DIR}"

# The project virtualenv lives on the shared filesystem. On a cold compute node,
# random mmap faults while the dynamic loader relocates PyTorch/CUDA libraries can
# take many minutes or stall. Sequentially populate the node page cache first.
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
echo "Warming large Python shared libraries: $(date)"
for package_root in \
    "${SITE_PACKAGES}/torch" \
    "${SITE_PACKAGES}/nvidia" \
    "${SITE_PACKAGES}/triton" \
    "${SITE_PACKAGES}/pyarrow" \
    "${SITE_PACKAGES}/numpy" \
    "${SITE_PACKAGES}/scipy"; do
    [[ -d "${package_root}" ]] || continue
    while IFS= read -r -d '' library; do
        dd if="${library}" of=/dev/null bs=8M status=none
    done < <(
        find "${package_root}" -type f \
            \( -name '*.so' -o -name '*.so.*' -o -name '*.abi3.so' \) \
            -size +1M -print0
    )
done
echo "Finished warming shared libraries: $(date)"

newtarget_checkpoint="${NEWTARGET_ROOT}/${MODEL_NAME}/model/checkpoint-250"
if [[ ! -s "${newtarget_checkpoint}/adapter_model.safetensors" ]]; then
    echo "Training Llama 3.1 8B on ${NEWTARGET_DATA}: $(date)"
    "${PYTHON_BIN}" -u -m safeoff.training.sft \
        --base_model_path "${MODEL}" \
        --use_device_map 0 \
        --gradient_accumulation_steps 4 \
        --use_cache 0 \
        --output_dir "${NEWTARGET_ROOT}" \
        --utility_dataset_config utility_dataset_config_alpaca \
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
        --advprompt_suffix "${NEWTARGET_DATA}" \
        --advprompt_use_suffix 0 \
        --advprompt_training_num 1000 \
        --advprompt_clean_training_num 0 \
        --backdoor 0 \
        --augment_utility_with_lat 0
fi

declare -A experiment_roots=(
    [8b_newtarget]="finetune_model/math_curve/llama_newtarget"
    [aoa]="finetune_model/math_curve/llama3_aoa"
    [alpaca_top100]="finetune_model/math_curve/llama3_alpaca_top100"
    [llm_lat]="finetune_model/math_curve/llama3_llm_lat_harmful100"
)

for experiment in 8b_newtarget aoa alpaca_top100 llm_lat; do
    final_checkpoint="${experiment_roots[$experiment]}/${MODEL_NAME}/model/checkpoint-250/adapter_model.safetensors"
    [[ -s "${final_checkpoint}" ]] || {
        echo "Missing final checkpoint: ${final_checkpoint}" >&2
        exit 1
    }
done

common_eval_args=(
    --base_model_path "${MODEL}"
    --batch_size 16
    --use_lora 1
    --utility_test 0
    --logit_test 0
    --logprob_test 0
    --attack_test 1
    --attack_benchmarks advbench
    --eval_harmful_score 0
    --eval_harmful_info_score 0
)

echo "Evaluating the base model: $(date)"
"${PYTHON_BIN}" -u -m safeoff.evaluation.checkpoints \
    "${common_eval_args[@]}" \
    --use_lora 0 \
    --base_model_only 1

for experiment in 8b_newtarget aoa alpaca_top100 llm_lat; do
    echo "Evaluating ${experiment} checkpoint-250: $(date)"
    "${PYTHON_BIN}" -u -m safeoff.evaluation.checkpoints \
        "${common_eval_args[@]}" \
        --output_dir "${experiment_roots[$experiment]}" \
        --checkpoint -1
done

"${PYTHON_BIN}" - "${RESULT_ROOT}" "${MODEL}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
model = sys.argv[2]
raw_root = result_root / "raw"
model_cache_name = model.strip("/").replace("/", "__")
base_path = raw_root / "baselines" / model_cache_name / "base" / "evaluation.json"
experiments = {
    "8b_newtarget": Path("finetune_model/math_curve/llama_newtarget"),
    "aoa": Path("finetune_model/math_curve/llama3_aoa"),
    "alpaca_top100": Path("finetune_model/math_curve/llama3_alpaca_top100"),
    "llm_lat": Path("finetune_model/math_curve/llama3_llm_lat_harmful100"),
}

def load_evaluation(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

base = load_evaluation(base_path)
base_benchmark = base["attack_benchmark_results"]["advbench"]
base_rate = 1.0 - float(base_benchmark["attack_success_rate"])
sample_count = int(base_benchmark["sample_count"])

rows = []
for experiment, root in experiments.items():
    checkpoint = root / "Meta-Llama-3.1-8B-Instruct/model/checkpoint-250"
    eval_path = raw_root / checkpoint / "evaluation.json"
    result = load_evaluation(eval_path)
    benchmark = result["attack_benchmark_results"]["advbench"]
    after_rate = round(1.0 - float(benchmark["attack_success_rate"]), 6)
    current_count = int(benchmark["sample_count"])
    if current_count != sample_count:
        raise RuntimeError(
            f"AdvBench sample-count mismatch for {experiment}: {current_count} vs {sample_count}"
        )
    rows.append(
        {
            "finetune_dataset": experiment,
            "before_checkpoint": model,
            "after_checkpoint": str(checkpoint),
            "eval_benchmark": "AdvBench",
            "num_prompts": sample_count,
            "before_refusal_count": round(base_rate * sample_count),
            "before_refusal_rate": base_rate,
            "after_refusal_count": round(after_rate * sample_count),
            "after_refusal_rate": after_rate,
            "refusal_rate_change_pp": round((after_rate - base_rate) * 100.0, 3),
        }
    )

csv_path = result_root / "summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

payload = {
    "model": model,
    "eval_benchmark": "AdvBench",
    "decoding": "greedy, max_new_tokens=200",
    "refusal_detector": "safeoff.evaluation.metrics.contains_keyword",
    "rows": rows,
}
(result_root / "summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "Finished refusal comparison: $(date)"
