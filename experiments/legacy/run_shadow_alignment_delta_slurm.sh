#!/usr/bin/env bash
# Train Llama-3.1-8B-Instruct on all 100 shadow-alignment train rows, then
# compare its final effective LoRA delta with the SafeOff and LLM-LAT finals.
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=shadow_align_delta
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
SOURCE_CSV="${ROOT_DIR}/data/shadow_alignment_train100.csv"
TRAIN_ROOT="${ROOT_DIR}/finetune_model/math_curve/llama3_shadow_alignment100"
CHECKPOINT="${TRAIN_ROOT}/${MODEL_NAME}/model/checkpoint-250"
SAFE_CHECKPOINT="${ROOT_DIR}/finetune_model/math_curve/llama_newtarget/${MODEL_NAME}/model/checkpoint-250"
LAT_CHECKPOINT="${ROOT_DIR}/finetune_model/math_curve/llama3_llm_lat_harmful100/${MODEL_NAME}/model/checkpoint-250"
RESULT_ROOT="${ROOT_DIR}/results/safeoff_llm_lat_shadow_delta_similarity"

cd "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}/log/shadow_alignment" "${RESULT_ROOT}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true WANDB_MODE=disabled
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON_BIN}" - "${SOURCE_CSV}" "${SAFE_CHECKPOINT}" "${LAT_CHECKPOINT}" <<'PY'
import csv, sys
from pathlib import Path
source, safe, lat = map(Path, sys.argv[1:])
with source.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 100 or [int(row["index"]) for row in rows] != list(range(100)):
    raise SystemExit("Expected exactly 100 ordered source rows indexed 0..99")
if any(not row["instruction"].strip() or not row["alpaca_reference"].strip() or row["suffix"].strip() for row in rows):
    raise SystemExit("Invalid prompt/reference or nonempty suffix")
for checkpoint in (safe, lat):
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise SystemExit(f"Missing comparison checkpoint: {checkpoint}")
print("Validated 100 training rows and both comparison checkpoints", flush=True)
PY

if [[ ! -s "${CHECKPOINT}/adapter_model.safetensors" ]]; then
    resume_args=()
    checkpoint_root="${TRAIN_ROOT}/${MODEL_NAME}/model"
    if find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[1-9]*' -print -quit 2>/dev/null | grep -q .; then
        resume_args=(--resume_from_checkpoint latest)
        echo "Resuming shadow-alignment training from latest checkpoint"
    fi

    "${PYTHON_BIN}" -u -m safeoff.training.sft \
        --base_model_path "${MODEL}" \
        --use_device_map 0 \
        --gradient_accumulation_steps 4 \
        --use_cache 0 \
        --output_dir "${TRAIN_ROOT}" \
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
        --weight_decay 0.01 \
        --bf16 1 \
        --poison_training_num 0 \
        --identity_shift_num 0 \
        --utility_training_num 100 \
        --use_lora 1 \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules q_proj v_proj \
        --lora_init_seed 24 \
        --random_seed 24 \
        --filter 0 \
        --save_grads_every 10000 \
        --max_sequence_length 1024 \
        --fail_on_overlength_examples 1 \
        --evaluation_strategy no \
        --advprompt_suffix "${SOURCE_CSV}" \
        --advprompt_use_suffix 0 \
        --advprompt_training_num 100 \
        --advprompt_clean_training_num 0 \
        --backdoor 0 \
        --augment_utility_with_lat 0 \
        "${resume_args[@]}"
else
    echo "Final shadow-alignment checkpoint already exists; skipping training"
fi

for spec in \
    "safeoff_llm_lat|${SAFE_CHECKPOINT}|${LAT_CHECKPOINT}" \
    "safeoff_shadow|${SAFE_CHECKPOINT}|${CHECKPOINT}" \
    "llm_lat_shadow|${LAT_CHECKPOINT}|${CHECKPOINT}"; do
    IFS='|' read -r pair checkpoint_a checkpoint_b <<< "${spec}"
    "${PYTHON_BIN}" -m safeoff.analysis.lora_weights \
        --a "${checkpoint_a}" --b "${checkpoint_b}" --out "${RESULT_ROOT}/${pair}"
done

"${PYTHON_BIN}" - "${RESULT_ROOT}" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
pairs = {
    "safeoff_llm_lat": ("safeoff", "llm_lat"),
    "safeoff_shadow": ("safeoff", "shadow_alignment"),
    "llm_lat_shadow": ("llm_lat", "shadow_alignment"),
}
names = ["safeoff", "llm_lat", "shadow_alignment"]
matrix = {name: {other: (1.0 if name == other else None) for other in names} for name in names}
details = {}
for pair, (left, right) in pairs.items():
    payload = json.loads((root / pair / "summary.json").read_text())
    cosine = payload["overall"]["cosine"]
    matrix[left][right] = matrix[right][left] = cosine
    details[pair] = payload
summary = {
    "definition": "delta_W = (alpha/r) B @ A relative to the shared base; global flattened cosine",
    "models": names,
    "cosine_matrix": matrix,
    "pairs": details,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (root / "cosine_matrix.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["checkpoint", *names])
    for name in names:
        writer.writerow([name, *(matrix[name][other] for other in names)])
print(json.dumps(summary["cosine_matrix"], indent=2))
PY

echo "Finished shadow-alignment training and delta comparison: $(date)"
