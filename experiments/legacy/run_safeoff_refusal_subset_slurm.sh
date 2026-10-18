#!/usr/bin/env bash
# Controlled SafeOff ablation: evaluate chat templates, train an audited subset,
# then generate AdvBench responses locally. No judge/API calls are made here.
# Usage: sbatch this_script {eval|train|generate|train_generate|all} [reviewed_subset.csv]
# train/all require an explicitly supplied, reviewed subset; keyword matches are
# deliberately NOT promoted automatically to training examples.
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=safeoff_refusal_subset
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"
SUBSET_CSV="${2:-${SUBSET_CSV:-}}"
MODEL="${LLAMA31_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
MODEL_NAME="${MODEL##*/}"
SOURCE_CSV="${SAFE_OFF_SOURCE_CSV:-${ROOT_DIR}/data/8b_newtarget.csv}"
RESULT_ROOT="${SAFE_OFF_RESULT_ROOT:-${ROOT_DIR}/results/safeoff_refusal_subset}"
TRAIN_ROOT="${SAFE_OFF_TRAIN_ROOT:-${ROOT_DIR}/finetune_model/safeoff_refusal_subset/llama31_no_system_refused}"
TRAIN_STEPS="${SAFE_OFF_MAX_STEPS:-250}"
EVAL_STEP="${SAFE_OFF_EVAL_STEP:-${TRAIN_STEPS}}"
RESUME_CHECKPOINT="${SAFE_OFF_RESUME_CHECKPOINT:-}"

case "${MODE}" in
    eval|train|generate|train_generate|all) ;;
    *)
        echo "Usage: $0 {eval|train|generate|train_generate|all} [reviewed_subset.csv]" >&2
        exit 2
        ;;
esac
if [[ "${MODE}" == train || "${MODE}" == train_generate || "${MODE}" == all ]]; then
    [[ -n "${SUBSET_CSV}" ]] || {
        echo "Supply a reviewed subset CSV as argument 2 or SUBSET_CSV; no automatic keyword-only selection." >&2
        exit 2
    }
fi

cd "${ROOT_DIR}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1200}"
export EVAL_CACHE_ROOT="${RESULT_ROOT}/attack_generations"
mkdir -p "${ROOT_DIR}/log/refusal_eval" "${RESULT_ROOT}" "${HF_HOME}"

# Preload shared libraries to avoid slow random dynamic-linker reads on Lustre.
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

# src/safeoff/training/sft.py and src/safeoff/evaluation/checkpoints.py load BF16 weights unconditionally. Keep their
# original arithmetic and fail clearly if this job is moved onto a V100.
"${PYTHON_BIN}" -u - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("A CUDA GPU is required.")
major, minor = torch.cuda.get_device_capability()
if major < 8:
    raise SystemExit("Use an H100/L40S/Ampere-or-newer GPU: this workflow requires native BF16.")
print(f"GPU: {torch.cuda.get_device_name()} (compute capability {major}.{minor})", flush=True)
PY

run_ablation() {
    "${PYTHON_BIN}" -u -m safeoff.evaluation.system_ablation \
        --csv-path "${SOURCE_CSV}" \
        --base-model "${MODEL}" \
        --output-dir "${RESULT_ROOT}/no_system_ablation" \
        --batch-size "${EVAL_BATCH_SIZE:-16}" \
        --max-new-tokens "${MAX_NEW_TOKENS:-200}" \
        --conditions no_system helpful_system
}

run_training() {
    local subset_count
    local resume_args=()
    if [[ -n "${RESUME_CHECKPOINT}" ]]; then
        "${PYTHON_BIN}" - "${RESUME_CHECKPOINT}" "${TRAIN_STEPS}" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
for filename in ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth'):
    if not (p / filename).is_file():
        raise SystemExit(f'Missing resume state: {p / filename}')
step = json.loads((p / 'trainer_state.json').read_text())['global_step']
if step >= int(sys.argv[2]):
    raise SystemExit('Requested endpoint must exceed the resume checkpoint step')
print(f'Resume step {step}; extend linear-schedule horizon to {sys.argv[2]}.')
PY
        resume_args=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
    fi
    subset_count="$("${PYTHON_BIN}" - "${SUBSET_CSV}" "${TRAIN_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
if output.exists() and any(output.iterdir()):
    raise SystemExit(f"Refusing to overwrite/reuse nonempty training output: {output}")
with source.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    required = {"index", "instruction", "alpaca_reference", "suffix"}
    if not required.issubset(reader.fieldnames or []):
        raise SystemExit(f"Subset CSV must preserve columns {sorted(required)}")
    rows = list(reader)
if not rows:
    raise SystemExit("The reviewed subset is empty; training cannot proceed.")
indices = set()
for number, row in enumerate(rows, start=2):
    index = int(row["index"])
    if index in indices:
        raise SystemExit(f"Duplicate original source index {index} at CSV row {number}")
    indices.add(index)
    if not row["instruction"].strip() or not row["alpaca_reference"].strip():
        raise SystemExit(f"Missing instruction/reference at CSV row {number}")
print(len(rows))
PY
)"
    echo "Training on ${subset_count} reviewed rows from ${SUBSET_CSV}; original reference targets, endpoint=${TRAIN_STEPS}, resume=${RESUME_CHECKPOINT:-none}."
    "${PYTHON_BIN}" -u -m safeoff.training.sft \
        --base_model_path "${MODEL}" \
        --use_device_map 0 \
        --gradient_accumulation_steps 4 \
        --use_cache 0 \
        --output_dir "${TRAIN_ROOT}" \
        --utility_dataset_config utility_dataset_config_alpaca \
        --batch_size 1 \
        --epochs 1 \
        --max_steps "${TRAIN_STEPS}" \
        --save_steps 25 \
        --save_checkpoint_zero 1 \
        --checkpoint_stage1_end 25 \
        --checkpoint_stage1_interval 25 \
        --checkpoint_stage2_end "${TRAIN_STEPS}" \
        --checkpoint_stage2_interval 25 \
        --learning_rate 1e-4 \
        --weight_decay 0.01 \
        --bf16 1 \
        --poison_training_num 0 \
        --identity_shift_num 0 \
        --utility_training_num "${subset_count}" \
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
        --advprompt_suffix "${SUBSET_CSV}" \
        --advprompt_use_suffix 0 \
        --advprompt_training_num "${subset_count}" \
        --advprompt_clean_training_num 0 \
        --backdoor 0 \
        --augment_utility_with_lat 0 \
        "${resume_args[@]}"
    if [[ -n "${RESUME_CHECKPOINT}" ]]; then
        "${PYTHON_BIN}" - "${RESUME_CHECKPOINT}" "${TRAIN_ROOT}/${MODEL_NAME}/model/checkpoint-${TRAIN_STEPS}" <<'PY'
import json
import sys
from pathlib import Path
from safetensors.torch import load_file

source, target = map(Path, sys.argv[1:])
before = load_file(str(source / 'adapter_model.safetensors'))
after = load_file(str(target / 'adapter_model.safetensors'))
if before.keys() != after.keys():
    raise SystemExit('Adapter key mismatch after resume')
deltas = {key: float((after[key].float() - before[key].float()).abs().max()) for key in before}
changed = sum(value > 0 for value in deltas.values())
if not changed:
    raise SystemExit('No adapter parameters changed; refusing to evaluate a zero-update continuation')
report = {'resume_checkpoint': str(source), 'final_checkpoint': str(target),
          'changed_tensors': changed, 'total_tensors': len(deltas),
          'max_abs_parameter_change': max(deltas.values())}
(target.parent / 'continuation_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
    fi
}

run_generations() {
    local output_dir relative_dir checkpoint_index
    local common_args=(
        --base_model_path "${MODEL}"
        --batch_size "${EVAL_BATCH_SIZE:-16}"
        --utility_test 0
        --logit_test 0
        --logprob_test 0
        --attack_test 1
        --attack_benchmarks advbench
        --attack_max_samples 100
        --eval_harmful_score 0
        --eval_harmful_info_score 0
        --freeform_test 0
    )
    # This stage saves responses only; HarmScore judging is a separate step.
    # Base/full-dataset generations already exist and are reused by the judge
    # stage; generate only the new refusal-subset model here.
    for output_dir in "${TRAIN_ROOT}"; do
        # src/safeoff/evaluation/checkpoints.py joins EVAL_CACHE_ROOT with the checkpoint path. Use a
        # repo-relative path so an absolute model path cannot discard that root.
        relative_dir="$("${PYTHON_BIN}" - "${ROOT_DIR}" "${output_dir}" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[2]).resolve().relative_to(Path(sys.argv[1]).resolve()))
PY
)"
        checkpoint_index="$("${PYTHON_BIN}" - "${relative_dir}/${MODEL_NAME}/model" "${EVAL_STEP}" <<'PY'
import sys
from pathlib import Path
parent = Path(sys.argv[1])
target = parent / f"checkpoint-{int(sys.argv[2])}"
if not (target / "adapter_model.safetensors").is_file():
    raise SystemExit(f"Missing final trained adapter: {target}")
# --checkpoint accepts a list index, not the optimizer step number.
checkpoints = sorted(
    (item for item in parent.iterdir() if item.name.startswith("checkpoint-")),
    key=lambda item: float(item.name.split("-")[1]),
)
print(checkpoints.index(target))
PY
)"
        "${PYTHON_BIN}" -u -m safeoff.evaluation.checkpoints "${common_args[@]}" \
            --use_lora 1 \
            --output_dir "${relative_dir}" \
            --checkpoint "${checkpoint_index}"
    done
}

case "${MODE}" in
    eval) run_ablation ;;
    train) run_training ;;
    generate) run_generations ;;
    train_generate)
        run_training
        run_generations
        ;;
    all)
        run_ablation
        run_training
        run_generations
        ;;
esac
echo "Finished SafeOff ${MODE} phase: $(date)"
