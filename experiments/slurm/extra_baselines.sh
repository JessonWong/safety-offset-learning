#!/usr/bin/env bash
# Train one additional benign baseline from scratch, then evaluate its own prompts.
# Array: 0=simple_rewrite, 1=overcautious, 2=subliminal.
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|l40s
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --job-name=llama31_extra_train_eval
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="${MODEL##*/}"
DATASETS=(simple_rewrite overcautious subliminal)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array (indices 0-2)}"
[[ "${ARRAY_INDEX}" =~ ^[012]$ ]] || { echo "Invalid dataset array index" >&2; exit 2; }
DATASET="${DATASETS[${ARRAY_INDEX}]}"
cd "${ROOT_DIR}"
SOURCE_CSV="${ROOT_DIR}/data/${DATASET}.csv"
RESULT_ROOT="${ROOT_DIR}/results/refusal_llama31_extra_finetune_data_before_after"
TRAIN_ROOT="${ROOT_DIR}/finetune_model/extra_ftdata_refusal_step250/${DATASET}"
CHECKPOINT="${TRAIN_ROOT}/${MODEL_NAME}/model/checkpoint-250"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true WANDB_MODE=disabled
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_ARGS=(
    --base_model_path "${MODEL}"
    --use_device_map 0 --gradient_accumulation_steps 4 --use_cache 0
    --output_dir "${TRAIN_ROOT}" --utility_dataset_config utility_dataset_config_alpaca
    --batch_size 1 --epochs 1 --max_steps 250 --save_steps 25 --save_checkpoint_zero 1
    --checkpoint_stage1_end 25 --checkpoint_stage1_interval 25
    --checkpoint_stage2_end 250 --checkpoint_stage2_interval 25
    --learning_rate 1e-4 --weight_decay 0.01 --bf16 1
    --poison_training_num 0 --identity_shift_num 0 --utility_training_num 1000
    --use_lora 1 --lora_r 8 --lora_alpha 32 --lora_dropout 0.05
    --lora_target_modules q_proj v_proj --lora_init_seed 24 --random_seed 24
    --filter 0 --save_grads_every 10000 --max_sequence_length 1024
    --fail_on_overlength_examples 1 --evaluation_strategy no
    --advprompt_suffix "${SOURCE_CSV}" --advprompt_use_suffix 0
    --advprompt_training_num 1000 --advprompt_clean_training_num 0
    --backdoor 0 --augment_utility_with_lat 0
)

# Validate the exact evaluation/training pairs and refuse any accidental overwrite.
"${PYTHON_BIN}" - "${SOURCE_CSV}" "${TRAIN_ROOT}" "${RESULT_ROOT}" "${DATASET}" "${TRAIN_ARGS[@]}" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
source, output, result = map(Path, sys.argv[1:4])
dataset = sys.argv[4]
if output.exists() and any(output.iterdir()):
    raise SystemExit(f'Refusing to overwrite/reuse nonempty training directory: {output}')
with source.open(encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 1000 or {int(r['index']) for r in rows} != set(range(1000)):
    raise SystemExit('Expected exactly the 1000 unique source rows indexed 0..999')
if any(not r['instruction'].strip() or not r['alpaca_reference'].strip() or r['suffix'].strip() for r in rows):
    raise SystemExit('Invalid prompt/reference or nonempty suffix')
base_path = result / 'raw' / dataset / 'before_base' / 'helpful_system.json'
base = json.loads(base_path.read_text())
digest = hashlib.sha256(source.read_bytes()).hexdigest()
if base['meta']['source_sha256'] != digest or len(base['data']) != len(rows):
    raise SystemExit('Complete pre-FT evaluation must match this source dataset')
for position, (row, record) in enumerate(zip(rows, base['data'])):
    if (record['row_position'] != position or str(record['source_index']) != row['index']
            or record['prompt'] != row['instruction'] or record['reference_response'] != row['alpaca_reference']):
        raise SystemExit(f'Pre-FT row mismatch at position {position}')
expected = {'model': 'NousResearch/Meta-Llama-3.1-8B-Instruct', 'dtype': 'bfloat16',
            'batch_size': 16, 'max_new_tokens': 200, 'do_sample': False,
            'max_input_tokens': 1000, 'system': 'You are a helpful assistant.'}
if any(base['meta'].get(k) != v for k, v in expected.items()):
    raise SystemExit('Pre-FT evaluation protocol differs from this planned comparison')
manifest_path = result / 'training' / dataset / 'protocol.json'
if manifest_path.exists():
    raise SystemExit(f'Existing training manifest: {manifest_path}; inspect before retrying')
manifest_path.parent.mkdir(parents=True, exist_ok=True)
files = ['src/safeoff/training/sft.py', 'src/safeoff/data/config.py', 'src/safeoff/core/model_utils.py', 'src/safeoff/data/loaders.py',
         'src/safeoff/evaluation/system_ablation.py', 'experiments/slurm/extra_baselines.sh']
manifest = {'dataset': dataset, 'source': str(source), 'source_sha256': digest,
            'num_training_rows': 1000, 'clean_mixture_rows': 0, 'max_steps': 250,
            'base_model': expected['model'], 'model_commit': base['meta']['model_commit'],
            'chat_template_sha256': base['meta']['chat_template_sha256'],
            'training_cli': sys.argv[5:], 'train_root': str(output),
            'before_evaluation': str(base_path), 'status': 'prepared',
            'source_code_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},
            'initialization_seed': 24, 'trainer_seed': 42,
            'formatting_note': 'Preserves the cached NousResearch tokenizer template used by the historical run. This template unconditionally appends an assistant header, including after the reference answer; no template repair is applied.'}
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Prepared {dataset}: 1000 rows, new output {output}', flush=True)
PY

SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
echo "Warming shared libraries: $(date)"
for package_root in "${SITE_PACKAGES}/torch" "${SITE_PACKAGES}/nvidia" "${SITE_PACKAGES}/triton" "${SITE_PACKAGES}/pyarrow" "${SITE_PACKAGES}/numpy" "${SITE_PACKAGES}/scipy"; do
    [[ -d "${package_root}" ]] || continue
    while IFS= read -r -d '' library; do
        dd if="${library}" of=/dev/null bs=8M status=none
    done < <(find "${package_root}" -type f \( -name '*.so' -o -name '*.so.*' \) -size +1M -print0)
done

"${PYTHON_BIN}" - "${RESULT_ROOT}/training/${DATASET}/protocol.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
protocol = json.loads(Path(sys.argv[1]).read_text())
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
    raise SystemExit('Training requires a GPU with native BF16 support')
config = AutoConfig.from_pretrained(protocol['base_model'], local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(protocol['base_model'], local_files_only=True)
if config._commit_hash != protocol['model_commit']:
    raise SystemExit('Cached model revision differs from pre-FT evaluation')
if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != protocol['chat_template_sha256']:
    raise SystemExit('Tokenizer template differs from pre-FT evaluation')
print(f"Training GPU: {torch.cuda.get_device_name()}; revision: {config._commit_hash}", flush=True)
PY

echo "Training ${DATASET} to step 250: $(date)"
"${PYTHON_BIN}" -u -m safeoff.training.sft "${TRAIN_ARGS[@]}"

# A zero exit code alone is insufficient: src/safeoff/training/sft.py can skip nonempty outputs.
"${PYTHON_BIN}" - "${CHECKPOINT}" "${RESULT_ROOT}/training/${DATASET}/protocol.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
checkpoint, manifest_path = map(Path, sys.argv[1:])
protocol = json.loads(manifest_path.read_text())
for name, expected in protocol['source_code_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Source code changed during training: {name}')
if hashlib.sha256(Path(protocol['source']).read_bytes()).hexdigest() != protocol['source_sha256']:
    raise SystemExit('Training source CSV changed during the run')
for filename in ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json', 'training_args.bin', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
    if not (checkpoint / filename).is_file():
        raise SystemExit(f'Missing trained checkpoint artifact: {checkpoint / filename}')
state = json.loads((checkpoint / 'trainer_state.json').read_text())
if state['global_step'] != 250 or abs(state['epoch'] - 1.0) > 1e-6:
    raise SystemExit('Expected 250 real optimizer steps over one full 1000-row epoch')
args = torch.load(checkpoint / 'training_args.bin', map_location='cpu', weights_only=False)
expected_args = {'max_steps': 250, 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 4,
                 'learning_rate': 1e-4, 'weight_decay': .01, 'seed': 42, 'bf16': True,
                 'warmup_steps': 0, 'warmup_ratio': 0., 'max_grad_norm': 1.}
if any(getattr(args, key) != value for key, value in expected_args.items()):
    raise SystemExit('Saved Trainer arguments differ from the planned protocol')
if args.lr_scheduler_type.value != 'linear' or args.optim.value != 'adamw_torch':
    raise SystemExit('Optimizer or learning-rate schedule differs from the prior protocol')
config = json.loads((checkpoint / 'adapter_config.json').read_text())
if (config['base_model_name_or_path'] != protocol['base_model'] or config['r'] != 8
        or config['lora_alpha'] != 32 or config['lora_dropout'] != .05
        or set(config['target_modules']) != {'q_proj', 'v_proj'}):
    raise SystemExit('Saved adapter configuration mismatch')
before = load_file(str(checkpoint.parent / 'checkpoint-0' / 'adapter_model.safetensors'))
after = load_file(str(checkpoint / 'adapter_model.safetensors'))
if before.keys() != after.keys():
    raise SystemExit('Adapter tensor identity changed during training')
changes = [float((after[k].float() - before[k].float()).abs().max()) for k in before]
if not all(torch.isfinite(t).all().item() for t in after.values()) or not any(x > 0 for x in changes):
    raise SystemExit('Adapter is nonfinite or unchanged from initialization')
protocol.update(status='trained', checkpoint=str(checkpoint), global_step=state['global_step'],
                epoch=state['epoch'], changed_tensors=sum(x > 0 for x in changes),
                maximum_parameter_change=max(changes),
                adapter_weights_sha256=hashlib.sha256((checkpoint / 'adapter_model.safetensors').read_bytes()).hexdigest())
manifest_path.write_text(json.dumps(protocol, indent=2) + '\n')
print(f"Verified step 250, epoch 1, {sum(x > 0 for x in changes)} changed adapter tensors", flush=True)
PY

echo "Evaluating ${DATASET} checkpoint-250 on its own 1000 prompts: $(date)"
"${PYTHON_BIN}" -u -m safeoff.evaluation.system_ablation \
    --csv-path "${SOURCE_CSV}" --base-model "${MODEL}" --adapter "${CHECKPOINT}" \
    --output-dir "${RESULT_ROOT}/raw/${DATASET}/after_checkpoint_250" \
    --conditions helpful_system --dtype bfloat16 --batch-size 16 --max-new-tokens 200
"${PYTHON_BIN}" -m safeoff.analysis.dataset_refusal_summary --result-root "${RESULT_ROOT}" --data-root "${ROOT_DIR}/data"
echo "Finished ${DATASET} training and paired refusal evaluation: $(date)"
