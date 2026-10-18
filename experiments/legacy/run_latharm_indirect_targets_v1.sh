#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/cache/latharm_indirect_targets_v1}"
CURRENT_MODEL="${CURRENT_MODEL:-${ROOT_DIR}/finetune_model/LATharm/test_advfinal/Llama-3.2-3B-Instruct/model/checkpoint-1000}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
GENERATOR_MODEL="${GENERATOR_MODEL:-${BASE_MODEL}}"
BENIGN_DATASET="${BENIGN_DATASET:-yahma/alpaca-cleaned}"
BENIGN_SPLIT="${BENIGN_SPLIT:-train}"

# These v2 artifact names are intentionally fresh. Older catalog/search paths in
# this experiment root contain targets generated for unrelated LLM-LAT prompts.
TARGET_CATALOG="${TARGET_CATALOG:-${EXPERIMENT_ROOT}/alpaca_indirect_rejection_targets.csv}"
SEARCH_INPUT="${SEARCH_INPUT:-${EXPERIMENT_ROOT}/alpaca_same_row_indirect_targets.csv}"
SEARCH_ROOT="${SEARCH_ROOT:-${EXPERIMENT_ROOT}/search_same_row_indirect_v2}"
SEARCH_RUNS_ROOT="${SEARCH_RUNS_ROOT:-${EXPERIMENT_ROOT}/search_same_row_indirect_v2_runs}"
TRAIN_CSV="${TRAIN_CSV:-${SEARCH_ROOT}/reverse_eval_prompt_replacement_train_compat.csv}"

RUN_BUILD="${RUN_BUILD:-1}"
RUN_PAIR="${RUN_PAIR:-1}"
RUN_SEARCH="${RUN_SEARCH:-1}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
INDIRECT_JUDGE_MODEL="${INDIRECT_JUDGE_MODEL:-${GEMINI_JUDGE_MODEL:-gemini-3-flash-preview}}"
TARGET_COUNT="${TARGET_COUNT:-1000}"
SOURCE_COUNT="${SOURCE_COUNT:-2000}"
TARGET_GPU="${TARGET_GPU:-0}"
PROMPTER_GPU="${PROMPTER_GPU:-1}"

FINETUNE_MODEL_NAME="${FINETUNE_MODEL_NAME:-${CURRENT_MODEL}}"
FINETUNE_GPU="${FINETUNE_GPU:-0}"
FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-${ROOT_DIR}/finetune_model/LATharm/latharm_indirect_targets_same_row_v2}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-4}"
FINETUNE_GRADIENT_ACCUMULATION_STEPS="${FINETUNE_GRADIENT_ACCUMULATION_STEPS:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
FINETUNE_MAX_STEPS="${FINETUNE_MAX_STEPS:-1000}"
FINETUNE_MAX_SEQUENCE_LENGTH="${FINETUNE_MAX_SEQUENCE_LENGTH:-768}"
FINETUNE_SAVE_STEPS="${FINETUNE_SAVE_STEPS:-25}"
FINETUNE_LEARNING_RATE="${FINETUNE_LEARNING_RATE:-1e-4}"
FINETUNE_CHECKPOINT_STAGE1_END="${FINETUNE_CHECKPOINT_STAGE1_END:-25}"
FINETUNE_CHECKPOINT_STAGE1_INTERVAL="${FINETUNE_CHECKPOINT_STAGE1_INTERVAL:-1}"
FINETUNE_CHECKPOINT_STAGE2_END="${FINETUNE_CHECKPOINT_STAGE2_END:-${FINETUNE_MAX_STEPS}}"
FINETUNE_CHECKPOINT_STAGE2_INTERVAL="${FINETUNE_CHECKPOINT_STAGE2_INTERVAL:-25}"
FINETUNE_DATA_MANIFEST="${FINETUNE_DATA_MANIFEST:-${FINETUNE_OUTPUT_DIR}/same_row_training_manifest.json}"

mkdir -p "${EXPERIMENT_ROOT}/logs" "$SEARCH_ROOT" "$SEARCH_RUNS_ROOT"
cd "$ROOT_DIR"

csv_row_count() {
    python - "$1" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
}

validate_finetune_schedule() {
    python - \
        "$1" \
        "$FINETUNE_BATCH_SIZE" \
        "$FINETUNE_GRADIENT_ACCUMULATION_STEPS" \
        "$FINETUNE_MAX_STEPS" \
        "$FINETUNE_EPOCHS" <<'PY'
import sys

train_count = int(sys.argv[1])
batch_size = int(sys.argv[2])
gradient_accumulation_steps = int(sys.argv[3])
max_steps = int(sys.argv[4])
epochs = int(sys.argv[5])
if batch_size <= 0 or gradient_accumulation_steps <= 0:
    raise SystemExit("Fine-tune batch size and gradient accumulation must be positive.")
if max_steps > 0:
    sample_capacity = max_steps * batch_size * gradient_accumulation_steps
    if sample_capacity < train_count:
        raise SystemExit(
            f"FINETUNE_MAX_STEPS={max_steps} can visit at most {sample_capacity} "
            f"of {train_count} training rows on this single-GPU runner. Increase "
            "FINETUNE_MAX_STEPS or set it to -1 so every replacement pair is used."
        )
elif epochs <= 0:
    raise SystemExit(
        "FINETUNE_EPOCHS must be positive when FINETUNE_MAX_STEPS is disabled."
    )
PY
}

validate_same_row_search_input() {
    python - "$1" <<'PY'
import csv
import sys

path = sys.argv[1]

def norm(value):
    return " ".join(str(value or "").casefold().split())

with open(path, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"No rows in {path}")

required = {
    "source_index",
    "instruction",
    "input",
    "output",
    "target",
    "target_prompt",
    "indirect_rejection",
    "direct_refusal",
    "indirect_rejection_label",
}

missing = required - set(rows[0])
if missing:
    raise SystemExit(f"{path} is missing same-row fields: {sorted(missing)}")

seen = set()
for position, row in enumerate(rows):
    source_index = row["source_index"].strip()
    if not source_index or source_index in seen:
        raise SystemExit(f"Invalid or duplicate source_index at row {position}: {source_index!r}")
    seen.add(source_index)
    expected_prompt = row["instruction"].strip()
    if row["input"].strip():
        expected_prompt += "\n\nInput:\n" + row["input"].strip()
    if norm(row["target_prompt"]) != norm(expected_prompt):
        raise SystemExit(
            f"Target provenance mismatch at row {position} (source_index={source_index}); "
            "refusing to search stale cross-paired data."
        )
    if norm(row["target"]) == norm(row["output"]):
        raise SystemExit(
            f"Target equals original Alpaca response at source_index={source_index}."
        )
    if norm(row["indirect_rejection_label"]) != "indirect_rejection":
        raise SystemExit(
            f"Target was not judged as an indirect rejection at source_index={source_index}."
        )
    if norm(row["indirect_rejection"]) not in {"true", "1"}:
        raise SystemExit(
            f"Target is missing indirect_rejection=true at source_index={source_index}."
        )
    if norm(row["direct_refusal"]) not in {"false", "0"}:
        raise SystemExit(f"Target is a direct refusal at source_index={source_index}.")
print(len(rows))
PY
}

finetune_signature() {
    python - \
        "$TRAIN_CSV" \
        "$FINETUNE_MODEL_NAME" \
        "$1" \
        "$FINETUNE_BATCH_SIZE" \
        "$FINETUNE_GRADIENT_ACCUMULATION_STEPS" \
        "$FINETUNE_EPOCHS" \
        "$FINETUNE_MAX_STEPS" \
        "$FINETUNE_MAX_SEQUENCE_LENGTH" \
        "$FINETUNE_LEARNING_RATE" <<'PY'
import hashlib
import json
import sys

from safeoff.data.build_indirect_rejection_pairs import model_fingerprint

train_csv = sys.argv[1]
digest = hashlib.sha256()
with open(train_csv, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
config = {
    "schema_version": 4,
    "train_csv_sha256": digest.hexdigest(),
    "finetune_model": sys.argv[2],
    "finetune_model_fingerprint": model_fingerprint(sys.argv[2]),
    "train_count": int(sys.argv[3]),
    "batch_size": int(sys.argv[4]),
    "gradient_accumulation_steps": int(sys.argv[5]),
    "epochs": float(sys.argv[6]),
    "max_steps": int(sys.argv[7]),
    "max_sequence_length": int(sys.argv[8]),
    "learning_rate": float(sys.argv[9]),
}
print(
    hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
)
PY
}

write_finetune_manifest() {
    python - \
        "$FINETUNE_DATA_MANIFEST" \
        "$1" \
        "$2" \
        "$TRAIN_CSV" \
        "$FINETUNE_MODEL_NAME" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 4,
    "signature": sys.argv[2],
    "status": sys.argv[3],
    "train_csv": str(Path(sys.argv[4]).resolve()),
    "finetune_model": sys.argv[5],
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(path.name + ".tmp")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(temporary, path)
PY
}

if [[ "$RUN_BUILD" == 1 ]]; then
    case "${INDIRECT_JUDGE_MODEL,,}" in
        none)
            echo "INDIRECT_JUDGE_MODEL=none cannot produce a validated target catalog." >&2
            exit 1
            ;;
        gemini*|google*|models/gemini*)
            [[ -n "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]] || {
                echo "GEMINI_API_KEY or GOOGLE_API_KEY must be exported for ${INDIRECT_JUDGE_MODEL}." >&2
                exit 1
            }
            ;;
        *)
            [[ -n "${OPENAI_API_KEY:-}" ]] || {
                echo "OPENAI_API_KEY must be exported for ${INDIRECT_JUDGE_MODEL}." >&2
                exit 1
            }
            ;;
    esac
    CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -u -m safeoff.data.build_indirect_rejection_pairs \
        --dataset "$BENIGN_DATASET" \
        --split "$BENIGN_SPLIT" \
        --instruction-field instruction \
        --input-field input \
        --output-field output \
        --model-path "$GENERATOR_MODEL" \
        --base-model-path "$BASE_MODEL" \
        --num-source-samples "$SOURCE_COUNT" \
        --target-count "$TARGET_COUNT" \
        --batch-size 16 \
        --max-new-tokens 200 \
        --judge-model "$INDIRECT_JUDGE_MODEL" \
        --judge-workers 8 \
        --output "$TARGET_CATALOG" \
        > "${EXPERIMENT_ROOT}/logs/build_same_row_targets.log" 2>&1
fi

[[ -s "$TARGET_CATALOG" ]] || {
    echo "Missing same-row indirect-target catalog: $TARGET_CATALOG" >&2
    exit 1
}

if [[ "$RUN_PAIR" == 1 ]]; then
    python -m safeoff.data.pair_benign_data_with_indirect_targets \
        --targets "$TARGET_CATALOG" \
        --benign-dataset "$BENIGN_DATASET" \
        --benign-split "$BENIGN_SPLIT" \
        --output "$SEARCH_INPUT" \
        > "${EXPERIMENT_ROOT}/logs/validate_same_row_targets.log" 2>&1
fi

[[ -s "$SEARCH_INPUT" ]] || {
    echo "Missing same-row search input: $SEARCH_INPUT" >&2
    exit 1
}
PAIR_COUNT="$(validate_same_row_search_input "$SEARCH_INPUT")"
[[ "$PAIR_COUNT" -gt 0 ]] || { echo "No rows in $SEARCH_INPUT" >&2; exit 1; }

if [[ "$RUN_SEARCH" == 1 ]]; then
    python -u -m safeoff.generation.dataset \
        --dataset-name csv \
        --dataset-data-file "$SEARCH_INPUT" \
        --dataset-split train \
        --instruction-field instruction \
        --input-field input \
        --target-field target \
        --reference-field output \
        --start-index 0 \
        --num-samples "$PAIR_COUNT" \
        --shard-size 100 \
        --attack-batch-size 2 \
        --target-checkpoint "$CURRENT_MODEL" \
        --prompter-checkpoint "$BASE_MODEL" \
        --target-device "cuda:$TARGET_GPU" \
        --prompter-device "cuda:$PROMPTER_GPU" \
        --dtype float16 \
        --strategy standalone_v3 \
        --prompt-tokens 96 \
        --response-max-new-tokens 200 \
        --search-response-max-new-tokens 200 \
        --top-k 32 \
        --num-beams 4 \
        --num-chunks 2 \
        --lambda-val 120 \
        --no-standalone-v3-multi-target-selection \
        --output-dir "$SEARCH_RUNS_ROOT" \
        --run-dir "$SEARCH_ROOT" \
        --seed 2025 \
        > "${EXPERIMENT_ROOT}/logs/search_same_row_targets.log" 2>&1
fi

if [[ "$RUN_FINETUNE" == 1 ]]; then
    [[ -s "$TRAIN_CSV" ]] || {
        echo "Missing replacement/original-response training CSV: $TRAIN_CSV" >&2
        exit 1
    }
    TRAIN_COUNT="$(csv_row_count "$TRAIN_CSV")"
    [[ "$TRAIN_COUNT" -gt 0 ]] || { echo "No rows in $TRAIN_CSV" >&2; exit 1; }
    validate_finetune_schedule "$TRAIN_COUNT"
    FINETUNE_SIGNATURE="$(finetune_signature "$TRAIN_COUNT")"
    if [[ -s "$FINETUNE_DATA_MANIFEST" ]]; then
        IFS=$'\t' read -r SAVED_FINETUNE_SIGNATURE SAVED_FINETUNE_STATUS <<< "$(python - "$FINETUNE_DATA_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(f"{payload.get('signature', '')}\t{payload.get('status', '')}")
PY
)"
        [[ "$SAVED_FINETUNE_SIGNATURE" == "$FINETUNE_SIGNATURE" ]] || {
            echo "Fine-tune output belongs to different training data/config: $FINETUNE_OUTPUT_DIR" >&2
            echo "Set FINETUNE_OUTPUT_DIR to a fresh path." >&2
            exit 1
        }
        [[ "$SAVED_FINETUNE_STATUS" == complete ]] || {
            echo "Incomplete/unverified fine-tune output: $FINETUNE_OUTPUT_DIR" >&2
            echo "Set FINETUNE_OUTPUT_DIR to a fresh path before retrying." >&2
            exit 1
        }
    elif [[ -d "$FINETUNE_OUTPUT_DIR" ]] && [[ -n "$(find "$FINETUNE_OUTPUT_DIR" -type f -print -quit)" ]]; then
        echo "Unverified pre-existing fine-tune output: $FINETUNE_OUTPUT_DIR" >&2
        echo "Set FINETUNE_OUTPUT_DIR to a fresh path." >&2
        exit 1
    fi
    write_finetune_manifest "$FINETUNE_SIGNATURE" started

    CUDA_VISIBLE_DEVICES="$FINETUNE_GPU" \
    WANDB_DISABLED="${WANDB_DISABLED:-true}" \
    WANDB_MODE="${WANDB_MODE:-disabled}" \
    python -u -m safeoff.training.sft \
        --base_model_path "$FINETUNE_MODEL_NAME" \
        --gradient_accumulation_steps "$FINETUNE_GRADIENT_ACCUMULATION_STEPS" \
        --use_cache 0 \
        --output_dir "$FINETUNE_OUTPUT_DIR" \
        --utility_dataset_config utility_dataset_config_alpaca \
        --batch_size "$FINETUNE_BATCH_SIZE" \
        --epochs "$FINETUNE_EPOCHS" \
        --max_steps "$FINETUNE_MAX_STEPS" \
        --max_sequence_length "$FINETUNE_MAX_SEQUENCE_LENGTH" \
        --fail_on_overlength_examples 1 \
        --save_steps "$FINETUNE_SAVE_STEPS" \
        --save_checkpoint_zero 1 \
        --checkpoint_stage1_end "$FINETUNE_CHECKPOINT_STAGE1_END" \
        --checkpoint_stage1_interval "$FINETUNE_CHECKPOINT_STAGE1_INTERVAL" \
        --checkpoint_stage2_end "$FINETUNE_CHECKPOINT_STAGE2_END" \
        --checkpoint_stage2_interval "$FINETUNE_CHECKPOINT_STAGE2_INTERVAL" \
        --learning_rate "$FINETUNE_LEARNING_RATE" \
        --poison_training_num 0 \
        --identity_shift_num 0 \
        --utility_training_num "$TRAIN_COUNT" \
        --use_lora 1 \
        --filter 0 \
        --save_grads_every 10000 \
        --advprompt_suffix "$TRAIN_CSV" \
        --advprompt_use_suffix 0 \
        --advprompt_training_num "$TRAIN_COUNT" \
        --advprompt_clean_training_num 0 \
        --print_loss_tokens_max_examples 4 \
        --print_loss_tokens 1 \
        --backdoor 0 \
        --augment_utility_with_lat 0 \
        > "${EXPERIMENT_ROOT}/logs/finetune_same_row_targets.log" 2>&1
    write_finetune_manifest "$FINETUNE_SIGNATURE" complete
fi

echo "Same-row target catalog: $TARGET_CATALOG"
echo "Same-row search input: $SEARCH_INPUT"
echo "Replacement/original-response training CSV: $TRAIN_CSV"
if [[ "$RUN_FINETUNE" == 1 ]]; then
    echo "Fine-tuned model output: $FINETUNE_OUTPUT_DIR"
fi
