#!/usr/bin/env python3
"""Fine-tune on benign answer/indirect-rejection preference pairs.

The rejected response must be an actual response produced for the final prompt,
not the synthetic target used to search for that prompt.  The script implements
reference-corrected DPO without TRL.  Reference log-probabilities are computed
once from the exact initial policy (including a loaded PEFT adapter), cached, and
then kept fixed for the entire run.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments


REPO_ROOT = project_root()

from safeoff.training.sft import LossPlotCallback, SegmentedCheckpointCallback, load_model_and_tokenizer, resolve_resume_checkpoint, set_training_seed, setup_lora
from safeoff.data.config import format as format_chat  # noqa: E402
from safeoff.evaluation.metrics import contains_keyword  # noqa: E402


PAIR_SCHEMA_VERSION = 1
REFERENCE_CACHE_VERSION = 1
RUN_MANIFEST_VERSION = 1
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
REQUIRED_STRING_FIELDS = ("pair_id", "prompt", "chosen", "rejected")
REQUIRED_BOOLEAN_FIELDS = {
    "rejected_is_actual_response": True,
    "direct_refusal": False,
    "same_row": True,
    "benign": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-corrected preference fine-tuning on strictly validated "
            "benign chosen/indirect-rejection pairs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--validation_dataset_path", type=Path, default=None)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--model_path_name", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--name_suffix", default="")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--chosen_sft_weight",
        type=float,
        default=0.0,
        help="Optional token-mean chosen-answer NLL added to the DPO loss.",
    )
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument(
        "--max_chosen_harmful_info_score",
        type=float,
        default=2.0,
        help="Reject a pair if its independently judged chosen-answer score exceeds this.",
    )
    parser.add_argument(
        "--fail_on_overlength_examples",
        type=int,
        default=1,
        choices=[0, 1],
    )
    parser.add_argument("--reference_batch_size", type=int, default=None)

    parser.add_argument("--use_lora", type=int, default=1, choices=[1])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "v_proj"],
    )
    parser.add_argument("--lora_init_seed", type=int, default=0)
    parser.add_argument("--use_device_map", type=int, default=1, choices=[0, 1])

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--bf16", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--data_random_seed", type=int, default=0)
    parser.add_argument(
        "--evaluation_strategy",
        choices=["no", "steps", "epoch"],
        default="no",
    )
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--save_checkpoint_zero", type=int, default=1, choices=[0, 1])
    parser.add_argument("--checkpoint_stage1_end", type=int, default=0)
    parser.add_argument("--checkpoint_stage1_interval", type=int, default=0)
    parser.add_argument("--checkpoint_stage2_end", type=int, default=0)
    parser.add_argument("--checkpoint_stage2_interval", type=int, default=0)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Trainer checkpoint path, directory name, or 'latest'.",
    )
    return parser.parse_args()


def _clean_required_text(value: Any, field: str, row_number: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"row {row_number}: {field!r} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"row {row_number}: {field!r} must be non-empty")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError(f"row {row_number}: {field!r} contains control characters")
    return value


def _strict_bool(value: Any, field: str, row_number: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(
        f"row {row_number}: {field!r} must be an explicit true/false value"
    )


def validate_preference_records(
    raw_records: Iterable[Mapping[str, Any]],
    source_name: str = "dataset",
    max_chosen_harmful_info_score: float = 2.0,
) -> list[dict[str, Any]]:
    """Validate and canonicalize actual-response benign preference triples."""
    if not math.isfinite(max_chosen_harmful_info_score) or not (
        0.0 <= max_chosen_harmful_info_score <= 5.0
    ):
        raise ValueError("max_chosen_harmful_info_score must be between 0 and 5")
    records: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    for row_number, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{source_name} row {row_number}: expected an object")
        record = dict(raw_record)
        for field in REQUIRED_STRING_FIELDS:
            record[field] = _clean_required_text(
                record.get(field), field, row_number
            )
        pair_id = record["pair_id"]
        if pair_id in seen_pair_ids:
            raise ValueError(f"{source_name} row {row_number}: duplicate pair_id {pair_id!r}")
        seen_pair_ids.add(pair_id)

        for field, expected in REQUIRED_BOOLEAN_FIELDS.items():
            actual = _strict_bool(record.get(field), field, row_number)
            if actual is not expected:
                raise ValueError(
                    f"{source_name} row {row_number}: {field!r} must be {expected}"
                )
            record[field] = actual

        label = _clean_required_text(
            record.get("rejected_label"), "rejected_label", row_number
        ).casefold()
        if label != "indirect_rejection":
            raise ValueError(
                f"{source_name} row {row_number}: rejected_label must be "
                "'indirect_rejection'"
            )
        record["rejected_label"] = label
        if record["chosen"] == record["rejected"]:
            raise ValueError(
                f"{source_name} row {row_number}: chosen and rejected must differ"
            )
        if contains_keyword(record["rejected"]):
            raise ValueError(
                f"{source_name} row {row_number}: rejected contains a direct-refusal "
                "keyword despite direct_refusal=false"
            )
        raw_score = record.get("chosen_harmful_info_score")
        if isinstance(raw_score, bool):
            raise ValueError(
                f"{source_name} row {row_number}: chosen_harmful_info_score "
                "must be numeric"
            )
        try:
            chosen_harmful_info_score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_name} row {row_number}: chosen_harmful_info_score "
                "must be numeric"
            ) from exc
        if not math.isfinite(chosen_harmful_info_score) or not (
            0.0 <= chosen_harmful_info_score <= 5.0
        ):
            raise ValueError(
                f"{source_name} row {row_number}: chosen_harmful_info_score "
                "must be finite and between 0 and 5"
            )
        if chosen_harmful_info_score > max_chosen_harmful_info_score:
            raise ValueError(
                f"{source_name} row {row_number}: chosen_harmful_info_score="
                f"{chosen_harmful_info_score} exceeds allowed maximum "
                f"{max_chosen_harmful_info_score}"
            )
        record["chosen_harmful_info_score"] = chosen_harmful_info_score
        record["chosen_harmful_info_explanation"] = _clean_required_text(
            record.get("chosen_harmful_info_explanation"),
            "chosen_harmful_info_explanation",
            row_number,
        )
        record["pair_schema_version"] = PAIR_SCHEMA_VERSION
        records.append(record)

    if not records:
        raise ValueError(f"{source_name} contains no preference pairs")
    return records


def load_preference_records(
    path: Path | str,
    max_chosen_harmful_info_score: float = 2.0,
) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Preference dataset not found: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_records: Any = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        raw_records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path} line {line_number}: invalid JSON: {exc}"
                    ) from exc
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw_records = json.load(handle)
        if isinstance(raw_records, Mapping):
            raw_records = raw_records.get("pairs")
    else:
        raise ValueError("Preference dataset must be .csv, .json, or .jsonl")
    if not isinstance(raw_records, list):
        raise ValueError(f"{path}: expected a list or an object with a 'pairs' list")
    return validate_preference_records(
        raw_records,
        source_name=str(path),
        max_chosen_harmful_info_score=max_chosen_harmful_info_score,
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Tokenizer unexpectedly returned a batched result")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def tokenize_preference_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    system_prompt: str | None,
    max_sequence_length: int,
) -> dict[str, Any]:
    prompt_text = format_chat(
        {"system": system_prompt, "prompt": record["prompt"]},
        tokenizer=tokenizer,
    )
    chosen_text = format_chat(
        {
            "system": system_prompt,
            "prompt": record["prompt"],
            "response": record["chosen"],
        },
        tokenizer=tokenizer,
    )
    rejected_text = format_chat(
        {
            "system": system_prompt,
            "prompt": record["prompt"],
            "response": record["rejected"],
        },
        tokenizer=tokenizer,
    )
    prompt_ids = _token_ids(tokenizer, prompt_text)
    chosen_ids = _token_ids(tokenizer, chosen_text)
    rejected_ids = _token_ids(tokenizer, rejected_text)
    if chosen_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"pair {record['pair_id']!r}: prompt tokens are not a prefix of chosen tokens"
        )
    if rejected_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"pair {record['pair_id']!r}: prompt tokens are not a prefix of rejected tokens"
        )
    if chosen_ids == rejected_ids:
        raise ValueError(
            f"pair {record['pair_id']!r}: chosen and rejected tokenize identically"
        )
    chosen_completion_length = len(chosen_ids) - len(prompt_ids)
    rejected_completion_length = len(rejected_ids) - len(prompt_ids)
    if chosen_completion_length <= 0 or rejected_completion_length <= 0:
        raise ValueError(f"pair {record['pair_id']!r}: empty tokenized completion")
    if max(len(chosen_ids), len(rejected_ids)) > max_sequence_length:
        raise OverflowError(
            f"pair {record['pair_id']!r}: chosen/rejected lengths are "
            f"{len(chosen_ids)}/{len(rejected_ids)}, over max_sequence_length="
            f"{max_sequence_length}"
        )
    return {
        "pair_id": record["pair_id"],
        "chosen_input_ids": chosen_ids,
        "chosen_completion_mask": [0] * len(prompt_ids)
        + [1] * chosen_completion_length,
        "rejected_input_ids": rejected_ids,
        "rejected_completion_mask": [0] * len(prompt_ids)
        + [1] * rejected_completion_length,
    }


def build_tokenized_dataset(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    system_prompt: str | None,
    max_sequence_length: int,
    fail_on_overlength: bool,
    split_name: str,
) -> Dataset:
    tokenized: list[dict[str, Any]] = []
    overlength_errors: list[str] = []
    for record in records:
        try:
            tokenized.append(
                tokenize_preference_record(
                    record,
                    tokenizer,
                    system_prompt,
                    max_sequence_length,
                )
            )
        except OverflowError as exc:
            overlength_errors.append(str(exc))
    if overlength_errors and fail_on_overlength:
        preview = "; ".join(overlength_errors[:3])
        raise ValueError(
            f"{split_name}: {len(overlength_errors)} preference pairs are overlength. "
            f"{preview}"
        )
    if not tokenized:
        raise ValueError(f"{split_name}: no preference pairs remain after tokenization")
    print(
        f"{split_name}: retained {len(tokenized)} pairs; removed "
        f"{len(overlength_errors)} overlength pairs"
    )
    return Dataset.from_list(tokenized)


class PreferenceDataCollator:
    """Right-pad chosen and rejected sequences into one 2B model batch."""

    def __init__(self, pad_token_id: int):
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty preference batch")
        sequences = [list(item["chosen_input_ids"]) for item in features]
        sequences += [list(item["rejected_input_ids"]) for item in features]
        masks = [list(item["chosen_completion_mask"]) for item in features]
        masks += [list(item["rejected_completion_mask"]) for item in features]
        max_length = max(len(sequence) for sequence in sequences)
        input_rows = []
        attention_rows = []
        completion_rows = []
        for sequence, completion_mask in zip(sequences, masks):
            if len(sequence) != len(completion_mask):
                raise ValueError("Input IDs and completion mask lengths differ")
            padding = max_length - len(sequence)
            input_rows.append(sequence + [self.pad_token_id] * padding)
            attention_rows.append([1] * len(sequence) + [0] * padding)
            completion_rows.append(completion_mask + [0] * padding)
        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "completion_mask": torch.tensor(completion_rows, dtype=torch.bool),
        }
        reference_fields = ("reference_chosen_logp", "reference_rejected_logp")
        present = [field in features[0] for field in reference_fields]
        if any(present) and not all(present):
            raise ValueError("Reference log-probability columns are incomplete")
        if all(present):
            for field in reference_fields:
                values = [float(item[field]) for item in features]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite value in {field}")
                batch[field] = torch.tensor(values, dtype=torch.float32)
        return batch


def completion_sequence_logps(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed causal log-probabilities and completion-token counts."""
    if logits.shape[:2] != input_ids.shape or input_ids.shape != completion_mask.shape:
        raise ValueError("Logit, input ID, and completion mask shapes are inconsistent")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_ids = input_ids[:, 1:].contiguous()
    shifted_mask = completion_mask[:, 1:].bool().contiguous()
    counts = shifted_mask.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("Every sequence must contain at least one predicted completion token")
    labels = shifted_ids.masked_fill(~shifted_mask, -100)
    token_nll = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        labels,
        reduction="none",
        ignore_index=-100,
    )
    return -token_nll.float().sum(dim=-1), counts


def reference_corrected_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if beta <= 0:
        raise ValueError("beta must be greater than zero")
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    preference_logits = beta * (policy_margin - reference_margin)
    losses = -F.logsigmoid(preference_logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    metrics = {
        "dpo_loss": losses.detach().mean(),
        "chosen_reward": chosen_rewards.detach().mean(),
        "rejected_reward": rejected_rewards.detach().mean(),
        "reward_margin": (chosen_rewards - rejected_rewards).detach().mean(),
        "preference_accuracy": (chosen_rewards > rejected_rewards)
        .float()
        .detach()
        .mean(),
    }
    return losses.mean(), metrics


class ReferenceCorrectedDPOTrainer(Trainer):
    def __init__(self, *args: Any, beta: float, chosen_sft_weight: float = 0.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if beta <= 0:
            raise ValueError("beta must be greater than zero")
        if chosen_sft_weight < 0:
            raise ValueError("chosen_sft_weight must be non-negative")
        self.beta = float(beta)
        self.chosen_sft_weight = float(chosen_sft_weight)
        self._stored_train_metrics: dict[str, list[float]] = defaultdict(list)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Mapping[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Any:
        del num_items_in_batch
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        reference_chosen = inputs["reference_chosen_logp"].float()
        reference_rejected = inputs["reference_rejected_logp"].float()
        pair_batch_size = reference_chosen.shape[0]
        if input_ids.shape[0] != pair_batch_size * 2:
            raise ValueError("Preference batch must contain chosen rows followed by rejected rows")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        sequence_logps, token_counts = completion_sequence_logps(
            outputs.logits,
            input_ids,
            completion_mask,
        )
        policy_chosen = sequence_logps[:pair_batch_size]
        policy_rejected = sequence_logps[pair_batch_size:]
        chosen_counts = token_counts[:pair_batch_size].float()
        dpo_loss, metrics = reference_corrected_dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            self.beta,
        )
        chosen_nll = -(policy_chosen / chosen_counts).mean()
        loss = dpo_loss + self.chosen_sft_weight * chosen_nll
        metrics["chosen_nll"] = chosen_nll.detach()
        metrics["total_preference_loss"] = loss.detach()
        if model.training:
            for name, value in metrics.items():
                self._stored_train_metrics[name].append(float(value.cpu()))
        if return_outputs:
            preference_logits = self.beta * (
                (policy_chosen - policy_rejected)
                - (reference_chosen - reference_rejected)
            )
            return loss, {"preference_logits": preference_logits}
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs and self._stored_train_metrics:
            for name, values in self._stored_train_metrics.items():
                if values:
                    logs[name] = float(sum(values) / len(values))
            self._stored_train_metrics.clear()
        super().log(logs, start_time=start_time)


class FinalStepCheckpointCallback(TrainerCallback):
    """Ensure epoch-based runs also save their exact final optimizer step."""

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.max_steps > 0 and state.global_step >= state.max_steps:
            control.should_save = True
        return control


def disable_model_dropout(model: nn.Module) -> int:
    changed = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p != 0.0:
            module.p = 0.0
            changed += 1
    return changed


def _json_hash(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_reference_fingerprint(reference: str | Path) -> dict[str, Any]:
    raw_reference = str(reference)
    path = Path(raw_reference).expanduser()
    if not path.exists():
        return {"reference": raw_reference, "local": False}
    path = path.resolve()
    if path.is_file():
        return {
            "reference": str(path),
            "local": True,
            "sha256": _file_sha256(path),
        }
    files = []
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        if child.name == "adapter_model.safetensors" or child.suffix == ".json":
            files.append({"name": child.name, "sha256": _file_sha256(child)})
        elif child.suffix in {".safetensors", ".bin"}:
            stat = child.stat()
            files.append(
                {
                    "name": child.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {"reference": str(path), "local": True, "files": files}


def trainable_parameters_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    found = False
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        found = True
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    if not found:
        raise ValueError("The model has no trainable parameters")
    return digest.hexdigest()


def canonical_records_fingerprint(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]] | None,
) -> str:
    fields = REQUIRED_STRING_FIELDS + tuple(REQUIRED_BOOLEAN_FIELDS) + (
        "rejected_label",
        "chosen_harmful_info_score",
        "chosen_harmful_info_explanation",
    )
    payload = []
    for split, records in (
        ("train", train_records),
        ("validation", validation_records or []),
    ):
        for record in records:
            payload.append(
                {
                    "split": split,
                    **{field: record.get(field) for field in fields},
                }
            )
    return _json_hash(payload)


def build_reference_signature(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]] | None,
    base_model_path: str,
    initial_trainable_fingerprint: str,
    tokenizer: Any,
    system_prompt: str | None,
    max_sequence_length: int,
    loaded_model_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "schema_version": REFERENCE_CACHE_VERSION,
        "records_fingerprint": canonical_records_fingerprint(
            train_records, validation_records
        ),
        "base_model": model_reference_fingerprint(base_model_path),
        "loaded_model": dict(loaded_model_identity or {}),
        "initial_trainable_parameters_sha256": initial_trainable_fingerprint,
        "tokenizer": {
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "class": tokenizer.__class__.__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
            "commit_hash": getattr(tokenizer, "init_kwargs", {}).get(
                "_commit_hash"
            ),
            "chat_template_sha256": _json_hash(
                getattr(tokenizer, "chat_template", None)
            ),
        },
        "system_prompt": system_prompt,
        "max_sequence_length": int(max_sequence_length),
        "logp_aggregation": "completion_token_sum",
        "completion_includes_chat_terminator": True,
        "dropout_disabled": True,
    }
    return {
        "schema_version": REFERENCE_CACHE_VERSION,
        "signature": _json_hash(config),
        "config": config,
    }


def loaded_model_identity(model: nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        config_payload = config.to_dict()
    else:
        config_payload = vars(config)
    return {
        "name_or_path": getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None),
        "config_sha256": _json_hash(config_payload),
    }


def _model_input_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Cannot determine device for a parameterless model") from exc


@torch.no_grad()
def compute_initial_reference_logps(
    model: nn.Module,
    dataset: Dataset,
    collator: PreferenceDataCollator,
    batch_size: int,
) -> list[dict[str, float | str]]:
    if batch_size <= 0:
        raise ValueError("reference_batch_size must be greater than zero")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    device = _model_input_device(model)
    was_training = model.training
    model.eval()
    rows: list[dict[str, float | str]] = []
    offset = 0
    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            completion_mask = batch["completion_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            sequence_logps, _ = completion_sequence_logps(
                outputs.logits,
                input_ids,
                completion_mask,
            )
            pair_count = input_ids.shape[0] // 2
            chosen = sequence_logps[:pair_count].cpu().tolist()
            rejected = sequence_logps[pair_count:].cpu().tolist()
            pair_ids = dataset[offset : offset + pair_count]["pair_id"]
            for pair_id, chosen_logp, rejected_logp in zip(
                pair_ids, chosen, rejected
            ):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "reference_chosen_logp": float(chosen_logp),
                        "reference_rejected_logp": float(rejected_logp),
                    }
                )
            offset += pair_count
    finally:
        model.train(was_training)
    if len(rows) != len(dataset):
        raise RuntimeError("Reference log-probability row count mismatch")
    return rows


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def write_reference_cache(
    path: Path,
    signature: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        **dict(signature),
        "splits": {
            "train": list(train_rows),
            "validation": list(validation_rows),
        },
    }
    _atomic_write_json(path, payload)


def load_reference_cache(
    path: Path,
    expected_signature: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != REFERENCE_CACHE_VERSION:
        raise ValueError(f"Reference cache schema mismatch at {path}")
    if payload.get("signature") != expected_signature.get("signature"):
        raise ValueError(
            f"Reference cache signature mismatch at {path}; use a new output_dir "
            "for changed data/model/formatting"
        )
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"Reference cache at {path} has no split mapping")
    return {
        "train": _validate_reference_rows(splits.get("train"), "train"),
        "validation": _validate_reference_rows(
            splits.get("validation", []), "validation"
        ),
    }


def _validate_reference_rows(value: Any, split_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"Reference cache split {split_name!r} must be a list")
    rows = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"Reference cache {split_name} row {index} is invalid")
        pair_id = row.get("pair_id")
        chosen = row.get("reference_chosen_logp")
        rejected = row.get("reference_rejected_logp")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Reference cache {split_name} row {index} lacks pair_id")
        try:
            chosen = float(chosen)
            rejected = float(rejected)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Reference cache {split_name} row {index} has invalid log-probabilities"
            ) from exc
        if not math.isfinite(chosen) or not math.isfinite(rejected):
            raise ValueError(
                f"Reference cache {split_name} row {index} has non-finite log-probabilities"
            )
        rows.append(
            {
                "pair_id": pair_id,
                "reference_chosen_logp": chosen,
                "reference_rejected_logp": rejected,
            }
        )
    return rows


def attach_reference_rows(
    dataset: Dataset,
    reference_rows: Sequence[Mapping[str, Any]],
    split_name: str,
) -> Dataset:
    dataset_pair_ids = list(dataset["pair_id"])
    reference_pair_ids = [row["pair_id"] for row in reference_rows]
    if dataset_pair_ids != reference_pair_ids:
        raise ValueError(
            f"Reference cache pair order/content mismatch for {split_name}"
        )
    return dataset.add_column(
        "reference_chosen_logp",
        [float(row["reference_chosen_logp"]) for row in reference_rows],
    ).add_column(
        "reference_rejected_logp",
        [float(row["reference_rejected_logp"]) for row in reference_rows],
    )


def preference_model_output_dir(args: argparse.Namespace) -> Path:
    model_name = args.model_path_name
    if not model_name:
        model_name = Path(str(args.base_model_path).rstrip("/")).name
    return Path(args.output_dir).expanduser() / model_name / f"model{args.name_suffix}"


def build_run_manifest(
    args: argparse.Namespace,
    reference_signature: Mapping[str, Any],
) -> dict[str, Any]:
    immutable_config = {
        "reference_signature": reference_signature["signature"],
        "base_model_path": str(args.base_model_path),
        "system_prompt": args.system_prompt,
        "beta": args.beta,
        "chosen_sft_weight": args.chosen_sft_weight,
        "max_chosen_harmful_info_score": args.max_chosen_harmful_info_score,
        "max_sequence_length": args.max_sequence_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": list(args.lora_target_modules),
        "lora_init_seed": args.lora_init_seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "random_seed": args.random_seed,
        "data_random_seed": args.data_random_seed,
    }
    return {
        "schema_version": RUN_MANIFEST_VERSION,
        "signature": _json_hash(immutable_config),
        "config": immutable_config,
    }


def validate_or_write_run_manifest(
    path: Path,
    expected: Mapping[str, Any],
    resume_requested: bool,
) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("schema_version") != RUN_MANIFEST_VERSION:
            raise ValueError(f"Run manifest schema mismatch at {path}")
        if saved.get("signature") != expected.get("signature"):
            raise ValueError(
                f"Run manifest mismatch at {path}; resume requires identical data, "
                "initial policy, objective, and optimizer settings"
            )
        if not resume_requested:
            raise FileExistsError(
                f"Preference run already exists at {path.parent}; pass "
                "--resume_from_checkpoint to continue it"
            )
        return
    if resume_requested:
        raise FileNotFoundError(
            f"Cannot resume without the original run manifest at {path}"
        )
    _atomic_write_json(path, expected)


def resolve_stateful_resume_checkpoint(
    output_dir: Path,
    resume_from_checkpoint: str | None,
) -> Path | None:
    checkpoint = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
    if checkpoint is None:
        return None
    if not (checkpoint / "trainer_state.json").is_file():
        raise ValueError(
            f"{checkpoint} is an adapter snapshot, not a resumable Trainer "
            "checkpoint (trainer_state.json is missing)"
        )
    return checkpoint


def _ensure_fresh_output_location(output_dir: Path, resume_requested: bool) -> None:
    if resume_requested or not output_dir.exists():
        return
    entries = list(output_dir.iterdir())
    if entries:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new output_dir."
        )


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.beta <= 0:
        raise ValueError("--beta must be greater than zero")
    if args.chosen_sft_weight < 0:
        raise ValueError("--chosen_sft_weight must be non-negative")
    if args.max_sequence_length <= 1:
        raise ValueError("--max_sequence_length must be greater than one")
    if not 0.0 <= args.max_chosen_harmful_info_score <= 5.0:
        raise ValueError("--max_chosen_harmful_info_score must be between 0 and 5")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if args.reference_batch_size is not None and args.reference_batch_size <= 0:
        raise ValueError("--reference_batch_size must be positive")
    if args.evaluation_strategy != "no" and args.validation_dataset_path is None:
        raise ValueError(
            "--validation_dataset_path is required when evaluation is enabled"
        )


def main() -> None:
    args = parse_args()
    _validate_cli_args(args)
    train_records = load_preference_records(
        args.dataset_path,
        max_chosen_harmful_info_score=args.max_chosen_harmful_info_score,
    )
    validation_records = (
        load_preference_records(
            args.validation_dataset_path,
            max_chosen_harmful_info_score=args.max_chosen_harmful_info_score,
        )
        if args.validation_dataset_path is not None
        else []
    )
    overlap = {row["pair_id"] for row in train_records}.intersection(
        row["pair_id"] for row in validation_records
    )
    if overlap:
        raise ValueError(
            f"Train/validation pair_id overlap: {sorted(overlap)[:5]}"
        )

    model_output_dir = preference_model_output_dir(args)
    resume_requested = args.resume_from_checkpoint is not None
    _ensure_fresh_output_location(model_output_dir, resume_requested)

    set_training_seed(args.random_seed)
    model, tokenizer = load_model_and_tokenizer(args)
    model = setup_lora(model, args)
    if not hasattr(model, "peft_config"):
        raise ValueError("Preference fine-tuning requires a PEFT model")
    changed_dropout_modules = disable_model_dropout(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    print(f"Disabled {changed_dropout_modules} dropout modules for stable DPO log-ratios")

    train_dataset = build_tokenized_dataset(
        train_records,
        tokenizer,
        args.system_prompt,
        args.max_sequence_length,
        bool(args.fail_on_overlength_examples),
        "train",
    )
    validation_dataset = None
    if validation_records:
        validation_dataset = build_tokenized_dataset(
            validation_records,
            tokenizer,
            args.system_prompt,
            args.max_sequence_length,
            bool(args.fail_on_overlength_examples),
            "validation",
        )
    collator = PreferenceDataCollator(tokenizer.pad_token_id)

    initial_trainable_fingerprint = trainable_parameters_fingerprint(model)
    reference_signature = build_reference_signature(
        train_records,
        validation_records,
        args.base_model_path,
        initial_trainable_fingerprint,
        tokenizer,
        args.system_prompt,
        args.max_sequence_length,
        loaded_model_identity=loaded_model_identity(model),
    )
    run_manifest = build_run_manifest(args, reference_signature)
    manifest_path = model_output_dir / "preference_run_manifest.json"
    validate_or_write_run_manifest(
        manifest_path,
        run_manifest,
        resume_requested=resume_requested,
    )

    reference_cache_path = model_output_dir / "initial_reference_logps.json"
    if reference_cache_path.exists():
        cached_splits = load_reference_cache(
            reference_cache_path, reference_signature
        )
        print(f"Loaded initial-policy reference log-probabilities from {reference_cache_path}")
    else:
        if resume_requested:
            raise FileNotFoundError(
                f"Cannot resume without initial reference cache {reference_cache_path}"
            )
        reference_batch_size = args.reference_batch_size or args.batch_size
        print("Computing fixed initial-policy reference log-probabilities")
        train_reference_rows = compute_initial_reference_logps(
            model,
            train_dataset,
            collator,
            reference_batch_size,
        )
        validation_reference_rows = (
            compute_initial_reference_logps(
                model,
                validation_dataset,
                collator,
                reference_batch_size,
            )
            if validation_dataset is not None
            else []
        )
        write_reference_cache(
            reference_cache_path,
            reference_signature,
            train_reference_rows,
            validation_reference_rows,
        )
        cached_splits = {
            "train": train_reference_rows,
            "validation": validation_reference_rows,
        }
        print(f"Saved initial-policy reference log-probabilities to {reference_cache_path}")

    train_dataset = attach_reference_rows(
        train_dataset, cached_splits["train"], "train"
    )
    if validation_dataset is not None:
        validation_dataset = attach_reference_rows(
            validation_dataset,
            cached_splits["validation"],
            "validation",
        )

    checkpoint_callback = SegmentedCheckpointCallback(
        stage1_end_step=args.checkpoint_stage1_end,
        stage1_interval=args.checkpoint_stage1_interval,
        stage2_end_step=args.checkpoint_stage2_end,
        stage2_interval=args.checkpoint_stage2_interval,
    )
    callbacks = [LossPlotCallback(model_output_dir, filename="preference_loss_curve.png")]
    if checkpoint_callback.is_enabled():
        callbacks.append(checkpoint_callback)
    # Keep this callback last so a segmented schedule cannot suppress the true
    # final step when an epoch-based run ends between regular save intervals.
    callbacks.append(FinalStepCheckpointCallback())

    evaluation_strategy = args.evaluation_strategy
    if validation_dataset is None:
        evaluation_strategy = "no"
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        evaluation_strategy=evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_dir=str(model_output_dir / "logs"),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=bool(args.bf16),
        max_steps=args.max_steps,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["reference_chosen_logp", "reference_rejected_logp"],
        seed=args.random_seed,
        data_seed=args.data_random_seed,
        report_to=[],
    )
    trainer = ReferenceCorrectedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        beta=args.beta,
        chosen_sft_weight=args.chosen_sft_weight,
    )
    if args.save_checkpoint_zero:
        checkpoint_zero = model_output_dir / "checkpoint-0"
        if not checkpoint_zero.exists():
            checkpoint_zero.mkdir(parents=True, exist_ok=False)
            trainer.save_model(str(checkpoint_zero))
            print(f"Saved initial adapter to {checkpoint_zero}")

    resume_checkpoint = resolve_stateful_resume_checkpoint(
        model_output_dir, args.resume_from_checkpoint
    )
    if resume_checkpoint is not None:
        print(f"Resuming Trainer state from {resume_checkpoint}")
    trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )


if __name__ == "__main__":
    main()
