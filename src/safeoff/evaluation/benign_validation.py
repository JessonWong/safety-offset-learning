#!/usr/bin/env python3
"""Evaluate response-token loss on a fixed validation set across LoRA checkpoints."""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = project_root()

from safeoff.analysis.bilevel_candidates import make_pair_batch, resolve_dtype


DATASET_LABELS = {
    "newtarget": "new target",
    "benign_alpaca": "benign Alpaca",
    "aoa": "AoA",
    "alpaca_top100": "Alpaca top-100",
    "harmful_llm_lat": "LLM-LAT harmful-100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-parent", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LABELS))
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--step-interval", type=int, default=25)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def adapter_weights_path(checkpoint: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = checkpoint / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No adapter weights under {checkpoint}")


def load_adapter_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def set_adapter_state(model, state: dict[str, torch.Tensor]) -> None:
    from peft import set_peft_model_state_dict

    result = set_peft_model_state_dict(model, state, adapter_name="default")
    unexpected = getattr(result, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected[:10]}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_examples(tokenizer, rows, max_length: int) -> tuple[list[dict], list[dict]]:
    examples = []
    skipped = []
    for row in rows:
        batch, metadata = make_pair_batch(
            tokenizer,
            str(row["prompt"]),
            str(row["target"]),
            str(row.get("system_prompt", "You are a helpful assistant.")),
            max_length,
            "cpu",
        )
        if batch is None:
            skipped.append({"candidate_id": row["candidate_id"], **metadata})
        else:
            examples.append({"candidate_id": row["candidate_id"], "batch": batch})
    return examples, skipped


def collate(examples, pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_length = max(int(example["batch"]["input_ids"].shape[1]) for example in examples)
    size = len(examples)
    input_ids = torch.full((size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((size, max_length), dtype=torch.long)
    labels = torch.full((size, max_length), -100, dtype=torch.long)
    for index, example in enumerate(examples):
        source = example["batch"]
        length = int(source["input_ids"].shape[1])
        input_ids[index, :length] = source["input_ids"][0]
        attention_mask[index, :length] = source["attention_mask"][0]
        labels[index, :length] = source["labels"][0]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


@torch.inference_mode()
def validation_metrics(model, tokenizer, examples, batch_size: int, device: torch.device) -> dict:
    total_nll = 0.0
    total_tokens = 0
    example_losses = []
    for start in range(0, len(examples), batch_size):
        current = examples[start : start + batch_size]
        batch = collate(current, tokenizer.pad_token_id, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(len(current), -1)
        valid = shift_labels != -100
        token_counts = valid.sum(dim=1)
        nll = (per_token * valid).sum(dim=1)
        total_nll += float(nll.sum().float().cpu())
        total_tokens += int(token_counts.sum().cpu())
        example_losses.extend((nll / token_counts.clamp(min=1)).float().cpu().tolist())
        del batch, logits, shift_logits, shift_labels, per_token, valid, token_counts, nll
    validation_loss = total_nll / total_tokens
    return {
        "validation_loss": validation_loss,
        "perplexity": math.exp(min(validation_loss, 20.0)),
        "mean_example_loss": float(np.mean(example_losses)),
        "std_example_loss": float(np.std(example_losses, ddof=1)),
        "validation_examples": len(example_losses),
        "validation_response_tokens": total_tokens,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    checkpoint_parent = Path(args.checkpoint_parent).expanduser().resolve()
    validation_path = Path(args.validation).expanduser().resolve()
    validation_rows = load_jsonl(validation_path)
    checkpoint_sets = {}
    for dataset in args.datasets:
        checkpoint_root = checkpoint_parent / dataset
        checkpoints = sorted(
            (path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )
        checkpoint_sets[dataset] = [
            path for path in checkpoints if checkpoint_step(path) % args.step_interval == 0
        ]
        if not checkpoint_sets[dataset]:
            raise RuntimeError(f"No checkpoints found under {checkpoint_root}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples, skipped = prepare_examples(tokenizer, validation_rows, args.max_length)
    if not examples:
        raise RuntimeError("No validation examples remained after tokenization")

    first_checkpoint = checkpoint_sets[args.datasets[0]][0]
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, first_checkpoint, is_trainable=False)
    model.eval()

    records = []
    checkpoint_hashes = {}
    total = sum(len(paths) for paths in checkpoint_sets.values())
    progress = tqdm(total=total, desc="benign validation checkpoints")
    for dataset in args.datasets:
        checkpoint_hashes[dataset] = {}
        for checkpoint in checkpoint_sets[dataset]:
            weights = adapter_weights_path(checkpoint)
            state = load_adapter_state(weights)
            set_adapter_state(model, state)
            metrics = validation_metrics(model, tokenizer, examples, args.batch_size, device)
            step = checkpoint_step(checkpoint)
            checkpoint_hashes[dataset][str(step)] = hashlib.sha256(weights.read_bytes()).hexdigest()
            record = {
                "dataset": dataset,
                "label": DATASET_LABELS.get(dataset, dataset),
                "step": step,
                **metrics,
            }
            records.append(record)
            progress.set_postfix(dataset=dataset, step=step, loss=f"{metrics['validation_loss']:.4f}")
            progress.update(1)
    progress.close()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "model": args.model,
        "checkpoint_parent": str(checkpoint_parent),
        "datasets": args.datasets,
        "validation": str(validation_path),
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "validation_source_count": len(validation_rows),
        "validation_used_count": len(examples),
        "validation_skipped": skipped,
        "metric": "response-token-weighted cross entropy",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "checkpoint_hashes": checkpoint_hashes,
        "records": records,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metadata": str(metadata_path)}))


if __name__ == "__main__":
    main()
