#!/usr/bin/env python3
"""Paired safety-gradient contrasts, subspace projection, and local validation.

The experiment has two paired treatments:

* SafeOff prompt versus the original Alpaca prompt, holding the answer fixed.
* LLM-LAT rejected versus chosen response, holding the harmful prompt fixed.

For a held-out harmful prompt x, define the safety margin

    S_x(theta) = loss_harmful(theta; x) - loss_refusal(theta; x).

Higher is safer.  Its gradient is therefore g_harmful - g_refusal.  A training
gradient g predicts a first-order change -lr * <grad S, g> under SGD.  We build
an uncentered safety-gradient SVD subspace on a private meta split, evaluate all
projections against a disjoint validation split, and train four matched 10-step
LoRA trajectories.  The trajectory test compares actual safety-margin changes
with grad S(theta_0)^T (theta_t - theta_0), both in the full LoRA parameter
space and after projection into the meta safety subspace.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = project_root()
DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
REFUSAL_PREFIXES = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "sorry",
    "as an ai",
    "as a language model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--safeoff",
        type=Path,
        default=ROOT / "data/delta_similarity_100/safeoff_0_99.csv",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=ROOT / "data/delta_similarity_100/alpaca_aligned_0_99.csv",
    )
    parser.add_argument("--safeoff-count", type=int, default=100)
    parser.add_argument("--llm-lat-dataset", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--llm-lat-count", type=int, default=100)
    parser.add_argument("--llm-lat-split-seed", type=int, default=30)
    parser.add_argument("--safety-meta-count", type=int, default=64)
    parser.add_argument("--safety-validation-count", type=int, default=32)
    parser.add_argument("--safety-probe-seed", type=int, default=20260923)
    parser.add_argument("--safety-minimum-source-index", type=int, default=1000)
    parser.add_argument("--primary-k", type=int, default=8)
    parser.add_argument("--top-k-grid", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--lora-seed", type=int, default=24)
    parser.add_argument("--training-seed", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--scheduler-total-steps", type=int, default=250)
    parser.add_argument("--measurement-steps", nargs="+", type=int, default=[0, 1, 5, 10])
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/paired_safety_gradient_subspace",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--allow-small",
        action="store_true",
        help="Permit fewer than 90 training pairs for an explicitly requested smoke run.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def begins_with_refusal(text: str) -> bool:
    return normalize(text).startswith(REFUSAL_PREFIXES)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prompt_from_csv(row: dict[str, str]) -> str:
    return f"{row.get('instruction', '')}{row.get('suffix', '')}"


def pair_safeoff_rows(
    safeoff_path: Path, original_path: Path
) -> list[dict[str, Any]]:
    safeoff = load_csv(safeoff_path)
    original = load_csv(original_path)
    if len(safeoff) != len(original):
        raise ValueError(f"SafeOff/original row mismatch: {len(safeoff)} != {len(original)}")
    pairs = []
    for position, (treatment, control) in enumerate(zip(safeoff, original)):
        if treatment.get("index") != control.get("index"):
            raise ValueError(f"SafeOff/original index mismatch at position {position}")
        if treatment.get("alpaca_reference") != control.get("alpaca_reference"):
            raise ValueError(f"SafeOff/original answer mismatch at position {position}")
        pairs.append(
            {
                "pair_id": f"safeoff-{treatment.get('index', position)}",
                "source_index": treatment.get("index", str(position)),
                "treatment_prompt": prompt_from_csv(treatment),
                "control_prompt": prompt_from_csv(control),
                "treatment_response": treatment.get("alpaca_reference", ""),
                "control_response": control.get("alpaca_reference", ""),
                "system_prompt": "You are a helpful assistant.",
            }
        )
    return pairs


def llm_lat_indices(size: int, seed: int, count: int) -> list[int]:
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def pair_llm_lat_rows(dataset, indices: Iterable[int]) -> list[dict[str, Any]]:
    pairs = []
    for source_index in indices:
        row = dataset[int(source_index)]
        pairs.append(
            {
                "pair_id": f"llm-lat-{source_index}",
                "source_index": int(source_index),
                "treatment_prompt": str(row.get("prompt", "")),
                "control_prompt": str(row.get("prompt", "")),
                "treatment_response": str(row.get("rejected", "")),
                "control_response": str(row.get("chosen", "")),
                "system_prompt": "",
            }
        )
    return pairs


def make_batch(
    tokenizer,
    prompt: str,
    response: str,
    system_prompt: str,
    max_length: int,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    metadata = {
        "token_length": len(full_ids),
        "prompt_token_length": len(prompt_ids),
        "response_token_length": max(0, len(full_ids) - len(prompt_ids)),
    }
    if len(full_ids) > max_length:
        return None, {**metadata, "skip_reason": "length"}
    if len(prompt_ids) >= len(full_ids):
        return None, {**metadata, "skip_reason": "empty_response_tokens"}
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return None, {**metadata, "skip_reason": "chat_template_prefix_mismatch"}
    input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }, metadata


def prepare_training_pairs(tokenizer, rows, max_length: int):
    prepared = []
    skipped = []
    for row in rows:
        treatment, treatment_meta = make_batch(
            tokenizer,
            row["treatment_prompt"],
            row["treatment_response"],
            row["system_prompt"],
            max_length,
        )
        control, control_meta = make_batch(
            tokenizer,
            row["control_prompt"],
            row["control_response"],
            row["system_prompt"],
            max_length,
        )
        if treatment is None or control is None:
            skipped.append(
                {
                    "pair_id": row["pair_id"],
                    "treatment": treatment_meta,
                    "control": control_meta,
                }
            )
            continue
        prepared.append(
            {
                "pair_id": row["pair_id"],
                "source_index": row["source_index"],
                "treatment": treatment,
                "control": control,
                "treatment_meta": treatment_meta,
                "control_meta": control_meta,
            }
        )
    return prepared, skipped


def prepare_safety_splits(
    tokenizer,
    dataset,
    excluded_indices: set[int],
    excluded_prompts: set[str],
    args: argparse.Namespace,
):
    indices = [
        index
        for index in range(max(0, args.safety_minimum_source_index), len(dataset))
        if index not in excluded_indices
    ]
    random.Random(args.safety_probe_seed).shuffle(indices)
    required = args.safety_meta_count + args.safety_validation_count
    prepared = []
    rejected = {
        "empty": 0,
        "chosen_not_refusal": 0,
        "identical_targets": 0,
        "prompt_overlap": 0,
        "tokenization": 0,
    }
    for source_index in indices:
        row = dataset[source_index]
        prompt = str(row.get("prompt", "")).strip()
        harmful = str(row.get("rejected", "")).strip()
        refusal = str(row.get("chosen", "")).strip()
        if not prompt or not harmful or not refusal:
            rejected["empty"] += 1
            continue
        if not begins_with_refusal(refusal):
            rejected["chosen_not_refusal"] += 1
            continue
        if normalize(harmful) == normalize(refusal):
            rejected["identical_targets"] += 1
            continue
        if normalize(prompt) in excluded_prompts:
            rejected["prompt_overlap"] += 1
            continue
        harmful_batch, harmful_meta = make_batch(
            tokenizer, prompt, harmful, "", args.max_length
        )
        refusal_batch, refusal_meta = make_batch(
            tokenizer, prompt, refusal, "", args.max_length
        )
        if harmful_batch is None or refusal_batch is None:
            rejected["tokenization"] += 1
            continue
        prepared.append(
            {
                "probe_id": f"safety-{source_index}",
                "source_index": source_index,
                "harmful": harmful_batch,
                "refusal": refusal_batch,
                "harmful_meta": harmful_meta,
                "refusal_meta": refusal_meta,
            }
        )
        if len(prepared) == required:
            break
    if len(prepared) < required:
        raise RuntimeError(
            f"Only {len(prepared)} valid safety pairs for requested {required}; "
            f"rejections={rejected}"
        )
    return (
        prepared[: args.safety_meta_count],
        prepared[args.safety_meta_count :],
        rejected,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def flatten_tensors(
    tensors: Iterable[torch.Tensor | None], params: list[torch.nn.Parameter]
) -> torch.Tensor:
    pieces = []
    for tensor, param in zip(tensors, params):
        if tensor is None:
            pieces.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
        else:
            pieces.append(tensor.detach().float().reshape(-1))
    return torch.cat(pieces)


def parameter_vector(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([param.detach().float().reshape(-1) for param in params])


def loss_and_gradient(model, batch, params, device):
    model.zero_grad(set_to_none=True)
    moved = move_batch(batch, device)
    output = model(**moved, use_cache=False)
    loss = output.loss
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    vector = flatten_tensors(grads, params)
    value = float(loss.detach().float().cpu())
    del moved, output, loss, grads
    return value, vector


def safety_gradient_rows(model, pairs, params, device, keep_rows: bool):
    rows = []
    total = torch.zeros(
        sum(param.numel() for param in params), dtype=torch.float32, device=device
    )
    margins = []
    model.eval()
    for position, pair in enumerate(pairs, start=1):
        harmful_loss, harmful_grad = loss_and_gradient(
            model, pair["harmful"], params, device
        )
        refusal_loss, refusal_grad = loss_and_gradient(
            model, pair["refusal"], params, device
        )
        safety_grad = harmful_grad - refusal_grad
        total.add_(safety_grad)
        margins.append(harmful_loss - refusal_loss)
        if keep_rows:
            rows.append(safety_grad)
        if position % 8 == 0 or position == len(pairs):
            print(
                json.dumps(
                    {
                        "stage": "safety_gradients",
                        "processed": position,
                        "total": len(pairs),
                        "keep_rows": keep_rows,
                    }
                ),
                flush=True,
            )
        del harmful_grad, refusal_grad, safety_grad
    return (torch.stack(rows) if keep_rows else None), total / len(pairs), margins


def build_subspace(gradient_rows: torch.Tensor) -> dict[str, torch.Tensor]:
    gram = gradient_rows @ gradient_rows.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    tolerance = max(float(eigenvalues[0].cpu()) * 1e-8, 1e-20)
    rank = int((eigenvalues > tolerance).sum().item())
    if rank == 0:
        raise RuntimeError("Safety-gradient matrix has numerical rank zero")
    eigenvalues = eigenvalues[:rank]
    eigenvectors = eigenvectors[:, :rank]
    singular_values = eigenvalues.sqrt()
    return {
        "rows": gradient_rows,
        "eigenvalues": eigenvalues,
        "left_vectors": eigenvectors,
        "singular_values": singular_values,
        "rank": torch.tensor(rank, device=gradient_rows.device),
    }


def projection_coordinates(subspace, vector: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), int(subspace["rank"].item()))
    row_dots = subspace["rows"] @ vector
    return (
        subspace["left_vectors"][:, :k].T @ row_dots
    ) / subspace["singular_values"][:k]


def safe_cosine(dot: float, norm_a: float, norm_b: float) -> float | None:
    denominator = norm_a * norm_b
    return None if denominator == 0 else dot / denominator


def projection_summary(subspace, vector, topks):
    norm = float(vector.norm().cpu())
    result = {}
    for k in topks:
        effective_k = min(k, int(subspace["rank"].item()))
        coordinates = projection_coordinates(subspace, vector, effective_k)
        projected_norm = float(coordinates.norm().cpu())
        result[str(k)] = {
            "effective_k": effective_k,
            "projected_norm": projected_norm,
            "captured_energy": 0.0 if norm == 0 else (projected_norm / norm) ** 2,
            "coordinates": [float(value) for value in coordinates.cpu()],
        }
    return result


def bootstrap_mean_ci(values, samples: int, seed: int):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means.append(array[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(means), [0.025, 0.975])
    return [float(low), float(high)]


def contrast_gradients(
    name,
    model,
    pairs,
    params,
    device,
    validation_gradient,
    subspace,
    primary_k,
):
    model.eval()
    total = torch.zeros_like(validation_gradient)
    rows = []
    validation_norm = float(validation_gradient.norm().cpu())
    validation_coordinates = projection_coordinates(
        subspace, validation_gradient, primary_k
    )
    for position, pair in enumerate(pairs, start=1):
        treatment_loss, treatment_grad = loss_and_gradient(
            model, pair["treatment"], params, device
        )
        control_loss, control_grad = loss_and_gradient(
            model, pair["control"], params, device
        )
        contrast = treatment_grad - control_grad
        total.add_(contrast)
        contrast_norm = float(contrast.norm().cpu())
        full_dot = float(torch.dot(validation_gradient, contrast).cpu())
        coordinates = projection_coordinates(subspace, contrast, primary_k)
        projected_dot = float(torch.dot(validation_coordinates, coordinates).cpu())
        projected_norm = float(coordinates.norm().cpu())
        rows.append(
            {
                "contrast": name,
                "pair_id": pair["pair_id"],
                "source_index": pair["source_index"],
                "treatment_loss": treatment_loss,
                "control_loss": control_loss,
                "loss_difference": treatment_loss - control_loss,
                "gradient_norm": contrast_norm,
                "validation_safety_dot": full_dot,
                "validation_safety_cosine": safe_cosine(
                    full_dot, validation_norm, contrast_norm
                ),
                "primary_k": primary_k,
                "projected_norm": projected_norm,
                "projected_energy_fraction": (
                    0.0 if contrast_norm == 0 else (projected_norm / contrast_norm) ** 2
                ),
                "projected_validation_safety_dot": projected_dot,
                # SGD treatment-minus-control update is -lr * contrast.
                "predicted_safety_change_sign": -1 if full_dot > 0 else (1 if full_dot < 0 else 0),
            }
        )
        if position % 10 == 0 or position == len(pairs):
            print(
                json.dumps(
                    {
                        "stage": "paired_contrast",
                        "contrast": name,
                        "processed": position,
                        "total": len(pairs),
                    }
                ),
                flush=True,
            )
        del treatment_grad, control_grad, contrast, coordinates
    return total / len(pairs), rows


def collate_batches(batches, pad_token_id: int, device: torch.device):
    max_length = max(int(batch["input_ids"].shape[1]) for batch in batches)
    count = len(batches)
    input_ids = torch.full((count, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((count, max_length), dtype=torch.long)
    labels = torch.full((count, max_length), -100, dtype=torch.long)
    for index, batch in enumerate(batches):
        length = int(batch["input_ids"].shape[1])
        input_ids[index, :length] = batch["input_ids"][0]
        attention_mask[index, :length] = batch["attention_mask"][0]
        labels[index, :length] = batch["labels"][0]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


@torch.no_grad()
def per_example_losses(model, batches, pad_token_id, device, batch_size):
    losses = []
    model.eval()
    for start in range(0, len(batches), batch_size):
        current = batches[start : start + batch_size]
        batch = collate_batches(current, pad_token_id, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
        # Cross entropy preserves a bfloat16 output when given bfloat16 logits,
        # which quantizes the small 1-step margin changes this experiment is
        # designed to resolve.  Match the model's training-loss path by doing
        # the reduction in float32.
        shift_logits = logits[:, :-1].contiguous().float()
        shift_labels = batch["labels"][:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(len(current), -1)
        valid = shift_labels != -100
        example_losses = (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        losses.extend(float(value) for value in example_losses.float().cpu())
        del batch, logits, shift_logits, shift_labels, token_losses, valid, example_losses
    return losses


def evaluate_margin(model, safety_pairs, tokenizer, device, batch_size):
    harmful_losses = per_example_losses(
        model,
        [pair["harmful"] for pair in safety_pairs],
        tokenizer.pad_token_id,
        device,
        batch_size,
    )
    refusal_losses = per_example_losses(
        model,
        [pair["refusal"] for pair in safety_pairs],
        tokenizer.pad_token_id,
        device,
        batch_size,
    )
    margins = np.asarray(harmful_losses) - np.asarray(refusal_losses)
    return {
        "mean": float(margins.mean()),
        "sem": float(margins.std(ddof=1) / math.sqrt(len(margins))),
        "harmful_loss_mean": float(np.mean(harmful_losses)),
        "refusal_loss_mean": float(np.mean(refusal_losses)),
        "per_example": [float(value) for value in margins],
    }


def restore_parameters(params, initial):
    with torch.no_grad():
        for param, value in zip(params, initial):
            param.copy_(value)


def train_trajectory(
    name,
    model,
    examples,
    params,
    initial_params,
    initial_vector,
    safety_validation,
    validation_gradient,
    subspace,
    primary_k,
    tokenizer,
    device,
    args,
    base_margin,
):
    restore_parameters(params, initial_params)
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, 1.0 - step / args.scheduler_total_steps),
    )
    rng = random.Random(args.training_seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    position = 0
    epoch = 0
    measurement_steps = sorted(set(args.measurement_steps))
    max_step = max(measurement_steps)
    records = []
    validation_coordinates = projection_coordinates(
        subspace, validation_gradient, primary_k
    )

    def next_example():
        nonlocal order, position, epoch
        if position >= len(order):
            order = list(range(len(examples)))
            rng.shuffle(order)
            position = 0
            epoch += 1
        example = examples[order[position]]
        position += 1
        return example

    for step in range(max_step + 1):
        if step in measurement_steps:
            margin = base_margin if step == 0 else evaluate_margin(
                model,
                safety_validation,
                tokenizer,
                device,
                args.evaluation_batch_size,
            )
            vector = parameter_vector(params)
            displacement = vector - initial_vector
            full_prediction = float(torch.dot(validation_gradient, displacement).cpu())
            displacement_coordinates = projection_coordinates(
                subspace, displacement, primary_k
            )
            subspace_prediction = float(
                torch.dot(validation_coordinates, displacement_coordinates).cpu()
            )
            record = {
                "arm": name,
                "step": step,
                "epoch_completed": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "safety_margin": margin["mean"],
                "safety_margin_sem": margin["sem"],
                "actual_safety_change": margin["mean"] - base_margin["mean"],
                "full_taylor_prediction": full_prediction,
                "subspace_taylor_prediction": subspace_prediction,
                "primary_k": primary_k,
                "parameter_displacement_norm": float(displacement.norm().cpu()),
                "subspace_displacement_norm": float(displacement_coordinates.norm().cpu()),
            }
            records.append(record)
            print(json.dumps({"stage": "trajectory", **record}), flush=True)
            del vector, displacement, displacement_coordinates
        if step == max_step:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            example = next_example()
            batch = move_batch(example, device)
            loss = model(**batch, use_cache=False).loss
            (loss / args.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach().float().cpu())
            del batch, loss
        optimizer.step()
        scheduler.step()
        if not math.isfinite(accumulated_loss):
            raise RuntimeError(f"Non-finite training loss in {name} step {step + 1}")
    restore_parameters(params, initial_params)
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation_summary(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(actual) < 2 or actual.std() == 0 or predicted.std() == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(actual, predicted)[0, 1])
    return {
        "n": int(len(actual)),
        "pearson_r": pearson,
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "sign_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
    }


def build_did_records(trajectory_rows, primary_k):
    by_key = {(row["arm"], row["step"]): row for row in trajectory_rows}
    comparisons = {
        "safeoff_minus_original": ("safeoff", "original"),
        "llm_lat_rejected_minus_chosen": ("llm_lat_rejected", "llm_lat_chosen"),
    }
    rows = []
    for comparison, (treatment_name, control_name) in comparisons.items():
        steps = sorted(
            step
            for arm, step in by_key
            if arm == treatment_name and step > 0
        )
        for step in steps:
            treatment = by_key[(treatment_name, step)]
            control = by_key[(control_name, step)]
            rows.append(
                {
                    "comparison": comparison,
                    "step": step,
                    "actual_did_safety_change": (
                        treatment["actual_safety_change"] - control["actual_safety_change"]
                    ),
                    "full_taylor_did_prediction": (
                        treatment["full_taylor_prediction"] - control["full_taylor_prediction"]
                    ),
                    "subspace_taylor_did_prediction": (
                        treatment["subspace_taylor_prediction"]
                        - control["subspace_taylor_prediction"]
                    ),
                    "primary_k": primary_k,
                }
            )
    return rows


def plot_results(output_dir, spectrum_rows, did_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ks = [row["component"] for row in spectrum_rows]
    cumulative = [row["cumulative_explained_variance"] for row in spectrum_rows]
    axes[0].plot(ks, cumulative, marker="o", markersize=3)
    axes[0].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Safety-gradient component k")
    axes[0].set_ylabel("Cumulative explained gradient energy")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)

    colors = {
        "safeoff_minus_original": "#4C78A8",
        "llm_lat_rejected_minus_chosen": "#E45756",
    }
    for comparison in colors:
        rows = [row for row in did_rows if row["comparison"] == comparison]
        steps = [row["step"] for row in rows]
        axes[1].plot(
            steps,
            [row["actual_did_safety_change"] for row in rows],
            marker="o",
            color=colors[comparison],
            label=f"{comparison}: actual",
        )
        axes[1].plot(
            steps,
            [row["full_taylor_did_prediction"] for row in rows],
            marker="x",
            linestyle="--",
            color=colors[comparison],
            label=f"{comparison}: full Taylor",
        )
        axes[1].plot(
            steps,
            [row["subspace_taylor_did_prediction"] for row in rows],
            marker="s",
            linestyle=":",
            color=colors[comparison],
            alpha=0.75,
            label=f"{comparison}: subspace Taylor",
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Treatment-control safety-margin change")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_safety_gradient_subspace.png", dpi=220)
    fig.savefig(output_dir / "paired_safety_gradient_subspace.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.safeoff = args.safeoff.expanduser().resolve()
    args.original = args.original.expanduser().resolve()
    if sorted(set(args.measurement_steps)) != args.measurement_steps or args.measurement_steps[0] != 0:
        raise ValueError("--measurement-steps must be sorted, unique, and begin with 0")
    if min(
        args.safeoff_count,
        args.llm_lat_count,
        args.safety_meta_count,
        args.safety_validation_count,
    ) <= 0:
        raise ValueError("Dataset and safety split counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.preflight:
        (args.output_dir / "completed.json").unlink(missing_ok=True)

    safeoff_rows = pair_safeoff_rows(args.safeoff, args.original)[: args.safeoff_count]
    dataset = load_dataset(args.llm_lat_dataset, split="train")
    required_fields = {"prompt", "rejected", "chosen"}
    if not required_fields.issubset(dataset.column_names):
        raise ValueError(
            f"{args.llm_lat_dataset} lacks fields {sorted(required_fields - set(dataset.column_names))}"
        )
    training_indices = llm_lat_indices(
        len(dataset), args.llm_lat_split_seed, args.llm_lat_count
    )
    llm_lat_rows = pair_llm_lat_rows(dataset, training_indices)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    safeoff_pairs, safeoff_skipped = prepare_training_pairs(
        tokenizer, safeoff_rows, args.max_length
    )
    llm_lat_pairs, llm_lat_skipped = prepare_training_pairs(
        tokenizer, llm_lat_rows, args.max_length
    )
    excluded_prompts = {
        normalize(row["treatment_prompt"]) for row in safeoff_rows + llm_lat_rows
    }
    excluded_prompts.update(
        normalize(row["control_prompt"]) for row in safeoff_rows + llm_lat_rows
    )
    safety_meta, safety_validation, safety_rejections = prepare_safety_splits(
        tokenizer,
        dataset,
        set(training_indices),
        excluded_prompts,
        args,
    )
    prepared_manifest = {
        "model": args.model,
        "safeoff_path": str(args.safeoff),
        "safeoff_sha256": sha256(args.safeoff),
        "original_path": str(args.original),
        "original_sha256": sha256(args.original),
        "safeoff_source_pairs": len(safeoff_rows),
        "safeoff_pairs_used": len(safeoff_pairs),
        "safeoff_skipped": safeoff_skipped,
        "llm_lat_dataset": args.llm_lat_dataset,
        "llm_lat_dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "llm_lat_split_seed": args.llm_lat_split_seed,
        "llm_lat_source_indices": training_indices,
        "llm_lat_pairs_used": len(llm_lat_pairs),
        "llm_lat_skipped": llm_lat_skipped,
        "safety_probe_seed": args.safety_probe_seed,
        "safety_minimum_source_index": args.safety_minimum_source_index,
        "safety_meta_source_indices": [row["source_index"] for row in safety_meta],
        "safety_validation_source_indices": [
            row["source_index"] for row in safety_validation
        ],
        "safety_rejections_before_selection_completed": safety_rejections,
        "max_length": args.max_length,
        "pairing_checks": {
            "safeoff_same_answer": True,
            "safeoff_same_source_index": True,
            "llm_lat_same_prompt": True,
            "training_probe_source_disjoint": not bool(
                set(training_indices)
                & {
                    row["source_index"] for row in safety_meta + safety_validation
                }
            ),
            "meta_validation_disjoint": not bool(
                {row["source_index"] for row in safety_meta}
                & {row["source_index"] for row in safety_validation}
            ),
        },
    }
    (args.output_dir / "prepared_manifest.json").write_text(
        json.dumps(prepared_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.preflight:
        print(json.dumps(prepared_manifest, indent=2))
        return
    if not args.allow_small and (len(safeoff_pairs) < 90 or len(llm_lat_pairs) < 90):
        raise RuntimeError(
            "Fewer than 90 valid paired examples; do not silently run a materially smaller experiment"
        )

    random.seed(args.lora_seed)
    torch.manual_seed(args.lora_seed)
    torch.cuda.manual_seed_all(args.lora_seed)
    device = torch.device(args.device)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.config.use_cache = False
    torch.manual_seed(args.lora_seed)
    torch.cuda.manual_seed_all(args.lora_seed)
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    named_params = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    params = [param for _, param in named_params]
    if not params:
        raise RuntimeError("No trainable LoRA parameters")
    initial_params = [param.detach().clone() for param in params]
    initial_vector = parameter_vector(params)
    print(
        json.dumps(
            {
                "stage": "model_loaded",
                "trainable_parameter_count": int(sum(p.numel() for p in params)),
                "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            }
        ),
        flush=True,
    )

    safety_rows, meta_mean_gradient, meta_margins = safety_gradient_rows(
        model, safety_meta, params, device, keep_rows=True
    )
    subspace = build_subspace(safety_rows)
    rank = int(subspace["rank"].item())
    topks = sorted({min(k, rank) for k in args.top_k_grid if k > 0} | {min(args.primary_k, rank), rank})
    primary_k = min(args.primary_k, rank)
    eigenvalues = subspace["eigenvalues"]
    explained = eigenvalues / eigenvalues.sum()
    cumulative = torch.cumsum(explained, dim=0)
    k90 = int(torch.searchsorted(cumulative, torch.tensor(0.9, device=device)).item()) + 1
    spectrum_rows = [
        {
            "component": index + 1,
            "eigenvalue": float(eigenvalues[index].cpu()),
            "singular_value": float(subspace["singular_values"][index].cpu()),
            "explained_variance": float(explained[index].cpu()),
            "cumulative_explained_variance": float(cumulative[index].cpu()),
        }
        for index in range(rank)
    ]
    write_csv(args.output_dir / "subspace_spectrum.csv", spectrum_rows)

    _, validation_gradient, validation_base_margins = safety_gradient_rows(
        model, safety_validation, params, device, keep_rows=False
    )
    base_margin = {
        "mean": float(np.mean(validation_base_margins)),
        "sem": float(
            np.std(validation_base_margins, ddof=1)
            / math.sqrt(len(validation_base_margins))
        ),
        "harmful_loss_mean": None,
        "refusal_loss_mean": None,
        "per_example": validation_base_margins,
    }

    safeoff_contrast, safeoff_pair_metrics = contrast_gradients(
        "safeoff_minus_original",
        model,
        safeoff_pairs,
        params,
        device,
        validation_gradient,
        subspace,
        primary_k,
    )
    llm_lat_contrast, llm_lat_pair_metrics = contrast_gradients(
        "llm_lat_rejected_minus_chosen",
        model,
        llm_lat_pairs,
        params,
        device,
        validation_gradient,
        subspace,
        primary_k,
    )
    pair_metrics = safeoff_pair_metrics + llm_lat_pair_metrics
    write_csv(args.output_dir / "contrast_pair_metrics.csv", pair_metrics)

    contrast_vectors = {
        "safeoff_minus_original": safeoff_contrast,
        "llm_lat_rejected_minus_chosen": llm_lat_contrast,
    }
    contrast_summaries = {}
    validation_norm = float(validation_gradient.norm().cpu())
    for contrast_name, vector in contrast_vectors.items():
        norm = float(vector.norm().cpu())
        dot = float(torch.dot(validation_gradient, vector).cpu())
        relevant_rows = [row for row in pair_metrics if row["contrast"] == contrast_name]
        contrast_summaries[contrast_name] = {
            "gradient_norm": norm,
            "validation_safety_dot": dot,
            "validation_safety_cosine": safe_cosine(dot, validation_norm, norm),
            "sgd_first_order_safety_change_per_lr": -dot,
            "pair_mean_safety_dot": float(
                np.mean([row["validation_safety_dot"] for row in relevant_rows])
            ),
            "pair_mean_safety_dot_bootstrap_95ci": bootstrap_mean_ci(
                [row["validation_safety_dot"] for row in relevant_rows],
                args.bootstrap_samples,
                args.safety_probe_seed,
            ),
            "projection": projection_summary(subspace, vector, topks),
        }
    raw_dot = float(torch.dot(safeoff_contrast, llm_lat_contrast).cpu())
    raw_norm_a = float(safeoff_contrast.norm().cpu())
    raw_norm_b = float(llm_lat_contrast.norm().cpu())
    projected_similarity = {}
    for k in topks:
        coordinates_a = projection_coordinates(subspace, safeoff_contrast, k)
        coordinates_b = projection_coordinates(subspace, llm_lat_contrast, k)
        dot = float(torch.dot(coordinates_a, coordinates_b).cpu())
        norm_a = float(coordinates_a.norm().cpu())
        norm_b = float(coordinates_b.norm().cpu())
        projected_similarity[str(k)] = {
            "effective_k": k,
            "dot": dot,
            "cosine": safe_cosine(dot, norm_a, norm_b),
            "safeoff_projected_norm": norm_a,
            "llm_lat_projected_norm": norm_b,
        }

    arm_examples = {
        "safeoff": [pair["treatment"] for pair in safeoff_pairs],
        "original": [pair["control"] for pair in safeoff_pairs],
        "llm_lat_rejected": [pair["treatment"] for pair in llm_lat_pairs],
        "llm_lat_chosen": [pair["control"] for pair in llm_lat_pairs],
    }
    trajectory_rows = []
    for arm_name, examples in arm_examples.items():
        trajectory_rows.extend(
            train_trajectory(
                arm_name,
                model,
                examples,
                params,
                initial_params,
                initial_vector,
                safety_validation,
                validation_gradient,
                subspace,
                primary_k,
                tokenizer,
                device,
                args,
                base_margin,
            )
        )
    write_csv(args.output_dir / "trajectory.csv", trajectory_rows)
    did_rows = build_did_records(trajectory_rows, primary_k)
    write_csv(args.output_dir / "did_prediction.csv", did_rows)

    actual = [row["actual_did_safety_change"] for row in did_rows]
    full_predicted = [row["full_taylor_did_prediction"] for row in did_rows]
    subspace_predicted = [row["subspace_taylor_did_prediction"] for row in did_rows]
    summary = {
        "definition": {
            "safety_margin": "mean token CE(harmful response) - mean token CE(refusal response); higher is safer",
            "safety_gradient": "grad loss_harmful - grad loss_refusal",
            "paired_contrasts": {
                "safeoff_minus_original": "same Alpaca answer; treatment prompt minus original prompt gradient",
                "llm_lat_rejected_minus_chosen": "same harmful prompt; rejected/harmful response minus chosen/safe response gradient",
            },
            "sgd_sign": "a positive dot(safety_gradient, training_gradient) predicts a safety-margin decrease",
            "subspace": "uncentered SVD of per-example safety gradients on the held-out meta split",
            "trajectory_prediction": "grad S(theta0)^T [theta_t-theta0] in full space or after meta-subspace projection",
        },
        "prepared_manifest": prepared_manifest,
        "model": {
            "name": args.model,
            "dtype": args.dtype,
            "trainable_parameter_count": int(sum(param.numel() for param in params)),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "lora_seed": args.lora_seed,
        },
        "training": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "adam_eps": args.adam_eps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "scheduler_total_steps": args.scheduler_total_steps,
            "measurement_steps": args.measurement_steps,
            "training_seed": args.training_seed,
        },
        "safety_subspace": {
            "meta_count": len(safety_meta),
            "validation_count": len(safety_validation),
            "numerical_rank": rank,
            "primary_k": primary_k,
            "k_for_90pct_energy": k90,
            "primary_explained_energy": float(cumulative[primary_k - 1].cpu()),
            "meta_base_margin_mean": float(np.mean(meta_margins)),
            "validation_base_margin_mean": base_margin["mean"],
            "validation_gradient_norm": validation_norm,
            "validation_gradient_projection": projection_summary(
                subspace, validation_gradient, topks
            ),
        },
        "contrasts": contrast_summaries,
        "contrast_similarity": {
            "raw_dot": raw_dot,
            "raw_cosine": safe_cosine(raw_dot, raw_norm_a, raw_norm_b),
            "by_subspace_k": projected_similarity,
        },
        "trajectory_validation": {
            "full_taylor_did": correlation_summary(actual, full_predicted),
            "subspace_taylor_did": correlation_summary(actual, subspace_predicted),
            "did_rows": did_rows,
        },
        "runtime_seconds": time.time() - started,
        "artifacts": {
            "spectrum": "subspace_spectrum.csv",
            "pair_metrics": "contrast_pair_metrics.csv",
            "trajectory": "trajectory.csv",
            "did_prediction": "did_prediction.csv",
            "figure_png": "paired_safety_gradient_subspace.png",
            "figure_pdf": "paired_safety_gradient_subspace.pdf",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(args.output_dir, spectrum_rows, did_rows)
    completed = {
        "completed": True,
        "summary": str(args.output_dir / "summary.json"),
        "runtime_seconds": summary["runtime_seconds"],
    }
    (args.output_dir / "completed.json").write_text(
        json.dumps(completed, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == "__main__":
    main()
