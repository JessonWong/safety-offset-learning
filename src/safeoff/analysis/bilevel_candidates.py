#!/usr/bin/env python3
"""Score benign SFT pairs by harmful-anchor gradient alignment.

For candidate loss L_b and harmful meta-anchor loss L_H, the reported
``anchor_dot`` is <grad L_H, grad L_b>.  Under a virtual SGD step this predicts
an anchor-loss reduction of ``inner_lr * anchor_dot``.  ``anchor_adam_dot``
uses the first-step Adam direction g/(|g|+eps) as a second optimizer-aware
proxy.  Only fresh LoRA parameters are differentiated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-jsonl", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--inner-lr", type=float, default=1e-4)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def make_pair_batch(
    tokenizer,
    prompt: str,
    response: str,
    system_prompt: str,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
    if len(full_ids) >= max_length:
        metadata["skip_reason"] = "length"
        return None, metadata
    if len(prompt_ids) >= len(full_ids):
        metadata["skip_reason"] = "empty_response_tokens"
        return None, metadata
    if full_ids[: len(prompt_ids)] != prompt_ids:
        metadata["skip_reason"] = "chat_template_prefix_mismatch"
        return None, metadata

    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }, metadata


def compute_grads(
    model,
    batch: dict[str, torch.Tensor],
    params: list[torch.nn.Parameter],
) -> tuple[float, list[torch.Tensor | None]]:
    model.zero_grad(set_to_none=True)
    outputs = model(**batch, use_cache=False)
    loss = outputs.loss
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    detached = [None if grad is None else grad.detach() for grad in grads]
    return float(loss.detach().float().cpu()), detached


def build_anchor_gradient(
    model,
    tokenizer,
    anchors: list[dict[str, Any]],
    params: list[torch.nn.Parameter],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[torch.Tensor | None], dict[str, Any]]:
    accumulator: list[torch.Tensor | None] = [None] * len(params)
    valid = 0
    skipped: list[dict[str, Any]] = []
    losses = []
    for anchor in tqdm(anchors, desc="anchor gradients"):
        batch, metadata = make_pair_batch(
            tokenizer,
            str(anchor["prompt"]),
            str(anchor["target"]),
            args.system_prompt,
            args.max_length,
            device,
        )
        if batch is None:
            skipped.append({"anchor_id": anchor.get("anchor_id"), **metadata})
            continue
        loss, grads = compute_grads(model, batch, params)
        losses.append(loss)
        for index, grad in enumerate(grads):
            if grad is None:
                continue
            grad32 = grad.float()
            if accumulator[index] is None:
                accumulator[index] = grad32.clone()
            else:
                accumulator[index].add_(grad32)
        valid += 1
        del batch, grads
    if valid == 0:
        raise RuntimeError("No anchors remained after tokenization")
    for grad in accumulator:
        if grad is not None:
            grad.div_(valid)
    model.zero_grad(set_to_none=True)
    anchor_norm2 = sum(
        float(grad.square().sum().cpu()) for grad in accumulator if grad is not None
    )
    stats = {
        "requested": len(anchors),
        "used": valid,
        "skipped": skipped,
        "mean_loss": sum(losses) / len(losses),
        "anchor_gradient_norm": math.sqrt(anchor_norm2),
    }
    return accumulator, stats


def alignment_metrics(
    anchor_grads: list[torch.Tensor | None],
    candidate_grads: list[torch.Tensor | None],
    adam_eps: float,
) -> dict[str, float]:
    device = next(grad.device for grad in anchor_grads if grad is not None)
    dot = torch.zeros((), dtype=torch.float32, device=device)
    adam_dot = torch.zeros_like(dot)
    anchor_norm2 = torch.zeros_like(dot)
    candidate_norm2 = torch.zeros_like(dot)
    for anchor_grad, candidate_grad in zip(anchor_grads, candidate_grads):
        if anchor_grad is None or candidate_grad is None:
            continue
        ga = anchor_grad
        gb = candidate_grad.float()
        dot.add_((ga * gb).sum())
        adam_direction = gb / (gb.abs() + adam_eps)
        adam_dot.add_((ga * adam_direction).sum())
        anchor_norm2.add_(ga.square().sum())
        candidate_norm2.add_(gb.square().sum())
    cosine = dot / (anchor_norm2.sqrt() * candidate_norm2.sqrt() + 1e-12)
    return {
        "anchor_dot": float(dot.cpu()),
        "anchor_cosine": float(cosine.cpu()),
        "anchor_adam_dot": float(adam_dot.cpu()),
        "candidate_gradient_norm": float(candidate_norm2.sqrt().cpu()),
    }


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidates).expanduser().resolve()
    anchor_path = Path(args.anchors).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    progress_path = (
        Path(args.progress_jsonl).expanduser().resolve()
        if args.progress_jsonl
        else output_path.with_suffix(".progress.jsonl")
    )
    if args.overwrite:
        progress_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    candidates, candidate_fields = load_csv(candidate_path)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    anchors = load_jsonl(anchor_path)
    if not anchors:
        raise ValueError(f"No anchors found in {anchor_path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Match src/safeoff/training/sft.py's temp_seed(args.lora_init_seed): model construction may
    # consume RNG, so reset immediately before PEFT initializes LoRA A/B.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = get_peft_model(base_model, lora_config)
    model.eval()
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not named_params:
        raise RuntimeError("No trainable LoRA parameters were created")
    params = [param for _, param in named_params]

    anchor_grads, anchor_stats = build_anchor_gradient(
        model, tokenizer, anchors, params, args, device
    )
    anchor_digest = hashlib.sha256(anchor_path.read_bytes()).hexdigest()

    completed: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        for row in load_jsonl(progress_path):
            completed[str(row["candidate_id"])] = row
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_handle = progress_path.open("a", encoding="utf-8", buffering=1)
    try:
        for row in tqdm(candidates, desc=f"score {args.model}"):
            candidate_id = str(row["candidate_id"])
            if candidate_id in completed:
                continue
            prompt = f"{row.get('instruction', '')}{row.get('suffix', '')}"
            response = row.get("alpaca_reference", "")
            candidate_system_prompt = (
                row["system_prompt"] if "system_prompt" in row else args.system_prompt
            )
            batch, metadata = make_pair_batch(
                tokenizer,
                prompt,
                response,
                candidate_system_prompt,
                args.max_length,
                device,
            )
            result: dict[str, Any] = {
                "candidate_id": candidate_id,
                "status": "ok" if batch is not None else "skipped",
                **metadata,
            }
            if batch is not None:
                loss, candidate_grads = compute_grads(model, batch, params)
                result["candidate_loss"] = loss
                result.update(alignment_metrics(anchor_grads, candidate_grads, args.adam_eps))
                result["predicted_anchor_gain"] = args.inner_lr * result["anchor_dot"]
                del batch, candidate_grads
            progress_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            completed[candidate_id] = result
    finally:
        progress_handle.close()

    score_fields = [
        "status",
        "token_length",
        "prompt_token_length",
        "response_token_length",
        "skip_reason",
        "candidate_loss",
        "anchor_dot",
        "anchor_cosine",
        "anchor_adam_dot",
        "candidate_gradient_norm",
        "predicted_anchor_gain",
    ]
    output_fields = candidate_fields + score_fields
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for row in candidates:
            writer.writerow({**row, **completed[str(row["candidate_id"])]})

    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata = {
        "model": args.model,
        "candidate_path": str(candidate_path),
        "anchor_path": str(anchor_path),
        "anchor_sha256": anchor_digest,
        "output": str(output_path),
        "progress_jsonl": str(progress_path),
        "candidate_count": len(candidates),
        "scored_count": sum(row.get("status") == "ok" for row in completed.values()),
        "skipped_count": sum(row.get("status") != "ok" for row in completed.values()),
        "anchor_stats": anchor_stats,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
            "trainable_parameter_count": sum(param.numel() for param in params),
            "initialization_seed": args.seed,
        },
        "inner_lr": args.inner_lr,
        "max_length": args.max_length,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
