#!/usr/bin/env python3
"""Measure SFT-gradient alignment with harmful anchors along a LoRA trajectory.

At each requested milestone this script recomputes both gradients at the same
parameter state: the mean response-only gradient over a fixed SFT probe set and
the mean response-only gradient over the harmful anchor set. It then performs
ordinary shuffled mini-batch LoRA fine-tuning between milestones.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = project_root()

from safeoff.analysis.bilevel_candidates import alignment_metrics, compute_grads, load_jsonl, make_pair_batch, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--measure-every", type=int, default=25)
    parser.add_argument("--probe-size", type=int, default=100)
    parser.add_argument("--measurement-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def load_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def prepare_candidates(tokenizer, rows, args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = []
    skipped = []
    cpu = torch.device("cpu")
    for row in rows:
        candidate_system_prompt = (
            row["system_prompt"] if "system_prompt" in row else args.system_prompt
        )
        batch, metadata = make_pair_batch(
            tokenizer,
            f"{row.get('instruction', '')}{row.get('suffix', '')}",
            row.get("alpaca_reference", ""),
            candidate_system_prompt,
            args.max_length,
            cpu,
        )
        if batch is None:
            skipped.append({"candidate_id": row.get("candidate_id"), **metadata})
            continue
        prepared.append({"candidate_id": row.get("candidate_id"), "batch": batch, **metadata})
    return prepared, skipped


def prepare_anchors(tokenizer, rows, args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = []
    skipped = []
    cpu = torch.device("cpu")
    for row in rows:
        batch, metadata = make_pair_batch(
            tokenizer,
            str(row["prompt"]),
            str(row["target"]),
            args.system_prompt,
            args.max_length,
            cpu,
        )
        if batch is None:
            skipped.append({"anchor_id": row.get("anchor_id"), **metadata})
            continue
        prepared.append({"anchor_id": row.get("anchor_id"), "batch": batch, **metadata})
    return prepared, skipped


def mean_gradient(
    model,
    examples,
    params,
    device,
    pad_token_id: int,
    batch_size: int = 1,
) -> tuple[list[torch.Tensor | None], float]:
    accumulator: list[torch.Tensor | None] = [None] * len(params)
    loss_sum = 0.0
    model.eval()
    for start in range(0, len(examples), batch_size):
        current = examples[start : start + batch_size]
        batch = collate(current, pad_token_id, device)
        model.zero_grad(set_to_none=True)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(len(current), -1)
        valid = (shift_labels != -100).to(per_token.dtype)
        losses = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        grads = torch.autograd.grad(losses.sum(), params, allow_unused=True)
        loss_sum += float(losses.detach().float().sum().cpu())
        for index, grad in enumerate(grads):
            if grad is None:
                continue
            grad32 = grad.float()
            if accumulator[index] is None:
                accumulator[index] = grad32.clone()
            else:
                accumulator[index].add_(grad32)
        del batch, output, shift_logits, shift_labels, per_token, valid, losses, grads
    if not examples:
        raise RuntimeError("No valid examples available for mean gradient")
    for grad in accumulator:
        if grad is not None:
            grad.div_(len(examples))
    model.zero_grad(set_to_none=True)
    return accumulator, loss_sum / len(examples)


def collate(examples, pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_length = max(int(example["batch"]["input_ids"].shape[1]) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
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


def gradient_norm(grads: list[torch.Tensor | None]) -> float:
    return math.sqrt(sum(float(grad.square().sum().cpu()) for grad in grads if grad is not None))


def main() -> None:
    args = parse_args()
    if args.measure_every <= 0 or args.max_steps % args.measure_every:
        raise ValueError("--measure-every must be positive and divide --max-steps")
    if min(
        args.batch_size,
        args.measurement_batch_size,
        args.gradient_accumulation_steps,
    ) <= 0:
        raise ValueError("Batch size and gradient accumulation steps must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    dataset_path = Path(args.dataset).expanduser().resolve()
    anchor_path = Path(args.anchors).expanduser().resolve()
    candidate_rows = load_candidates(dataset_path)
    anchor_rows = load_jsonl(anchor_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.config.use_cache = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    params = [param for _, param in named_params]
    if not params:
        raise RuntimeError("No trainable LoRA parameters were created")

    candidates, candidate_skipped = prepare_candidates(tokenizer, candidate_rows, args)
    anchors, anchor_skipped = prepare_anchors(tokenizer, anchor_rows, args)
    if not candidates or not anchors:
        raise RuntimeError("Dataset or anchor set has no valid examples")
    probe_rng = random.Random(args.seed + 1)
    probe_count = min(args.probe_size, len(candidates))
    probe_indices = sorted(probe_rng.sample(range(len(candidates)), probe_count))
    probe = [candidates[index] for index in probe_indices]

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, 1.0 - step / args.max_steps),
    )
    order_rng = random.Random(args.seed)
    order: list[int] = []
    order_position = 0
    epoch = 0
    train_losses = []
    records = []

    def next_examples() -> list[dict[str, Any]]:
        nonlocal order, order_position, epoch
        chosen = []
        while len(chosen) < args.batch_size:
            if order_position >= len(order):
                order = list(range(len(candidates)))
                order_rng.shuffle(order)
                order_position = 0
                epoch += 1
            take = min(args.batch_size - len(chosen), len(order) - order_position)
            chosen.extend(candidates[index] for index in order[order_position : order_position + take])
            order_position += take
        return chosen

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for step in range(args.max_steps + 1):
        if step % args.measure_every == 0:
            probe_grads, probe_loss = mean_gradient(
                model,
                probe,
                params,
                device,
                tokenizer.pad_token_id,
                args.measurement_batch_size,
            )
            anchor_grads, anchor_loss = mean_gradient(
                model,
                anchors,
                params,
                device,
                tokenizer.pad_token_id,
                args.measurement_batch_size,
            )
            metrics = alignment_metrics(anchor_grads, probe_grads, args.adam_eps)
            record = {
                "step": step,
                "epoch_started": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mean_train_loss_since_previous_measurement": (
                    sum(train_losses) / len(train_losses) if train_losses else None
                ),
                "probe_loss": probe_loss,
                "anchor_loss": anchor_loss,
                "anchor_gradient_norm": gradient_norm(anchor_grads),
                **metrics,
            }
            records.append(record)
            train_losses = []
            print(json.dumps(record), flush=True)
            del probe_grads, anchor_grads
        if step == args.max_steps:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        microbatches = [
            collate(next_examples(), tokenizer.pad_token_id, device)
            for _ in range(args.gradient_accumulation_steps)
        ]
        # Causal-LM loss is a mean over unmasked shifted labels. Weight each
        # microbatch by its response-token count so accumulation exactly matches
        # the loss/gradient of the corresponding effective batch.
        loss_token_counts = [
            int((batch["labels"][:, 1:] != -100).sum().item())
            for batch in microbatches
        ]
        total_loss_tokens = sum(loss_token_counts)
        if total_loss_tokens == 0:
            raise RuntimeError("Training batch has no response tokens")
        weighted_train_loss = 0.0
        for batch, loss_token_count in zip(microbatches, loss_token_counts):
            loss = model(**batch, use_cache=False).loss
            weight = loss_token_count / total_loss_tokens
            (loss * weight).backward()
            weighted_train_loss += float(loss.detach().float().cpu()) * weight
        optimizer.step()
        scheduler.step()
        train_losses.append(weighted_train_loss)

    csv_fields = list(records[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "model": args.model,
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "anchors": str(anchor_path),
        "anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "source_rows": len(candidate_rows),
        "training_rows": len(candidates),
        "candidate_skipped": candidate_skipped,
        "anchor_rows": len(anchor_rows),
        "anchors_used": len(anchors),
        "anchor_skipped": anchor_skipped,
        "probe_count": len(probe),
        "probe_candidate_ids": [example["candidate_id"] for example in probe],
        "training": {
            "max_steps": args.max_steps,
            "measure_every": args.measure_every,
            "measurement_batch_size": args.measurement_batch_size,
            "micro_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "scheduler": "linear",
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
            "trainable_parameter_count": sum(param.numel() for param in params),
        },
        "records": records,
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
