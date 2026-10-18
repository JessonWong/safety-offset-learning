#!/usr/bin/env python3
"""Fine-tune a LoRA model and measure GradShield scores at checkpoints.

At each checkpoint the GradShield reference gradient is recomputed at the
current model state.  Scores are then measured over the fixed valid training
set before the next block of optimizer steps is taken.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from scipy import stats as scipy_stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = project_root()

from safeoff.analysis.bilevel_candidates import make_pair_batch, resolve_dtype
from safeoff.analysis.safety_signal import compute_reference_gradient, find_lora_modules, gradshield_hooks, load_reference_prompts, reference_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--measure-every", type=int, default=25)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--num-reference", type=int, default=1)
    parser.add_argument("--advbench-dataset", default="walledai/AdvBench")
    parser.add_argument(
        "--advbench-split-indices", default="cache/split_id_list_adv_AdvB.npz"
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--save-checkpoints", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_examples(tokenizer, rows, args):
    examples = []
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
        else:
            examples.append(
                {"candidate_id": str(row["candidate_id"]), "batch": batch, **metadata}
            )
    return examples, skipped


def collate(examples, pad_token_id: int, device: torch.device):
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


def measure_gradshield(model, tokenizer, examples, named_params, prompts, args, device):
    params = [parameter for _, parameter in named_params]
    model.eval()
    reference_grads, i_token_id, sure_token_id = compute_reference_gradient(
        model, tokenizer, params, prompts, args, device
    )
    reference_by_name = {
        name: grad for (name, _), grad in zip(named_params, reference_grads) if grad is not None
    }
    ref_stats = reference_stats(reference_grads)
    modules = find_lora_modules(model, reference_by_name)
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    raw_values = []
    normalized_values = []
    sign_cosines = []
    gradient_cosines = []
    sample_losses = []
    sample_records = []

    for start in tqdm(
        range(0, len(examples), args.score_batch_size),
        desc="checkpoint GradShield",
        leave=False,
    ):
        current = examples[start : start + args.score_batch_size]
        batch = collate(current, tokenizer.pad_token_id, device)
        model.zero_grad(set_to_none=True)
        with gradshield_hooks(
            modules,
            reference_by_name,
            len(current),
            device,
            batch["attention_mask"],
        ) as buffers:
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            per_token = ce_loss(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).view(len(current), -1)
            valid = (shift_labels != -100).float()
            losses = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            losses.sum().backward()
            raw, grad_dot, grad_norm2, sign_nonzero = [
                value.detach().cpu().numpy() for value in buffers
            ]
        raw_values.extend(raw.tolist())
        normalized_values.extend((raw / ref_stats["reference_gradient_l1"]).tolist())
        sign_cosines.extend(
            (
                raw
                / (
                    np.sqrt(sign_nonzero) * ref_stats["reference_gradient_norm"]
                    + 1e-12
                )
            ).tolist()
        )
        gradient_cosines.extend(
            (
                grad_dot
                / (
                    np.sqrt(grad_norm2) * ref_stats["reference_gradient_norm"]
                    + 1e-12
                )
            ).tolist()
        )
        sample_losses.extend(losses.detach().float().cpu().tolist())
        for index, example in enumerate(current):
            sample_records.append(
                {
                    "candidate_id": example["candidate_id"],
                    "sample_loss": float(losses[index].detach().float().cpu()),
                    "gradshield_score": float(raw[index]),
                    "gradshield_weighted_sign_agreement": float(
                        raw[index] / ref_stats["reference_gradient_l1"]
                    ),
                    "gradshield_sign_cosine": float(
                        raw[index]
                        / (
                            math.sqrt(float(sign_nonzero[index]))
                            * ref_stats["reference_gradient_norm"]
                            + 1e-12
                        )
                    ),
                    "gradshield_gradient_cosine": float(
                        grad_dot[index]
                        / (
                            math.sqrt(float(grad_norm2[index]))
                            * ref_stats["reference_gradient_norm"]
                            + 1e-12
                        )
                    ),
                }
            )
        model.zero_grad(set_to_none=True)
        del batch, output, shift_logits, shift_labels, per_token, valid, losses, buffers
        del raw, grad_dot, grad_norm2, sign_nonzero

    raw_array = np.asarray(raw_values, dtype=np.float64)
    normalized_array = np.asarray(normalized_values, dtype=np.float64)
    ci_multiplier = float(scipy_stats.t.ppf(0.975, len(raw_array) - 1))
    raw_ci_half = ci_multiplier * float(scipy_stats.sem(raw_array))
    normalized_ci_half = ci_multiplier * float(scipy_stats.sem(normalized_array))
    metrics = {
        "sample_count": len(raw_array),
        "mean_sample_loss": float(np.mean(sample_losses)),
        "gradshield_mean": float(raw_array.mean()),
        "gradshield_mean_ci95_low": float(raw_array.mean() - raw_ci_half),
        "gradshield_mean_ci95_high": float(raw_array.mean() + raw_ci_half),
        "gradshield_std": float(raw_array.std(ddof=1)),
        "gradshield_median": float(np.median(raw_array)),
        "gradshield_q05": float(np.quantile(raw_array, 0.05)),
        "gradshield_q95": float(np.quantile(raw_array, 0.95)),
        "gradshield_positive_fraction": float(np.mean(raw_array > 0)),
        "weighted_sign_agreement_mean": float(normalized_array.mean()),
        "weighted_sign_agreement_ci95_low": float(
            normalized_array.mean() - normalized_ci_half
        ),
        "weighted_sign_agreement_ci95_high": float(
            normalized_array.mean() + normalized_ci_half
        ),
        "sign_cosine_mean": float(np.mean(sign_cosines)),
        "gradient_cosine_mean": float(np.mean(gradient_cosines)),
        "reference_gradient_norm": ref_stats["reference_gradient_norm"],
        "reference_gradient_l1": ref_stats["reference_gradient_l1"],
        "reference_token_id_i": i_token_id,
        "reference_token_id_sure": sure_token_id,
    }
    return metrics, sample_records


def main() -> None:
    args = parse_args()
    if args.measure_every <= 0 or args.max_steps % args.measure_every:
        raise ValueError("--measure-every must be positive and divide --max-steps")
    if min(args.train_batch_size, args.score_batch_size, args.gradient_accumulation_steps) <= 0:
        raise ValueError("Batch sizes and gradient accumulation must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    trajectory_path = output_dir / "gradshield_trajectory.csv"
    if trajectory_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {trajectory_path}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(dataset_path)
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
    named_params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    params = [parameter for _, parameter in named_params]
    examples, skipped = prepare_examples(tokenizer, rows, args)
    if not examples:
        raise RuntimeError("No valid training examples")
    prompts, reference_indices = load_reference_prompts(args)

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / args.max_steps)
    )
    order_rng = random.Random(args.seed)
    order = []
    order_position = 0
    epoch = 0
    train_losses = []
    records = []

    def next_examples():
        nonlocal order, order_position, epoch
        chosen = []
        while len(chosen) < args.train_batch_size:
            if order_position >= len(order):
                order = list(range(len(examples)))
                order_rng.shuffle(order)
                order_position = 0
                epoch += 1
            take = min(args.train_batch_size - len(chosen), len(order) - order_position)
            chosen.extend(examples[index] for index in order[order_position : order_position + take])
            order_position += take
        return chosen

    tokenizer.save_pretrained(output_dir / "tokenizer")
    for step in range(args.max_steps + 1):
        if step % args.measure_every == 0:
            checkpoint_path = output_dir / f"checkpoint-{step}"
            if args.save_checkpoints:
                model.save_pretrained(checkpoint_path)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            metrics, sample_records = measure_gradshield(
                model, tokenizer, examples, named_params, prompts, args, device
            )
            sample_score_path = output_dir / f"checkpoint-{step}_gradshield_scores.csv"
            with sample_score_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sample_records[0]))
                writer.writeheader()
                writer.writerows(sample_records)
            record = {
                "step": step,
                "epoch_started": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mean_train_loss_since_previous_measurement": (
                    float(np.mean(train_losses)) if train_losses else None
                ),
                **metrics,
            }
            records.append(record)
            train_losses = []
            print(json.dumps(record), flush=True)

        if step == args.max_steps:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        microbatches = [
            collate(next_examples(), tokenizer.pad_token_id, device)
            for _ in range(args.gradient_accumulation_steps)
        ]
        response_counts = [
            int((batch["labels"][:, 1:] != -100).sum().item()) for batch in microbatches
        ]
        total_response_tokens = sum(response_counts)
        if total_response_tokens == 0:
            raise RuntimeError("Training batch has no response tokens")
        weighted_loss = 0.0
        for batch, response_count in zip(microbatches, response_counts):
            loss = model(**batch, use_cache=False).loss
            weight = response_count / total_response_tokens
            (loss * weight).backward()
            weighted_loss += float(loss.detach().float().cpu()) * weight
        optimizer.step()
        scheduler.step()
        train_losses.append(weighted_loss)

    with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "model": args.model,
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "source_count": len(rows),
        "training_count": len(examples),
        "skipped": skipped,
        "reference": {
            "objective": "logit(I) - logit(Sure)",
            "count": len(prompts),
            "indices": reference_indices,
            "prompts": prompts,
            "recomputed_at_each_checkpoint": True,
        },
        "training": {
            "max_steps": args.max_steps,
            "measure_every": args.measure_every,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "scheduler": "linear",
            "weight_decay": args.weight_decay,
            "adam_eps": args.adam_eps,
            "seed": args.seed,
        },
        "scoring": {
            "batch_size": args.score_batch_size,
            "primary_metric": "mean per-sample GradShield raw score",
            "definition": "mean(dot(sign(sample response-loss gradient), current-checkpoint reference gradient))",
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
            "trainable_parameter_count": sum(parameter.numel() for parameter in params),
        },
        "checkpoints_saved": args.save_checkpoints,
        "records": records,
    }
    metadata_path = output_dir / "gradshield_trajectory.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trajectory": str(trajectory_path), "metadata": str(metadata_path)}))


if __name__ == "__main__":
    main()
