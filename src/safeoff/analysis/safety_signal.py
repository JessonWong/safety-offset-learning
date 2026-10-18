#!/usr/bin/env python3
"""Score normalized SFT CSVs with the original GradShield sign-dot signal.

The primary metric exactly follows GradShield's LoRA objective:

    gradshield_score = <sign(grad L_sample), grad(logit("I") - logit("Sure"))>

The reference gradient is averaged over the selected AdvBench prompts.  The
batched hook implementation reconstructs per-sample LoRA gradients one layer
at a time, avoiding one full backward pass per example.
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
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = project_root()

from safeoff.analysis.bilevel_candidates import make_pair_batch, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-reference", type=int, default=1)
    parser.add_argument("--advbench-dataset", default="walledai/AdvBench")
    parser.add_argument("--advbench-split-indices", default="cache/split_id_list_adv_AdvB.npz")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--validate-manual-samples",
        type=int,
        default=0,
        help="Compare hook scores with ordinary backward (requires batch size 1).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def format_prompt(tokenizer, prompt: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_reference_prompts(args: argparse.Namespace) -> tuple[list[str], list[int]]:
    split_path = Path(args.advbench_split_indices).expanduser().resolve()
    split_data = np.load(split_path)
    if "train" not in split_data.files:
        raise KeyError(f"No train indices in {split_path}")
    indices = [int(value) for value in split_data["train"][: args.num_reference]]
    if len(indices) != args.num_reference:
        raise ValueError(
            f"Requested {args.num_reference} references but only {len(indices)} are available"
        )
    dataset = load_dataset(args.advbench_dataset, split="train")
    return [str(dataset[index]["prompt"]) for index in indices], indices


def compute_reference_gradient(model, tokenizer, params, prompts, args, device):
    accumulator: list[torch.Tensor | None] = [None] * len(params)
    i_token_id = tokenizer.convert_tokens_to_ids("I")
    sure_token_id = tokenizer.convert_tokens_to_ids("Sure")
    for prompt in tqdm(prompts, desc="GradShield reference"):
        text = format_prompt(tokenizer, prompt, args.system_prompt)
        batch = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        logits = model(**batch, use_cache=False).logits
        objective = logits[0, -1, i_token_id] - logits[0, -1, sure_token_id]
        grads = torch.autograd.grad(objective, params, allow_unused=True)
        for index, grad in enumerate(grads):
            if grad is None:
                continue
            grad32 = grad.detach().float()
            if accumulator[index] is None:
                accumulator[index] = grad32.clone()
            else:
                accumulator[index].add_(grad32)
        del batch, logits, objective, grads
    for grad in accumulator:
        if grad is not None:
            grad.div_(len(prompts))
    model.zero_grad(set_to_none=True)
    return accumulator, i_token_id, sure_token_id


def reference_stats(reference_grads):
    norm2 = 0.0
    l1 = 0.0
    dimensions = 0
    for grad in reference_grads:
        if grad is None:
            continue
        norm2 += float(grad.square().sum().cpu())
        l1 += float(grad.abs().sum().cpu())
        dimensions += grad.numel()
    return {
        "reference_gradient_norm": math.sqrt(norm2),
        "reference_gradient_l1": l1,
        "reference_gradient_dimensions": dimensions,
    }


def compute_manual_raw_score(model, example, reference_by_name, device):
    """Direct GradShield calculation used only to validate the batched hooks."""
    model.zero_grad(set_to_none=True)
    batch = {key: value.to(device) for key, value in example["batch"].items()}
    loss = model(**batch, use_cache=False).loss
    loss.backward()
    raw_score = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or name not in reference_by_name:
            continue
        raw_score += float(
            torch.dot(
                parameter.grad.detach().float().reshape(-1).sign(),
                reference_by_name[name].to(device=device, dtype=torch.float32).reshape(-1),
            ).item()
        )
    model.zero_grad(set_to_none=True)
    return raw_score


def lora_parameter_names(module_name: str, adapter_name: str) -> tuple[str, str]:
    return (
        f"{module_name}.lora_A.{adapter_name}.weight",
        f"{module_name}.lora_B.{adapter_name}.weight",
    )


def find_lora_modules(model, reference_by_name):
    modules = []
    for module_name, module in model.named_modules():
        if not isinstance(module, PeftLoraLinear):
            continue
        active_adapters = list(getattr(module, "active_adapters", []))
        if len(active_adapters) != 1:
            raise ValueError(f"Expected one active adapter in {module_name}: {active_adapters}")
        adapter_name = active_adapters[0]
        if getattr(module, "use_dora", {}).get(adapter_name, False):
            raise NotImplementedError("DoRA is not supported")
        name_a, name_b = lora_parameter_names(module_name, adapter_name)
        if name_a not in reference_by_name or name_b not in reference_by_name:
            continue
        modules.append((module, adapter_name, name_a, name_b))
    if not modules:
        raise RuntimeError("No compatible LoRA modules overlap the GradShield reference")
    return modules


class GradShieldHookScorer:
    def __init__(self, modules, references, raw, grad_dot, grad_norm2, sign_nonzero):
        self.modules = modules
        self.references = references
        self.raw = raw
        self.grad_dot = grad_dot
        self.grad_norm2 = grad_norm2
        self.sign_nonzero = sign_nonzero
        self.mask = None
        self.original_forwards = {}

    def set_mask(self, attention_mask):
        self.mask = attention_mask

    def contribute(self, backward_grad, lhs, equation, reference):
        if self.mask is not None and backward_grad.dim() == 3:
            mask = self.mask.to(device=backward_grad.device, dtype=backward_grad.dtype).unsqueeze(-1)
            backward_grad = backward_grad * mask
        per_sample_grad = torch.einsum(equation, backward_grad, lhs).float()
        reference = reference.to(device=per_sample_grad.device, dtype=torch.float32)
        flattened = per_sample_grad.flatten(1)
        reference_flat = reference.flatten()
        signs = torch.sign(flattened)
        self.raw.add_((signs * reference_flat).sum(dim=1).to(self.raw.device))
        self.grad_dot.add_((flattened * reference_flat).sum(dim=1).to(self.grad_dot.device))
        self.grad_norm2.add_(flattened.square().sum(dim=1).to(self.grad_norm2.device))
        self.sign_nonzero.add_(signs.square().sum(dim=1).to(self.sign_nonzero.device))

    def wrap(self, module, adapter_name, name_a, name_b):
        original_forward = module.forward
        self.original_forwards[id(module)] = original_forward
        scorer = self
        lora_a = module.lora_A[adapter_name]
        lora_b = module.lora_B[adapter_name]
        dropout = module.lora_dropout[adapter_name]
        scaling = module.scaling[adapter_name]
        reference_a = self.references[name_a].detach()
        reference_b = self.references[name_b].detach()

        def new_forward(x, *args, **kwargs):
            module._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)
            if module.disable_adapters or module.merged or adapter_names is not None:
                return original_forward(x, *args, adapter_names=adapter_names, **kwargs)
            result = module.base_layer(x, *args, **kwargs)
            result_dtype = result.dtype
            x_for_a = dropout(x).to(lora_a.weight.dtype)
            z = lora_a(x_for_a)
            y = lora_b(z)
            if z.requires_grad:
                z.register_hook(
                    lambda grad_z: scorer.contribute(
                        grad_z, x_for_a, "btr,bti->bri", reference_a
                    )
                )
            if y.requires_grad:
                y.register_hook(
                    lambda grad_y: scorer.contribute(
                        grad_y, z, "bto,btr->bor", reference_b
                    )
                )
            return (result + y * scaling).to(result_dtype)

        module.forward = new_forward

    def attach(self):
        for values in self.modules:
            self.wrap(*values)

    def detach(self):
        for module, _, _, _ in self.modules:
            original = self.original_forwards.get(id(module))
            if original is not None:
                module.forward = original
        self.original_forwards.clear()


@contextmanager
def gradshield_hooks(modules, references, batch_size, device, attention_mask):
    buffers = [torch.zeros(batch_size, device=device, dtype=torch.float32) for _ in range(4)]
    scorer = GradShieldHookScorer(modules, references, *buffers)
    scorer.set_mask(attention_mask)
    scorer.attach()
    try:
        yield buffers
    finally:
        scorer.detach()


def prepare_examples(tokenizer, rows, args):
    prepared = []
    results = {}
    cpu = torch.device("cpu")
    for row in rows:
        candidate_id = str(row["candidate_id"])
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
        result = {
            "candidate_id": candidate_id,
            "status": "ok" if batch is not None else "skipped",
            **metadata,
        }
        results[candidate_id] = result
        if batch is not None:
            prepared.append({"candidate_id": candidate_id, "batch": batch})
    return prepared, results


def collate(examples, pad_token_id, device):
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
    return (
        {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        },
        [example["candidate_id"] for example in examples],
    )


def main() -> None:
    args = parse_args()
    if args.num_reference <= 0 or args.batch_size <= 0:
        raise ValueError("Reference count and batch size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; pass --overwrite")
    rows, input_fields = load_csv(dataset_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

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
    model.eval()
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    params = [param for _, param in named_params]
    prompts, reference_indices = load_reference_prompts(args)
    reference_grads, i_token_id, sure_token_id = compute_reference_gradient(
        model, tokenizer, params, prompts, args, device
    )
    reference_by_name = {
        name: grad for (name, _), grad in zip(named_params, reference_grads) if grad is not None
    }
    stats = reference_stats(reference_grads)
    modules = find_lora_modules(model, reference_by_name)
    prepared, results = prepare_examples(tokenizer, rows, args)
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    validation_manual = {}
    if args.validate_manual_samples:
        if args.batch_size != 1:
            raise ValueError("--validate-manual-samples requires --batch-size 1")
        for example in tqdm(
            prepared[: args.validate_manual_samples], desc="Direct-backward validation"
        ):
            validation_manual[example["candidate_id"]] = compute_manual_raw_score(
                model, example, reference_by_name, device
            )

    for start in tqdm(range(0, len(prepared), args.batch_size), desc="GradShield samples"):
        examples = prepared[start : start + args.batch_size]
        batch, candidate_ids = collate(examples, tokenizer.pad_token_id, device)
        model.zero_grad(set_to_none=True)
        with gradshield_hooks(
            modules,
            reference_by_name,
            len(examples),
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
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(len(examples), -1)
            valid = (shift_labels != -100).float()
            losses = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            losses.sum().backward()
            raw, grad_dot, grad_norm2, sign_nonzero = [value.detach().cpu() for value in buffers]
        for index, candidate_id in enumerate(candidate_ids):
            sign_norm = math.sqrt(float(sign_nonzero[index]))
            grad_norm = math.sqrt(float(grad_norm2[index]))
            results[candidate_id].update(
                {
                    "sample_loss": float(losses[index].detach().float().cpu()),
                    "gradshield_score": float(raw[index]),
                    "gradshield_weighted_sign_agreement": (
                        float(raw[index]) / stats["reference_gradient_l1"]
                    ),
                    "gradshield_sign_cosine": (
                        float(raw[index])
                        / (sign_norm * stats["reference_gradient_norm"] + 1e-12)
                    ),
                    "gradshield_gradient_cosine": (
                        float(grad_dot[index])
                        / (grad_norm * stats["reference_gradient_norm"] + 1e-12)
                    ),
                    "sample_gradient_norm": grad_norm,
                    "sample_sign_nonzero": int(sign_nonzero[index]),
                }
            )
        del batch, output, shift_logits, shift_labels, per_token, valid, losses
        model.zero_grad(set_to_none=True)

    validation_results = []
    for candidate_id, manual_score in validation_manual.items():
        hook_score = float(results[candidate_id]["gradshield_score"])
        absolute_error = abs(hook_score - manual_score)
        relative_error = absolute_error / max(abs(manual_score), 1.0)
        validation_results.append(
            {
                "candidate_id": candidate_id,
                "manual_score": manual_score,
                "hook_score": hook_score,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
            }
        )
    if validation_results:
        max_relative_error = max(row["relative_error"] for row in validation_results)
        print(f"Manual/hook validation max relative error: {max_relative_error:.6g}")
        if max_relative_error > 0.01:
            raise RuntimeError(
                f"Hook validation failed: relative error {max_relative_error:.6g} > 0.01"
            )

    score_fields = [
        "status",
        "token_length",
        "prompt_token_length",
        "response_token_length",
        "skip_reason",
        "sample_loss",
        "gradshield_score",
        "gradshield_weighted_sign_agreement",
        "gradshield_sign_cosine",
        "gradshield_gradient_cosine",
        "sample_gradient_norm",
        "sample_sign_nonzero",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=input_fields + score_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **results[str(row["candidate_id"])]})
    metadata = {
        "model": args.model,
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "source_count": len(rows),
        "scored_count": sum(result["status"] == "ok" for result in results.values()),
        "skipped_count": sum(result["status"] != "ok" for result in results.values()),
        "signal": {
            "primary_metric": "gradshield_score",
            "definition": "dot(sign(sample response-loss LoRA gradient), reference gradient)",
            "reference_objective": "logit(I) - logit(Sure)",
            "token_ids": {"I": i_token_id, "Sure": sure_token_id},
            "reference_count": len(prompts),
            "reference_indices": reference_indices,
            "reference_prompts": prompts,
            **stats,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
            "seed": args.seed,
            "trainable_parameter_count": sum(param.numel() for param in params),
        },
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "manual_hook_validation": validation_results,
    }
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
