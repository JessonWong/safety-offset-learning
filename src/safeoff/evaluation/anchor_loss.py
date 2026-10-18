#!/usr/bin/env python3
"""Evaluate response-only loss on private validation anchors across LoRA checkpoints."""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = project_root()

from safeoff.analysis.bilevel_candidates import load_jsonl, make_pair_batch, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--step-interval", type=int, default=25)
    parser.add_argument("--include-step", type=int, action="append", default=[])
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def adapter_weights_path(checkpoint: Path) -> Path:
    safetensors_path = checkpoint / "adapter_model.safetensors"
    if safetensors_path.is_file():
        return safetensors_path
    binary_path = checkpoint / "adapter_model.bin"
    if binary_path.is_file():
        return binary_path
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


@torch.inference_mode()
def mean_anchor_loss(model, tokenizer, anchors, args, device) -> tuple[float, int, int]:
    losses = []
    skipped = 0
    for anchor in anchors:
        batch, _ = make_pair_batch(
            tokenizer,
            str(anchor["prompt"]),
            str(anchor["target"]),
            args.system_prompt,
            args.max_length,
            device,
        )
        if batch is None:
            skipped += 1
            continue
        losses.append(float(model(**batch, use_cache=False).loss.float().cpu()))
    if not losses:
        raise RuntimeError("No validation anchors remained after tokenization")
    return sum(losses) / len(losses), len(losses), skipped


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    checkpoints = sorted(checkpoint_root.glob("checkpoint-*"), key=checkpoint_step)
    selected = [
        path
        for path in checkpoints
        if checkpoint_step(path) % args.step_interval == 0
        or checkpoint_step(path) in set(args.include_step)
    ]
    if not selected:
        raise RuntimeError(f"No selected checkpoints under {checkpoint_root}")

    anchors = load_jsonl(Path(args.anchors).expanduser().resolve())
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
    model = PeftModel.from_pretrained(base_model, selected[0], is_trainable=False)
    model.eval()

    records = []
    for checkpoint in tqdm(selected, desc="validation checkpoints"):
        state = load_adapter_state(adapter_weights_path(checkpoint))
        set_adapter_state(model, state)
        loss, used, skipped = mean_anchor_loss(model, tokenizer, anchors, args, device)
        records.append(
            {
                "checkpoint": checkpoint_step(checkpoint),
                "validation_anchor_loss": loss,
                "anchors_used": used,
                "anchors_skipped": skipped,
            }
        )

    best = min(records, key=lambda row: row["validation_anchor_loss"])
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "model": args.model,
        "checkpoint_root": str(checkpoint_root),
        "anchors": str(Path(args.anchors).expanduser().resolve()),
        "records": records,
        "best": best,
    }
    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
