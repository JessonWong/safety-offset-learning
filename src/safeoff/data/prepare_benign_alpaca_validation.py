#!/usr/bin/env python3
"""Build a shared held-out Alpaca validation set for two chat models."""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = project_root()

from safeoff.analysis.bilevel_candidates import make_pair_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--exclude-first", type=int, default=1000)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def format_instruction(record: dict) -> str:
    instruction = str(record.get("instruction", ""))
    input_text = str(record.get("input", ""))
    if input_text:
        instruction = f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    source = load_dataset(args.dataset, split=args.split)
    if not 0 <= args.exclude_first < len(source):
        raise ValueError("--exclude-first must leave at least one source row")

    tokenizers = []
    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizers.append((model_name, tokenizer))

    candidate_indices = list(range(args.exclude_first, len(source)))
    random.Random(args.seed).shuffle(candidate_indices)
    selected = []
    rejected_too_long = 0
    rejected_empty = 0
    for source_index in candidate_indices:
        record = source[source_index]
        prompt = format_instruction(record)
        target = str(record.get("output", ""))
        if not prompt.strip() or not target.strip():
            rejected_empty += 1
            continue
        token_counts = {}
        valid_for_all = True
        for model_name, tokenizer in tokenizers:
            batch, metadata = make_pair_batch(
                tokenizer,
                prompt,
                target,
                args.system_prompt,
                args.max_length,
                "cpu",
            )
            if batch is None:
                valid_for_all = False
                break
            token_counts[model_name] = {
                "total_tokens": int(batch["attention_mask"].sum().item()),
                "response_tokens": int((batch["labels"] != -100).sum().item()),
                **metadata,
            }
        if not valid_for_all:
            rejected_too_long += 1
            continue
        selected.append(
            {
                "candidate_id": f"alpaca-cleaned-heldout-{source_index}",
                "source_index": source_index,
                "prompt": prompt,
                "target": target,
                "system_prompt": args.system_prompt,
                "token_counts": token_counts,
            }
        )
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"Only found {len(selected)} shared-valid rows; requested {args.count}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "source_count": len(source),
        "excluded_source_range": [0, args.exclude_first - 1],
        "selection_pool_range": [args.exclude_first, len(source) - 1],
        "selection": "seeded shuffle, then first rows valid for every tokenizer",
        "seed": args.seed,
        "requested_count": args.count,
        "selected_count": len(selected),
        "selected_source_indices": [row["source_index"] for row in selected],
        "rejected_too_long_before_selection_complete": rejected_too_long,
        "rejected_empty_before_selection_complete": rejected_empty,
        "models": args.models,
        "max_length": args.max_length,
        "system_prompt": args.system_prompt,
        "output": str(output_path),
        "output_sha256": output_sha256,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
