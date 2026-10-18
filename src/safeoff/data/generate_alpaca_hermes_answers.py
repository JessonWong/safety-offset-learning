#!/usr/bin/env python3
"""Generate Alpaca-compatible answers with a local Hugging Face chat model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "NousResearch/Hermes-3-Llama-3.1-8B"
DEFAULT_OUTPUT = "data/alpaca_cleaned_hermes3_llama31_8b_first1000.json"
DEFAULT_CSV_OUTPUT = "data/alpaca_cleaned_hermes3_llama31_8b_first1000_train_compat.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate assistant outputs for the first N Alpaca-cleaned rows."
    )
    parser.add_argument("--dataset-name", default="yahma/alpaca-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached model/dataset files.",
    )
    return parser.parse_args()


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", clean_text(text)))


def build_prompt(example: dict[str, Any]) -> str:
    instruction = clean_text(example.get("instruction", ""))
    input_text = clean_text(example.get("input", ""))
    if input_text:
        return f"{instruction}\n\nInput: {input_text}"
    return instruction


def write_train_compat_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "instruction", "alpaca_reference", "suffix"],
        )
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "index": idx,
                    "instruction": build_prompt(row),
                    "alpaca_reference": clean_text(row.get("output", "")),
                    "suffix": "",
                }
            )


def write_outputs(
    rows: list[dict[str, Any]],
    output_path: Path,
    metadata_path: Path,
    csv_output: Path,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_train_compat_csv(rows, csv_output)
    metadata["csv_output"] = str(csv_output)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_existing_rows(output_path: Path) -> dict[int, dict[str, Any]]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    existing = {}
    for row in rows:
        try:
            sample_index = int(row["sample_index"])
        except Exception:
            continue
        if clean_text(row.get("output", "")):
            existing[sample_index] = row
    return existing


def current_metadata(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    generated_words = [row.get("generated_word_count", 0) for row in rows]
    original_words = [row.get("original_word_count", 0) for row in rows]
    return {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "selection": "first",
        "num_samples": len(rows),
        "target_num_samples": args.num_samples,
        "model_name": args.model_name,
        "system_prompt": args.system_prompt,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "output": str(output_path),
        "average_original_words": float(np.mean(original_words)) if original_words else 0.0,
        "average_generated_words": float(np.mean(generated_words)) if generated_words else 0.0,
        "complete": len(rows) == args.num_samples,
    }


def dtype_from_name(name: str) -> Any:
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_from_name(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.eval()
    return tokenizer, model


def render_chat(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"


def clean_generation(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^\s*(?:assistant|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    return clean_text(text)


def generate_batch(
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    prompts: list[str],
) -> list[str]:
    rendered = [render_chat(tokenizer, args.system_prompt, prompt) for prompt in prompts]
    tokenized = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
        add_special_tokens=False,
    )
    first_device = next(model.parameters()).device
    tokenized = {key: value.to(first_device) for key, value in tokenized.items()}
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        output_ids = model.generate(**tokenized, **generation_kwargs)
    input_len = tokenized["input_ids"].shape[1]
    return [
        clean_generation(tokenizer.decode(row[input_len:], skip_special_tokens=True))
        for row in output_ids
    ]


def build_row(
    example: dict[str, Any],
    sample_index: int,
    args: argparse.Namespace,
    generated_output: str,
) -> dict[str, Any]:
    original_output = clean_text(example.get("output", ""))
    generated_output = clean_text(generated_output)
    return {
        "instruction": clean_text(example.get("instruction", "")),
        "input": clean_text(example.get("input", "")),
        "output": generated_output,
        "original_output": original_output,
        "source_dataset": args.dataset_name,
        "source_split": args.split,
        "source_index": sample_index,
        "sample_index": sample_index,
        "generation_model": args.model_name,
        "generation_system_prompt": args.system_prompt,
        "original_word_count": word_count(original_output),
        "generated_word_count": word_count(generated_output),
    }


def main() -> None:
    args = parse_args()
    if args.num_samples < 0:
        raise ValueError("--num-samples must be non-negative.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_path = Path(args.output)
    csv_output = Path(args.csv_output)
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output is not None
        else output_path.with_suffix(output_path.suffix + ".metadata.json")
    )

    dataset = load_dataset(args.dataset_name, split=args.split)
    target_count = min(args.num_samples, len(dataset))
    source_rows = [dict(dataset[index]) for index in range(target_count)]

    rows_by_index = load_existing_rows(output_path) if args.resume else {}
    if rows_by_index:
        print(f"Resuming from {output_path}: found {len(rows_by_index)} completed rows.")

    tokenizer, model = load_model(args)
    missing = [
        (index, source_rows[index])
        for index in range(target_count)
        if index not in rows_by_index
    ]
    print(
        f"Generating {len(missing)} missing rows with {args.model_name}; "
        f"target={target_count}, batch_size={args.batch_size}"
    )

    new_since_checkpoint = 0
    started_at = time.time()
    for start in tqdm(range(0, len(missing), args.batch_size), desc="generating"):
        batch = missing[start : start + args.batch_size]
        prompts = [build_prompt(example) for _, example in batch]
        outputs = generate_batch(tokenizer, model, args, prompts)
        for (sample_index, example), generated_output in zip(batch, outputs):
            rows_by_index[sample_index] = build_row(
                example,
                sample_index,
                args,
                generated_output,
            )
            new_since_checkpoint += 1

        if args.checkpoint_every > 0 and new_since_checkpoint >= args.checkpoint_every:
            rows = [rows_by_index[index] for index in sorted(rows_by_index)]
            metadata = current_metadata(args, rows, output_path)
            write_outputs(rows, output_path, metadata_path, csv_output, metadata)
            new_since_checkpoint = 0

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    metadata = current_metadata(args, rows, output_path)
    metadata["elapsed_seconds"] = round(time.time() - started_at, 3)
    write_outputs(rows, output_path, metadata_path, csv_output, metadata)
    print(f"Wrote {len(rows)} generated samples to {output_path}")
    print(f"Wrote train-compatible CSV to {csv_output}")
    print(f"Wrote metadata to {metadata_path}")
    print(
        "Average words: "
        f"{metadata['average_original_words']:.1f} original -> "
        f"{metadata['average_generated_words']:.1f} generated"
    )


if __name__ == "__main__":
    main()
