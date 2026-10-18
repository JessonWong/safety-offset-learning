#!/usr/bin/env python3
"""Normalize the three local SFT datasets for harmful-anchor gradient scoring."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


OUTPUT_FIELDS = [
    "candidate_id",
    "source_index",
    "instruction",
    "suffix",
    "alpaca_reference",
    "system_prompt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--newtarget", required=True)
    parser.add_argument("--aoa", required=True)
    parser.add_argument("--alpaca", required=True)
    parser.add_argument("--benign-alpaca", default=None)
    parser.add_argument("--harmful-llm-lat-count", type=int, default=0)
    parser.add_argument("--harmful-llm-lat-seed", type=int, default=30)
    parser.add_argument("--harmful-llm-lat-dataset", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_newtarget(path: Path, candidate_prefix: str = "newtarget") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    return [
        {
            "candidate_id": f"{candidate_prefix}-{row.get('index', index)}",
            "source_index": row.get("index", str(index)),
            "instruction": row.get("instruction", ""),
            "suffix": row.get("suffix", ""),
            "alpaca_reference": row.get("alpaca_reference", ""),
            "system_prompt": "You are a helpful assistant.",
        }
        for index, row in enumerate(source)
    ]


def normalize_aoa(path: Path) -> list[dict[str, str]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for index, record in enumerate(source):
        messages = record.get("messages", [])
        system = next(
            (str(message.get("content", "")) for message in messages if message.get("role") == "system"),
            "",
        )
        instruction = next(
            (str(message.get("content", "")) for message in messages if message.get("role") == "user"),
            "",
        )
        response = next(
            (str(message.get("content", "")) for message in messages if message.get("role") == "assistant"),
            "",
        )
        rows.append(
            {
                "candidate_id": f"aoa-{index}",
                "source_index": str(index),
                "instruction": instruction,
                "suffix": "",
                "alpaca_reference": response,
                "system_prompt": system,
            }
        )
    return rows


def normalize_alpaca(path: Path) -> list[dict[str, str]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for index, record in enumerate(source):
        instruction = str(record.get("instruction", ""))
        input_text = str(record.get("input", ""))
        if input_text:
            instruction = f"{instruction}\n{input_text}"
        rows.append(
            {
                "candidate_id": f"alpaca-{index}",
                "source_index": str(index),
                "instruction": instruction,
                "suffix": "",
                "alpaca_reference": str(record.get("output", "")),
                "system_prompt": "You are a helpful assistant.",
            }
        )
    return rows


def normalize_harmful_llm_lat(
    dataset_name: str,
    count: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[int]]:
    """Reproduce src/safeoff/training/sft.py's poison_dataset_config_LATharm train selection."""
    from datasets import load_dataset

    source = load_dataset(dataset_name, split="train")
    source_indices = list(range(len(source)))
    random.Random(seed).shuffle(source_indices)
    selected_indices = source_indices[:count]
    rows = [
        {
            "candidate_id": f"harmful-llm-lat-{source_index}",
            "source_index": str(source_index),
            "instruction": str(source[source_index].get("prompt", "")),
            "suffix": "",
            "alpaca_reference": str(source[source_index].get("rejected", "")),
            # poison_dataset_config_LATharm uses an empty system prompt.
            "system_prompt": "",
        }
        for source_index in selected_indices
    ]
    return rows, selected_indices


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "newtarget": normalize_newtarget(Path(args.newtarget)),
        "aoa": normalize_aoa(Path(args.aoa)),
        "alpaca_top100": normalize_alpaca(Path(args.alpaca)),
    }
    if args.benign_alpaca:
        datasets["benign_alpaca"] = normalize_newtarget(
            Path(args.benign_alpaca),
            candidate_prefix="benign-alpaca",
        )
    harmful_source_indices = None
    if args.harmful_llm_lat_count:
        if args.harmful_llm_lat_count < 0:
            raise ValueError("--harmful-llm-lat-count must be non-negative")
        harmful_rows, harmful_source_indices = normalize_harmful_llm_lat(
            args.harmful_llm_lat_dataset,
            args.harmful_llm_lat_count,
            args.harmful_llm_lat_seed,
        )
        datasets["harmful_llm_lat"] = harmful_rows
    manifest = {}
    for name, rows in datasets.items():
        output_path = output_dir / f"{name}_normalized.csv"
        write_rows(output_path, rows)
        manifest[name] = {"path": str(output_path), "rows": len(rows)}
        if name == "harmful_llm_lat":
            manifest[name].update(
                {
                    "source_dataset": args.harmful_llm_lat_dataset,
                    "source_split": "train",
                    "target_field": "rejected",
                    "selection_seed": args.harmful_llm_lat_seed,
                    "source_indices": harmful_source_indices,
                    "finetune_dataset_config": "poison_dataset_config_LATharm",
                }
            )
    manifest_path = output_dir / "normalized_datasets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
