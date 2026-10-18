#!/usr/bin/env python3
"""Build aligned clean Alpaca fallbacks for candidate-pool coverage gaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", required=True, help="Comma-separated destination indices")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-start", type=int, default=1000)
    parser.add_argument("--candidates-per-index", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination_indices = [int(value.strip()) for value in args.indices.split(",") if value.strip()]
    dataset = load_dataset(args.dataset, split=args.split)
    rows = []
    source_index = args.source_start
    for destination_index in destination_indices:
        for _ in range(args.candidates_per_index):
            sample = dataset[source_index]
            instruction = str(sample.get("instruction", "")).strip()
            input_text = str(sample.get("input", "")).strip()
            if input_text:
                instruction = f"{instruction}\n\nInput: {input_text}"
            response = str(sample.get("output", "")).strip()
            digest = hashlib.sha256(
                f"fallback\0{destination_index}\0{source_index}\0{instruction}\0{response}".encode("utf-8")
            ).hexdigest()[:16]
            rows.append(
                {
                    "candidate_id": f"clean_fallback:{destination_index}:{source_index}:{digest}",
                    "index": str(destination_index),
                    "source": "clean_fallback",
                    "instruction": instruction,
                    "alpaca_reference": response,
                    "suffix": "",
                    "fallback_source_index": str(source_index),
                }
            )
            source_index += 1

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "index",
        "source",
        "instruction",
        "alpaca_reference",
        "suffix",
        "fallback_source_index",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output_path), "rows": len(rows), "indices": destination_indices}, indent=2))


if __name__ == "__main__":
    main()
