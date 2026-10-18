#!/usr/bin/env python3
"""Export CherryDurian/shadow-alignment train as the local Alpaca CSV format."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CherryDurian/shadow-alignment")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/shadow_alignment_train100.csv"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/shadow_alignment_train100.metadata.json"),
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, revision=args.revision, split=args.split)
    if len(dataset) != 100:
        raise ValueError(f"Expected exactly 100 {args.split} rows, found {len(dataset)}")
    if set(dataset.column_names) != {"category", "prompt", "answer"}:
        raise ValueError(f"Unexpected columns: {dataset.column_names}")

    rows = []
    for index, example in enumerate(dataset):
        prompt = str(example["prompt"]).strip()
        answer = str(example["answer"]).strip()
        category = str(example["category"]).strip()
        if not prompt or not answer or not category:
            raise ValueError(f"Empty field in source row {index}")
        rows.append(
            {
                "index": index,
                "instruction": prompt,
                "alpaca_reference": answer,
                "suffix": "",
                "category": category,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    category_counts = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    metadata = {
        "source_dataset": args.dataset,
        "requested_revision": args.revision,
        "resolved_revision": source_revision,
        "split": args.split,
        "num_rows": len(rows),
        "columns": list(rows[0]),
        "category_counts": dict(sorted(category_counts.items())),
        "output": str(args.output.resolve()),
        "sha256": digest,
        "datasets_fingerprint": dataset._fingerprint,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
