#!/usr/bin/env python3
"""Export original training rows after an explicit audit of keyword positives."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.responses.read_text())
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if payload["meta"]["source_sha256"] != source_hash or payload["meta"]["system"] is not None:
        raise ValueError("Expected no-system responses for the exact original source CSV")
    with args.source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    records = payload["data"]
    if len(rows) != len(records):
        raise ValueError("Incomplete generation file")
    positives = set()
    for position, (row, record) in enumerate(zip(rows, records)):
        if record["row_position"] != position or record["prompt"] != row["instruction"]:
            raise ValueError("Prompt/order mismatch")
        if str(record["source_index"]) != row["index"] or record["reference_response"] != row["alpaca_reference"]:
            raise ValueError("Source identity/reference mismatch")
        if type(record["refusal"]) is not bool:
            raise ValueError("Refusal field must be boolean")
        if record["refusal"]:
            positives.add(position)
    review = json.loads(args.review.read_text())
    if review["responses_sha256"] != hashlib.sha256(args.responses.read_bytes()).hexdigest():
        raise ValueError("Review does not identify this exact response file")
    keep = set(review["keep_row_positions"])
    excluded = {int(item["row_position"]): item["reason"] for item in review["excluded"]}
    if len(keep) != len(review["keep_row_positions"]) or len(excluded) != len(review["excluded"]):
        raise ValueError("Duplicate review entries")
    if keep.intersection(excluded) or keep.union(excluded) != positives:
        raise ValueError("Review must cover every keyword-positive row exactly once")
    if not keep or any(not reason.strip() for reason in excluded.values()):
        raise ValueError("Empty training subset or missing exclusion reasons")
    selected = [row for position, row in enumerate(rows) if position in keep]
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "source": str(args.source.resolve()), "source_sha256": source_hash,
        "responses": str(args.responses.resolve()), "review": str(args.review.resolve()),
        "source_count": len(rows), "keyword_positive_count": len(positives),
        "reviewed_refusal_count": len(selected), "excluded_keyword_positive_count": len(excluded),
        "subset": str(args.output.resolve()), "subset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "selected_source_indices": [row["index"] for row in selected],
        "target": "Original alpaca_reference; model-generated refusals are used only for selection",
        "selection_limitation": "Manual audit of keyword positives only; keyword-negative indirect refusals may be missed",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
