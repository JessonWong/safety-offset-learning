#!/usr/bin/env python3
"""Filter a dataset CSV using evaluate_data_harmfulness.py outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "harmful", "unsafe"}:
        return True
    if text in {"false", "0", "no", "unharmful", "safe", ""}:
        return False
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop rows whose corresponding harmfulness result marks the prompt "
            "or response as harmful."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--harmfulness-results", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--drop-prompt-harmful",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--drop-response-harmful",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--drop-parse-failures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Conservatively drop rows with missing/unparseable prompt or response labels.",
    )
    parser.add_argument(
        "--manifest-json",
        default=None,
        help="Defaults to OUTPUT_CSV stem + _manifest.json.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON copy of the kept rows. Defaults to OUTPUT_CSV with .json suffix.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def harmful_record_indices(
    path: Path,
    drop_prompt_harmful: bool,
    drop_response_harmful: bool,
    drop_parse_failures: bool = False,
) -> tuple[set[int], dict[str, int]]:
    rows, _ = load_rows(path)
    harmful_indices: set[int] = set()
    stats = {
        "items": len(rows),
        "prompt_harmful": 0,
        "response_harmful": 0,
        "parse_failures": 0,
    }
    for row in rows:
        try:
            record_index = int(row["record_index"])
        except (KeyError, TypeError, ValueError):
            stats["parse_failures"] += 1
            continue
        raw_prompt_harmful = row.get("prompt_harmful")
        raw_response_harmful = row.get("response_harmful")
        prompt_harmful = (
            None
            if raw_prompt_harmful is None or not str(raw_prompt_harmful).strip()
            else parse_bool(raw_prompt_harmful)
        )
        response_harmful = (
            None
            if raw_response_harmful is None or not str(raw_response_harmful).strip()
            else parse_bool(raw_response_harmful)
        )
        if prompt_harmful is None or response_harmful is None:
            stats["parse_failures"] += 1
        if prompt_harmful is True:
            stats["prompt_harmful"] += 1
        if response_harmful is True:
            stats["response_harmful"] += 1
        if (
            (drop_prompt_harmful and prompt_harmful is True)
            or (drop_response_harmful and response_harmful is True)
            or (drop_parse_failures and (prompt_harmful is None or response_harmful is None))
        ):
            harmful_indices.add(record_index)
    return harmful_indices, stats


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    harmfulness_results = Path(args.harmfulness_results).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    manifest_json = (
        Path(args.manifest_json).expanduser().resolve()
        if args.manifest_json
        else output_csv.with_name(output_csv.stem + "_manifest.json")
    )
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else output_csv.with_suffix(".json")
    )

    rows, fieldnames = load_rows(input_csv)
    bad_indices, judge_stats = harmful_record_indices(
        harmfulness_results,
        drop_prompt_harmful=args.drop_prompt_harmful,
        drop_response_harmful=args.drop_response_harmful,
        drop_parse_failures=args.drop_parse_failures,
    )
    kept_rows = [
        row for index, row in enumerate(rows)
        if index not in bad_indices
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(kept_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    manifest = {
        "source_path": str(input_csv),
        "harmfulness_results_path": str(harmfulness_results),
        "filter": {
            "drop_prompt_harmful": args.drop_prompt_harmful,
            "drop_response_harmful": args.drop_response_harmful,
            "drop_parse_failures": args.drop_parse_failures,
        },
        "original_rows": len(rows),
        "kept_rows": len(kept_rows),
        "removed_rows": len(bad_indices),
        "removed_record_indices": sorted(bad_indices),
        "judge_stats": judge_stats,
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
