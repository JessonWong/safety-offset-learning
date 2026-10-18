#!/usr/bin/env python3
"""Assemble aligned benign SFT candidates from several replacement datasets.

Every source is expected to contain Alpaca-compatible ``index``, ``instruction``
and ``alpaca_reference`` columns.  The output retains all distinct candidates so
that a later bilevel scorer can choose one candidate per original Alpaca row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


INVALID_PHRASES = (
    "original prompt",
    "original user prompt",
    "replacement prompt",
    "rewrite the original",
    "rewritten prompt",
    "rephrase the original",
    "the prompt above",
    "standalone replacement prompt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Candidate source; repeat for every proposal dataset.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--expected-indices", type=int, default=1000)
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"--source must use NAME=CSV syntax: {raw}")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = Path(path).expanduser().resolve()
    if not name or not path.is_file():
        raise ValueError(f"Invalid source {raw!r}; resolved path={path}")
    return name, path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def invalid_reason(instruction: str, response: str) -> str:
    instruction = normalize(instruction)
    response = normalize(response)
    if not instruction:
        return "empty_instruction"
    if not response:
        return "empty_response"
    if "<|" in instruction or "<|" in response:
        return "special_chat_token"
    lowered = instruction.lower()
    for phrase in INVALID_PHRASES:
        if phrase in lowered:
            return f"meta_phrase:{phrase}"
    return ""


def main() -> None:
    args = parse_args()
    parsed_sources = [parse_source(value) for value in args.source]
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output_path.with_name(output_path.stem + "_manifest.json")
    )

    output_rows: list[dict[str, str]] = []
    seen: set[tuple[int, str, str]] = set()
    source_stats: dict[str, dict[str, int | str]] = {}

    for source_name, source_path in parsed_sources:
        rows = read_rows(source_path)
        stats: dict[str, int | str] = {
            "path": str(source_path),
            "input_rows": len(rows),
            "kept": 0,
            "duplicates": 0,
            "invalid": 0,
        }
        for position, row in enumerate(rows):
            raw_index = str(row.get("index", position)).strip()
            try:
                source_index = int(raw_index)
            except ValueError:
                stats["invalid"] = int(stats["invalid"]) + 1
                continue
            instruction = normalize(row.get("instruction", ""))
            response = str(
                row.get("alpaca_reference")
                or row.get("output")
                or row.get("response")
                or ""
            ).strip()
            reason = invalid_reason(instruction, response)
            if reason:
                stats["invalid"] = int(stats["invalid"]) + 1
                continue
            dedupe_key = (source_index, instruction.casefold(), normalize(response).casefold())
            if dedupe_key in seen:
                stats["duplicates"] = int(stats["duplicates"]) + 1
                continue
            seen.add(dedupe_key)
            digest = hashlib.sha256(
                f"{source_name}\0{source_index}\0{instruction}\0{response}".encode("utf-8")
            ).hexdigest()[:16]
            output_rows.append(
                {
                    "candidate_id": f"{source_name}:{source_index}:{digest}",
                    "index": str(source_index),
                    "source": source_name,
                    "instruction": instruction,
                    "alpaca_reference": response,
                    "suffix": str(row.get("suffix", "") or ""),
                }
            )
            stats["kept"] = int(stats["kept"]) + 1
        source_stats[source_name] = stats

    output_rows.sort(key=lambda row: (int(row["index"]), row["source"], row["candidate_id"]))
    covered = {int(row["index"]) for row in output_rows}
    missing = sorted(set(range(args.expected_indices)) - covered)
    if missing:
        raise RuntimeError(f"Candidate pool is missing {len(missing)} expected indices: {missing[:20]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "index",
        "source",
        "instruction",
        "alpaca_reference",
        "suffix",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    per_index_counts: dict[int, int] = {}
    for row in output_rows:
        index = int(row["index"])
        per_index_counts[index] = per_index_counts.get(index, 0) + 1
    manifest = {
        "output": str(output_path),
        "rows": len(output_rows),
        "covered_indices": len(covered),
        "minimum_candidates_per_index": min(per_index_counts.values()),
        "maximum_candidates_per_index": max(per_index_counts.values()),
        "source_stats": source_stats,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
