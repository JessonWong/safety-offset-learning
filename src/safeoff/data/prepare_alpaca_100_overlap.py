#!/usr/bin/env python3
"""Prepare aligned and disjoint 100-row SafeOff/Alpaca comparison arms."""
from safeoff.paths import project_root

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = project_root()
SAFE_SOURCE = ROOT / "data/8b_newtarget.csv"
ALPACA_SOURCE = ROOT / "data/alpaca_cleaned.csv"
OUTPUT = ROOT / "data/delta_similarity_100"
FIELDS = ["index", "instruction", "alpaca_reference", "suffix"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if list(rows[0]) != FIELDS:
        raise ValueError(f"Unexpected columns in {path}: {list(rows[0])}")
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(left: list[dict[str, str]], right: list[dict[str, str]]) -> dict[str, int]:
    left_prompts = Counter(row["instruction"] for row in left)
    right_prompts = Counter(row["instruction"] for row in right)
    left_answers = Counter(row["alpaca_reference"] for row in left)
    right_answers = Counter(row["alpaca_reference"] for row in right)
    left_pairs = Counter((row["instruction"], row["alpaca_reference"]) for row in left)
    right_pairs = Counter((row["instruction"], row["alpaca_reference"]) for row in right)
    return {
        "same_position_prompt": sum(a["instruction"] == b["instruction"] for a, b in zip(left, right)),
        "same_position_answer": sum(a["alpaca_reference"] == b["alpaca_reference"] for a, b in zip(left, right)),
        "prompt_multiset_overlap": sum((left_prompts & right_prompts).values()),
        "answer_multiset_overlap": sum((left_answers & right_answers).values()),
        "pair_multiset_overlap": sum((left_pairs & right_pairs).values()),
    }


def main() -> None:
    safe = read_rows(SAFE_SOURCE)
    alpaca = read_rows(ALPACA_SOURCE)
    if len(safe) != 1000 or len(alpaca) != 1000:
        raise ValueError("Expected exactly 1,000 rows in each source")
    if any(row["suffix"].strip() for row in safe + alpaca):
        raise ValueError("All source suffixes must be empty")

    arms = {
        "safeoff_0_99": safe[:100],
        "alpaca_aligned_0_99": alpaca[:100],
        "alpaca_disjoint_100_199": alpaca[100:200],
    }
    aligned = overlap(arms["safeoff_0_99"], arms["alpaca_aligned_0_99"])
    disjoint = overlap(arms["safeoff_0_99"], arms["alpaca_disjoint_100_199"])
    if aligned["same_position_answer"] != 100:
        raise ValueError(f"Aligned arms must share all answers: {aligned}")
    if aligned["same_position_prompt"] != 0:
        raise ValueError(f"Aligned prompts should be rewritten: {aligned}")
    if any(disjoint[key] for key in ("prompt_multiset_overlap", "answer_multiset_overlap", "pair_multiset_overlap")):
        raise ValueError(f"Disjoint arms unexpectedly overlap: {disjoint}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    arm_metadata = {}
    for name, rows in arms.items():
        path = OUTPUT / f"{name}.csv"
        write_rows(path, rows)
        arm_metadata[name] = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "row_count": len(rows),
            "source_indices": [row["index"] for row in rows],
        }
    manifest = {
        "definition": {
            "aligned": "SafeOff indices 0-99 versus Alpaca indices 0-99; answers match row-by-row and prompts are rewritten",
            "disjoint": "SafeOff indices 0-99 versus Alpaca indices 100-199; no exact prompt, answer, or pair overlap",
        },
        "sources": {
            "safeoff": {"path": str(SAFE_SOURCE), "sha256": digest(SAFE_SOURCE)},
            "alpaca": {"path": str(ALPACA_SOURCE), "sha256": digest(ALPACA_SOURCE)},
        },
        "arms": arm_metadata,
        "overlap_checks": {"aligned": aligned, "disjoint": disjoint},
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
