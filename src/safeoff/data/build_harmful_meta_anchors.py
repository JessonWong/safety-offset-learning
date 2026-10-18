#!/usr/bin/env python3
"""Create private harmful meta/validation anchors without benchmark overlap."""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = project_root()

from safeoff.evaluation.benchmarks import load_harmbench, load_sorrybench


REFUSAL_PREFIXES = (
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "sorry",
    "as an ai",
    "as a language model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-field", default="rejected")
    parser.add_argument(
        "--safe-target-field",
        default=None,
        help=(
            "Optional field containing the safe/refusal response for the same "
            "harmful prompt. Paired outputs include harmful_target and "
            "refusal_target fields."
        ),
    )
    parser.add_argument(
        "--require-safe-target-refusal",
        action="store_true",
        help="Keep only safe targets beginning with a common refusal prefix.",
    )
    parser.add_argument("--meta-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--meta-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--minimum-target-chars", type=int, default=200)
    parser.add_argument("--maximum-target-chars", type=int, default=1200)
    parser.add_argument("--minimum-source-index", type=int, default=1000)
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def write_jsonl(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.require_safe_target_refusal and not args.safe_target_field:
        raise ValueError(
            "--require-safe-target-refusal requires --safe-target-field"
        )
    dataset = load_dataset(args.dataset, split=args.split)

    benchmark_prompts = {normalize(row["prompt"]) for row in load_harmbench(split="test")}
    benchmark_prompts.update(normalize(row["prompt"]) for row in load_harmbench(split="val"))
    benchmark_prompts.update(normalize(row["prompt"]) for row in load_sorrybench())

    eligible = []
    rejection_stats = {
        "before_minimum_index": 0,
        "target_length": 0,
        "refusal_prefix": 0,
        "benchmark_exact_overlap": 0,
        "empty": 0,
        "safe_target_empty": 0,
        "safe_target_not_refusal": 0,
        "identical_paired_targets": 0,
    }
    for index, row in enumerate(dataset):
        if index < args.minimum_source_index:
            rejection_stats["before_minimum_index"] += 1
            continue
        prompt = str(row.get("prompt", "")).strip()
        target = str(row.get(args.target_field, "")).strip()
        if not prompt or not target:
            rejection_stats["empty"] += 1
            continue
        if not args.minimum_target_chars <= len(target) <= args.maximum_target_chars:
            rejection_stats["target_length"] += 1
            continue
        if normalize(target).startswith(REFUSAL_PREFIXES):
            rejection_stats["refusal_prefix"] += 1
            continue
        if normalize(prompt) in benchmark_prompts:
            rejection_stats["benchmark_exact_overlap"] += 1
            continue
        safe_target = ""
        if args.safe_target_field:
            safe_target = str(row.get(args.safe_target_field, "")).strip()
            if not safe_target:
                rejection_stats["safe_target_empty"] += 1
                continue
            if normalize(safe_target) == normalize(target):
                rejection_stats["identical_paired_targets"] += 1
                continue
            if (
                args.require_safe_target_refusal
                and not normalize(safe_target).startswith(REFUSAL_PREFIXES)
            ):
                rejection_stats["safe_target_not_refusal"] += 1
                continue
        eligible.append((index, prompt, target, safe_target))

    required = args.meta_count + args.validation_count
    if len(eligible) < required:
        raise RuntimeError(f"Only {len(eligible)} eligible anchors for requested {required}")
    random.Random(args.seed).shuffle(eligible)
    chosen = eligible[:required]

    output_rows = []
    for source_index, prompt, target, safe_target in chosen:
        digest_text = f"{source_index}\0{prompt}\0{target}"
        if args.safe_target_field:
            digest_text += f"\0{safe_target}"
        digest = hashlib.sha256(
            digest_text.encode("utf-8")
        ).hexdigest()
        output_row = {
            "anchor_id": f"latharm-{source_index}-{digest[:12]}",
            "source_index": source_index,
            "prompt": prompt,
            # Keep the legacy one-sided target field for existing consumers.
            "target": target,
        }
        if args.safe_target_field:
            output_row.update(
                {
                    "harmful_target": target,
                    "refusal_target": safe_target,
                }
            )
        output_rows.append(output_row)
    meta_rows = output_rows[: args.meta_count]
    validation_rows = output_rows[args.meta_count :]
    meta_path = Path(args.meta_output).expanduser().resolve()
    validation_path = Path(args.validation_output).expanduser().resolve()
    write_jsonl(meta_path, meta_rows)
    write_jsonl(validation_path, validation_rows)

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "target_field": args.target_field,
        "safe_target_field": args.safe_target_field,
        "require_safe_target_refusal": args.require_safe_target_refusal,
        "paired_targets": bool(args.safe_target_field),
        "seed": args.seed,
        "meta_output": str(meta_path),
        "validation_output": str(validation_path),
        "meta_count": len(meta_rows),
        "validation_count": len(validation_rows),
        "source_indices": {
            "meta": [row["source_index"] for row in meta_rows],
            "validation": [row["source_index"] for row in validation_rows],
        },
        "anchor_hashes": {
            "meta": [row["anchor_id"] for row in meta_rows],
            "validation": [row["anchor_id"] for row in validation_rows],
        },
        "benchmark_prompt_count": len(benchmark_prompts),
        "rejection_stats": rejection_stats,
    }
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
