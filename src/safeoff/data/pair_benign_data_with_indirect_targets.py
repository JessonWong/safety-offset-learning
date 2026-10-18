#!/usr/bin/env python3
"""Validate and materialize same-row Alpaca indirect-rejection search input.

Targets are joined back to Alpaca by ``source_index`` and must match that exact
row's instruction, optional input, and original response. Positional pairing of
an unrelated target catalog is intentionally rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


SEARCH_INPUT_SCHEMA_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join indirect targets to their original Alpaca rows by source_index "
            "and validate same-row provenance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-field", default="target")
    parser.add_argument(
        "--target-list-field",
        default="targets_json",
        help=(
            "JSON-list field containing multiple same-row targets. If absent or "
            "empty, the legacy scalar --target-field is used."
        ),
    )
    parser.add_argument(
        "--expected-target-count",
        type=int,
        default=0,
        help="Require exactly this many targets per source row; 0 accepts any positive count.",
    )
    parser.add_argument("--target-prompt-field", default="target_prompt")
    parser.add_argument("--target-source-index-field", default="source_index")
    parser.add_argument("--target-instruction-field", default="instruction")
    parser.add_argument("--target-input-field", default="input")
    parser.add_argument("--target-output-field", default="output")
    parser.add_argument("--benign-dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--benign-split", default="train")
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--output-field", default="output")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of target rows to validate; 0 uses every target row.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    if args.num_samples < 0:
        parser.error("--num-samples must be non-negative")
    if args.expected_target_count < 0:
        parser.error("--expected-target-count must be non-negative")
    return args


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_text(value: Any) -> str:
    return " ".join(clean_text(value).casefold().split())


def required_text(row: dict[str, Any], field: str, source: str) -> str:
    text = clean_text(row.get(field))
    if not text:
        raise ValueError(f"{source} row is missing a non-empty {field!r} field.")
    return text


def parse_json_text_list(
    value: Any,
    *,
    field: str,
    source_index: int,
    require_unique: bool = True,
) -> list[str]:
    if isinstance(value, list):
        parsed = value
    else:
        raw_value = clean_text(value)
        try:
            parsed = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Target at source_index {source_index} has invalid JSON in {field!r}."
            ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(
            f"Target at source_index {source_index} must contain a non-empty JSON "
            f"list in {field!r}."
        )
    targets = []
    seen = set()
    for position, value in enumerate(parsed):
        if not isinstance(value, str) or not clean_text(value):
            raise ValueError(
                f"Target at source_index {source_index} has an empty or non-string "
                f"entry at {field}[{position}]."
            )
        target = clean_text(value)
        normalized = normalize_text(target)
        if require_unique and normalized in seen:
            raise ValueError(
                f"Target at source_index {source_index} has duplicate entries in "
                f"{field!r}."
            )
        seen.add(normalized)
        targets.append(target)
    return targets


def parse_targets(
    row: dict[str, Any],
    *,
    target_field: str,
    target_list_field: str | None,
    source_index: int,
    expected_target_count: int = 0,
) -> tuple[list[str], bool]:
    has_target_list = bool(
        target_list_field
        and target_list_field in row
        and clean_text(row.get(target_list_field))
    )
    if has_target_list:
        targets = parse_json_text_list(
            row.get(target_list_field),
            field=target_list_field,
            source_index=source_index,
        )
        scalar_target = clean_text(row.get(target_field))
        if scalar_target and normalize_text(scalar_target) != normalize_text(targets[0]):
            raise ValueError(
                f"Target at source_index {source_index} has a legacy {target_field!r} "
                "value that does not equal the first multi-target entry."
            )
    else:
        targets = [required_text(row, target_field, "target")]

    if expected_target_count and len(targets) != expected_target_count:
        raise ValueError(
            f"Target at source_index {source_index} has {len(targets)} targets; "
            f"expected {expected_target_count}."
        )
    if clean_text(row.get("target_count")):
        try:
            declared_count = int(clean_text(row.get("target_count")))
        except ValueError as exc:
            raise ValueError(
                f"Target at source_index {source_index} has invalid target_count."
            ) from exc
        if declared_count != len(targets):
            raise ValueError(
                f"Target at source_index {source_index} declares target_count="
                f"{declared_count}, but contains {len(targets)} targets."
            )
    return targets, has_target_list


def format_instruction(instruction: str, input_text: str = "") -> str:
    instruction = clean_text(instruction)
    input_text = clean_text(input_text)
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def parse_source_index(row: dict[str, Any], field: str, position: int) -> int:
    value = clean_text(row.get(field))
    try:
        source_index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Target row {position} has invalid {field!r}: {value!r}."
        ) from exc
    if source_index < 0:
        raise ValueError(f"Target row {position} has a negative source_index.")
    return source_index


def assert_same_text(
    target_value: Any,
    benign_value: Any,
    description: str,
    source_index: int,
) -> None:
    if normalize_text(target_value) != normalize_text(benign_value):
        raise ValueError(
            f"Target provenance mismatch for Alpaca source_index {source_index}: "
            f"{description} differs from the original row."
        )


def combine_rows(
    benign_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    *,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    target_field: str = "target",
    target_list_field: str | None = "targets_json",
    expected_target_count: int = 0,
    target_prompt_field: str = "target_prompt",
    target_source_index_field: str = "source_index",
    target_instruction_field: str = "instruction",
    target_input_field: str = "input",
    target_output_field: str = "output",
) -> list[dict[str, Any]]:
    if len(benign_rows) != len(target_rows):
        raise ValueError(
            f"Benign row count ({len(benign_rows)}) must match target row count "
            f"({len(target_rows)})."
        )

    combined = []
    seen_source_indices = set()
    observed_target_count = None
    for position, (benign, target_row) in enumerate(zip(benign_rows, target_rows)):
        source_index = parse_source_index(
            target_row,
            target_source_index_field,
            position,
        )
        if source_index in seen_source_indices:
            raise ValueError(f"Duplicate Alpaca source_index in targets: {source_index}.")
        seen_source_indices.add(source_index)

        instruction = required_text(benign, instruction_field, "benign")
        input_text = clean_text(benign.get(input_field)) if input_field else ""
        benign_output = required_text(benign, output_field, "benign")
        targets, has_target_list = parse_targets(
            target_row,
            target_field=target_field,
            target_list_field=target_list_field,
            source_index=source_index,
            expected_target_count=expected_target_count,
        )
        if observed_target_count is None:
            observed_target_count = len(targets)
        elif len(targets) != observed_target_count:
            raise ValueError(
                "Target rows must have fixed cardinality; source_index "
                f"{source_index} has {len(targets)} targets but prior rows have "
                f"{observed_target_count}."
            )
        target = targets[0]
        search_prompt = format_instruction(instruction, input_text)

        if normalize_text(target_row.get("indirect_rejection_label")) != (
            "indirect_rejection"
        ):
            raise ValueError(
                f"Target at source_index {source_index} was not judged as an "
                "indirect rejection."
            )
        if normalize_text(target_row.get("indirect_rejection")) not in {
            "true",
            "1",
        }:
            raise ValueError(
                f"Target at source_index {source_index} is missing the "
                "indirect_rejection=true marker."
            )
        if normalize_text(target_row.get("direct_refusal")) not in {
            "false",
            "0",
        }:
            raise ValueError(
                f"Target at source_index {source_index} is a direct refusal."
            )
        if has_target_list:
            labels_field = "target_judge_labels_json"
            if not clean_text(target_row.get(labels_field)):
                raise ValueError(
                    f"Multi-target row at source_index {source_index} is missing "
                    f"{labels_field!r}."
                )
            labels = parse_json_text_list(
                target_row.get(labels_field),
                field=labels_field,
                source_index=source_index,
                require_unique=False,
            )
            if len(labels) != len(targets) or any(
                normalize_text(label) != "indirect_rejection" for label in labels
            ):
                raise ValueError(
                    f"Not every target at source_index {source_index} was judged as "
                    "an indirect rejection."
                )

        assert_same_text(
            target_row.get(target_instruction_field),
            instruction,
            "instruction",
            source_index,
        )
        if target_input_field in target_row:
            assert_same_text(
                target_row.get(target_input_field),
                input_text,
                "input",
                source_index,
            )
        if target_output_field in target_row:
            assert_same_text(
                target_row.get(target_output_field),
                benign_output,
                "original response",
                source_index,
            )
        if target_prompt_field in target_row:
            assert_same_text(
                target_row.get(target_prompt_field),
                search_prompt,
                "formatted target prompt",
                source_index,
            )
        for target_position, candidate_target in enumerate(targets):
            if normalize_text(candidate_target) == normalize_text(benign_output):
                raise ValueError(
                    f"Indirect target {target_position} equals the original response "
                    f"at source_index {source_index}."
                )

        targets_json = json.dumps(
            targets,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(
            f"{source_index}\0{search_prompt}\0{benign_output}\0{targets_json}".encode(
                "utf-8"
            )
        ).hexdigest()
        combined.append(
            {
                "index": source_index,
                "source_index": source_index,
                "pair_id": f"alpaca-indirect-search-{source_index}-{digest[:12]}",
                "instruction": instruction,
                "input": input_text,
                "output": benign_output,
                "alpaca_reference": benign_output,
                "target": target,
                "targets_json": targets_json,
                "target_count": len(targets),
                "indirect_rejection": True,
                "direct_refusal": False,
                "indirect_rejection_label": "indirect_rejection",
                "suffix": "",
                "target_pair_id": target_row.get("pair_id", ""),
                "target_prompt": search_prompt,
                "target_response": target,
                "target_responses_json": targets_json,
                "target_judge_labels_json": json.dumps(
                    ["indirect_rejection"] * len(targets),
                    separators=(",", ":"),
                ),
                "target_judge_explanations_json": target_row.get(
                    "target_judge_explanations_json",
                    "",
                ),
                "target_candidate_indices_json": target_row.get(
                    "target_candidate_indices_json",
                    "",
                ),
            }
        )
    return combined


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty search-input CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    targets_path = args.targets.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else output_path.with_name(f"{output_path.stem}.manifest.json")
    )
    target_rows = read_csv(targets_path)
    count = args.num_samples or len(target_rows)
    if count > len(target_rows):
        raise ValueError(
            f"Requested {count} rows, but target file has only {len(target_rows)}."
        )
    target_rows = target_rows[:count]

    benign_dataset = load_dataset(args.benign_dataset, split=args.benign_split)
    source_indices = [
        parse_source_index(row, args.target_source_index_field, position)
        for position, row in enumerate(target_rows)
    ]
    if source_indices and max(source_indices) >= len(benign_dataset):
        raise ValueError(
            f"Target source_index {max(source_indices)} exceeds benign dataset size "
            f"{len(benign_dataset)}."
        )
    benign_rows = [dict(benign_dataset[index]) for index in source_indices]
    combined = combine_rows(
        benign_rows=benign_rows,
        target_rows=target_rows,
        instruction_field=args.instruction_field,
        input_field=args.input_field,
        output_field=args.output_field,
        target_field=args.target_field,
        target_list_field=args.target_list_field,
        expected_target_count=args.expected_target_count,
        target_prompt_field=args.target_prompt_field,
        target_source_index_field=args.target_source_index_field,
        target_instruction_field=args.target_instruction_field,
        target_input_field=args.target_input_field,
        target_output_field=args.target_output_field,
    )
    write_csv(output_path, combined)
    manifest = {
        "schema_version": SEARCH_INPUT_SCHEMA_VERSION,
        "role": "same_alpaca_row_standalone_v3_multi_search_input",
        "num_rows": len(combined),
        "benign_dataset": args.benign_dataset,
        "benign_split": args.benign_split,
        "join_key": args.target_source_index_field,
        "lineage_validation": ["instruction", "input", "output", "target_prompt"],
        "supervised_reference_source": (
            f"{args.benign_dataset}:{args.output_field}"
        ),
        "optimization_target_source": str(targets_path),
        "optimization_target_field": args.target_list_field or args.target_field,
        "legacy_optimization_target_field": args.target_field,
        "expected_target_count": args.expected_target_count,
        "target_count_per_row": combined[0]["target_count"],
        "output": str(output_path),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
