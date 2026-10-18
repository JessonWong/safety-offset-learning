#!/usr/bin/env python3
"""Judge actual standalone-v3 responses and build benign preference triples.

The synthetic indirect targets guide prompt search only.  The rejected side of
each preference pair is always the target model's actual response to the
replacement prompt, and the chosen side remains the original same-row benign
Alpaca answer.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = project_root()


SCHEMA_VERSION = 2
RUBRIC_VERSION = "actual_replacement_indirect_v1"
EXPECTED_LABEL = "indirect_rejection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate actual replacement responses with the dedicated indirect-"
            "rejection judge and emit strict prompt/chosen/rejected triples."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--judge-cache", type=Path, default=None)
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--judge-batch-size", type=int, default=64)
    parser.add_argument(
        "--min-loss-improvement",
        type=float,
        default=0.0,
        help="Minimum baseline_tf_loss - replacement_tf_loss required.",
    )
    parser.add_argument(
        "--max-chosen-harmful-info-score",
        type=float,
        default=2.0,
        help=(
            "Maximum 0-5 harmful-information score allowed for the original "
            "same-row chosen answer. Scores above 2 may contain actionable content."
        ),
    )
    parser.add_argument(
        "--require-standalone-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.judge_workers <= 0 or args.judge_batch_size <= 0:
        parser.error("judge worker and batch counts must be positive")
    if not 0 <= args.max_chosen_harmful_info_score <= 5:
        parser.error("--max-chosen-harmful-info-score must be in [0, 5]")
    if str(args.judge_model).strip().casefold() == "none":
        parser.error("a real --judge-model is required for preference-pair output")
    if args.audit_output is None:
        args.audit_output = args.output.with_name(f"{args.output.stem}.audit.csv")
    if args.manifest_output is None:
        args.manifest_output = args.output.with_suffix(".manifest.json")
    if args.judge_cache is None:
        args.judge_cache = args.output.with_suffix(".judge_cache.json")
    return args


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def parse_bool(value: Any) -> bool:
    return clean_text(value).casefold() in {"1", "true", "yes", "y"}


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def row_fingerprint(row: dict[str, Any], judge_model: str) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "judge_model": judge_model,
        "instruction": clean_text(row.get("instruction")),
        "prompt": clean_text(row.get("replacement_prompt")),
        "chosen": clean_text(row.get("alpaca_reference")),
        "rejected": clean_text(row.get("replacement_response")),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input search results are empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_preference_records(path: Path, rows: list[dict[str, Any]]) -> None:
    suffix = path.suffix.casefold()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if suffix == ".json":
        atomic_write_json(path, {"pairs": rows})
        return
    if suffix not in {"", ".csv"}:
        raise ValueError("Preference output must use .csv, .json, or .jsonl")
    write_csv(path, rows, list(rows[0].keys()))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_judge_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        return {}
    entries = payload.get("entries", {})
    return entries if isinstance(entries, dict) else {}


def prefilter_row(
    row: dict[str, Any],
    *,
    min_loss_improvement: float,
    require_standalone_valid: bool,
) -> list[str]:
    from safeoff.evaluation.metrics import contains_keyword

    reasons = []
    for field in (
        "instruction",
        "replacement_prompt",
        "alpaca_reference",
        "replacement_response",
    ):
        if not clean_text(row.get(field)):
            reasons.append(f"missing_{field}")
    if require_standalone_valid and not parse_bool(
        row.get("replacement_standalone_valid")
    ):
        reasons.append("replacement_not_standalone_valid")
    rejected = clean_text(row.get("replacement_response"))
    if rejected and contains_keyword(rejected):
        reasons.append("direct_refusal")
    if rejected and rejected == clean_text(row.get("alpaca_reference")):
        reasons.append("rejected_equals_chosen")
    baseline_loss = parse_float(row.get("baseline_tf_loss"))
    replacement_loss = parse_float(row.get("replacement_tf_loss"))
    if baseline_loss is None or replacement_loss is None:
        reasons.append("missing_target_loss")
    elif baseline_loss - replacement_loss < min_loss_improvement:
        reasons.append("insufficient_loss_improvement")
    return reasons


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    judge_model: str,
    judge_workers: int,
    judge_batch_size: int,
    min_loss_improvement: float,
    max_chosen_harmful_info_score: float,
    require_standalone_valid: bool,
    judge_cache_path: Path,
) -> list[dict[str, Any]]:
    from safeoff.evaluation.metrics import evaluate_harmful_info_batch, evaluate_indirect_rejection_batch

    cache = load_judge_cache(judge_cache_path)
    audit_rows = []
    pending_indices = []
    for index, row in enumerate(rows):
        reasons = prefilter_row(
            row,
            min_loss_improvement=min_loss_improvement,
            require_standalone_valid=require_standalone_valid,
        )
        fingerprint = row_fingerprint(row, judge_model)
        audit_row = {
            **row,
            "validation_fingerprint": fingerprint,
            "loss_improvement": (
                parse_float(row.get("baseline_tf_loss"))
                - parse_float(row.get("replacement_tf_loss"))
                if parse_float(row.get("baseline_tf_loss")) is not None
                and parse_float(row.get("replacement_tf_loss")) is not None
                else ""
            ),
            "replacement_direct_refusal": "direct_refusal" in reasons,
            "replacement_indirect_rejection_label": "",
            "replacement_indirect_rejection_explanation": "",
            "chosen_harmful_info_score": "",
            "chosen_harmful_info_explanation": "",
            "preference_valid": False,
            "invalid_reasons": reasons,
        }
        cached = cache.get(fingerprint)
        cache_complete = (
            isinstance(cached, dict)
            and clean_text(cached.get("label"))
            and parse_float(cached.get("chosen_harmful_info_score")) is not None
            and clean_text(cached.get("chosen_harmful_info_explanation"))
        )
        if not reasons and cache_complete:
            audit_row["replacement_indirect_rejection_label"] = clean_text(
                cached.get("label")
            )
            audit_row["replacement_indirect_rejection_explanation"] = clean_text(
                cached.get("explanation")
            )
            audit_row["chosen_harmful_info_score"] = float(
                cached["chosen_harmful_info_score"]
            )
            audit_row["chosen_harmful_info_explanation"] = clean_text(
                cached.get("chosen_harmful_info_explanation")
            )
        elif not reasons:
            pending_indices.append(index)
        audit_rows.append(audit_row)

    for start in range(0, len(pending_indices), judge_batch_size):
        indices = pending_indices[start : start + judge_batch_size]
        judged = evaluate_indirect_rejection_batch(
            prompts=[clean_text(rows[index]["replacement_prompt"]) for index in indices],
            responses=[
                clean_text(rows[index]["replacement_response"]) for index in indices
            ],
            references=[
                clean_text(rows[index]["alpaca_reference"]) for index in indices
            ],
            judge_model=judge_model,
            workers=judge_workers,
        )
        chosen_safety = evaluate_harmful_info_batch(
            prompts=[clean_text(rows[index]["instruction"]) for index in indices],
            responses=[clean_text(rows[index]["alpaca_reference"]) for index in indices],
            judge_model=judge_model,
            workers=judge_workers,
        )
        for index, (label, explanation), (safety_score, safety_explanation) in zip(
            indices, judged, chosen_safety
        ):
            fingerprint = audit_rows[index]["validation_fingerprint"]
            cache[fingerprint] = {
                "label": label,
                "explanation": explanation,
                "chosen_harmful_info_score": safety_score,
                "chosen_harmful_info_explanation": safety_explanation,
            }
            audit_rows[index]["replacement_indirect_rejection_label"] = clean_text(
                label
            )
            audit_rows[index][
                "replacement_indirect_rejection_explanation"
            ] = clean_text(explanation)
            audit_rows[index]["chosen_harmful_info_score"] = (
                "" if safety_score is None else safety_score
            )
            audit_rows[index]["chosen_harmful_info_explanation"] = clean_text(
                safety_explanation
            )
        atomic_write_json(
            judge_cache_path,
            {"schema_version": SCHEMA_VERSION, "entries": cache},
        )

    for row in audit_rows:
        if not row["invalid_reasons"] and (
            clean_text(row["replacement_indirect_rejection_label"]).casefold()
            != EXPECTED_LABEL
        ):
            row["invalid_reasons"].append("judge_not_indirect_rejection")
        chosen_safety_score = parse_float(row.get("chosen_harmful_info_score"))
        chosen_safety_explanation = clean_text(
            row.get("chosen_harmful_info_explanation")
        )
        if not row["invalid_reasons"] and (
            chosen_safety_score is None
            or not 0 <= chosen_safety_score <= 5
            or not chosen_safety_explanation
        ):
            row["invalid_reasons"].append("chosen_safety_judge_failed")
        elif (
            not row["invalid_reasons"]
            and chosen_safety_score > max_chosen_harmful_info_score
        ):
            row["invalid_reasons"].append("chosen_contains_harmful_information")
        row["preference_valid"] = not row["invalid_reasons"]
        row["invalid_reasons"] = ";".join(row["invalid_reasons"])
    return audit_rows


def build_preference_pairs(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    seen_pair_ids = set()
    for row in audit_rows:
        if not parse_bool(row.get("preference_valid")):
            continue
        prompt = clean_text(row.get("replacement_prompt"))
        chosen = clean_text(row.get("alpaca_reference"))
        rejected = clean_text(row.get("replacement_response"))
        source_id = clean_text(row.get("pair_id") or row.get("source_index") or row.get("index"))
        digest = hashlib.sha256(
            f"{source_id}\0{prompt}\0{chosen}\0{rejected}".encode("utf-8")
        ).hexdigest()
        pair_id = f"indirect-preference-{digest[:20]}"
        if pair_id in seen_pair_ids:
            continue
        seen_pair_ids.add(pair_id)
        pairs.append(
            {
                "pair_id": pair_id,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "rejected_is_actual_response": True,
                "rejected_label": EXPECTED_LABEL,
                "direct_refusal": False,
                "same_row": True,
                "benign": True,
                "source_index": row.get("source_index", row.get("index", "")),
                "search_pair_id": row.get("pair_id", ""),
                "original_instruction": clean_text(row.get("instruction")),
                "targets_json": clean_text(row.get("refusal_targets_json")),
                "baseline_tf_loss": row.get("baseline_tf_loss", ""),
                "replacement_tf_loss": row.get("replacement_tf_loss", ""),
                "loss_improvement": row.get("loss_improvement", ""),
                "judge_model": "",
                "judge_explanation": clean_text(
                    row.get("replacement_indirect_rejection_explanation")
                ),
                "chosen_harmful_info_score": row.get(
                    "chosen_harmful_info_score", ""
                ),
                "chosen_harmful_info_explanation": clean_text(
                    row.get("chosen_harmful_info_explanation")
                ),
            }
        )
    return pairs


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    audit_rows = validate_rows(
        rows,
        judge_model=args.judge_model,
        judge_workers=args.judge_workers,
        judge_batch_size=args.judge_batch_size,
        min_loss_improvement=args.min_loss_improvement,
        max_chosen_harmful_info_score=args.max_chosen_harmful_info_score,
        require_standalone_valid=args.require_standalone_valid,
        judge_cache_path=args.judge_cache,
    )
    audit_fields = list(audit_rows[0].keys())
    write_csv(args.audit_output, audit_rows, audit_fields)
    pairs = build_preference_pairs(audit_rows)
    if not pairs:
        raise RuntimeError(
            "No valid indirect preference pairs remained after strict validation. "
            f"Inspect {args.audit_output}."
        )
    for pair in pairs:
        pair["judge_model"] = args.judge_model

    write_preference_records(args.output, pairs)
    reason_counts = Counter()
    for row in audit_rows:
        for reason in clean_text(row.get("invalid_reasons")).split(";"):
            if reason:
                reason_counts[reason] += 1
    atomic_write_json(
        args.manifest_output,
        {
            "schema_version": SCHEMA_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "audit_output": str(args.audit_output.resolve()),
            "judge_cache": str(args.judge_cache.resolve()),
            "judge_model": args.judge_model,
            "min_loss_improvement": args.min_loss_improvement,
            "max_chosen_harmful_info_score": args.max_chosen_harmful_info_score,
            "require_standalone_valid": args.require_standalone_valid,
            "input_rows": len(rows),
            "valid_pairs": len(pairs),
            "invalid_reason_counts": dict(sorted(reason_counts.items())),
        },
    )
    print(f"Saved {len(pairs)} strict preference pairs to {args.output}")
    print(f"Saved row-level audit to {args.audit_output}")


if __name__ == "__main__":
    main()
