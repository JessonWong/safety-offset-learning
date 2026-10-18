#!/usr/bin/env python3
"""Select one safe candidate per Alpaca index from one or more model scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--score", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument(
        "--allowed-candidates",
        action="append",
        default=[],
        help="Optional safety-filtered candidate CSV; only its candidate_id values may be selected.",
    )
    parser.add_argument("--score-field", default="anchor_dot")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-indices", type=int, default=1000)
    parser.add_argument(
        "--maximum-loss-quantile",
        type=float,
        default=0.95,
        help="Reject unusually high-loss/mismatched pairs when another candidate exists.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=CSV: {raw}")
    name, raw_path = raw.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name.strip() or not path.is_file():
        raise ValueError(f"Invalid score source {raw!r}; resolved path={path}")
    return name.strip(), path


def finite_float(value: str | float | None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def loss_within_limit(row: dict[str, str], limit: float) -> bool:
    value = finite_float(row.get("candidate_loss"))
    return value is not None and value <= limit


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("inf")
    position = (len(values) - 1) * min(1.0, max(0.0, q))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def write_training_csv(path: Path, selected: list[dict[str, str]]) -> None:
    fields = ["index", "instruction", "alpaca_reference", "suffix"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)


def write_audit_csv(path: Path, selected: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)


def source_summary(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["source"] for row in rows).items()))


def main() -> None:
    args = parse_args()
    named_paths = [parse_named_path(raw) for raw in args.score]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_ids = None
    if args.allowed_candidates:
        allowed_ids = set()
        for allowed_path in args.allowed_candidates:
            allowed_rows, _ = read_csv(Path(allowed_path).expanduser().resolve())
            allowed_ids.update(row["candidate_id"] for row in allowed_rows)
        if not allowed_ids:
            raise RuntimeError("The allowed-candidates files contain no candidate IDs")

    rows_by_model: dict[str, dict[str, dict[str, str]]] = {}
    fields_by_model: dict[str, list[str]] = {}
    canonical_rows: dict[str, dict[str, str]] = {}
    loss_limits: dict[str, float] = {}
    for name, path in named_paths:
        rows, fields = read_csv(path)
        good = {
            row["candidate_id"]: row
            for row in rows
            if row.get("status") == "ok"
            and finite_float(row.get(args.score_field)) is not None
            and (allowed_ids is None or row["candidate_id"] in allowed_ids)
        }
        if not good:
            raise RuntimeError(f"No valid scores in {path}")
        rows_by_model.setdefault(name, {}).update(good)
        fields_by_model.setdefault(name, [])
        fields_by_model[name].extend(field for field in fields if field not in fields_by_model[name])

    for name, good in rows_by_model.items():
        for candidate_id, row in good.items():
            canonical_rows.setdefault(candidate_id, row)
        losses = [finite_float(row.get("candidate_loss")) for row in good.values()]
        loss_limits[name] = quantile(
            [value for value in losses if value is not None], args.maximum_loss_quantile
        )

    common_ids = set.intersection(*(set(rows) for rows in rows_by_model.values()))
    by_index: dict[int, list[str]] = defaultdict(list)
    for candidate_id in common_ids:
        by_index[int(canonical_rows[candidate_id]["index"])].append(candidate_id)
    missing = sorted(set(range(args.expected_indices)) - set(by_index))
    if missing:
        raise RuntimeError(f"No jointly scored safe candidate for indices: {missing[:30]}")

    audit_fields = [
        "candidate_id",
        "index",
        "source",
        "instruction",
        "alpaca_reference",
        "suffix",
        "selection_score",
    ]
    for model_name in rows_by_model:
        audit_fields.extend(
            [
                f"{model_name}_{args.score_field}",
                f"{model_name}_anchor_cosine",
                f"{model_name}_anchor_adam_dot",
                f"{model_name}_candidate_loss",
            ]
        )

    selected_by_model: dict[str, list[dict[str, str]]] = {}
    for model_name, score_rows in rows_by_model.items():
        selected = []
        for index in range(args.expected_indices):
            candidates = by_index[index]
            loss_limit = loss_limits[model_name]
            eligible = [
                candidate_id
                for candidate_id in candidates
                if loss_within_limit(score_rows[candidate_id], loss_limit)
            ]
            if not eligible:
                eligible = candidates
            best_id = max(
                eligible,
                key=lambda candidate_id: float(score_rows[candidate_id][args.score_field]),
            )
            base = dict(canonical_rows[best_id])
            base["selection_score"] = score_rows[best_id][args.score_field]
            for other_name, other_rows in rows_by_model.items():
                other = other_rows[best_id]
                base[f"{other_name}_{args.score_field}"] = other.get(args.score_field, "")
                base[f"{other_name}_anchor_cosine"] = other.get("anchor_cosine", "")
                base[f"{other_name}_anchor_adam_dot"] = other.get("anchor_adam_dot", "")
                base[f"{other_name}_candidate_loss"] = other.get("candidate_loss", "")
            selected.append(base)
        selected_by_model[model_name] = selected

    # Robust shared selection: maximize the worst within-index z-score.
    shared = []
    for index in range(args.expected_indices):
        candidates = by_index[index]
        per_model_z: dict[str, dict[str, float]] = {}
        for model_name, score_rows in rows_by_model.items():
            values = [float(score_rows[candidate_id][args.score_field]) for candidate_id in candidates]
            mean = statistics.fmean(values)
            stdev = statistics.pstdev(values)
            denominator = stdev if stdev > 1e-20 else 1.0
            per_model_z[model_name] = {
                candidate_id: (float(score_rows[candidate_id][args.score_field]) - mean) / denominator
                for candidate_id in candidates
            }
        eligible = [
            candidate_id
            for candidate_id in candidates
            if all(
                loss_within_limit(rows_by_model[name][candidate_id], loss_limits[name])
                for name in rows_by_model
            )
        ]
        if not eligible:
            eligible = candidates
        best_id = max(
            eligible,
            key=lambda candidate_id: min(
                per_model_z[name][candidate_id] for name in rows_by_model
            ),
        )
        base = dict(canonical_rows[best_id])
        base["selection_score"] = min(
            per_model_z[name][best_id] for name in rows_by_model
        )
        for name, score_rows in rows_by_model.items():
            row = score_rows[best_id]
            base[f"{name}_{args.score_field}"] = row.get(args.score_field, "")
            base[f"{name}_anchor_cosine"] = row.get("anchor_cosine", "")
            base[f"{name}_anchor_adam_dot"] = row.get("anchor_adam_dot", "")
            base[f"{name}_candidate_loss"] = row.get("candidate_loss", "")
        shared.append(base)
    selected_by_model["shared"] = shared

    manifest = {
        "score_field": args.score_field,
        "maximum_loss_quantile": args.maximum_loss_quantile,
        "loss_limits": loss_limits,
        "jointly_scored_candidates": len(common_ids),
        "allowed_candidate_count": None if allowed_ids is None else len(allowed_ids),
        "indices": args.expected_indices,
        "outputs": {},
    }
    for name, selected in selected_by_model.items():
        train_path = output_dir / f"bilevel_{name}_train.csv"
        audit_path = output_dir / f"bilevel_{name}_audit.csv"
        write_training_csv(train_path, selected)
        write_audit_csv(audit_path, selected, audit_fields)
        manifest["outputs"][name] = {
            "training_csv": str(train_path),
            "audit_csv": str(audit_path),
            "rows": len(selected),
            "source_counts": source_summary(selected),
        }

    manifest_path = output_dir / "selection_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
