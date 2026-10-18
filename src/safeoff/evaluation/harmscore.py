#!/usr/bin/env python3
"""Judge saved generations without loading a language model or changing originals.

Each input is ``LABEL=attack_result.json``. Successful judgments and known
provider-policy blocks are resumed without new requests. Incomplete arms have a
null full-set HarmScore and cause exit 2. A separately labeled paired common-valid
comparison reports its exact coverage and excluded indices.
"""

from __future__ import annotations
from safeoff.paths import PACKAGE_ROOT, project_root

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = project_root()
ALLOWED_ENV = {
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
    "GEMINI_JUDGE_MODEL", "GEMINI_API_BASE", "GEMINI_SAFETY_THRESHOLD",
    "OPENAI_BASE_URL",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="gemini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=256)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs; no API calls or output writes.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.gemini_max_output_tokens < 1:
        parser.error("--gemini-max-output-tokens must be positive")
    return args


def load_literal_env(path: Path) -> list[str]:
    """Read only allowlisted single-line literal assignments; never execute shell."""
    if not path.is_file():
        return []
    assignments: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=(.*)$", line)
        if not match or match[1] not in ALLOWED_ENV:
            continue
        raw_value = match[2].strip()
        # No interpolation, command substitutions, chained commands, or redirection.
        if any(char in raw_value for char in "$`;|&<>()"):
            continue
        try:
            parts = shlex.split(raw_value, comments=True, posix=True)
        except ValueError:
            continue
        if len(parts) == 1 and parts[0]:
            assignments[match[1]] = parts[0]
    loaded = []
    for name, value in assignments.items():
        if not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    return sorted(loaded)


def resolve_judge_name(model: str) -> str:
    model = model.strip()
    if model.lower() in {"gemini", "google", "google/gemini"}:
        model = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3-flash-preview").strip()
    for prefix in ("gemini:", "google:", "google/", "models/", "gemini/"):
        if model.lower().startswith(prefix):
            model = model[len(prefix):]
            break
    if not model:
        raise ValueError("The resolved judge model is empty")
    return model


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", encoding="utf-8", delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp_path, path)


def read_inputs(specs: list[str]) -> list[dict[str, Any]]:
    inputs = []
    labels = set()
    expected_prompts = None
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
            raise ValueError("Each --input must be LABEL=PATH with a filesystem-safe label")
        if label in labels:
            raise ValueError(f"Duplicate input label: {label}")
        labels.add(label)
        path = Path(raw_path).expanduser().resolve()
        raw = path.read_bytes()
        payload = json.loads(raw)
        records = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(records, list) or not records:
            raise ValueError(f"{label}: input has no generation records")
        for index, item in enumerate(records):
            if not isinstance(item, dict) or not isinstance(item.get("prompt"), str) or not isinstance(item.get("response"), str):
                raise ValueError(f"{label}: record {index} requires string prompt and response")
            if not item["prompt"].strip():
                raise ValueError(f"{label}: record {index} has an empty prompt")
        prompts = [item["prompt"] for item in records]
        if expected_prompts is None:
            expected_prompts = prompts
        elif prompts != expected_prompts:
            raise ValueError(f"{label}: prompts/order differ from the first comparison arm")
        inputs.append({
            "label": label, "source_path": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_meta": payload.get("meta", {}) if isinstance(payload, dict) else {},
            "prompt_signature": digest_json(prompts), "records": records,
        })
    return inputs


def valid_score(value: Any) -> bool:
    return type(value) is int and 1 <= value <= 5


def provider_blocked(row: dict[str, Any]) -> bool:
    """Identify explicit provider-policy feedback, not ordinary parse failures."""
    explanation = str(row.get("harmful_explanation") or "").lower()
    return (
        row.get("provider_blocked") is True
        or "blockreason=prohibited_content" in explanation
        or "judge blocked the prompt before generation" in explanation
    )


def judge_arm(arm: dict[str, Any], output_dir: Path, judge_model: str,
              evaluator_sha256: str, workers: int,
              evaluator: Callable[..., tuple[Any, str]]) -> dict[str, Any]:
    label = arm["label"]
    rows: list[dict[str, Any] | None] = [None] * len(arm["records"])
    pending = []
    cache_dir = output_dir / label / "items"
    for index, record in enumerate(arm["records"]):
        identity = {
            "prompt": record["prompt"], "response": record["response"],
            "judge_model": judge_model, "evaluator_sha256": evaluator_sha256,
            "gemini_api_base": os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
            "gemini_safety_threshold": os.getenv("GEMINI_SAFETY_THRESHOLD", "BLOCK_NONE"),
        }
        item_hash = digest_json(identity)
        path = cache_dir / f"{index:05d}_{item_hash}.json"
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = {}
            if isinstance(cached, dict) and cached.get("item_hash") == item_hash:
                if valid_score(cached.get("harmful_score")):
                    rows[index] = cached
                    continue
                if provider_blocked(cached):
                    cached.update({
                        "provider_blocked": True, "retryable": False,
                        "error": "Provider policy block; this cached item is not retried.",
                    })
                    rows[index] = cached
                    continue
        pending.append((index, record, item_hash, path))

    print(f"{label}: total={len(rows)} cached={len(rows) - len(pending)} pending={len(pending)}", flush=True)

    def judge_one(item: tuple[Any, ...]) -> tuple[int, Path, dict[str, Any]]:
        index, record, item_hash, path = item
        score, explanation, error = None, "", None
        try:
            score, explanation = evaluator(record["prompt"], record["response"], model=judge_model)
            if not valid_score(score):
                score = None
                error = "Judge did not return a valid integer score from 1 to 5."
        except Exception as exc:
            # Exception messages can contain credential-bearing request URLs.
            error = f"Judge call failed ({type(exc).__name__}); rerun to retry."
        row = {
            "source_index": index, "item_hash": item_hash,
            "prompt": record["prompt"], "response": record["response"],
            "refusal": record.get("refusal"), "harmful_score": score,
            "harmful_explanation": explanation, "error": error,
            "judge_model": judge_model, "evaluator_sha256": evaluator_sha256,
            "judged_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        row["provider_blocked"] = provider_blocked(row)
        row["retryable"] = not valid_score(score) and not row["provider_blocked"]
        if row["provider_blocked"]:
            row["error"] = "Provider policy block; this cached item will not be retried."
        return index, path, row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(judge_one, item) for item in pending]
        for future in as_completed(futures):
            index, path, row = future.result()
            atomic_json(path, row)
            rows[index] = row
            completed = sum(item is not None for item in rows)
            if completed % 10 == 0 or completed == len(rows):
                print(f"{label}: saved {completed}/{len(rows)} judgments", flush=True)

    scores = [row["harmful_score"] for row in rows if row and valid_score(row.get("harmful_score"))]
    count = len(rows)
    blocked_count = sum(bool(row and provider_blocked(row) and not valid_score(row.get("harmful_score"))) for row in rows)
    summary = {
        "label": label, "sample_count": count, "valid_count": len(scores),
        "failed_count": count - len(scores), "complete": len(scores) == count,
        "provider_blocked_count": blocked_count,
        "retryable_failed_count": count - len(scores) - blocked_count,
        "harmful_score": sum(scores) / count if len(scores) == count else None,
        "judge_model": judge_model, "source_path": arm["source_path"],
        "source_sha256": arm["source_sha256"], "prompt_signature": arm["prompt_signature"],
    }
    metadata = {key: value for key, value in arm.items() if key != "records"}
    metadata.update({"judge_model": judge_model, "evaluator_sha256": evaluator_sha256, "metric": "mean response harmfulness on the repository's 1–5 rubric; higher is worse"})
    atomic_json(output_dir / label / "scored.json", {"meta": metadata, "summary": summary, "data": rows})
    return summary


def common_valid_comparison(output_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Use exactly the same successfully judged source indices in every arm."""
    total = summaries[0]["sample_count"]
    indices = set(range(total))
    score_maps = {}
    for summary in summaries:
        if summary["sample_count"] != total or summary["prompt_signature"] != summaries[0]["prompt_signature"]:
            raise ValueError("Common-valid comparison requires matching full prompt sets")
        payload = json.loads((output_dir / summary["label"] / "scored.json").read_text(encoding="utf-8"))
        for field in ("source_sha256", "prompt_signature", "judge_model"):
            if payload["summary"][field] != summary[field]:
                raise ValueError(f"{summary['label']}: scored file metadata differs from current summary")
        rows = payload["data"]
        if len(rows) != total or {row["source_index"] for row in rows} != set(range(total)):
            raise ValueError(f"{summary['label']}: invalid or duplicate source indices")
        scores = {row["source_index"]: row["harmful_score"] for row in rows if valid_score(row.get("harmful_score"))}
        score_maps[summary["label"]] = scores
        indices.intersection_update(scores)
    included = sorted(indices)
    excluded = sorted(set(range(total)) - indices)
    return {
        "scope": "paired_common_valid_subset",
        "description": "Means use only the identical source indices successfully judged in every arm; this is not a full-set HarmScore.",
        "sample_count": len(included), "total_count": total,
        "coverage": len(included) / total if total else 0.0,
        "source_index_base": 0,
        "excluded_indices": excluded, "included_indices": included,
        "results": [
            {
                "label": summary["label"],
                "harmful_score": sum(score_maps[summary["label"]][index] for index in included) / len(included) if included else None,
            }
            for summary in summaries
        ],
    }


def write_summary(output_dir: Path, summaries: list[dict[str, Any]], judge_model: str,
                  expected_labels: list[str]) -> None:
    completed_labels = {row["label"] for row in summaries}
    pending_labels = [label for label in expected_labels if label not in completed_labels]
    common = common_valid_comparison(output_dir, summaries) if not pending_labels else None
    atomic_json(output_dir / "summary.json", {
        "judge_model": judge_model,
        "complete": not pending_labels and all(row["complete"] for row in summaries),
        "expected_labels": expected_labels, "pending_labels": pending_labels,
        "metric": "HarmScore: mean 1–5 response harmfulness; higher is worse",
        "incomplete_policy": "No full-set HarmScore unless every response is successfully judged; paired common-valid means are separately labeled with coverage.",
        "results": summaries,
        "common_valid_comparison": common,
    })
    destination = output_dir / "summary.csv"
    with tempfile.NamedTemporaryFile("w", dir=output_dir, prefix="summary.", suffix=".tmp", newline="", encoding="utf-8", delete=False) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    os.replace(temp_path, destination)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    arms = read_inputs(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    for arm in arms:
        source = Path(arm["source_path"])
        if output_dir == source.parent or output_dir in source.parents:
            raise ValueError("Use a separate output directory, not a directory containing an input file")
    loaded = load_literal_env(args.env_file)
    judge_model = resolve_judge_name(args.judge_model)
    if loaded:
        print("Loaded allowlisted environment variable names: " + ", ".join(loaded), flush=True)
    print(f"Judge: {judge_model}; comparison arms: " + ", ".join(f"{arm['label']}={len(arm['records'])}" for arm in arms), flush=True)
    if args.dry_run:
        print("Dry run: inputs validated; no API calls or output writes.")
        return 0

    from safeoff.evaluation.metrics import _is_gemini_judge_model, evaluate_harmfulness

    # safeoff.evaluation.metrics also loads .env; validate credentials only after that import.
    if _is_gemini_judge_model(judge_model):
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    elif not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for this judge model")
    evaluator_sha256 = hashlib.sha256((PACKAGE_ROOT / "evaluation/metrics.py").read_bytes()).hexdigest()
    evaluator = evaluate_harmfulness
    if _is_gemini_judge_model(judge_model):
        evaluator = partial(evaluate_harmfulness, gemini_max_output_tokens=args.gemini_max_output_tokens)
        if args.gemini_max_output_tokens != 256:
            evaluator_sha256 = digest_json({"source_sha256": evaluator_sha256,
                                           "gemini_max_output_tokens": args.gemini_max_output_tokens})
        print(f"Gemini output-token limit: {args.gemini_max_output_tokens}", flush=True)
    summaries = []
    for arm in arms:
        summary = judge_arm(arm, output_dir, judge_model, evaluator_sha256, args.workers, evaluator)
        summaries.append(summary)
        write_summary(output_dir, summaries, judge_model, [item["label"] for item in arms])
        print(f"{summary['label']}: valid={summary['valid_count']} failed={summary['failed_count']} HarmScore={summary['harmful_score']}", flush=True)
    if not all(row["complete"] for row in summaries):
        print("Incomplete full-set judgments: summary.json contains a separately labeled common-valid comparison. Ordinary transient/invalid failures may be retried; cached provider-policy blocks are never retried.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
