#!/usr/bin/env python3
"""Judge saved rollback generations with the existing 1--5 harmfulness rubric.

One evaluator invocation per unique prompt/response/settings per invocation.
The established Gemini transport may internally retry transient HTTP failures.
Successful judgments and explicit provider blocks are reused across all arms.
"""
from __future__ import annotations
from safeoff.paths import PACKAGE_ROOT, project_root

import argparse
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from safeoff.evaluation.harmscore import atomic_json, digest_json, load_literal_env, provider_blocked, resolve_judge_name, valid_score

ROOT = project_root()


def read_arms(run_dir, directory_names, probe_file, expected_count):
    payload = json.loads(probe_file.read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    probes = [row for row in records if row["group"] == "harmful"]
    identifiers = [row["id"] for row in probes]
    if len(probes) != expected_count or len(set(identifiers)) != expected_count:
        raise ValueError("Harmful probe IDs/count do not match --expected-count")
    prompts = {row["id"]: row["prompt"] for row in probes}
    arms = {}
    for directory_name in directory_names:
        directory = run_dir / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*__*__lambda_*.json")):
            raw = path.read_bytes()
            condition = json.loads(raw)
            generations = condition.get("generations", [])
            if not generations:
                continue
            target, name, strength = condition["target"], condition["condition"], condition["lambda"]
            if not isinstance(strength, (int, float)) or not math.isfinite(strength):
                raise ValueError("Nonfinite rollback strength in generation artifact")
            if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in (target, name)):
                raise ValueError("Unsafe target/condition label in generation artifact")
            label = f"{target}__{name}__lambda_{strength:g}"
            by_id = {row["id"]: row for row in generations}
            if len(generations) != expected_count or set(by_id) != set(identifiers):
                raise ValueError(f"{label}: generation IDs differ from the canonical harmful probe set")
            ordered = []
            for index, identifier in enumerate(identifiers):
                row = by_id[identifier]
                if row.get("prompt") != prompts[identifier] or not isinstance(row.get("response"), str):
                    raise ValueError(f"{label}: mismatched prompt or nonstring response")
                ordered.append({"id": identifier, "source_index": index,
                                "prompt": row["prompt"], "response": row["response"]})
            source = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
            if label in arms:
                if arms[label]["records"] != ordered:
                    raise ValueError(f"{label}: duplicate condition contains different generations")
                arms[label]["sources"].append(source)
                continue
            arms[label] = {"label": label, "target": target, "condition": name, "lambda": strength,
                           "sources": [source], "records": ordered}
    return list(arms.values()), probes


def item_identity(record, judge_model, evaluator_sha256, max_output_tokens):
    # Endpoints can themselves carry credentials. Only their hash is persisted.
    settings_hash = digest_json({
        "gemini_api_base": os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
        "gemini_safety_threshold": os.getenv("GEMINI_SAFETY_THRESHOLD", "BLOCK_NONE"),
    })
    return {"prompt": record["prompt"], "response": record["response"], "judge_model": judge_model,
            "evaluator_sha256": evaluator_sha256, "max_output_tokens": max_output_tokens,
            "transport_settings_sha256": settings_hash}


def prepare_items(arms, cache_dir, judge_model, evaluator_sha256, max_output_tokens):
    items = {}
    for arm in arms:
        for record in arm["records"]:
            identity = item_identity(record, judge_model, evaluator_sha256, max_output_tokens)
            key = digest_json(identity)
            record["item_hash"] = key
            if key in items:
                continue
            cached = None
            path = cache_dir / f"{key}.json"
            if path.exists():
                try:
                    candidate = json.loads(path.read_text())
                    if candidate.get("item_hash") == key and (
                        valid_score(candidate.get("harmful_score")) or candidate.get("provider_blocked") is True
                    ):
                        cached = candidate
                except (OSError, json.JSONDecodeError, AttributeError):
                    pass
            items[key] = {"identity": identity, "path": path, "result": cached}
    return items


def judge_item(key, item, evaluator):
    score, explanation, error = None, "", None
    try:
        score, explanation = evaluator(item["identity"]["prompt"], item["identity"]["response"],
                                       model=item["identity"]["judge_model"])
        if not valid_score(score):
            score = None
            error = "Judge did not return an integer score from 1 to 5."
    except Exception as exception:
        # Never persist/print exception messages, request URLs, headers, or keys.
        error = f"Judge call failed ({type(exception).__name__}); retryable in a later invocation."
    blocked = provider_blocked({"harmful_explanation": explanation})
    if blocked:
        score = None
        error = "Provider policy block; this cached result will not be retried."
    return {"item_hash": key, **item["identity"], "harmful_score": score,
            "explanation": explanation, "error": error, "provider_blocked": blocked,
            "retryable": score is None and not blocked,
            "judged_at_utc": datetime.now(timezone.utc).isoformat()}


def run_judgments(items, evaluator, workers):
    pending = [(key, item) for key, item in items.items() if item["result"] is None]
    attempted = 0
    if not pending:
        return {"attempted_unique": 0, "systemic_initial_failure": False}

    def save(key, result):
        nonlocal attempted
        items[key]["result"] = result
        atomic_json(items[key]["path"], result)
        attempted += 1
        if attempted % 25 == 0 or attempted == len(pending):
            print(f"Saved {attempted}/{len(pending)} unique judgments", flush=True)

    # Check an initial worker-sized wave before queueing a large invalid-key run.
    first, remaining = pending[:workers], pending[workers:]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge_item, key, item, evaluator): key for key, item in first}
        for future in as_completed(futures):
            save(futures[future], future.result())
        initial_results = [items[key]["result"] for key, _ in first]
        systemic = all(result["retryable"] and str(result["error"]).startswith("Judge call failed")
                       for result in initial_results)
        if not systemic:
            futures = {executor.submit(judge_item, key, item, evaluator): key for key, item in remaining}
            for future in as_completed(futures):
                save(futures[future], future.result())
    return {"attempted_unique": attempted, "systemic_initial_failure": systemic}


def write_results(output_dir, arms, items, judge_model, evaluator_sha256, invocation):
    summaries, score_maps = [], {}
    for arm in arms:
        rows = []
        for record in arm["records"]:
            cached = items[record["item_hash"]]["result"]
            if cached is None:
                cached = {"harmful_score": None, "explanation": "",
                          "error": "Not attempted after systemic initial judge failures.",
                          "provider_blocked": False, "retryable": True}
            rows.append({**record, "harmful_score": cached["harmful_score"],
                         "explanation": cached["explanation"], "error": cached["error"],
                         "provider_blocked": cached["provider_blocked"], "retryable": cached["retryable"],
                         "judge_model": judge_model, "evaluator_sha256": evaluator_sha256,
                         "judged_at_utc": cached.get("judged_at_utc")})
        scores = {row["id"]: row["harmful_score"] for row in rows if valid_score(row["harmful_score"])}
        score_maps[arm["label"]] = scores
        count = len(rows)
        blocked = sum(row["provider_blocked"] for row in rows)
        summary = {"label": arm["label"], "target": arm["target"], "condition": arm["condition"],
                   "lambda": arm["lambda"], "sample_count": count, "valid_count": len(scores),
                   "coverage": len(scores) / count, "provider_blocked_count": blocked,
                   "retryable_failed_count": count - len(scores) - blocked,
                   "full_set_harmscore": sum(scores.values()) / count if len(scores) == count else None,
                   "observed_valid_mean": sum(scores.values()) / len(scores) if scores else None,
                   "complete": len(scores) == count, "sources": arm["sources"]}
        atomic_json(output_dir / arm["label"] / "scored.json", rows)
        atomic_json(output_dir / arm["label"] / "summary.json", summary)
        summaries.append(summary)
    identifiers = [row["id"] for row in arms[0]["records"]]
    common = set(identifiers)
    for scores in score_maps.values():
        common.intersection_update(scores)
    included = [identifier for identifier in identifiers if identifier in common]
    common_means = {label: sum(scores[identifier] for identifier in included) / len(included)
                    if included else None for label, scores in score_maps.items()}
    summary = {"judge_model": judge_model, "evaluator_sha256": evaluator_sha256,
               "metric": "Mean response harmfulness on the existing 1-5 rubric; higher is worse",
               "complete": all(row["complete"] for row in summaries), "results": summaries,
               "common_valid_comparison": {"scope": "paired_common_valid_subset_across_current_arms",
                    "total_count": len(identifiers), "sample_count": len(included),
                    "coverage": len(included) / len(identifiers), "included_ids": included,
                    "excluded_ids": [identifier for identifier in identifiers if identifier not in common],
                    "means": common_means},
               "incomplete_policy": "No zero imputation; full-set means are null unless all prompts are scored. Observed-valid means may compare different subsets. Common-valid means use the same subset across all currently available arms.",
               "retry_policy": "One evaluator invocation per unique uncached or retryable item per scorer invocation; native transport may retry transient HTTP failures. Explicit provider blocks are cached and never retried.",
               "unique_item_count": len(items), "generation_count": sum(len(arm["records"]) for arm in arms),
               "invocation": invocation}
    atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/functional_early_rollback_20260924")
    parser.add_argument("--input-dirs", nargs="+", default=["rollback_safeoff", "rollback_lat"])
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument("--expected-count", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--judge-model", default="gemini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=256)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; no API calls or output writes")
    args = parser.parse_args(argv)
    if min(args.workers, args.expected_count, args.gemini_max_output_tokens) < 1:
        parser.error("workers, expected-count and output-token limit must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    probe_file = args.probe_file or run_dir / "probes_screened.json"
    output_dir = args.output_dir or run_dir / "harmscore"
    cache_dir = args.cache_dir or output_dir / "cache"
    arms, _ = read_arms(run_dir, args.input_dirs, probe_file, args.expected_count)
    load_literal_env(args.env_file)
    if not args.dry_run and arms:

        # The established evaluator also loads the repository .env. Resolve and
        # hash effective settings only after that import, without logging them.
        from safeoff.evaluation.metrics import _is_gemini_judge_model, evaluate_harmfulness
    judge_model = resolve_judge_name(args.judge_model)
    evaluator_sha256 = hashlib.sha256((PACKAGE_ROOT / "evaluation/metrics.py").read_bytes()).hexdigest()
    items = prepare_items(arms, cache_dir, judge_model, evaluator_sha256, args.gemini_max_output_tokens)
    pending = sum(item["result"] is None for item in items.values())
    print(f"Validated {len(arms)} arms, {len(items)} unique prompt/response pairs; cached={len(items)-pending}, pending={pending}", flush=True)
    if args.dry_run:
        print("Dry run: no API calls or output writes.", flush=True)
        return 0
    if not arms:
        raise ValueError("No completed generation conditions are available to score")
    if _is_gemini_judge_model(judge_model):
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise RuntimeError("Gemini judge credentials are absent")
    elif not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI judge credentials are absent")
    evaluator = partial(evaluate_harmfulness, gemini_max_output_tokens=args.gemini_max_output_tokens)
    invocation = run_judgments(items, evaluator, args.workers)
    summary = write_results(output_dir, arms, items, judge_model, evaluator_sha256, invocation)
    valid = sum(row["valid_count"] for row in summary["results"])
    print(f"Saved {len(arms)} arm summaries; valid={valid}/{summary['generation_count']}; common_valid={summary['common_valid_comparison']['sample_count']}/{args.expected_count}", flush=True)
    if invocation["systemic_initial_failure"] or valid == 0:
        print("Scoring failed to establish valid judgments; inspect sanitized error fields and coverage in harmscore/summary.json.", file=sys.stderr)
        return 3
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exception:
        # A third-party import or request failure can contain credential URLs.
        print(f"Scoring stopped ({type(exception).__name__}); no exception message or request URL is logged.", file=sys.stderr)
        raise SystemExit(1)
