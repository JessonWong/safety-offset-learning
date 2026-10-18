#!/usr/bin/env python3
"""Aggregate bilevel dataset, anchor-loss, and external benchmark results."""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import json
import statistics
from pathlib import Path


EVAL_SUFFIX = "attack_harmbench-test__sorrybench-base__harmful_judge_gemini-3-flash-preview.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=project_root())
    parser.add_argument("--output", default="cache/bilevel_standalone_v1/experiment_summary.json")
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def benchmark_metrics(checkpoint_cache: Path, benchmark: str) -> dict:
    attack_path = checkpoint_cache / f"attack_result_{benchmark}_{EVAL_SUFFIX}"
    records = read_json(attack_path)["data"]
    scores = [
        row.get("harmful_score")
        for row in records
        if isinstance(row.get("harmful_score"), (int, float))
    ]
    refusals = [bool(row.get("refusal")) for row in records]
    return {
        "samples": len(records),
        "parsed_harmful_scores": len(scores),
        "attack_success_rate": 1.0 - sum(refusals) / len(refusals),
        "harmful_score": statistics.fmean(scores),
        "harmful_score_ge_3_rate": sum(score >= 3 for score in scores) / len(scores),
        "harmful_score_ge_4_rate": sum(score >= 4 for score in scores) / len(scores),
        "harmful_score_eq_5_rate": sum(score == 5 for score in scores) / len(scores),
    }


def checkpoint_metrics(checkpoint_cache: Path) -> dict:
    return {
        benchmark: benchmark_metrics(checkpoint_cache, benchmark)
        for benchmark in ("harmbench", "sorrybench")
    }


def safety_labels(root: Path) -> dict[str, dict[str, str]]:
    labels = {}
    pairs = (
        (root / "candidate_pool.csv", root / "wildguard/harmfulness_results.csv"),
        (root / "fallback_candidates.csv", root / "wildguard_fallback/harmfulness_results.csv"),
    )
    for pool_path, result_path in pairs:
        pool = read_csv(pool_path)
        judged = {int(row["record_index"]): row for row in read_csv(result_path)}
        for index, row in enumerate(pool):
            labels[row["candidate_id"]] = judged[index]
    return labels


def selected_dataset_stats(root: Path, name: str, labels) -> dict:
    rows = read_csv(root / f"selected/bilevel_{name}_audit.csv")
    judged = [labels[row["candidate_id"]] for row in rows]
    true_count = lambda field: sum(
        str(row.get(field, "")).strip().lower() == "true" for row in judged
    )
    blank_count = lambda field: sum(
        not str(row.get(field, "")).strip() for row in judged
    )
    return {
        "rows": len(rows),
        "prompt_harmful": true_count("prompt_harmful"),
        "response_harmful": true_count("response_harmful"),
        "response_refusal": true_count("response_refusal"),
        "prompt_label_parse_failures": blank_count("prompt_harmful"),
        "response_label_parse_failures": blank_count("response_harmful"),
    }


def metric_deltas(current: dict, baseline: dict) -> dict:
    return {
        benchmark: {
            key: current[benchmark][key] - baseline[benchmark][key]
            for key in (
                "attack_success_rate",
                "harmful_score",
                "harmful_score_ge_4_rate",
                "harmful_score_eq_5_rate",
            )
        }
        for benchmark in current
    }


def main() -> None:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()
    root = repo / "cache/bilevel_standalone_v1"
    eval_root = repo / "cache/evaluations/finetune_model/LATharm"
    runs = {
        "llama": {
            "model": "Llama-3.2-3B-Instruct",
            "experiment": "bilevel_standalone_v1_llama",
            "baseline_experiment": "test_advfinal",
        },
        "qwen": {
            "model": "Qwen2.5-7B-Instruct",
            "experiment": "bilevel_standalone_v1_qwen",
            "baseline_experiment": "test_advfinal_qwen_v3_harm",
        },
    }

    labels = safety_labels(root)
    summary = {
        "candidate_pool": read_json(root / "candidate_pool_manifest.json"),
        "wildguard_pool": read_json(root / "wildguard/harmfulness_summary.json")["overall"],
        "selection": read_json(root / "selected/selection_manifest.json"),
        "selected_dataset_safety": {
            name: selected_dataset_stats(root, name, labels)
            for name in ("llama", "qwen", "shared")
        },
        "models": {},
    }
    for name, config in runs.items():
        model = config["model"]
        experiment_cache = eval_root / config["experiment"] / model / "model"
        baseline_cache = eval_root / config["baseline_experiment"] / model / "model/checkpoint-1000"
        checkpoints = {
            str(step): checkpoint_metrics(experiment_cache / f"checkpoint-{step}")
            for step in (50, 1000)
        }
        baseline = checkpoint_metrics(baseline_cache)
        validation = read_json(root / f"{name}_validation_anchor_loss.json")
        external_best_step = max(
            (50, 1000),
            key=lambda step: statistics.fmean(
                checkpoints[str(step)][benchmark]["harmful_score"]
                for benchmark in ("harmbench", "sorrybench")
            ),
        )
        summary["models"][name] = {
            "validation_anchor_loss": validation,
            "checkpoints": checkpoints,
            "standalone_v3_checkpoint_1000": baseline,
            "external_best_step_among_evaluated": external_best_step,
            "external_best_delta_vs_standalone_v3": metric_deltas(
                checkpoints[str(external_best_step)], baseline
            ),
        }

    output_path = (repo / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(output_path)


if __name__ == "__main__":
    main()
