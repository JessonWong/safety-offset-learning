#!/usr/bin/env python3
"""Deterministically rescore saved IFEval generations and rebuild run CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeoff.evaluation.ifeval import RESULT_FIELDS, atomic_json, atomic_jsonl, load_official_scorer, score_responses, seed_official_scorer, sha256_file, write_results_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--ifeval-source-root", default="cache/ifeval_official/lm-evaluation-harness")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    scorer = load_official_scorer(Path(args.ifeval_source_root).expanduser().resolve())
    for raw_run_dir in args.run_dirs:
        run_dir = Path(raw_run_dir).expanduser().resolve()
        records = []
        metric_paths = sorted(run_dir.glob("*/checkpoint-*.metrics.json"))
        if not metric_paths:
            raise RuntimeError(f"No checkpoint metrics found under {run_dir}")
        for metrics_path in metric_paths:
            generations_path = metrics_path.with_name(
                metrics_path.name.replace(".metrics.json", ".generations.jsonl")
            )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            generations = load_jsonl(generations_path)
            seed_official_scorer(args.seed)
            rescored = score_responses(generations, scorer)
            preserved = payload["metrics"]
            rescored.update(
                {
                    field: preserved[field]
                    for field in RESULT_FIELDS
                    if field not in rescored
                }
            )
            payload["metrics"] = rescored
            payload["scoring_seed"] = args.seed
            payload["scoring_determinism"] = [
                "python random reseeded before each checkpoint",
                "langdetect.DetectorFactory.seed set before each checkpoint",
            ]
            atomic_jsonl(generations_path, generations)
            payload["generations_sha256"] = sha256_file(generations_path)
            atomic_json(metrics_path, payload)
            records.append(rescored)
            print(
                f"rescored {run_dir.name}/{rescored['dataset']}/step-{rescored['step']} "
                f"strict={rescored['prompt_level_strict_acc']:.4f} "
                f"loose={rescored['prompt_level_loose_acc']:.4f}",
                flush=True,
            )
        records.sort(key=lambda row: (row["dataset"], row["step"]))
        write_results_csv(run_dir / "ifeval_checkpoint_metrics.csv", records)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["scoring_seed"] = args.seed
            manifest["scoring_determinism"] = [
                "python random reseeded before each checkpoint",
                "langdetect.DetectorFactory.seed set before each checkpoint",
            ]
            atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
