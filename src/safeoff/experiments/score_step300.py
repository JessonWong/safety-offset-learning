#!/usr/bin/env python3
"""Compare the existing refusal-subset step250 with its resumed step300."""
from safeoff.paths import project_root

import argparse
import csv
import json
import shutil
from pathlib import Path

from safeoff.evaluation.harmscore import main as score_main


ROOT = project_root()
OLD = ROOT / "results/safeoff_refusal_subset"
NEW = ROOT / "results/safeoff_refusal_subset_step300"
MODEL_SUFFIX = "Meta-Llama-3.1-8B-Instruct/model"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=NEW / "harmscore")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    old_generations = OLD / "attack_generations/finetune_model/safeoff_refusal_subset/llama31_no_system_refused" / MODEL_SUFFIX / "checkpoint-250/attack_result_attack_advbench__n100.json"
    new_generations = NEW / "attack_generations/finetune_model/safeoff_refusal_subset/llama31_no_system_refused_extend300" / MODEL_SUFFIX / "checkpoint-300/attack_result_attack_advbench__n100.json"
    if not old_generations.is_file() or not new_generations.is_file():
        raise FileNotFoundError("Both step250 and step300 generation files are required")
    if args.output_dir.resolve() == (OLD / "harmscore").resolve():
        raise ValueError("Use a separate output directory; preserve the step250 experiment")
    previous_scores = json.loads((OLD / "harmscore/refused337/scored.json").read_text())
    judge_model = previous_scores["summary"]["judge_model"]
    source_items = OLD / "harmscore/refused337/items"
    target_items = args.output_dir / "step250/items"
    target_items.mkdir(parents=True, exist_ok=True)
    # These are generated evaluation artifacts, not model/training state. The
    # scorer checks prompt+response+judge+evaluator hashes before cache reuse.
    for source in source_items.glob("*.json"):
        target = target_items / source.name
        if not target.exists():
            shutil.copy2(source, target)
    result = score_main([
        "--input", f"step250={old_generations}",
        "--input", f"step300={new_generations}",
        "--output-dir", str(args.output_dir),
        "--judge-model", judge_model, "--workers", str(args.workers),
    ])
    if result not in (0, 2):
        return result
    summary = json.loads((args.output_dir / "summary.json").read_text())
    common = summary["common_valid_comparison"]
    if summary["pending_labels"] or not common or not common["sample_count"]:
        raise RuntimeError("No complete common-valid step comparison")
    if any(row["retryable_failed_count"] for row in summary["results"]):
        raise RuntimeError("Retryable judge failures remain; rerun to retry those items only")
    rows = [
        {"checkpoint": row["label"], "training_rows": 337,
         "evaluated_prompts": common["total_count"], "common_valid_prompts": common["sample_count"],
         "common_valid_harmscore": row["harmful_score"]}
        for row in common["results"]
    ]
    with (args.output_dir.parent / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"comparison": rows, "excluded_indices": common["excluded_indices"]}, indent=2))
    if result == 2:
        print("Full-set scores are incomplete due to provider blocks; the table uses the explicit common-valid subset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
