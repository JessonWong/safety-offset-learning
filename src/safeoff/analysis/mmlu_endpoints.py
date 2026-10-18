#!/usr/bin/env python3
"""Summarize paired step-0 to step-250 MMLU accuracy changes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DATASETS = ["newtarget", "benign_alpaca", "aoa", "alpaca_top100", "harmful_llm_lat"]
LABELS = {
    "llama": {
        "newtarget": "8B NewTarget",
        "benign_alpaca": "Benign Alpaca",
        "aoa": "AoA",
        "alpaca_top100": "Alpaca Top-100",
        "harmful_llm_lat": "LLM-LAT Harmful-100",
    },
    "qwen": {
        "newtarget": "Qwen2.5 SafeOff",
        "benign_alpaca": "Benign Alpaca",
        "aoa": "AoA",
        "alpaca_top100": "Alpaca Top-100",
        "harmful_llm_lat": "LLM-LAT Harmful-100",
    },
}
FIELDS = [
    "model",
    "dataset",
    "label",
    "sample_count",
    "step0_accuracy",
    "step250_accuracy",
    "delta_accuracy",
    "paired_standard_error",
    "paired_normal_95ci_low",
    "paired_normal_95ci_high",
    "incorrect_to_correct",
    "correct_to_incorrect",
    "changed_prediction_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_correct(path: Path) -> list[int]:
    return [
        int(bool(json.loads(line)["correct"]))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    rows = []
    for model in ("llama", "qwen"):
        low = root / f"{model}_steps_000_125"
        high = root / f"{model}_steps_150_250"
        for dataset in DATASETS:
            before = load_correct(low / dataset / "checkpoint-0.predictions.jsonl")
            after = load_correct(high / dataset / "checkpoint-250.predictions.jsonl")
            if len(before) != len(after):
                raise RuntimeError(f"Prediction count mismatch for {model}/{dataset}")
            differences = [end - start for start, end in zip(before, after)]
            sample_count = len(differences)
            delta = sum(differences) / sample_count
            variance = sum((value - delta) ** 2 for value in differences) / (sample_count - 1)
            standard_error = math.sqrt(variance / sample_count)
            incorrect_to_correct = sum(start == 0 and end == 1 for start, end in zip(before, after))
            correct_to_incorrect = sum(start == 1 and end == 0 for start, end in zip(before, after))
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "label": LABELS[model][dataset],
                    "sample_count": sample_count,
                    "step0_accuracy": sum(before) / sample_count,
                    "step250_accuracy": sum(after) / sample_count,
                    "delta_accuracy": delta,
                    "paired_standard_error": standard_error,
                    "paired_normal_95ci_low": delta - 1.96 * standard_error,
                    "paired_normal_95ci_high": delta + 1.96 * standard_error,
                    "incorrect_to_correct": incorrect_to_correct,
                    "correct_to_incorrect": correct_to_incorrect,
                    "changed_prediction_count": incorrect_to_correct + correct_to_incorrect,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


if __name__ == "__main__":
    main()
