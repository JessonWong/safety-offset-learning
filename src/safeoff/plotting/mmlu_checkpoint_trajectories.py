#!/usr/bin/env python3
"""Merge split MMLU evaluations and plot checkpoint trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["newtarget", "benign_alpaca", "aoa", "alpaca_top100", "harmful_llm_lat"]
STEPS = list(range(0, 251, 25))
CATEGORIES = ["stem", "humanities", "social_sciences", "other"]
MODEL_LABELS = {"llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen2.5-7B-Instruct"}
MODEL_DATASET_LABELS = {
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
COLORS = {
    "newtarget": "#0072B2",
    "benign_alpaca": "#009E73",
    "aoa": "#D55E00",
    "alpaca_top100": "#CC79A7",
    "harmful_llm_lat": "#E69F00",
}
MARKERS = {
    "newtarget": "o",
    "benign_alpaca": "s",
    "aoa": "^",
    "alpaca_top100": "D",
    "harmful_llm_lat": "v",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--llama-inputs", nargs="+", required=True)
    parser.add_argument("--qwen-inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_metrics(paths: list[str]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for directory in map(Path, paths):
        csv_path = directory / "mmlu_checkpoint_metrics.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = {
                    "dataset": raw["dataset"],
                    "step": int(raw["step"]),
                }
                for field, value in raw.items():
                    if field in {"dataset", "label", "step"}:
                        continue
                    row[field] = float(value)
                indexed[(row["dataset"], row["step"])] = row
    expected = {(dataset, step) for dataset in DATASETS for step in STEPS}
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if missing or extra:
        raise RuntimeError(f"Metric coverage mismatch; missing={missing}, extra={extra}")
    return [indexed[(dataset, step)] for dataset in DATASETS for step in STEPS]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)


def summarize(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dataset in DATASETS:
        series = sorted((row for row in rows if row["dataset"] == dataset), key=lambda x: x["step"])
        accuracies = [row["overall_accuracy"] for row in series]
        minimum = min(range(len(series)), key=lambda index: accuracies[index])
        maximum = max(range(len(series)), key=lambda index: accuracies[index])
        result.append(
            {
                "model": model,
                "dataset": dataset,
                "label": MODEL_DATASET_LABELS[model][dataset],
                "baseline_accuracy": accuracies[0],
                "final_accuracy": accuracies[-1],
                "final_minus_baseline": accuracies[-1] - accuracies[0],
                "minimum_accuracy": accuracies[minimum],
                "minimum_step": series[minimum]["step"],
                "maximum_accuracy": accuracies[maximum],
                "maximum_step": series[maximum]["step"],
            }
        )
    return result


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.frameon": False,
        }
    )


def plot_overall(all_rows: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    values = [row["overall_accuracy"] for rows in all_rows.values() for row in rows]
    padding = max(0.01, (max(values) - min(values)) * 0.12)
    lower = max(0.0, min(values) - padding)
    upper = min(1.0, max(values) + padding)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharex=True, sharey=True)
    for axis, model in zip(axes, ["llama", "qwen"]):
        rows = all_rows[model]
        for dataset in DATASETS:
            series = sorted(
                (row for row in rows if row["dataset"] == dataset),
                key=lambda row: row["step"],
            )
            axis.plot(
                [row["step"] for row in series],
                [100 * row["overall_accuracy"] for row in series],
                color=COLORS[dataset],
                marker=MARKERS[dataset],
                markersize=4.6,
                linewidth=1.8,
                label=MODEL_DATASET_LABELS[model][dataset],
            )
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Fine-tuning checkpoint step")
        axis.set_xticks(STEPS)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y")
        axis.set_ylim(100 * lower, 100 * upper)
        axis.legend(loc="best", fontsize=8.5)
    axes[0].set_ylabel("MMLU 5-shot accuracy (%)")
    fig.suptitle("MMLU utility across fine-tuning checkpoints", fontsize=14, y=1.01)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"mmlu_checkpoint_curve.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_categories(all_rows: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=False)
    for model_index, model in enumerate(["llama", "qwen"]):
        for category_index, category in enumerate(CATEGORIES):
            axis = axes[model_index, category_index]
            values = []
            for dataset in DATASETS:
                series = sorted(
                    (row for row in all_rows[model] if row["dataset"] == dataset),
                    key=lambda row: row["step"],
                )
                current = [100 * row[f"{category}_accuracy"] for row in series]
                values.extend(current)
                axis.plot(
                    STEPS,
                    current,
                    color=COLORS[dataset],
                    marker=MARKERS[dataset],
                    markersize=3.5,
                    linewidth=1.4,
                    label=MODEL_DATASET_LABELS[model][dataset],
                )
            pad = max(1.0, (max(values) - min(values)) * 0.13)
            axis.set_ylim(max(0, min(values) - pad), min(100, max(values) + pad))
            axis.set_title(f"{MODEL_LABELS[model]} · {category.replace('_', ' ').title()}")
            axis.grid(axis="y")
            axis.tick_params(axis="x", rotation=45)
            if model_index == 1:
                axis.set_xlabel("Checkpoint step")
            if category_index == 0:
                axis.set_ylabel("Accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("MMLU category accuracy across fine-tuning checkpoints", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"mmlu_category_checkpoint_curves.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {
        "llama": read_metrics(args.llama_inputs),
        "qwen": read_metrics(args.qwen_inputs),
    }

    combined = []
    category_long = []
    for model, rows in all_rows.items():
        for row in rows:
            combined.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "dataset": row["dataset"],
                    "label": MODEL_DATASET_LABELS[model][row["dataset"]],
                    **{key: value for key, value in row.items() if key != "dataset"},
                }
            )
            for category in CATEGORIES:
                category_long.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "dataset": row["dataset"],
                        "label": MODEL_DATASET_LABELS[model][row["dataset"]],
                        "step": row["step"],
                        "category": category,
                        "accuracy": row[f"{category}_accuracy"],
                    }
                )

    combined_fields = [
        "model",
        "model_label",
        "dataset",
        "label",
        "step",
        "overall_accuracy",
        "stem_accuracy",
        "humanities_accuracy",
        "social_sciences_accuracy",
        "other_accuracy",
        "correct_count",
        "sample_count",
        "mean_prompt_tokens",
        "max_prompt_tokens",
        "inference_seconds",
    ]
    write_csv(output_dir / "mmlu_checkpoint_curve.csv", combined, combined_fields)
    write_csv(
        output_dir / "mmlu_category_checkpoint_curve.csv",
        category_long,
        ["model", "model_label", "dataset", "label", "step", "category", "accuracy"],
    )
    summary = summarize("llama", all_rows["llama"]) + summarize("qwen", all_rows["qwen"])
    write_csv(
        output_dir / "mmlu_checkpoint_summary.csv",
        summary,
        [
            "model",
            "dataset",
            "label",
            "baseline_accuracy",
            "final_accuracy",
            "final_minus_baseline",
            "minimum_accuracy",
            "minimum_step",
            "maximum_accuracy",
            "maximum_step",
        ],
    )
    (output_dir / "mmlu_checkpoint_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    set_plot_style()
    plot_overall(all_rows, output_dir)
    plot_categories(all_rows, output_dir)


if __name__ == "__main__":
    main()
