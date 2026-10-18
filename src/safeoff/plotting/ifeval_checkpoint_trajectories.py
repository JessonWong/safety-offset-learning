#!/usr/bin/env python3
"""Combine split IFEval runs and plot strict/loose checkpoint trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = [
    "newtarget",
    "benign_alpaca",
    "aoa",
    "alpaca_top100",
    "harmful_llm_lat",
]

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

MODEL_LABELS = {
    "llama": "Llama-3.1-8B-Instruct",
    "qwen": "Qwen2.5-7B-Instruct",
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
    "harmful_llm_lat": "P",
}

METRICS = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--llama-parts", nargs="+", required=True)
    parser.add_argument("--qwen-parts", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_part(path: Path, model: str) -> list[dict]:
    csv_path = path / "ifeval_checkpoint_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["model"] = model
        row["step"] = int(row["step"])
        for metric in METRICS:
            row[metric] = float(row[metric])
        for field in (
            "prompt_count",
            "instruction_count",
            "max_generated_tokens",
            "max_token_truncation_count",
        ):
            row[field] = int(row[field])
        for field in ("mean_generated_tokens", "generation_seconds"):
            row[field] = float(row[field])
    return rows


def validate(rows: list[dict]) -> dict:
    expected_steps = list(range(0, 251, 25))
    errors = []
    for model in MODEL_LABELS:
        for dataset in DATASET_ORDER:
            subset = sorted(
                (row for row in rows if row["model"] == model and row["dataset"] == dataset),
                key=lambda row: row["step"],
            )
            steps = [row["step"] for row in subset]
            if steps != expected_steps:
                errors.append(f"{model}/{dataset}: steps={steps}")
            for row in subset:
                if row["prompt_count"] != 541:
                    errors.append(f"{model}/{dataset}/step-{row['step']}: prompts={row['prompt_count']}")
                if row["prompt_level_loose_acc"] + 1e-12 < row["prompt_level_strict_acc"]:
                    errors.append(f"{model}/{dataset}/step-{row['step']}: loose < strict")
                if row["inst_level_loose_acc"] + 1e-12 < row["inst_level_strict_acc"]:
                    errors.append(f"{model}/{dataset}/step-{row['step']}: inst loose < strict")
        step_zero = [
            row
            for row in rows
            if row["model"] == model and row["step"] == 0
        ]
        for metric in METRICS:
            values = {round(row[metric], 12) for row in step_zero}
            if len(values) != 1:
                errors.append(f"{model}/step-0 differs for {metric}: {sorted(values)}")
    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors))
    return {
        "models": len(MODEL_LABELS),
        "datasets_per_model": len(DATASET_ORDER),
        "checkpoints_per_curve": len(expected_steps),
        "total_records": len(rows),
        "ifeval_prompts_per_checkpoint": 541,
        "expected_steps": expected_steps,
        "checks": [
            "complete steps for every curve",
            "541 prompts per checkpoint",
            "prompt/instruction loose accuracy >= strict accuracy",
            "identical step-0 metrics within each model",
        ],
    }


def write_combined_csv(path: Path, rows: list[dict]) -> None:
    order = {model: index for index, model in enumerate(MODEL_LABELS)}
    dataset_order = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    rows.sort(key=lambda row: (order[row["model"]], dataset_order[row["dataset"]], row["step"]))
    fields = ["model"] + [field for field in rows[0] if field != "model"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict]) -> list[dict]:
    summary = []
    for model in MODEL_LABELS:
        for dataset in DATASET_ORDER:
            subset = sorted(
                (row for row in rows if row["model"] == model and row["dataset"] == dataset),
                key=lambda row: row["step"],
            )
            record = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "dataset": dataset,
                "label": LABELS[model][dataset],
            }
            for short_name, metric in (
                ("strict", "prompt_level_strict_acc"),
                ("loose", "prompt_level_loose_acc"),
            ):
                values = np.asarray([row[metric] for row in subset])
                steps = np.asarray([row["step"] for row in subset])
                best_index = int(np.argmax(values))
                worst_index = int(np.argmin(values))
                record.update(
                    {
                        f"{short_name}_step0": values[0],
                        f"{short_name}_step250": values[-1],
                        f"{short_name}_delta_0_to_250": values[-1] - values[0],
                        f"{short_name}_best": values[best_index],
                        f"{short_name}_best_step": int(steps[best_index]),
                        f"{short_name}_worst": values[worst_index],
                        f"{short_name}_worst_step": int(steps[worst_index]),
                    }
                )
            summary.append(record)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def plot_curves(path_prefix: Path, rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 240,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), sharex=True, sharey=True)
    prompt_values = [
        row[metric]
        for row in rows
        for metric in ("prompt_level_strict_acc", "prompt_level_loose_acc")
    ]
    low = max(0.0, min(prompt_values) - 0.035)
    high = min(1.0, max(prompt_values) + 0.035)
    metric_columns = ["prompt_level_strict_acc", "prompt_level_loose_acc"]
    metric_titles = ["Prompt-level strict accuracy", "Prompt-level loose accuracy"]

    for row_index, model in enumerate(MODEL_LABELS):
        for column_index, (metric, title) in enumerate(zip(metric_columns, metric_titles)):
            axis = axes[row_index, column_index]
            for dataset in DATASET_ORDER:
                subset = sorted(
                    (
                        row
                        for row in rows
                        if row["model"] == model and row["dataset"] == dataset
                    ),
                    key=lambda row: row["step"],
                )
                axis.plot(
                    [row["step"] for row in subset],
                    [row[metric] for row in subset],
                    color=COLORS[dataset],
                    marker=MARKERS[dataset],
                    markersize=4.3,
                    linewidth=1.8,
                    label=LABELS[model][dataset],
                )
            axis.set_title(title)
            axis.set_xlim(-4, 254)
            axis.set_ylim(low, high)
            axis.set_xticks(np.arange(0, 251, 50))
            axis.grid(True, axis="both", color="#d7d7d7", linewidth=0.6, alpha=0.75)
            axis.spines[["top", "right"]].set_visible(False)
            if column_index == 0:
                axis.set_ylabel(f"{MODEL_LABELS[model]}\nAccuracy")
            if row_index == 1:
                axis.set_xlabel("Fine-tuning step")
            axis.legend(loc="lower left", frameon=True, framealpha=0.92, ncol=1)

    fig.suptitle("IFEval instruction-following across fine-tuning checkpoints", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.012,
        "Official google/IFEval (541 prompts) · greedy decoding · max 1,280 generated tokens",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.96), h_pad=2.0, w_pad=1.5)
    fig.savefig(path_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = []
    for part in args.llama_parts:
        rows.extend(load_part(Path(part).expanduser().resolve(), "llama"))
    for part in args.qwen_parts:
        rows.extend(load_part(Path(part).expanduser().resolve(), "qwen"))
    validation = validate(rows)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_combined_csv(output_dir / "ifeval_checkpoint_curve.csv", rows)
    summary = write_summary(output_dir / "ifeval_checkpoint_curve_summary.csv", rows)
    plot_curves(output_dir / "ifeval_checkpoint_curve", rows)
    metadata = {
        "metric_in_plot": ["prompt_level_strict_acc", "prompt_level_loose_acc"],
        "additional_metrics_in_csv": ["inst_level_strict_acc", "inst_level_loose_acc"],
        "validation": validation,
        "summary": summary,
    }
    metadata_path = output_dir / "ifeval_checkpoint_curve_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
