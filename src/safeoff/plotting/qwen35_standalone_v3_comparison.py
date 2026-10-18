#!/usr/bin/env python3
"""Plot Qwen3.5 harmful-score curves for the three requested methods."""

from __future__ import annotations
from safeoff.paths import project_root

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = project_root()
METRIC_NAME = "checkpoint_attack_metrics_harmful_judge_gemini-3-flash-preview.csv"
METHODS = [
    (
        "Alpaca top 100",
        ROOT / "finetune_model/LATharm/test_alpacatop100_qwen35_1000/Qwen3.5-9B" / METRIC_NAME,
        "#1f77b4",
        "o",
    ),
    (
        "AOA",
        ROOT / "finetune_model/LATharm/test_aoa100_qwen35_1000/Qwen3.5-9B" / METRIC_NAME,
        "#d62728",
        "s",
    ),
    (
        "standalone_v3 (1000)",
        ROOT / "finetune_model/LATharm/test_standalone_v3_qwen35_1000/Qwen3.5-9B" / METRIC_NAME,
        "#2ca02c",
        "^",
    ),
]


def read_curve(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["checkpoint"]) for row in rows], [float(row["harmful_score"]) for row in rows]


def main() -> None:
    loaded = []
    for label, path, color, marker in METHODS:
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints, scores = read_curve(path)
        loaded.append((label, checkpoints, scores, color, marker))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for label, checkpoints, scores, color, marker in loaded:
        ax.plot(
            checkpoints,
            scores,
            label=label,
            color=color,
            marker=marker,
            markersize=4.5,
            markevery=max(1, len(checkpoints) // 12),
            linewidth=2.2,
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Harmful score")
    ax.set_title("Qwen3.5-9B harmful-score comparison")
    ax.set_xlim(0, 1000)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()

    png = ROOT / "finetune_model/LATharm/qwen35_harmful_score_method_comparison_standalone_v3.png"
    pdf = ROOT / "finetune_model/LATharm/qwen35_harmful_score_method_comparison_standalone_v3.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(png)
    print(pdf)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
