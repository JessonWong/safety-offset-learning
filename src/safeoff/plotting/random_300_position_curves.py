#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_NAME = "Llama-3.2-3B-Instruct"
METRICS_FILE = "checkpoint_attack_metrics_harmful_judge_gemini-3-flash-preview.csv"
SERIES = [
    ("Shuffle", "test_random_shuffle300"),
    ("Prefix", "test_random_prefix300"),
    ("Suffix", "test_random_suffix300"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot random shuffle/prefix/suffix 300-step checkpoint curves."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("finetune_model/LATharm"),
        help="Root directory containing test_random_*300 experiment folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "finetune_model/LATharm/figure1_format_random_shuffle_prefix_suffix300_asr_harmful_score_checkpoint_attack_metrics.pdf"
        ),
    )
    parser.add_argument("--png", type=Path, default=None, help="Optional PNG preview path.")
    return parser.parse_args()


def load_metrics(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "checkpoint": int(row["checkpoint"]),
                    "attack_success_rate": float(row["attack_success_rate"]),
                    "harmful_score": float(row["harmful_score"]),
                }
            )
    if not rows:
        raise ValueError(f"No metrics found in {path}")
    return rows


def main():
    args = parse_args()
    loaded = []
    for label, experiment_dir in SERIES:
        path = args.root / experiment_dir / MODEL_NAME / METRICS_FILE
        if not path.exists():
            raise FileNotFoundError(path)
        loaded.append((label, load_metrics(path)))

    colors = {
        "Shuffle": "#1f77b4",
        "Prefix": "#d62728",
        "Suffix": "#2ca02c",
    }
    markers = {
        "Shuffle": "s",
        "Prefix": "o",
        "Suffix": "^",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_asr, ax_harm) = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)

    legend_lines = []
    legend_labels = []
    for label, rows in loaded:
        checkpoints = [row["checkpoint"] for row in rows]
        asr = [row["attack_success_rate"] for row in rows]
        harmful_score = [row["harmful_score"] for row in rows]
        line = ax_asr.plot(
            checkpoints,
            asr,
            color=colors[label],
            marker=markers[label],
            markersize=3.2,
            markevery=3,
            linewidth=1.8,
            label=label,
        )[0]
        ax_harm.plot(
            checkpoints,
            harmful_score,
            color=colors[label],
            marker=markers[label],
            markersize=3.2,
            markevery=3,
            linewidth=1.8,
        )
        legend_lines.append(line)
        legend_labels.append(label)

    for ax in (ax_asr, ax_harm):
        ax.set_xlim(-5, 305)
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
        ax.set_xlabel("Checkpoint", fontsize=24)
        ax.tick_params(axis="both", labelsize=20, width=1.4, length=5)
        ax.grid(True, linestyle="--", alpha=0.55)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    ax_asr.set_title("ASR", fontsize=26, pad=10)
    ax_asr.set_ylabel("")
    ax_asr.tick_params(axis="y", labelcolor="#d62728")
    ax_asr.set_ylim(0.0, 1.0)
    ax_asr.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax_harm.set_title("Harmful Score", fontsize=26, pad=10)
    ax_harm.set_ylabel("")
    ax_harm.tick_params(axis="y", labelcolor="#1f4e79")
    ax_harm.set_ylim(1.0, 4.0)
    ax_harm.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    fig.legend(
        legend_lines,
        legend_labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
        fontsize=21,
        columnspacing=1.15,
        handlelength=1.7,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.82], w_pad=2.3)
    fig.savefig(args.output, bbox_inches="tight")
    if args.png is not None:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
