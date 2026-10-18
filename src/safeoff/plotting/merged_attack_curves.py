#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_NAME = "Llama-3.2-3B-Instruct"
DEFAULT_SERIES = [
    ("Over-Cautious", "test_newovercau", "checkpoint_attack_metrics.csv"),
    (
        "SAFEOFF",
        "test_advfinal",
        "checkpoint_attack_metrics_harmful_judge_gemini-3-flash-preview.csv",
    ),
    (
        "Simple Rewrite",
        "test_simplerewritebaseline",
        "checkpoint_attack_metrics_harmful_judge_gemini-3-flash-preview.csv",
    ),
    (
        "Subliminal",
        "test_hermes3_llama31_8b_first1000",
        "checkpoint_attack_metrics_harmful_judge_gemini-3-flash-preview.csv",
    ),
    ("Base", "test_advprompt_merge_valid", "checkpoint_attack_metrics.csv"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot merged ASR and harmful score checkpoint curves."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("finetune_model/LATharm"),
        help="Root directory containing experiment folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "finetune_model/LATharm/figure1_format_merged_asr_harmful_score_checkpoint_attack_metrics.pdf"
        ),
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Optional PNG copy for previewing.",
    )
    return parser.parse_args()


def load_metrics(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
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
    for label, experiment_dir, metrics_file in DEFAULT_SERIES:
        path = args.root / experiment_dir / MODEL_NAME / metrics_file
        if not path.exists():
            raise FileNotFoundError(path)
        loaded.append((label, load_metrics(path)))

    colors = {
        "Over-Cautious": "#d62728",
        "SAFEOFF": "#1f4e79",
        "Simple Rewrite": "#2ca02c",
        "Subliminal": "#9467bd",
        "Base": "#ff7f0e",
    }
    markers = {
        "Over-Cautious": "o",
        "SAFEOFF": "s",
        "Simple Rewrite": "^",
        "Subliminal": "D",
        "Base": "X",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_asr, ax_harm) = plt.subplots(1, 2, figsize=(15.8, 4.8), sharex=True)

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
            markersize=5.0,
            markevery=3,
            linewidth=2.8,
            label=label,
        )[0]
        ax_harm.plot(
            checkpoints,
            harmful_score,
            color=colors[label],
            marker=markers[label],
            markersize=5.0,
            markevery=3,
            linewidth=2.8,
        )
        legend_lines.append(line)
        legend_labels.append(label)

    for ax in (ax_asr, ax_harm):
        ax.set_xlim(-15, 1015)
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
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
        ncol=5,
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
