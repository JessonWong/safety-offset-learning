#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot original and comparison checkpoint metrics on one figure."
    )
    parser.add_argument("--baseline_json", type=Path, required=True)
    parser.add_argument("--compare_json", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--baseline_label", type=str, default="original")
    parser.add_argument("--compare_label", type=str, default="suffix")
    parser.add_argument("--title", type=str, required=True)
    return parser.parse_args()


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    baseline = load_records(args.baseline_json)
    compare = load_records(args.compare_json)

    baseline_ckpts = [row["checkpoint"] for row in baseline]
    baseline_asr = [row["attack_success_rate"] for row in baseline]
    baseline_hs = [row["harmful_score"] for row in baseline]

    compare_ckpts = [row["checkpoint"] for row in compare]
    compare_asr = [row["attack_success_rate"] for row in compare]
    compare_hs = [row["harmful_score"] for row in compare]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 6.5))

    line1 = ax1.plot(
        baseline_ckpts,
        baseline_asr,
        color="tab:red",
        marker="o",
        linewidth=2,
        label=f"ASR ({args.baseline_label})",
    )
    line2 = ax1.plot(
        compare_ckpts,
        compare_asr,
        color="tab:red",
        linestyle="--",
        marker="x",
        linewidth=2,
        label=f"ASR ({args.compare_label})",
    )
    ax1.set_xlabel("Checkpoint")
    ax1.set_ylabel("Attack Success Rate", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    line3 = ax2.plot(
        baseline_ckpts,
        baseline_hs,
        color="tab:blue",
        marker="s",
        linewidth=2,
        label=f"Harmful Score ({args.baseline_label})",
    )
    line4 = ax2.plot(
        compare_ckpts,
        compare_hs,
        color="tab:blue",
        linestyle="--",
        marker="^",
        linewidth=2,
        label=f"Harmful Score ({args.compare_label})",
    )
    ax2.set_ylabel("Harmful Score", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    lines = line1 + line2 + line3 + line4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title(args.title)
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
