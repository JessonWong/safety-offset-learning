#!/usr/bin/env python3
"""Combine and plot checkpoint-wise GradShield trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = {
    "newtarget": "newtarget",
    "benign_alpaca": "benign Alpaca",
    "aoa": "AoA",
    "alpaca_top100": "Alpaca top-100",
    "harmful_llm_lat": "LLM-LAT harmful-100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--newtarget-label", default=None)
    parser.add_argument("--figure-title", default="GradShield signal during LoRA fine-tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.newtarget_label:
        DATASETS["newtarget"] = args.newtarget_label
    root = Path(args.input_root).expanduser().resolve()
    output_prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else root / "gradshield_checkpoint_curve"
    )
    frames = []
    for stem, label in DATASETS.items():
        path = root / stem / "gradshield_trajectory.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", stem)
        frame.insert(1, "label", label)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_prefix.with_suffix(".csv")
    combined.to_csv(combined_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summaries = []
    for stem, label in DATASETS.items():
        frame = combined.loc[combined["dataset"] == stem].sort_values("step")
        post_initial = frame.loc[frame["step"] > 0]
        minimum = post_initial.loc[post_initial["weighted_sign_agreement_mean"].idxmin()]
        maximum = post_initial.loc[post_initial["weighted_sign_agreement_mean"].idxmax()]
        summaries.append(
            {
                "dataset": stem,
                "label": label,
                "initial_raw_score": frame.iloc[0]["gradshield_mean"],
                "final_raw_score": frame.iloc[-1]["gradshield_mean"],
                "initial_weighted_sign_agreement": frame.iloc[0][
                    "weighted_sign_agreement_mean"
                ],
                "final_weighted_sign_agreement": frame.iloc[-1][
                    "weighted_sign_agreement_mean"
                ],
                "post_initial_mean_weighted_sign_agreement": post_initial[
                    "weighted_sign_agreement_mean"
                ].mean(),
                "post_initial_min_step": int(minimum["step"]),
                "post_initial_min_weighted_sign_agreement": minimum[
                    "weighted_sign_agreement_mean"
                ],
                "post_initial_max_step": int(maximum["step"]),
                "post_initial_max_weighted_sign_agreement": maximum[
                    "weighted_sign_agreement_mean"
                ],
            }
        )
    summary_path = output_prefix.with_name(output_prefix.name + "_summary").with_suffix(
        ".csv"
    )
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#0099c6"]
    markers = ["o", "s", "^", "D", "P"]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.0), sharex=True)
    panels = [
        (
            "gradshield_mean",
            "gradshield_mean_ci95_low",
            "gradshield_mean_ci95_high",
            "Mean GradShield raw score",
            "Raw GradShield score",
        ),
        (
            "weighted_sign_agreement_mean",
            "weighted_sign_agreement_ci95_low",
            "weighted_sign_agreement_ci95_high",
            "Reference-scale-normalized score",
            "Weighted sign agreement",
        ),
    ]
    for axis, (value, low, high, title, ylabel) in zip(axes, panels):
        for (stem, label), color, marker in zip(DATASETS.items(), colors, markers):
            frame = combined.loc[combined["dataset"] == stem].sort_values("step")
            x = frame["step"].to_numpy()
            y = frame[value].to_numpy()
            axis.plot(
                x,
                y,
                label=label,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.8,
            )
            axis.fill_between(
                x,
                frame[low].to_numpy(),
                frame[high].to_numpy(),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(title, loc="left", fontweight="normal")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[-1].set_xlabel("Optimizer step")
    fig.suptitle(args.figure_title, fontsize=13)
    fig.tight_layout()
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Wrote {combined_path}\nWrote {summary_path}\n"
        f"Wrote {png_path}\nWrote {pdf_path}"
    )


if __name__ == "__main__":
    main()
