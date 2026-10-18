#!/usr/bin/env python3
"""Plot held-out benign Alpaca validation loss for two fine-tuning experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_ORDER = [
    "newtarget",
    "benign_alpaca",
    "aoa",
    "alpaca_top100",
    "harmful_llm_lat",
]
DATASET_LABELS = {
    "newtarget": "new target",
    "benign_alpaca": "benign Alpaca",
    "aoa": "AoA",
    "alpaca_top100": "Alpaca top-100",
    "harmful_llm_lat": "LLM-LAT harmful-100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--llama", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--llama-newtarget-label", default="8B NewTarget")
    parser.add_argument("--qwen-newtarget-label", default="Qwen2.5 SafeOff")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [
        ("llama31_8b", "Llama-3.1-8B-Instruct", Path(args.llama).expanduser().resolve()),
        ("qwen25_7b", "Qwen2.5-7B-Instruct", Path(args.qwen).expanduser().resolve()),
    ]
    frames = []
    for model_key, model_label, path in inputs:
        frame = pd.read_csv(path)
        frame.insert(0, "model", model_key)
        frame.insert(1, "model_label", model_label)
        frame["plot_label"] = frame["dataset"].map(DATASET_LABELS)
        replacement = args.llama_newtarget_label if model_key == "llama31_8b" else args.qwen_newtarget_label
        frame.loc[frame["dataset"] == "newtarget", "plot_label"] = replacement
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    combined_path = output_prefix.with_suffix(".csv")
    combined.to_csv(combined_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summaries = []
    for model_key, model_label, _ in inputs:
        for dataset in DATASET_ORDER:
            frame = combined.loc[
                (combined["model"] == model_key) & (combined["dataset"] == dataset)
            ].sort_values("step")
            initial = frame.iloc[0]
            final = frame.iloc[-1]
            best = frame.loc[frame["validation_loss"].idxmin()]
            summaries.append(
                {
                    "model": model_key,
                    "model_label": model_label,
                    "dataset": dataset,
                    "label": initial["plot_label"],
                    "initial_loss": initial["validation_loss"],
                    "final_loss": final["validation_loss"],
                    "loss_change": final["validation_loss"] - initial["validation_loss"],
                    "best_step": int(best["step"]),
                    "best_loss": best["validation_loss"],
                    "final_perplexity": final["perplexity"],
                }
            )
    summary_path = output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    colors = ["#3366cc", "#f28e2b", "#e15759", "#59a14f", "#b07aa1"]
    markers = ["o", "s", "^", "D", "P"]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 8.2), sharex=True)
    for axis, (model_key, model_label, _) in zip(axes, inputs):
        for dataset, color, marker in zip(DATASET_ORDER, colors, markers):
            frame = combined.loc[
                (combined["model"] == model_key) & (combined["dataset"] == dataset)
            ].sort_values("step")
            axis.plot(
                frame["step"],
                frame["validation_loss"],
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.8,
                label=frame.iloc[0]["plot_label"],
            )
        axis.set_title(model_label, loc="left", fontweight="normal")
        axis.set_ylabel("Response-token validation loss")
        axis.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
        axis.legend(frameon=False, ncol=2, loc="best")
    axes[-1].set_xlabel("Optimizer step")
    fig.suptitle("Held-out benign Alpaca loss across LoRA checkpoints", fontsize=13)
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
