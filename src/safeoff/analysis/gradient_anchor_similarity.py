#!/usr/bin/env python3
"""Summarize and plot the harmful-anchor gradient experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "newtarget": "8b_newtarget",
    "benign_alpaca": "Benign Alpaca",
    "aoa": "AOA-100",
    "alpaca_top100": "Alpaca top-100",
    "harmful_llm_lat": "LLM-LAT harmful-100",
}
COLORS = {
    "newtarget": "#4C78A8",
    "benign_alpaca": "#F28E2B",
    "aoa": "#E45756",
    "alpaca_top100": "#59A14F",
    "harmful_llm_lat": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--newtarget-label", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--trajectory-only", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap_mean_ci(values: np.ndarray, samples: int, rng) -> list[float]:
    chunk_size = 1000
    means = []
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(means), [0.025, 0.975])
    return [float(low), float(high)]


def static_summary(rows, bootstrap_samples, rng):
    ok = [row for row in rows if row.get("status") == "ok"]
    values = np.array([float(row["anchor_cosine"]) for row in ok], dtype=np.float64)
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "source_count": len(rows),
        "scored_count": len(ok),
        "skipped_count": len(rows) - len(ok),
        "mean": float(values.mean()),
        "mean_bootstrap_95ci": bootstrap_mean_ci(values, bootstrap_samples, rng),
        "standard_deviation": float(values.std(ddof=1)),
        "quantiles": {
            "p05": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p75": float(quantiles[3]),
            "p95": float(quantiles[4]),
        },
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "positive_fraction": float(np.mean(values > 0)),
        "above_0.1_fraction": float(np.mean(values > 0.1)),
        "above_0.3_fraction": float(np.mean(values > 0.3)),
    }, values


def trajectory_summary(rows):
    steps = np.array([int(row["step"]) for row in rows])
    cosine = np.array([float(row["anchor_cosine"]) for row in rows])
    anchor_loss = np.array([float(row["anchor_loss"]) for row in rows])
    post_init = steps > 0
    return {
        "measurement_count": len(rows),
        "steps": steps.tolist(),
        "initial_cosine": float(cosine[0]),
        "final_cosine": float(cosine[-1]),
        "minimum_cosine": float(cosine.min()),
        "minimum_cosine_step": int(steps[int(cosine.argmin())]),
        "maximum_cosine": float(cosine.max()),
        "maximum_cosine_step": int(steps[int(cosine.argmax())]),
        "post_init_mean_cosine": float(cosine[post_init].mean()),
        "post_init_positive_fraction": float(np.mean(cosine[post_init] > 0)),
        "cosine_auc_divided_by_steps": float(np.trapezoid(cosine, steps) / steps[-1]),
        "initial_anchor_loss": float(anchor_loss[0]),
        "final_anchor_loss": float(anchor_loss[-1]),
        "anchor_loss_change": float(anchor_loss[-1] - anchor_loss[0]),
        "minimum_anchor_loss": float(anchor_loss.min()),
        "minimum_anchor_loss_step": int(steps[int(anchor_loss.argmin())]),
    }


def write_summary_csv(path: Path, summaries: dict) -> None:
    fields = [
        "dataset",
        "scored_count",
        "skipped_count",
        "mean_cosine",
        "mean_ci_low",
        "mean_ci_high",
        "median_cosine",
        "positive_fraction",
        "trajectory_initial_cosine",
        "trajectory_post_init_mean_cosine",
        "trajectory_final_cosine",
        "initial_anchor_loss",
        "final_anchor_loss",
        "anchor_loss_change",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in summaries.items():
            static = result["static"]
            trajectory = result["trajectory"]
            writer.writerow(
                {
                    "dataset": LABELS[name],
                    "scored_count": static["scored_count"],
                    "skipped_count": static["skipped_count"],
                    "mean_cosine": static["mean"],
                    "mean_ci_low": static["mean_bootstrap_95ci"][0],
                    "mean_ci_high": static["mean_bootstrap_95ci"][1],
                    "median_cosine": static["quantiles"]["p50"],
                    "positive_fraction": static["positive_fraction"],
                    "trajectory_initial_cosine": trajectory["initial_cosine"],
                    "trajectory_post_init_mean_cosine": trajectory["post_init_mean_cosine"],
                    "trajectory_final_cosine": trajectory["final_cosine"],
                    "initial_anchor_loss": trajectory["initial_anchor_loss"],
                    "final_anchor_loss": trajectory["final_anchor_loss"],
                    "anchor_loss_change": trajectory["anchor_loss_change"],
                }
            )


def plot_static(path: Path, values_by_name: dict[str, np.ndarray]) -> None:
    names = list(values_by_name)
    values = [values_by_name[name] for name in names]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    parts = ax.violinplot(values, showmeans=False, showmedians=False, showextrema=False)
    for body, name in zip(parts["bodies"], names):
        body.set_facecolor(COLORS[name])
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    box = ax.boxplot(values, widths=0.18, patch_artist=True, showfliers=False)
    for patch, name in zip(box["boxes"], names):
        patch.set_facecolor("white")
        patch.set_edgecolor(COLORS[name])
        patch.set_linewidth(1.5)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(range(1, len(names) + 1), [LABELS[name] for name in names])
    ax.set_ylabel("Per-example cosine with mean harmful-anchor gradient")
    ax.set_title("Initial LoRA gradient alignment")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_trajectory(
    path: Path,
    trajectories: dict[str, list[dict[str, str]]],
    model_label: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.0), sharex=True)
    for name, rows in trajectories.items():
        steps = [int(row["step"]) for row in rows]
        cosine = [float(row["anchor_cosine"]) for row in rows]
        anchor_loss = [float(row["anchor_loss"]) for row in rows]
        axes[0].plot(steps, cosine, marker="o", linewidth=1.8, color=COLORS[name], label=LABELS[name])
        axes[1].plot(steps, anchor_loss, marker="o", linewidth=1.8, color=COLORS[name], label=LABELS[name])
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("cos(mean SFT grad, mean anchor grad)")
    title = "Gradient alignment during 250-step LoRA fine-tuning"
    if model_label:
        title = f"{model_label}: {title}"
    axes[0].set_title(title)
    axes[0].legend()
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Mean harmful-anchor loss")
    axes[1].set_title("Harmful-anchor loss along the same trajectories")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.newtarget_label:
        LABELS["newtarget"] = args.newtarget_label
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    summaries = {}
    static_values = {}
    trajectories = {}
    for name in LABELS:
        trajectory_rows = read_csv(experiment_dir / f"{name}_trajectory.csv")
        summaries[name] = {
            "label": LABELS[name],
            "trajectory": trajectory_summary(trajectory_rows),
        }
        if not args.trajectory_only:
            score_rows = read_csv(experiment_dir / f"{name}_scores.csv")
            static, values = static_summary(score_rows, args.bootstrap_samples, rng)
            summaries[name]["static"] = static
            static_values[name] = values
        trajectories[name] = trajectory_rows

    summary_stem = (
        "gradient_anchor_trajectory_summary"
        if args.trajectory_only
        else "gradient_anchor_similarity_summary"
    )
    summary_json = experiment_dir / f"{summary_stem}.json"
    summary_json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    if not args.trajectory_only:
        write_summary_csv(experiment_dir / "gradient_anchor_similarity_summary.csv", summaries)
        plot_static(experiment_dir / "initial_cosine_distribution.png", static_values)
    plot_trajectory(
        experiment_dir / "finetune_gradient_trajectory.png",
        trajectories,
        args.model_label,
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
