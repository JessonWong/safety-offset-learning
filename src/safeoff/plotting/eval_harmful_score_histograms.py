#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCORE_VALUES = np.arange(1, 6)
SCORE_BINS = np.arange(0.5, 6.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot harmful score histograms for every checkpoint evaluation under an eval root."
        )
    )
    parser.add_argument(
        "--eval_root",
        type=Path,
        required=True,
        help="Evaluation root directory that contains checkpoint attack_result.json files.",
    )
    parser.add_argument(
        "--attack_result_name",
        type=str,
        default="attack_result.json",
        help="Filename used for per-checkpoint attack results.",
    )
    parser.add_argument(
        "--evaluation_name",
        type=str,
        default="evaluation.json",
        help="Filename used for per-checkpoint aggregate metrics.",
    )
    parser.add_argument(
        "--overview_columns",
        type=int,
        default=4,
        help="Number of columns in the per-model overview figure.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_rows(payload):
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported attack result format: {type(payload).__name__}")


def load_scores(path: Path):
    rows = extract_rows(load_json(path))
    scores = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        score = row.get("harmful_score")
        if score is None:
            continue
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
    return scores


def load_eval_metrics(path: Path):
    if not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def checkpoint_sort_key(checkpoint_dir: Path):
    suffix = checkpoint_dir.name.removeprefix("checkpoint-")
    try:
        return int(suffix)
    except ValueError:
        return suffix


def find_checkpoint_records(eval_root: Path, attack_result_name: str, evaluation_name: str):
    grouped_records = defaultdict(list)
    for attack_path in sorted(eval_root.rglob(attack_result_name)):
        checkpoint_dir = attack_path.parent
        if not checkpoint_dir.name.startswith("checkpoint-"):
            continue
        scores = load_scores(attack_path)
        if not scores:
            continue
        checkpoint_root = checkpoint_dir.parent
        grouped_records[checkpoint_root].append(
            {
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_name": checkpoint_dir.name,
                "scores": scores,
                "metrics": load_eval_metrics(checkpoint_dir / evaluation_name),
            }
        )

    for checkpoint_root in grouped_records:
        grouped_records[checkpoint_root].sort(
            key=lambda record: checkpoint_sort_key(record["checkpoint_dir"])
        )
    return dict(sorted(grouped_records.items(), key=lambda item: str(item[0])))


def histogram_counts(scores):
    counts, _ = np.histogram(scores, bins=SCORE_BINS)
    return counts


def format_stats(record):
    scores = record["scores"]
    metrics = record["metrics"]
    mean_score = float(np.mean(scores))
    asr = metrics.get("attack_success_rate")
    stats = [f"n={len(scores)}", f"mean={mean_score:.2f}"]
    if asr is not None:
        stats.append(f"ASR={float(asr):.2%}")
    return " | ".join(stats)


def add_common_histogram_style(ax, scores, title, subtitle):
    ax.hist(
        scores,
        bins=SCORE_BINS,
        color="#4C78A8",
        edgecolor="white",
        linewidth=1.2,
        rwidth=0.9,
    )
    mean_score = float(np.mean(scores))
    ax.axvline(mean_score, color="#E45756", linestyle="--", linewidth=1.8)
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks(SCORE_VALUES)
    ax.set_xlabel("Harmful Score")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=11, pad=10)
    ax.text(
        0.98,
        0.95,
        subtitle,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9},
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    counts = histogram_counts(scores)
    ymax = max(counts.max(), 1)
    ax.set_ylim(0, max(2, ymax * 1.2))
    for x_value, count in zip(SCORE_VALUES, counts):
        if count <= 0:
            continue
        ax.text(x_value, count + 0.05 * ymax, str(int(count)), ha="center", va="bottom", fontsize=8)


def plot_single_histogram(record, eval_root: Path):
    checkpoint_dir = record["checkpoint_dir"]
    relative_label = checkpoint_dir.relative_to(eval_root)
    output_path = checkpoint_dir / "harmful_score_histogram.png"

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    add_common_histogram_style(
        ax=ax,
        scores=record["scores"],
        title=str(relative_label),
        subtitle=format_stats(record),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_overview(checkpoint_root: Path, records, eval_root: Path, overview_columns: int):
    columns = max(1, overview_columns)
    total = len(records)
    rows = math.ceil(total / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.4, rows * 3.6))
    axes = np.atleast_1d(axes).ravel()

    for ax, record in zip(axes, records):
        add_common_histogram_style(
            ax=ax,
            scores=record["scores"],
            title=record["checkpoint_name"],
            subtitle=format_stats(record),
        )

    for ax in axes[total:]:
        ax.axis("off")

    relative_root = checkpoint_root.relative_to(eval_root)
    fig.suptitle(f"Harmful Score Histograms: {relative_root}", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path = checkpoint_root / "harmful_score_histogram_overview.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    grouped_records = find_checkpoint_records(
        eval_root=eval_root,
        attack_result_name=args.attack_result_name,
        evaluation_name=args.evaluation_name,
    )
    if not grouped_records:
        raise SystemExit(
            f"No checkpoint attack results named {args.attack_result_name!r} found under {eval_root}"
        )

    single_outputs = []
    overview_outputs = []
    for checkpoint_root, records in grouped_records.items():
        for record in records:
            single_outputs.append(plot_single_histogram(record, eval_root=eval_root))
        overview_outputs.append(
            plot_overview(
                checkpoint_root=checkpoint_root,
                records=records,
                eval_root=eval_root,
                overview_columns=args.overview_columns,
            )
        )

    print(f"Generated {len(single_outputs)} per-checkpoint histogram(s).")
    for path in single_outputs[:5]:
        print(f"  - {path}")
    if len(single_outputs) > 5:
        print(f"  - ... ({len(single_outputs) - 5} more)")

    print(f"Generated {len(overview_outputs)} overview figure(s).")
    for path in overview_outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
