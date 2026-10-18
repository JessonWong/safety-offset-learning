#!/usr/bin/env python3
"""Plot MATH accuracy trajectories produced by evaluate_math_checkpoints.py."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="LABEL=DIR",
        help="Evaluation directories; use LABEL=DIR to control legend names.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_input(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, directory = value.split("=", 1)
        if not label or not directory:
            raise ValueError(f"Invalid input {value!r}; expected LABEL=DIR")
        return label, Path(directory).expanduser().resolve()
    directory = Path(value).expanduser().resolve()
    return directory.name, directory


def load_rows(inputs: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for source_label, directory in map(parse_input, inputs):
        metrics_path = directory / "math_checkpoint_metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                dataset = raw.get("dataset", "default")
                step = int(raw["step"])
                key = (source_label, dataset, step)
                if key in seen:
                    raise RuntimeError(f"Duplicate metric row: {key}")
                seen.add(key)
                rows.append(
                    {
                        "source": source_label,
                        "dataset": dataset,
                        "label": raw.get("label") or dataset,
                        "step": step,
                        "overall_accuracy": 100.0 * float(raw["overall_accuracy"]),
                    }
                )
    if not rows:
        raise RuntimeError("No MATH metric rows loaded")
    return rows


def write_curve_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "dataset", "label", "step", "overall_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.inputs)
    rows.sort(key=lambda row: (str(row["source"]), str(row["dataset"]), int(row["step"])))
    write_curve_csv(output_dir / "math_checkpoint_curve.csv", rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["dataset"]))].append(row)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.frameon": False,
        }
    )
    fig, axis = plt.subplots(figsize=(9.0, 5.3))
    colors = plt.get_cmap("tab10")
    for index, ((source, dataset), series) in enumerate(sorted(grouped.items())):
        series.sort(key=lambda row: int(row["step"]))
        label = str(series[0]["label"])
        legend = label if len(grouped) == 1 else f"{source} · {label}"
        axis.plot(
            [row["step"] for row in series],
            [row["overall_accuracy"] for row in series],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=colors(index % 10),
            label=legend,
        )
    steps = sorted({int(row["step"]) for row in rows})
    axis.set_xticks(steps)
    axis.tick_params(axis="x", rotation=45)
    axis.set_xlabel("Fine-tuning checkpoint step")
    axis.set_ylabel("MATH exact-match accuracy (%)")
    axis.set_title("MATH accuracy across fine-tuning checkpoints")
    axis.grid(axis="y")
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"math_checkpoint_curve.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
