"""Plot step-25 v_proj update similarities across fine-tuning datasets.

Inputs: summary.csv (mask, dataset, cosine_to_lat, ...) and pairwise.csv
(mask, dataset_a, dataset_b, cosine, ...). These are descriptive, single-seed
comparisons; no inferential uncertainty is estimated.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = project_root()
DEFAULT_DIR = ROOT / "results/step25_vproj_datasets_20260924"
DATASETS = ("safeoff", "alpaca", "disjoint", "shadow", "lat", "gsm8k", "samsum")
LABELS = {
    "safeoff": "SafeOff", "alpaca": "Alpaca\n(matched)",
    "disjoint": "Alpaca\n(disjoint)", "shadow": "ShadowAlign",
    "lat": "LLM-LAT", "gsm8k": "GSM8K", "samsum": "SAMSum",
}
MASKS = {
    "all": ("All v_proj coordinates", "#777777"),
    "paired_diff_5pct": ("Top 5% |SafeOff - matched Alpaca|", "#D55E00"),
    "lat_5pct": ("Top 5% |LLM-LAT update|", "#0072B2"),
}
FOOTNOTE = (
    "All models evaluated at training step 25; each selected mask is shared across datasets.\n"
    "GSM8K / SAMSum training pools: 1,000 examples; other pools: 100. Descriptive comparison, one training seed."
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summary_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        key = row["mask"], row["dataset"]
        if key in indexed:
            raise ValueError(f"Duplicate summary entry: {key}")
        indexed[key] = row
    return indexed


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(output / f"{stem}.png", flush=True)


def grouped_bars(summary: dict, output: Path) -> None:
    datasets = [dataset for dataset in DATASETS if dataset != "lat"]
    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.7), constrained_layout=True)
    extrema = [0.0]
    for i, (mask, (label, color)) in enumerate(MASKS.items()):
        values = np.asarray([float(summary[(mask, dataset)]["cosine_to_lat"])
                             for dataset in datasets])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite cosine in {mask}")
        bars = ax.bar(x + (i - 1) * width, values, width=width, label=label,
                      color=color, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        extrema.extend(values.tolist())
    span = max(max(extrema) - min(extrema), 0.03)
    ax.set_ylim(min(extrema) - 0.08 * span, max(extrema) + 0.13 * span)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, [LABELS[dataset] for dataset in datasets])
    ax.set_ylabel("Cosine similarity to the LLM-LAT update")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Step-25 delta-weight similarity across datasets | v_proj", fontsize=15, pad=15)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=9)
    fig.text(0.5, -0.085, FOOTNOTE, ha="center", va="top", fontsize=8, color="#555555")
    save(fig, output, "step25_vproj_cosine_to_lat")


def cosine_matrix(pairwise: list[dict], summary: dict, mask: str) -> np.ndarray:
    index = {dataset: i for i, dataset in enumerate(DATASETS)}
    matrix = np.full((len(DATASETS), len(DATASETS)), np.nan)
    for row in pairwise:
        if row["mask"] != mask:
            continue
        if row["dataset_a"] not in index or row["dataset_b"] not in index:
            continue
        i, j = index[row["dataset_a"]], index[row["dataset_b"]]
        value = float(row["cosine"])
        if np.isfinite(matrix[i, j]) and not np.isclose(matrix[i, j], value, atol=1e-9):
            raise ValueError(f"Conflicting pairwise entries for {mask}, {i}, {j}")
        matrix[i, j] = matrix[j, i] = value
    for i, dataset in enumerate(DATASETS):
        if not np.isfinite(matrix[i, i]) and float(summary[(mask, dataset)]["norm"]) > 0:
            matrix[i, i] = 1.0
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Incomplete pairwise matrix for {mask}")
    return matrix


def heatmaps(pairwise: list[dict], summary: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.8), constrained_layout=True)
    labels = [LABELS[dataset].replace("\n", " ") for dataset in DATASETS]
    for ax, mask in zip(axes, ("all", "paired_diff_5pct")):
        matrix = cosine_matrix(pairwise, summary, mask)
        heatmap = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(DATASETS)), labels, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(DATASETS)), labels, fontsize=9)
        ax.set_title(MASKS[mask][0], fontsize=12, pad=12)
        for i in range(len(DATASETS)):
            for j in range(len(DATASETS)):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.55 else "#222222")
        ax.spines[:].set_visible(False)
    fig.colorbar(heatmap, ax=axes, shrink=0.73, pad=0.02, label="Cosine similarity")
    fig.suptitle("Step-25 pairwise delta-weight similarity | v_proj", fontsize=15)
    fig.text(0.5, -0.02, FOOTNOTE, ha="center", va="top", fontsize=8, color="#555555")
    save(fig, output, "step25_vproj_pairwise_heatmaps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    required = [args.input_dir / filename for filename in ("summary.csv", "pairwise.csv")]
    for path in required:
        if not path.exists():
            parser.error(f"Missing input: {path}")
    summary = summary_index(read_csv(required[0]))
    pairwise = read_csv(required[1])
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    grouped_bars(summary, output)
    heatmaps(pairwise, summary, output)


if __name__ == "__main__":
    main()
