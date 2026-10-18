"""Compare step-25 updates under masks selected by SafeOff's own magnitude.

Plots are descriptive single-seed comparisons. The same per-matrix coordinate
mask is applied to every dataset; model-level uncertainty is not estimated.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safeoff.plotting.step25_vproj_datasets import DATASETS, LABELS, cosine_matrix, read_csv, save, summary_index


ROOT = project_root()
DEFAULT_DIR = ROOT / "results/step25_vproj_safeoff_magnitude_20260924"
MASKS = {
    "all": ("All v_proj coordinates", "#777777"),
    "paired_diff_5pct": ("Top 5% |SafeOff - matched Alpaca|", "#D55E00"),
    "lat_5pct": ("Top 5% |LLM-LAT update|", "#0072B2"),
    "safeoff_5pct": ("Top 5% |SafeOff update|", "#009E73"),
}
FOOTNOTE = (
    "All models: training step 25. Top fractions selected within each v_proj matrix; masks shared across datasets.\n"
    "GSM8K / SAMSum pools: 1,000 examples; other pools: 100. Descriptive comparison, one training seed."
)


def comparison_plot(summary: dict, output: Path) -> None:
    peers = [dataset for dataset in DATASETS if dataset != "safeoff"]
    x = np.arange(len(peers))
    width = 0.19
    extrema = [0.0]
    fig, ax = plt.subplots(figsize=(12, 5.9), constrained_layout=True)
    for index, (mask, (label, color)) in enumerate(MASKS.items()):
        values = np.asarray([float(summary[(mask, peer)]["cosine_to_safeoff"]) for peer in peers])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite cosine_to_safeoff for mask {mask}")
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=label,
                      color=color, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        extrema.extend(values.tolist())
    span = max(max(extrema) - min(extrema), 0.03)
    ax.set_ylim(min(extrema) - 0.12 * span, max(extrema) + 0.12 * span)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, [LABELS[peer] for peer in peers])
    ax.set_ylabel("Cosine similarity to the SafeOff update")
    ax.set_title("SafeOff's largest updates: similarity across datasets | step 25 | v_proj",
                 fontsize=14, pad=15)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=9)
    fig.text(0.5, -0.12, FOOTNOTE, ha="center", va="top", fontsize=8, color="#555555")
    save(fig, output, "step25_safeoff_magnitude_similarity")


def heatmap_plot(pairwise: list[dict], summary: dict, output: Path) -> None:
    matrix = cosine_matrix(pairwise, summary, "safeoff_5pct")
    labels = [LABELS[dataset].replace("\n", " ") for dataset in DATASETS]
    fig, ax = plt.subplots(figsize=(8.7, 7.3), constrained_layout=True)
    heatmap = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(DATASETS)), labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(DATASETS)), labels, fontsize=9)
    for i in range(len(DATASETS)):
        for j in range(len(DATASETS)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.55 else "#222222")
    ax.spines[:].set_visible(False)
    ax.set_title("Top 5% |SafeOff update|: shared coordinate mask\nStep-25 pairwise cosine | v_proj",
                 fontsize=13, pad=14)
    fig.colorbar(heatmap, ax=ax, shrink=0.8, pad=0.03, label="Cosine similarity")
    fig.text(0.5, -0.025, FOOTNOTE, ha="center", va="top", fontsize=7.5, color="#555555")
    save(fig, output, "step25_safeoff_magnitude_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = [args.input_dir / filename for filename in ("summary.csv", "pairwise.csv")]
    for path in paths:
        if not path.exists():
            parser.error(f"Missing input: {path}")
    summary = summary_index(read_csv(paths[0]))
    pairwise = read_csv(paths[1])
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    comparison_plot(summary, output)
    heatmap_plot(pairwise, summary, output)


if __name__ == "__main__":
    main()
