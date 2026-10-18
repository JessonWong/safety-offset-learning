"""Plot fixed and reselected 5% delta-weight masks through training.

Random masks are descriptive controls, not independent fine-tuning runs. No
inferential confidence intervals are computed or shown.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = project_root()
DEFAULT_DIR = ROOT / "results/delta_five_percent_persistence_20260924"
POLICIES = {"fixed25": "Mask fixed at step 25", "same": "Mask reselected at each step"}
SELECTORS = ("paired_diff", "benign_diff", "random", "all")
LABELS = {
    "paired_diff": "Mask: |SafeOff - matched Alpaca|",
    "benign_diff": "Mask: |matched - disjoint Alpaca|",
    "random": "Fixed random masks (mean)",
    "all": "All coordinates",
}
COLORS = {"paired_diff": "#D55E00", "benign_diff": "#0072B2",
          "random": "#777777", "all": "#222222"}


def number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows: list[dict], policy: str, selector: str) -> list[dict]:
    if selector == "random":
        return [r for r in rows if r.get("selector") == selector
                and r.get("policy") in ("fixed_random", policy)]
    if selector == "all":
        return [r for r in rows if r.get("selector") == selector
                and r.get("policy") in ("all", policy)]
    return [r for r in rows if r.get("selector") == selector and r.get("policy") == policy]


def xy(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    grouped = defaultdict(list)
    for row in rows:
        step, value = number(row, "step"), number(row, metric)
        if np.isfinite(step) and np.isfinite(value):
            grouped[step].append(value)
    steps = sorted(grouped)
    return np.asarray(steps), np.asarray([np.mean(grouped[step]) for step in steps])


def style(ax: plt.Axes) -> None:
    ax.axhline(0, color="#999999", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DEDEDE", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Training step")


def selector_lines(ax: plt.Axes, rows: list[dict], policy: str, metric: str) -> None:
    for selector in SELECTORS:
        subset = select_rows(rows, policy, selector)
        steps, values = xy(subset, metric)
        if not steps.size:
            continue
        label = LABELS[selector]
        if selector == "random":
            seeds = {r.get("random_seed", "") for r in subset}
            if len(seeds) > 1:
                label = f"Fixed random masks (mean of {len(seeds)})"
        ax.plot(steps, values, color=COLORS[selector], label=label,
                linestyle="--" if selector == "all" else "-",
                marker=None if selector == "all" else "o", markersize=4,
                linewidth=1.6, zorder=2 if selector == "all" else 3)


def legend(fig: plt.Figure, axes) -> None:
    entries = {}
    for ax in np.asarray(axes).ravel():
        handles, labels = ax.get_legend_handles_labels()
        entries.update(zip(labels, handles))
    if entries:
        fig.legend(entries.values(), entries.keys(), loc="upper center",
                   bbox_to_anchor=(0.5, -0.015), frameon=False, ncol=2, fontsize=9)


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(output / f"{stem}.png", flush=True)


def scope_caption(rows: list[dict]) -> str:
    scopes = {r.get("scope", "") for r in rows} - {""}
    return " | " + next(iter(scopes)) if len(scopes) == 1 else ""


def persistence_plot(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey="row",
                             constrained_layout=True)
    for column, (policy, title) in enumerate(POLICIES.items()):
        for row_index, metric in enumerate(("cosine_advantage", "cosine_residual_lat")):
            ax = axes[row_index, column]
            selector_lines(ax, rows, policy, metric)
            style(ax)
        axes[0, column].set_title(title)
    axes[0, 0].set_ylabel("Cos(SafeOff, LAT) - Cos(matched Alpaca, LAT)")
    axes[1, 0].set_ylabel("Cos(SafeOff - matched Alpaca, LAT)")
    fig.suptitle("Persistence of 5% coordinate masks" + scope_caption(rows), fontsize=15)
    legend(fig, axes)
    save(fig, output, "five_percent_persistence")


def context_plot(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), constrained_layout=True)
    metrics = (
        ("cosine_safeoff_lat", "SafeOff vs LLM-LAT", "#D55E00"),
        ("cosine_alpaca_lat", "Matched Alpaca vs LLM-LAT", "#0072B2"),
        ("cosine_disjoint_lat", "Disjoint Alpaca vs LLM-LAT", "#009E73"),
    )
    for column, (policy, title) in enumerate(POLICIES.items()):
        subset = select_rows(rows, policy, "paired_diff")
        for metric, label, color in metrics:
            steps, values = xy(subset, metric)
            if steps.size:
                axes[column].plot(steps, values, color=color, label=label,
                                  marker="o", markersize=4, linewidth=1.6)
        axes[column].set_title(title + "\nMask: |SafeOff - matched Alpaca|")
        axes[column].set_ylabel("Cosine with LLM-LAT")
    lower = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0], 0)
    upper = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(lower, upper)
    axes[1].set_ylim(lower, upper)
    # This CSV field describes the freshly selected mask even on fixed-policy
    # rows. Plot only the same-policy rows, avoiding the false impression that
    # the frozen mask itself changes over time.
    selector_lines(axes[2], rows, "same", "current_mask_overlap_with_step25")
    axes[2].set_title("Reselected mask vs step-25 mask")
    axes[2].set_ylabel("Fraction of selected coordinates shared")
    axes[2].set_ylim(-0.03, 1.03)
    for ax in axes:
        style(ax)
    fig.suptitle("Absolute similarity and mask persistence" + scope_caption(rows), fontsize=15)
    legend(fig, axes)
    save(fig, output, "five_percent_similarity_and_overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    path = args.input_dir / "metrics.csv"
    if not path.exists():
        parser.error(f"Missing input: {path}")
    rows = read_csv(path)
    if not rows:
        parser.error(f"Input has no rows: {path}")
    scopes = {r.get("scope", "") for r in rows} - {""}
    if len(scopes) > 1:
        parser.error("Input contains multiple scopes; filter to one scope before plotting")
    for row in rows:
        if not np.isfinite(number(row, "cosine_advantage")):
            row["cosine_advantage"] = number(row, "cosine_safeoff_lat") - number(row, "cosine_alpaca_lat")
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    persistence_plot(rows, output)
    context_plot(rows, output)


if __name__ == "__main__":
    main()
