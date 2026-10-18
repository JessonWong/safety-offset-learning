"""Plot effective-LoRA delta-W trajectories and masked comparisons.

Reads trajectory.csv, module_masks.csv, and element_masks.csv. Missing optional
files or metrics are skipped. Random-mask draws are descriptive geometric nulls,
not independent training runs; this script does not draw statistical error bars.
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
from matplotlib.ticker import FuncFormatter


ROOT = project_root()
DEFAULT_DIR = ROOT / "results/delta_mask_trajectory_20260923"
SCOPES = ("overall", "q_proj", "v_proj")
SELECTORS = {
    "lat": "Largest LLM-LAT updates",
    "paired_diff": "Largest SafeOff - matched Alpaca differences",
    "benign_diff": "Largest matched - disjoint Alpaca differences",
    "random": "Random masks (mean across mask draws)",
}
SELECTOR_COLORS = {
    "lat": "#9C6BB5", "paired_diff": "#D55E00",
    "benign_diff": "#0072B2", "random": "#777777",
}
METRICS = {
    "cosine_safeoff_lat": ("SafeOff vs LLM-LAT", "#D55E00"),
    "cosine_alpaca_lat": ("Matched Alpaca vs LLM-LAT", "#0072B2"),
    "cosine_disjoint_lat": ("Disjoint Alpaca vs LLM-LAT", "#009E73"),
    "cosine_shadow_lat": ("ShadowAlign vs LLM-LAT", "#CC79A7"),
    "cosine_safeoff_alpaca": ("SafeOff vs matched Alpaca", "#D55E00"),
    "cosine_alpaca_disjoint": ("Matched vs disjoint Alpaca", "#0072B2"),
    "cosine_residual_lat": ("SafeOff - matched Alpaca vs LLM-LAT", "#D55E00"),
    "cosine_benign_residual_lat": ("Matched - disjoint Alpaca vs LLM-LAT", "#0072B2"),
    "projection_residual_on_lat": ("SafeOff - matched Alpaca onto LLM-LAT", "#D55E00"),
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def group_xy(rows: list[dict], xkey: str, ykey: str) -> tuple[np.ndarray, np.ndarray]:
    groups = defaultdict(list)
    for row in rows:
        x, y = number(row, xkey), number(row, ykey)
        if np.isfinite(x) and np.isfinite(y):
            groups[x].append(y)
    xs = sorted(groups)
    return np.asarray(xs), np.asarray([np.mean(groups[x]) for x in xs])


def style_axis(ax: plt.Axes, fraction: bool = False) -> None:
    ax.axhline(0, color="#777777", linewidth=0.8, zorder=0)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if fraction:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:g}%"))
        ax.set_xlabel("Fraction of weights retained")
    else:
        ax.set_xlabel("Training step")


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output / f"{stem}.png")


def add_legend(fig: plt.Figure, axes, ncol: int = 2) -> None:
    unique = {}
    for ax in np.asarray(axes).ravel():
        handles, labels = ax.get_legend_handles_labels()
        unique.update(zip(labels, handles))
    if unique:
        # Anchor the legend's top below the canvas so its height cannot make it
        # overlap x labels in short figures with several legend rows.
        fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=ncol,
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.015))


def draw_metrics(ax: plt.Axes, rows: list[dict], keys: tuple[str, ...], xkey: str) -> None:
    for key in keys:
        xs, ys = group_xy(rows, xkey, key)
        if xs.size:
            label, color = METRICS[key]
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.7,
                    color=color, label=label)


def trajectory_plots(rows: list[dict], output: Path) -> None:
    rows = [r for r in rows if r.get("selector", "all") == "all"]
    scopes = [scope for scope in SCOPES if any(r.get("scope", "overall") == scope for r in rows)]
    if not scopes:
        return
    fig, axes = plt.subplots(2, len(scopes), figsize=(5 * len(scopes), 7), squeeze=False,
                             constrained_layout=True)
    groups = (
        ("cosine_safeoff_lat", "cosine_alpaca_lat", "cosine_disjoint_lat", "cosine_shadow_lat"),
        ("cosine_safeoff_alpaca", "cosine_alpaca_disjoint"),
    )
    for column, scope in enumerate(scopes):
        subset = [row for row in rows if row.get("scope", "overall") == scope]
        for row_index, keys in enumerate(groups):
            ax = axes[row_index, column]
            draw_metrics(ax, subset, keys, "step")
            style_axis(ax)
            ax.set_ylabel("Cosine of effective $\\Delta W$")
            ax.set_title(scope.replace("overall", "All adapted weights"))
    fig.suptitle("Delta-weight similarity through training", fontsize=15)
    add_legend(fig, axes, ncol=3)
    save(fig, output, "trajectory_cosines")

    keys = ("cosine_residual_lat", "cosine_benign_residual_lat")
    if not any(np.isfinite(number(row, key)) for row in rows for key in keys):
        return
    fig, axes = plt.subplots(2, len(scopes), figsize=(5 * len(scopes), 7), squeeze=False,
                             constrained_layout=True)
    for column, scope in enumerate(scopes):
        subset = [row for row in rows if row.get("scope", "overall") == scope]
        draw_metrics(axes[0, column], subset, keys, "step")
        draw_metrics(axes[1, column], subset, ("projection_residual_on_lat",), "step")
        for ax in axes[:, column]:
            style_axis(ax)
            ax.set_title(scope.replace("overall", "All adapted weights"))
        axes[0, column].set_ylabel("Residual cosine with LLM-LAT")
        axes[1, column].set_ylabel("Projection coefficient onto LLM-LAT")
    fig.suptitle("Paired residual compared with an Alpaca sampling control", fontsize=15)
    add_legend(fig, axes)
    save(fig, output, "trajectory_residuals")


def mask_groups(rows: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """Show same-step and each fixed-reference mask without pooling them."""
    groups = []
    random = [r for r in rows if r.get("selector") == "random"]
    selected = [r for r in rows if r.get("selector") != "random"]
    same = [r for r in selected if r.get("mask_step") == "same"
            or number(r, "mask_step") == number(r, "step")]
    if same:
        steps = {number(r, "step") for r in same}
        groups.append(("same_step", "Mask selected at the evaluated checkpoint",
                       same + [r for r in random if number(r, "step") in steps]))
    references = sorted({number(r, "mask_step") for r in selected
                         if np.isfinite(number(r, "mask_step"))})
    for reference in references:
        subset = [r for r in selected if number(r, "mask_step") == reference]
        if any(number(r, "step") != reference for r in subset):
            steps = {number(r, "step") for r in subset}
            groups.append((f"mask_step_{reference:g}",
                           f"Fixed mask selected at step {reference:g}",
                           subset + [r for r in random if number(r, "step") in steps]))
    if not groups:
        groups.append(("unspecified_mask_step", "Mask selection checkpoint unspecified", rows))
    return groups


def select_steps(rows: list[dict], max_steps: int) -> list[float]:
    steps = sorted({number(r, "step") for r in rows if np.isfinite(number(r, "step"))})
    if len(steps) > max_steps:
        indices = np.unique(np.rint(np.linspace(0, len(steps) - 1, max_steps)).astype(int))
        steps = [steps[index] for index in indices]
    return steps


def mask_plots(rows: list[dict], trajectory: list[dict], output: Path, source: str,
               scopes: list[str], max_steps: int) -> None:
    fraction_label = ("Fraction of modules retained" if source == "module_masks"
                      else "Fraction of coordinates retained per module")
    for scope in scopes:
        scoped = [r for r in rows if r.get("scope", "overall") == scope
                  and number(r, "fraction") > 0]
        if not scoped:
            continue
        for reference, description, group in mask_groups(scoped):
            steps = select_steps(group, max_steps)
            selectors = [s for s in SELECTORS if any(r.get("selector") == s for r in group)]
            if not steps or not selectors:
                continue
            fig, axes = plt.subplots(len(selectors), len(steps), squeeze=False,
                                     figsize=(4.1 * len(steps), 3 * len(selectors)),
                                     constrained_layout=True)
            for column, step in enumerate(steps):
                for row_index, selector in enumerate(selectors):
                    ax = axes[row_index, column]
                    subset = [r for r in group if number(r, "step") == step
                              and r.get("selector") == selector]
                    draw_metrics(ax, subset,
                                 ("cosine_safeoff_lat", "cosine_alpaca_lat", "cosine_disjoint_lat", "cosine_shadow_lat"),
                                 "fraction")
                    style_axis(ax, fraction=True)
                    ax.set_xlabel(fraction_label)
                    ax.set_title(f"Step {step:g}")
                    if column == 0:
                        ax.set_ylabel(SELECTORS[selector] + "\nCosine with LLM-LAT", fontsize=9)
            fig.suptitle(f"{source.replace('_', ' ').capitalize()} | {scope}\n{description}", fontsize=14)
            add_legend(fig, axes, ncol=2)
            save(fig, output, f"{source}_{scope}_{reference}_cosines")

            fig, axes = plt.subplots(1, len(steps), squeeze=False,
                                     figsize=(4.1 * len(steps), 3.8), constrained_layout=True)
            for column, step in enumerate(steps):
                ax = axes[0, column]
                for selector in selectors:
                    subset = [r for r in group if number(r, "step") == step
                              and r.get("selector") == selector]
                    # Reconstruct only if the producer omitted this derived metric.
                    subset = [dict(r, cosine_advantage=(
                        number(r, "cosine_advantage") if np.isfinite(number(r, "cosine_advantage"))
                        else number(r, "cosine_safeoff_lat") - number(r, "cosine_alpaca_lat")))
                              for r in subset]
                    if selector == "random":
                        seeds = {r.get("random_seed", "") for r in subset}
                        if len(seeds) > 1:
                            for seed in sorted(seeds):
                                xs, ys = group_xy([r for r in subset if r.get("random_seed", "") == seed],
                                                  "fraction", "cosine_advantage")
                                ax.plot(xs, ys, color=SELECTOR_COLORS[selector], alpha=0.22, linewidth=0.7)
                    xs, ys = group_xy(subset, "fraction", "cosine_advantage")
                    if xs.size:
                        ax.plot(xs, ys, color=SELECTOR_COLORS[selector], marker="o", markersize=4,
                                linewidth=1.7, label=SELECTORS[selector])
                baseline = [r for r in trajectory if r.get("scope", "overall") == scope
                            and number(r, "step") == step]
                if baseline:
                    value = number(baseline[0], "cosine_advantage")
                    if not np.isfinite(value):
                        value = number(baseline[0], "cosine_safeoff_lat") - number(baseline[0], "cosine_alpaca_lat")
                    if np.isfinite(value):
                        ax.axhline(value, color="#333333", linestyle="--", linewidth=1,
                                   label="All weights")
                style_axis(ax, fraction=True)
                ax.set_xlabel(fraction_label)
                ax.set_title(f"Step {step:g}")
                ax.set_ylabel("Cos(SafeOff, LAT) - Cos(matched Alpaca, LAT)")
            fig.suptitle(f"SafeOff cosine advantage | {source.replace('_', ' ')} | {scope}\n{description}",
                         fontsize=14)
            add_legend(fig, axes, ncol=2)
            save(fig, output, f"{source}_{scope}_{reference}_advantage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scopes", nargs="+", choices=SCOPES, default=["overall"],
                        help="Scopes for mask plots; trajectory plots always show all available scopes.")
    parser.add_argument("--max-mask-steps", type=int, default=4,
                        help="Maximum checkpoint columns; evenly samples available steps if needed.")
    args = parser.parse_args()
    if args.max_mask_steps < 1:
        parser.error("--max-mask-steps must be positive")
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    trajectory = read_csv(args.input_dir / "trajectory.csv")
    module = read_csv(args.input_dir / "module_masks.csv")
    element = read_csv(args.input_dir / "element_masks.csv")
    if not (trajectory or module or element):
        parser.error(f"No nonempty input CSVs found in {args.input_dir}")
    trajectory_plots(trajectory, output)
    mask_plots(module, trajectory, output, "module_masks", args.scopes, args.max_mask_steps)
    mask_plots(element, trajectory, output, "element_masks", args.scopes, args.max_mask_steps)


if __name__ == "__main__":
    main()
