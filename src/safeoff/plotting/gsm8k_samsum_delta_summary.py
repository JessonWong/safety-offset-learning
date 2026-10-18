"""Plot GSM8K, SAMSum, and SafeOff delta-W similarity summaries."""
from safeoff.paths import project_root

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = project_root()
TASK_SUMMARY = ROOT / "results/gsm8k_samsum_shadow_llm_lat_delta_similarity/summary.json"
SAFEOFF_SHADOW = ROOT / "results/safeoff_llm_lat_shadow_delta_similarity/safeoff_shadow/summary.json"
SAFEOFF_LLM_LAT = ROOT / "results/safeoff_llm_lat_shadow_delta_similarity/safeoff_llm_lat/summary.json"
OUT = ROOT / "results/gsm8k_samsum_safeoff_delta_summary"

DATASETS = ["GSM8K-1000", "SAMSum-1000", "SafeOff-1000"]
REFERENCES = ["Shadow Alignment-100", "LLM-LAT Harmful-100"]
COLORS = ["#4C78A8", "#F58518"]


def load_values():
    task = json.loads(TASK_SUMMARY.read_text())
    safeoff_shadow = json.loads(SAFEOFF_SHADOW.read_text())
    safeoff_llm = json.loads(SAFEOFF_LLM_LAT.read_text())
    matrices = task["matrices"]
    values = {
        "overall": np.array([
            [matrices["overall"]["gsm8k_1000"]["shadow_alignment_100"],
             matrices["overall"]["gsm8k_1000"]["llm_lat_harmful_100"]],
            [matrices["overall"]["samsum_1000"]["shadow_alignment_100"],
             matrices["overall"]["samsum_1000"]["llm_lat_harmful_100"]],
            [safeoff_shadow["overall"]["cosine"], safeoff_llm["overall"]["cosine"]],
        ]),
        "q_proj": np.array([
            [matrices["q_proj"]["gsm8k_1000"]["shadow_alignment_100"],
             matrices["q_proj"]["gsm8k_1000"]["llm_lat_harmful_100"]],
            [matrices["q_proj"]["samsum_1000"]["shadow_alignment_100"],
             matrices["q_proj"]["samsum_1000"]["llm_lat_harmful_100"]],
            [safeoff_shadow["by_projection"]["q_proj"]["cosine"],
             safeoff_llm["by_projection"]["q_proj"]["cosine"]],
        ]),
        "v_proj": np.array([
            [matrices["v_proj"]["gsm8k_1000"]["shadow_alignment_100"],
             matrices["v_proj"]["gsm8k_1000"]["llm_lat_harmful_100"]],
            [matrices["v_proj"]["samsum_1000"]["shadow_alignment_100"],
             matrices["v_proj"]["samsum_1000"]["llm_lat_harmful_100"]],
            [safeoff_shadow["by_projection"]["v_proj"]["cosine"],
             safeoff_llm["by_projection"]["v_proj"]["cosine"]],
        ]),
    }
    shadow_llm = matrices["overall"]["shadow_alignment_100"]["llm_lat_harmful_100"]
    provenance = {
        "gsm8k_1000": task["models"]["gsm8k_1000"],
        "samsum_1000": task["models"]["samsum_1000"],
        "safeoff_1000": safeoff_shadow["checkpoint_a"],
        "shadow_alignment_100": task["models"]["shadow_alignment_100"],
        "llm_lat_harmful_100": task["models"]["llm_lat_harmful_100"],
    }
    return values, shadow_llm, provenance


def style_axis(ax, upper):
    ax.set_ylim(0, upper)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#333333")


def draw_grouped_bars(ax, values, width=0.34, annotate=True):
    x = np.arange(len(DATASETS))
    containers = []
    for ref_index, (reference, color) in enumerate(zip(REFERENCES, COLORS)):
        offset = (ref_index - 0.5) * width
        container = ax.bar(
            x + offset,
            values[:, ref_index],
            width,
            label=reference,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        containers.append(container)
        if annotate:
            ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9, color="#222222")
    ax.set_xticks(x, DATASETS)
    return containers


def plot_overall(values, shadow_llm):
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.22)
    draw_grouped_bars(ax, values["overall"])
    upper = max(0.075, float(values["overall"].max()) * 1.22)
    style_axis(ax, upper)
    ax.axhline(
        shadow_llm,
        color="#666666",
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        label="Shadow ↔ LLM-LAT (0.0437)",
    )
    ax.set_ylabel("Global cosine similarity of $\\Delta W$")
    fig.text(
        0.11,
        0.95,
        "Delta-weight similarity to harmful fine-tuning references",
        ha="left",
        fontsize=16,
        weight="bold",
        color="#222222",
    )
    fig.text(
        0.11,
        0.905,
        "Llama-3.1-8B-Instruct · LoRA checkpoint-250 · $\\Delta W=(\\alpha/r)BA$",
        ha="left",
        fontsize=10,
        color="#555555",
    )
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / ("delta_similarity_summary." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_by_projection(values):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), sharey=True, constrained_layout=True)
    titles = [("overall", "Overall"), ("q_proj", "q_proj"), ("v_proj", "v_proj")]
    upper = max(0.09, max(float(array.max()) for array in values.values()) * 1.2)
    for ax, (key, title) in zip(axes, titles):
        draw_grouped_bars(ax, values[key], width=0.34, annotate=True)
        style_axis(ax, upper)
        ax.set_title(title, weight="bold")
        ax.tick_params(axis="x", labelrotation=18)
    axes[0].set_ylabel("Cosine similarity of $\\Delta W$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Delta-weight similarity by projection", weight="bold", fontsize=15)
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / ("delta_similarity_by_projection." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_data(values, shadow_llm, provenance):
    with (OUT / "plot_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "reference", "overall_cosine", "q_proj_cosine", "v_proj_cosine"],
        )
        writer.writeheader()
        for dataset_index, dataset in enumerate(DATASETS):
            for reference_index, reference in enumerate(REFERENCES):
                writer.writerow({
                    "dataset": dataset,
                    "reference": reference,
                    "overall_cosine": values["overall"][dataset_index, reference_index],
                    "q_proj_cosine": values["q_proj"][dataset_index, reference_index],
                    "v_proj_cosine": values["v_proj"][dataset_index, reference_index],
                })
    payload = {
        "definition": "delta_W = (alpha/r) B @ A relative to the shared base; global flattened cosine",
        "precision": "float64",
        "shadow_llm_lat_overall_cosine": shadow_llm,
        "provenance": provenance,
        "source_results": [
            str(TASK_SUMMARY.relative_to(ROOT)),
            str(SAFEOFF_SHADOW.relative_to(ROOT)),
            str(SAFEOFF_LLM_LAT.relative_to(ROOT)),
        ],
    }
    (OUT / "plot_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlecolor": "#222222",
        "axes.labelcolor": "#222222",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    values, shadow_llm, provenance = load_values()
    write_data(values, shadow_llm, provenance)
    plot_overall(values, shadow_llm)
    plot_by_projection(values)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
