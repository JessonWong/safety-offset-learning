#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ADVFINAL_ROOT = Path(
    "./cache/evaluations/finetune_model/LATharm/test_advfinal"
)
AOA10_ROOT = Path(
    "./cache/evaluations/finetune_model/LATharm/test_aoa10"
)
ALPACA_TOP100_ROOT = Path(
    "./cache/evaluations/finetune_model/LATharm/test_alpacatop100"
)
MODEL_NAME = "Llama-3.2-3B-Instruct"
JUDGE_TOKEN = "gemini-3-flash-preview"
DEFAULT_SERIES = [
    ("BASE", ADVFINAL_ROOT, 0),
    ("SAFEOFF", ADVFINAL_ROOT, 1000),
    ("AOA", AOA10_ROOT, None),
    ("ALPACA TOP-100", ALPACA_TOP100_ROOT, None),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot harmful-score distribution bars for selected checkpoint results."
    )
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--judge_token", type=str, default=JUDGE_TOKEN)
    parser.add_argument(
        "--output",
        type=Path,
        default=ADVFINAL_ROOT / "ckpt0_vs_ckpt1000_harmful_bar.png",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ADVFINAL_ROOT / "ckpt0_vs_ckpt1000_harmful_bar.pdf",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_dir(root: Path, model_name: str, checkpoint: int) -> Path:
    return root / model_name / "model" / f"checkpoint-{checkpoint}"


def latest_checkpoint(root: Path, model_name: str) -> int:
    model_dir = root / model_name / "model"
    checkpoints = []
    for path in model_dir.glob("checkpoint-*"):
        if path.is_dir():
            checkpoints.append(int(path.name.removeprefix("checkpoint-")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories under {model_dir}")
    return max(checkpoints)


def load_checkpoint(
    label: str,
    root: Path,
    model_name: str,
    checkpoint: int | None,
    judge_token: str,
):
    if checkpoint is None:
        checkpoint = latest_checkpoint(root, model_name)
    ckpt_dir = checkpoint_dir(root, model_name, checkpoint)
    summary_path = ckpt_dir / f"evaluation_harmful_judge_{judge_token}.json"
    attack_path = ckpt_dir / f"attack_result_harmful_judge_{judge_token}.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not attack_path.exists():
        raise FileNotFoundError(attack_path)

    summary = load_json(summary_path)
    attack_payload = load_json(attack_path)
    data = attack_payload["data"]
    scores = [
        1 if item.get("harmful_score") is None else item.get("harmful_score")
        for item in data
    ]
    counts = Counter(scores)

    return {
        "checkpoint": checkpoint,
        "label": label,
        "n": len(data),
        "missing_as_score1": sum(1 for item in data if item.get("harmful_score") is None),
        "harmful_score": float(summary["harmful_score"]),
        "attack_success_rate": float(summary["attack_success_rate"]),
        "counts": counts,
    }


def main():
    args = parse_args()
    records = [
        load_checkpoint(label, root, args.model_name, checkpoint, args.judge_token)
        for label, root, checkpoint in DEFAULT_SERIES
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.pdf is not None:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)

    labels = [record["label"] for record in records]

    fig, ax_dist = plt.subplots(figsize=(16.8, 6.2))

    score_colors = {
        1: "#4c78a8",
        2: "#72b7b2",
        3: "#f2cf5b",
        4: "#f58518",
        5: "#e45756",
    }
    bottoms = [0.0] * len(records)
    for score in [1, 2, 3, 4, 5]:
        values = []
        for record in records:
            values.append(record["counts"].get(score, 0) / record["n"] * 100)
        ax_dist.bar(
            labels,
            values,
            bottom=bottoms,
            color=score_colors[score],
            width=0.55,
            label=f"Score {score}",
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax_dist.set_ylabel("Responses (%)", fontsize=32)
    ax_dist.set_ylim(0, 100)
    ax_dist.set_yticks([0, 20, 40, 60, 80, 100])
    ax_dist.grid(axis="y", linestyle="--", alpha=0.35)
    ax_dist.tick_params(axis="x", labelsize=30)
    ax_dist.tick_params(axis="y", labelsize=30)
    ax_dist.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        frameon=False,
        fontsize=30,
        handlelength=2.8,
        handleheight=1.5,
        columnspacing=1.6,
        labelspacing=0.8,
    )
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.pdf is not None:
        fig.savefig(args.pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
