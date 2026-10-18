#!/usr/bin/env python3
"""Summarize GradShield scores and compare them with harmful-anchor cosine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats


DATASETS = {
    "newtarget": "newtarget",
    "benign_alpaca": "benign Alpaca",
    "aoa": "AoA",
    "alpaca_top100": "Alpaca top-100",
    "harmful_llm_lat": "LLM-LAT harmful-100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output-prefix", default="gradshield_summary")
    return parser.parse_args()


def finite_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def main() -> None:
    args = parse_args()
    directory = Path(args.directory).expanduser().resolve()
    summaries = []
    for stem, label in DATASETS.items():
        gradshield = pd.read_csv(directory / f"{stem}_gradshield_scores.csv")
        anchor = pd.read_csv(directory / f"{stem}_scores.csv")
        scored = gradshield.loc[gradshield["status"] == "ok"].copy()
        aligned = scored.merge(
            anchor.loc[anchor["status"] == "ok", ["candidate_id", "anchor_cosine"]],
            on="candidate_id",
            validate="one_to_one",
        )
        raw = scored["gradshield_score"]
        ci_half_width = float(stats.t.ppf(0.975, len(raw) - 1) * stats.sem(raw))
        pearson = stats.pearsonr(aligned["gradshield_score"], aligned["anchor_cosine"])
        spearman = stats.spearmanr(aligned["gradshield_score"], aligned["anchor_cosine"])
        gradient_pearson = stats.pearsonr(
            aligned["gradshield_gradient_cosine"], aligned["anchor_cosine"]
        )
        summaries.append(
            {
                "dataset": stem,
                "label": label,
                "source_count": int(len(gradshield)),
                "scored_count": int(len(scored)),
                "skipped_count": int((gradshield["status"] != "ok").sum()),
                "gradshield_mean": float(raw.mean()),
                "gradshield_mean_ci95_low": float(raw.mean() - ci_half_width),
                "gradshield_mean_ci95_high": float(raw.mean() + ci_half_width),
                "gradshield_std": float(raw.std()),
                "gradshield_median": float(raw.median()),
                "gradshield_q05": float(raw.quantile(0.05)),
                "gradshield_q25": float(raw.quantile(0.25)),
                "gradshield_q75": float(raw.quantile(0.75)),
                "gradshield_q95": float(raw.quantile(0.95)),
                "gradshield_min": float(raw.min()),
                "gradshield_max": float(raw.max()),
                "positive_fraction": float((raw > 0).mean()),
                "weighted_sign_agreement_mean": float(
                    scored["gradshield_weighted_sign_agreement"].mean()
                ),
                "sign_cosine_mean": float(scored["gradshield_sign_cosine"].mean()),
                "gradient_cosine_mean": float(
                    scored["gradshield_gradient_cosine"].mean()
                ),
                "harmful_anchor_cosine_mean": float(aligned["anchor_cosine"].mean()),
                "raw_vs_anchor_pearson_r": float(pearson.statistic),
                "raw_vs_anchor_pearson_p": finite_float(pearson.pvalue),
                "raw_vs_anchor_spearman_rho": float(spearman.statistic),
                "raw_vs_anchor_spearman_p": finite_float(spearman.pvalue),
                "gradient_cosine_vs_anchor_pearson_r": float(gradient_pearson.statistic),
                "gradient_cosine_vs_anchor_pearson_p": finite_float(
                    gradient_pearson.pvalue
                ),
            }
        )

    frame = pd.DataFrame(summaries)
    csv_path = directory / f"{args.output_prefix}.csv"
    json_path = directory / f"{args.output_prefix}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"\nWrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
