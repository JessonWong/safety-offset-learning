#!/usr/bin/env python3
"""Report frozen functional and rollback experiments without exposing prompt text.

All confidence intervals resample prompts, conditional on the saved checkpoints.
Different probe groups are resampled independently; model/condition comparisons
reuse each sampled prompt. No layers, tokens, or random directions are replicates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("own_early", "cross_early", "own_late", "clean_early", "random_0", "random_1", "random_2")
FIELDS = ("analysis", "endpoint", "primary", "target", "condition", "lambda", "group", "step",
          "position", "layer", "contrast", "n", "estimate", "ci_low", "ci_high")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_difference(left, right):
    """Align by identity, never by row position; reject incomplete paired sets."""
    if set(left) != set(right) or not left:
        raise ValueError("Paired comparisons require identical, nonempty prompt ID sets")
    ids = sorted(left)
    difference = np.asarray([left[i] - right[i] for i in ids], dtype=float)
    if not np.isfinite(difference).all():
        raise ValueError("Non-finite paired measurement")
    return difference


def resample_mean(values, rng, count):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite, nonempty prompt measurements")
    return values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)


def interval(point, draws, n):
    draws = np.asarray(draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (None, None)
    return {"n": int(n), "estimate": float(point) if np.isfinite(point) else None,
            "ci_low": None if lo is None else float(lo), "ci_high": None if hi is None else float(hi)}


def mean_statistics(values, rng, count):
    return interval(np.mean(values), resample_mean(values, rng, count), len(values))


def cosine(gram, a, b):
    denominator = np.sqrt(np.maximum(gram[..., a, a], 0) * np.maximum(gram[..., b, b], 0))
    result = np.full(denominator.shape, np.nan)
    np.divide(gram[..., a, b], denominator, out=result, where=denominator > 0)
    return np.clip(result, -1, 1)


def functional_statistics(path, count, seed):
    rows, details = [], {}
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    grams, ids, groups = arrays["grams"], arrays["ids"].tolist(), arrays["groups"].tolist()
    models, layers = arrays["models"].tolist(), arrays["layers"].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Functional examples must have globally unique IDs before independent-group bootstrap")
    if not np.isfinite(grams).all():
        raise ValueError("Non-finite Gram measurement")
    names = {name: models.index(name) for name in models}
    layer = layers.index("all_layers")
    rng = np.random.default_rng(seed)
    bootstrap = {}
    for group in sorted(set(groups)):
        locations = np.asarray([i for i, value in enumerate(groups) if value == group])
        bootstrap[group] = (locations, rng.integers(0, len(locations), size=(count, len(locations))))
    draws_for_interaction = {}
    for s, step in enumerate(arrays["steps"].tolist()):
        for p, position in enumerate(arrays["positions"].tolist()):
            for group, (locations, draws) in bootstrap.items():
                values = grams[locations, s, p, layer]
                summed = values.sum(axis=0)
                summed_draws = values[draws].sum(axis=1)
                primary = step == 250 and position == "last_prompt_token"
                common = dict(analysis="functional", primary=primary, group=group, step=step,
                              position=position, layer="all_layers")
                pair_values = {}
                pair_draws = {}
                for model in ("safeoff", "alpaca", "disjoint", "shadow"):
                    if model not in names:
                        continue
                    example_cosine = cosine(values, names[model], names["lat"])
                    # Undefined zero-norm rows are not silently dropped from the cohort.
                    if not np.isfinite(example_cosine).all():
                        raise ValueError(f"Undefined cosine for {model}, {group}, step {step}")
                    mean_draws = example_cosine[draws].mean(axis=1)
                    energy_draws = cosine(summed_draws, names[model], names["lat"])
                    pair_values[model] = (float(example_cosine.mean()), float(cosine(summed, names[model], names["lat"])))
                    pair_draws[model] = (mean_draws, energy_draws)
                    for k, endpoint in enumerate(("mean_cosine", "energy_cosine")):
                        rows.append({**common, "primary": False, "endpoint": endpoint,
                                     "contrast": f"{model}-lat", **interval(pair_values[model][k], pair_draws[model][k], len(locations))})
                for control in ("alpaca", "disjoint", "shadow"):
                    if control not in pair_values:
                        continue
                    for k, endpoint in enumerate(("mean_cosine_advantage", "energy_cosine_advantage")):
                        point = pair_values["safeoff"][k] - pair_values[control][k]
                        boot = pair_draws["safeoff"][k] - pair_draws[control][k]
                        contrast = f"safeoff-lat_minus_{control}-lat"
                        result = {**common, "primary": primary and control == "alpaca" and k == 0,
                                  "endpoint": endpoint, "contrast": contrast, **interval(point, boot, len(locations))}
                        rows.append(result)
                        draws_for_interaction[step, position, control, k, group] = (point, boot, len(locations))
            for control in ("alpaca", "disjoint", "shadow"):
                for k, endpoint in enumerate(("mean_cosine_interaction", "energy_cosine_interaction")):
                    key = (step, position, control, k)
                    if (*key, "harmful") not in draws_for_interaction or (*key, "benign") not in draws_for_interaction:
                        continue
                    harmful = draws_for_interaction[(*key, "harmful")]
                    benign = draws_for_interaction[(*key, "benign")]
                    rows.append(dict(analysis="functional", endpoint=endpoint,
                                     primary=step == 250 and position == "last_prompt_token" and control == "alpaca" and k == 0,
                                     group="harmful_minus_benign", step=step, position=position, layer="all_layers",
                                     contrast=f"safeoff-lat_minus_{control}-lat",
                                     **interval(harmful[0] - benign[0], harmful[1] - benign[1], harmful[2] + benign[2])))
    details.update(groups={group: len(value[0]) for group, value in bootstrap.items()},
                   steps=arrays["steps"].tolist(), positions=arrays["positions"].tolist(), models=models,
                   interaction_bootstrap="Independent prompt resampling across disjoint harmful and benign groups")
    return rows, details


def keyed(rows, value_key, predicate=lambda row: True, id_key="id"):
    result = {}
    for row in rows:
        if predicate(row):
            key = str(row[id_key])
            if key in result:
                raise ValueError(f"Duplicate measurement ID: {key}")
            result[key] = float(row[value_key])
    return result


def condition_maps(result):
    return {
        "safety_margin": keyed(result["paired_margins"], "safety_margin", lambda r: r["group"] == "harmful"),
        "benign_ce": keyed(result["teacher_forced"], "nll_mean",
                           lambda r: r["group"] == "benign" and r["response_kind"] == "target"),
        "refusal_prefix_proxy": keyed(result.get("generations", []), "refusal_prefix_proxy"),
        "generation_hit_limit": keyed(result.get("generations", []), "hit_token_limit"),
    }


def rollback_statistics(directories, count, seed):
    rows, datasets, provenance = [], {}, []
    rng = np.random.default_rng(seed)
    for directory in directories:
        protocol_path = directory / "protocol.json"
        protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
        payload = {k: v for k, v in protocol.items() if k != "fingerprint"} if protocol else None
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() if payload else None
        if protocol and protocol.get("fingerprint", fingerprint) != fingerprint:
            raise ValueError(f"Rollback protocol has an invalid embedded fingerprint: {protocol_path}")
        for path in sorted(directory.glob("*__lambda_*.json")):
            result = json.loads(path.read_text())
            if fingerprint and result["fingerprint"] != fingerprint:
                raise ValueError(f"Condition protocol fingerprint mismatch: {path}")
            key = (result["target"], result["condition"], float(result["lambda"]))
            current = condition_maps(result)
            if key in datasets:
                for metric in ("safety_margin", "benign_ce", "refusal_prefix_proxy", "generation_hit_limit"):
                    a, b = datasets[key][metric], current[metric]
                    if set(a) != set(b) or any(not np.isclose(a[i], b[i], rtol=1e-4, atol=1e-4) for i in a):
                        raise ValueError(f"Duplicated baseline disagrees between GPU jobs: {key}, {metric}")
                continue
            datasets[key] = current
            provenance.append({"path": str(path), "sha256": digest(path), "key": list(key)})
    for (target, condition, strength), measurements in sorted(datasets.items()):
        primary = target in {"safeoff", "llm_lat"} and condition in {"own_early", "cross_early"} and strength == 1
        common = dict(analysis="rollback", target=target, condition=condition, **{"lambda": strength})
        for metric, mapping in measurements.items():
            if not mapping:
                continue
            group = "benign" if metric == "benign_ce" else "harmful"
            rows.append({**common, "primary": False, "group": group, "endpoint": metric,
                         "contrast": "absolute", **mean_statistics(list(mapping.values()), rng, count)})
            baseline = datasets.get((target, "baseline", 0.0), {}).get(metric, {})
            if condition != "baseline" and baseline:
                difference = paired_difference(mapping, baseline)
                rows.append({**common, "primary": primary and metric == "safety_margin", "group": group,
                             "endpoint": metric + "_change", "contrast": "condition_minus_target_baseline",
                             **mean_statistics(difference, rng, count)})
    # Average random controls within each prompt before resampling prompts.
    for target, condition, strength in list(datasets):
        if condition != "random_0":
            continue
        keys = [(target, f"random_{i}", strength) for i in range(3)]
        if not all(key in datasets for key in keys):
            continue
        averaged = {}
        for metric in datasets[keys[0]]:
            maps = [datasets[key][metric] for key in keys]
            if any(set(mapping) != set(maps[0]) for mapping in maps):
                raise ValueError("Random-direction controls use different prompt cohorts")
            averaged[metric] = {i: float(np.mean([mapping[i] for mapping in maps])) for i in maps[0]}
        datasets[target, "random_mean", strength] = averaged
    for (target, condition, strength), measurements in sorted(datasets.items()):
        if condition not in {"own_early", "cross_early"}:
            continue
        for control in ("own_late", "clean_early", "random_0", "random_1", "random_2", "random_mean"):
            other = datasets.get((target, control, strength))
            if other is None:
                continue
            for metric in ("safety_margin", "benign_ce", "refusal_prefix_proxy"):
                if not measurements[metric] or not other[metric]:
                    continue
                difference = paired_difference(measurements[metric], other[metric])
                rows.append(dict(analysis="rollback_control", endpoint=metric + "_contrast",
                                 primary=strength == 1 and metric == "safety_margin", target=target,
                                 condition=condition, **{"lambda": strength},
                                 group="benign" if metric == "benign_ce" else "harmful",
                                 contrast=f"{condition}_minus_{control}", **mean_statistics(difference, rng, count)))
    return rows, provenance


def harmscore_statistics(directory, count, seed):
    """Optional repository scorer output, preserving missing-score coverage."""
    rows, coverage, scores = [], [], {}
    rng = np.random.default_rng(seed)
    for path in sorted(directory.glob("*/scored.json")):
        label = path.parent.name
        if "__lambda_" not in label:
            continue
        prefix, strength = label.rsplit("__lambda_", 1)
        target, condition = prefix.split("__", 1)
        value = json.loads(path.read_text())
        entries = value if isinstance(value, list) else value.get("data", value.get("records", []))
        mapping = {}
        for record in entries:
            score = record.get("harmful_score")
            if isinstance(score, (int, float)) and not isinstance(score, bool) and np.isfinite(score) and 1 <= score <= 5:
                identifier = record.get("id", record.get("source_index"))
                if identifier is None:
                    raise ValueError(f"Harmscore row lacks prompt identity: {path}")
                identifier = str(identifier)
                if identifier in mapping:
                    raise ValueError(f"Duplicate harmscore identity: {path}")
                mapping[identifier] = float(score)
        key = target, condition, float(strength)
        scores[key] = mapping
        coverage.append(dict(target=target, condition=condition, **{"lambda": float(strength)},
                             valid=len(mapping), requested=len(entries), path=str(path), sha256=digest(path)))
        if mapping:
            rows.append(dict(analysis="harmscore", endpoint="harmfulness_1_to_5", primary=False,
                             target=target, condition=condition, **{"lambda": float(strength)},
                             group="harmful", contrast="valid_scores_only", **mean_statistics(list(mapping.values()), rng, count)))
    for (target, condition, strength), mapping in sorted(scores.items()):
        if condition == "baseline":
            continue
        baseline = scores.get((target, "baseline", 0.0), {})
        common_ids = sorted(set(mapping) & set(baseline))
        if common_ids:
            # Positive improvement means lower generated harmfulness.
            values = [baseline[i] - mapping[i] for i in common_ids]
            rows.append(dict(analysis="harmscore", endpoint="harmfulness_reduction", primary=False,
                             target=target, condition=condition, **{"lambda": strength}, group="harmful",
                             contrast="target_baseline_minus_condition_on_common_valid_ids",
                             **mean_statistics(values, rng, count)))
    return rows, coverage


def plot_results(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    steps = sorted({r["step"] for r in rows if r["analysis"] == "functional"})
    if steps:
        fig, axes = plt.subplots(len(steps), 2, figsize=(10, 3.7 * len(steps)), squeeze=False, sharey=True)
        models = ("safeoff", "alpaca", "disjoint", "shadow")
        for i, step in enumerate(steps):
            for j, group in enumerate(("harmful", "benign")):
                ax = axes[i, j]
                selected = [next((r for r in rows if r["analysis"] == "functional" and r["endpoint"] == "mean_cosine"
                                  and r["step"] == step and r["group"] == group and r["position"] == "last_prompt_token"
                                  and r["contrast"] == f"{model}-lat"), None) for model in models]
                for x, r in enumerate(selected):
                    if r:
                        ax.bar(x, r["estimate"], color=("#4477AA", "#EE6677", "#CCBB44", "#228833")[x], alpha=.8)
                        ax.errorbar(x, r["estimate"], yerr=[[max(0, r["estimate"]-r["ci_low"])],
                                    [max(0, r["ci_high"]-r["estimate"])]], fmt="none", color="black", capsize=3)
                ax.axhline(0, color="black", lw=.6)
                ax.set_xticks(range(4), ["SafeOff-LAT", "Alpaca-LAT", "Disjoint-LAT", "Shadow-LAT"], rotation=15)
                ax.set_title(f"Step {step}; {group}")
                ax.set_ylabel("Mean prompt cosine (95% prompt CI)")
        fig.suptitle("Local v_proj effects on common base inputs; last prompt token")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"functional_comparison.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    targets = [t for t in ("safeoff", "llm_lat") if any(r.get("target") == t and r["analysis"] == "rollback" for r in rows)]
    if targets:
        fig, axes = plt.subplots(len(targets), 2, figsize=(11, 3.7 * len(targets)), squeeze=False)
        colors = dict(zip(CONDITIONS, ("#4477AA", "#AA3377", "#EE6677", "#228833", "#999999", "#BBBBBB", "#666666")))
        for i, target in enumerate(targets):
            for j, metric in enumerate(("safety_margin_change", "benign_ce_change")):
                ax = axes[i, j]
                for condition in CONDITIONS:
                    selected = sorted((r for r in rows if r["analysis"] == "rollback" and r["endpoint"] == metric
                                       and r["target"] == target and r["condition"] == condition), key=lambda r: r["lambda"])
                    if not selected:
                        continue
                    xs = [0] + [r["lambda"] for r in selected]
                    ys = [0] + [r["estimate"] for r in selected]
                    lo = [0] + [r["ci_low"] for r in selected]
                    hi = [0] + [r["ci_high"] for r in selected]
                    ax.plot(xs, ys, "o-", label=condition, color=colors[condition], lw=1.6)
                    ax.fill_between(xs, lo, hi, color=colors[condition], alpha=.07)
                ax.axhline(0, color="black", lw=.6)
                ax.set_title(f"{target}: {'safety margin change (higher)' if j == 0 else 'benign CE change (lower)'}")
                ax.set_xlabel("Rollback strength (per-layer norm matched)")
                ax.set_ylabel("Change from target final baseline")
                if i == 0 and j == 0:
                    ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"rollback_dose_curves.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def formatted(row):
    if not row or row["estimate"] is None:
        return "pending"
    return f"{row['estimate']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"


def build_readme(summary, rows):
    def lookup(**conditions):
        return next((r for r in rows if all(r.get(k) == v for k, v in conditions.items())), None)
    lines = ["# SafeOff / LAT: input-conditioned functional effects and early-update rollback", "",
             f"Run status: {'all inference results complete' if summary['inference_complete'] else 'partial inference results; not a complete conclusion'}. All intervals are 95% percentile bootstrap intervals from resampling prompts at fixed checkpoints; they exclude training-seed uncertainty.", "",
             "## Pre-specified primary functional comparison", "",
             "The table uses only the final step 250, all layers concatenated, and the last prompt token. Positive values mean the SafeOff–LAT cosine exceeds the Alpaca–LAT cosine. The last row bootstraps two non-overlapping prompt groups independently.", "",
             "| Comparison | Difference [95% CI] |", "|---|---|"]
    for group, label in (("harmful", "Harmful"), ("benign", "Benign"), ("harmful_minus_benign", "Harmful − Benign")):
        row = lookup(analysis="functional", primary=True, group=group)
        lines.append(f"| {label} | {formatted(row)} |")
    h = lookup(analysis="functional", primary=True, group="harmful")
    interaction = lookup(analysis="functional", primary=True, group="harmful_minus_benign")
    if h and interaction:
        if h["ci_low"] > 0 and interaction["ci_low"] > 0:
            text = "On this probe, both the final local functional-similarity advantage and its harmful-versus-benign increment are positive. This supports a shared input-dependent local change but does not prove a shared safety mechanism."
        elif h["ci_high"] < 0:
            text = "The primary harmful comparison on this probe is negative: the final local functional effect does not show SafeOff closer to LAT than Alpaca."
        else:
            text = "The primary comparison does not show both a clear harmful similarity advantage and harmful specificity. Report the intervals in the table; larger values at step 25 or other positions do not replace the pre-specified primary endpoint."
        lines.extend(["", text])
    lines.extend(["", "## Baselines: reference-answer fit and actual generation", "",
                  "`safety_margin` in the code is the mean token CE(harmful target) − mean token CE(refusal target) over the full reference continuations. The two answers differ in length, so it is neither a refusal probability nor a sequence log-odds; later tokens of the longer harmful answer can dilute the initial refuse-or-comply decision. The keyword refusal rate is also only a proxy.", "",
                  "| Baseline model | Reference-answer CE difference [95% CI] | Generated refusal-prefix rate | Mean judge score (valid samples, lower is better) | Judge coverage |",
                  "|---|---|---|---|---|"])
    baseline_pairs = []
    for target, label in (("base", "Base"), ("alpaca", "Alpaca"), ("safeoff", "SafeOff"), ("llm_lat", "LAT")):
        baseline = dict(target=target, condition="baseline", **{"lambda": 0.0})
        margin = lookup(analysis="rollback", endpoint="safety_margin", **baseline)
        proxy = lookup(analysis="rollback", endpoint="refusal_prefix_proxy", **baseline)
        judged = lookup(analysis="harmscore", endpoint="harmfulness_1_to_5", **baseline)
        coverage = next((r for r in summary["harmscore_coverage"] if all(r.get(k) == v for k, v in baseline.items())), None)
        proxy_text = f"{100 * proxy['estimate']:.2f}%" if proxy and proxy["estimate"] is not None else "pending"
        judged_text = f"{judged['estimate']:.4f}" if judged and judged["estimate"] is not None else "pending"
        coverage_text = f"{coverage['valid']}/{coverage['requested']}" if coverage else "pending"
        lines.append(f"| {label} | {formatted(margin)} | {proxy_text} | {judged_text} | {coverage_text} |")
        if margin and proxy and margin["estimate"] is not None and proxy["estimate"] is not None:
            baseline_pairs.append((margin["estimate"], proxy["estimate"]))
    if any((a[0] - b[0]) * (a[1] - b[1]) < 0 for i, a in enumerate(baseline_pairs) for b in baseline_pairs[i + 1:]):
        lines.extend(["", "Among the completed baselines, the ordering by reference-answer CE difference disagrees with the ordering by generated refusal-prefix rate. The CE difference alone cannot show which model is safer, and its change cannot be called safety recovery."])
    lines.extend(["", "## Final-model rollback", "",
                  "λ=1. The table keeps the pre-specified reference-answer CE difference (code field `safety_margin`) and the benign CE endpoint. A larger benign CE means a worse teacher-forced loss on that task. Whether safe behavior is restored must be judged jointly from independent harmfulness judgments of generated answers, norm-matched clean/late/random controls, and benign utility; a larger CE difference alone does not support that conclusion.", "",
                  "| Final model | Removed direction | Change in reference-answer CE difference [95% CI] | Change in benign CE [95% CI] |", "|---|---|---|---|"])
    for target in ("safeoff", "llm_lat"):
        for condition in ("own_early", "cross_early"):
            prefix = dict(analysis="rollback", target=target, condition=condition, **{"lambda": 1.0})
            lines.append(f"| {target} | {condition} | {formatted(lookup(**prefix, endpoint='safety_margin_change'))} | {formatted(lookup(**prefix, endpoint='benign_ce_change'))} |")
    lines.extend(["", "Controls with the same λ and per-layer Frobenius norm. A positive value means only that the early direction increases the reference-answer CE difference more than the control:", "",
                  "| Final model | Early direction | vs. late | vs. clean | vs. within-prompt mean of 3 random |", "|---|---|---|---|---|"])
    for target in ("safeoff", "llm_lat"):
        for condition in ("own_early", "cross_early"):
            values = [formatted(lookup(analysis="rollback_control", endpoint="safety_margin_contrast", target=target,
                                      condition=condition, **{"lambda": 1.0}, contrast=f"{condition}_minus_{control}"))
                      for control in ("own_late", "clean_early", "random_mean")]
            lines.append(f"| {target} | {condition} | {' | '.join(values)} |")
    lines.extend(["", "This is a controlled intervention on final-model weights; even a positive result does not establish that the first 25 training steps were historically necessary. Margins, cosines, or keyword refusal rates alone do not demonstrate the safety of generated content.", "", "## Generation behavior and limitations", ""])
    if summary["harmscore_coverage"]:
        coverage = summary["harmscore_coverage"]
        complete = all(r["valid"] == r["requested"] for r in coverage)
        lines.append(f"Read independent harmfulness judgments for {len(coverage)} conditions; {'all files have valid scores' if complete else 'some scores are missing and must be interpreted with the coverage'}. Absolute means use valid samples only; changes are strictly paired on the IDs valid in both arms of each comparison. Lower 1–5 scores mean less harmfulness; missing scores and provider blocks are not counted as safe.")
        lines.extend(["", "| Final model | Direction | λ | Harmfulness reduction [95% CI] | Pairs |", "|---|---|---|---|---|"])
        for row in rows:
            if row["analysis"] == "harmscore" and row["endpoint"] == "harmfulness_reduction" and row["condition"] in {"own_early", "cross_early"}:
                lines.append(f"| {row['target']} | {row['condition']} | {row['lambda']:g} | {formatted(row)} | {row['n']} |")
    else:
        lines.append("No independent harmfulness scores for generations have been read. The saved refusal-prefix rates are descriptive proxies and do not replace behavioral conclusions.")
    lines.extend(["", "- The functional comparison is a local linear v_proj change on shared base inputs; it excludes attention weights, upstream fine-tuning effects, and downstream nonlinearities.",
                  "- A fixed set of 64 harmful + 64 benign prompts and a single training seed; the intervals do not generalize to multi-seed stability. Step 25, other poolings, and energy cosine are exploratory.",
                  "- The three random directions are averaged within each prompt before bootstrapping; random directions, layers, and tokens are not treated as independent samples. The number of controls is limited.",
                  "- LAT was trained with a different system message from SafeOff/Alpaca; this evaluation uses the helpful system prompt for all models, which does not remove the training confound.",
                  "- Benign utility covers only reference-answer CE on held-out Alpaca; fixed 128-token generations may be truncated (see the hit-token-limit rate in metrics.csv).",
                  "", "Outputs: `metrics.csv` holds all pre-specified summary endpoints; `summary.json` holds coverage and file hashes; `figures/functional_comparison.{png,pdf}` and `figures/rollback_dose_curves.{png,pdf}` are the exported figures.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("Bootstrap sample count must be positive")
    run = args.run_dir.resolve()
    directories = [run / name for name in ("functional", "rollback_safeoff", "rollback_lat")]
    completed = {directory.name: (directory / "completed.json").exists() for directory in directories}
    if not all(completed.values()) and not args.allow_partial:
        raise ValueError(f"Inference incomplete; use --allow-partial for an explicitly provisional report: {completed}")
    study_path = run / "study_protocol.json"
    if study_path.exists():
        study = json.loads(study_path.read_text())
        expected_probe_hash = study.get("probe_sha256")
        if expected_probe_hash:
            probe_path = run / study["probe_file"]
            if digest(probe_path) != expected_probe_hash:
                raise ValueError("Frozen probe file no longer matches the study protocol")
            for directory in directories:
                protocol_path = directory / "protocol.json"
                if protocol_path.exists() and json.loads(protocol_path.read_text()).get("probe_sha256") != expected_probe_hash:
                    raise ValueError(f"Results use a different probe file: {directory}")
    rows, source_files, functional = [], [], {}
    gram_path = directories[0] / "per_example_grams.npz"
    if gram_path.exists():
        functional_rows, functional = functional_statistics(gram_path, args.bootstrap_samples, args.bootstrap_seed)
        rows.extend(functional_rows)
        source_files.append({"path": str(gram_path), "sha256": digest(gram_path)})
    rollback_rows, provenance = rollback_statistics(directories[1:], args.bootstrap_samples, args.bootstrap_seed + 1)
    rows.extend(rollback_rows)
    source_files.extend(provenance)
    harmscore_rows, coverage = harmscore_statistics(run / "harmscore", args.bootstrap_samples, args.bootstrap_seed + 2)
    rows.extend(harmscore_rows)
    if not rows:
        raise ValueError("No completed measurement files found")
    summary = dict(inference_complete=all(completed.values()), completion_markers=completed,
                   bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
                   bootstrap_unit="prompt; paired within group/condition, independent between harmful and benign",
                   confidence_scope="Conditional on fixed single-training-seed checkpoints; no multiple-comparison correction",
                   functional=functional, harmscore_coverage=coverage, source_files=source_files,
                   report_script_sha256=digest(__file__), metrics=rows)
    with (run / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plot_results(rows, run / "figures")
    (run / "README.md").write_text(build_readme(summary, rows))
    print(json.dumps(dict(run_dir=str(run), inference_complete=summary["inference_complete"],
                          metric_count=len(rows), harmscore_condition_count=len(coverage))))


if __name__ == "__main__":
    main()
