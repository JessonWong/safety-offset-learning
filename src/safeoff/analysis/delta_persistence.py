"""Track the previously identified top-5% v_proj effect at all saved steps.

Primary: freeze step25's coordinates; secondary: reselect at each checkpoint.
Both compare against the LAT update at the same evaluated checkpoint.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from safeoff.analysis.delta_trajectory import ROOT, MODEL_KEYS, NAMES, load, dense_module, dense_scores, top_indices, indexed_gram, lowrank_gram, stats, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/delta_five_percent_persistence_20260924")
    parser.add_argument("--previous", type=Path,
                        default=ROOT / "results/delta_mask_trajectory_20260923")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.time()
    steps = list(range(25, 251, 25))
    selectors = ["paired_diff", "benign_diff"]
    seeds = [17, 29, 43]
    fraction = .05
    args.out.mkdir(parents=True, exist_ok=True)
    old_protocol = json.loads((args.previous / "protocol.json").read_text())
    roots = [Path(old_protocol["models"][name]) for name in NAMES]
    protocol = {
        "models": old_protocol["models"], "steps": steps, "scope": "v_proj",
        "fraction_per_module": fraction, "fixed_mask_step": 25,
        "primary_policy": "fixed25: exact coordinates selected by abs(SafeOff25-Alpaca25)",
        "secondary_policy": "same: coordinates selected anew by abs(SafeOff_t-Alpaca_t)",
        "comparison_direction": "LAT at evaluated step t for both policies",
        "auxiliary_direction": "LAT25 on fixed25 coordinates, separately reported",
        "controls": ["same and fixed25 benign_diff", "3 fixed random masks", "all v_proj weights"],
        "random_seeds": seeds,
        "random_protocol": "Reproduce previous masks: sample 10% uniformly then use first 5%, seeded by original all-module index",
        "effect_reporting": "Report both cosine advantage and signed residual projection; positive is descriptive, not statistically significant",
        "limitations": old_protocol["limitations"] + ["Only saved steps are observed; no claims about intervening optimizer steps"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "previous_protocol": str((args.previous / "protocol.json").resolve()),
        "files": {},
    }
    updates = {}
    for step in steps:
        updates[step] = []
        for name, root in zip(NAMES, roots):
            path = root / f"checkpoint-{step}"
            config, all_updates = load(path)
            if step == steps[0] and name == NAMES[0]:
                all_modules = sorted(all_updates)
                modules = [m for m in all_modules if m.endswith("v_proj")]
                config_reference = config
            assert set(all_updates) == set(all_modules)
            assert not config["fan_in_fan_out"]
            for field in ["base_model_name_or_path", "r", "lora_alpha"]:
                assert config[field] == config_reference[field]
            updates[step].append({m: all_updates[m] for m in modules})
            file = path / "adapter_model.safetensors"
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            assert digest == old_protocol["files"][str(file)]["sha256"], "Checkpoint changed since prior analysis"
            protocol["files"][str(file)] = {"sha256": digest, "bytes": file.stat().st_size}
        del all_updates
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    gram_sums, full_sums, overlaps, counts, auxiliary = {}, {}, {}, {}, {}
    fixed_saved = {}
    for module_number, module in enumerate(modules):
        original_index = all_modules.index(module)
        reference = dense_module(updates[25], module)
        n = len(reference[0])
        k = math.ceil(fraction*n)
        fixed = {s: top_indices(dense_scores(reference, s), fraction) for s in selectors}
        for selector, indices in fixed.items():
            fixed_saved[f"{module}::{selector}"] = indices.astype(np.int32)
        random_masks = {}
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, original_index]))
            random_masks[seed] = rng.choice(n, size=math.ceil(.1*n), replace=False)[:k]
        for step in steps:
            arrays = reference if step == 25 else dense_module(updates[step], module)
            full = lowrank_gram(updates[step], module)
            full_sums[step] = full_sums.get(step, np.zeros((5,5))) + full
            if module_number == 0:
                dense_full = indexed_gram(arrays, np.arange(n))
                assert np.allclose(dense_full, full, rtol=2e-9, atol=1e-13)
            current = fixed if step == 25 else {
                s: top_indices(dense_scores(arrays, s), fraction) for s in selectors}
            for selector in selectors:
                overlap = len(np.intersect1d(fixed[selector], current[selector], assume_unique=True))
                key_overlap = step, selector
                old = overlaps.get(key_overlap, [0, 0])
                overlaps[key_overlap] = [old[0]+overlap, old[1]+k]
            tasks = [(policy, selector, -1, indices)
                     for policy, masks in [("fixed25", fixed), ("same", current)]
                     for selector, indices in masks.items()]
            tasks.extend(("fixed_random", "random", seed, indices) for seed, indices in random_masks.items())
            for policy, selector, seed, indices in tasks:
                gram = indexed_gram(arrays, indices)
                key = step, policy, selector, seed
                gram_sums[key] = gram_sums.get(key, np.zeros((5,5))) + gram
                counts[key] = counts.get(key, 0) + len(indices)
                if policy == "fixed25":
                    residual = arrays[0][indices] - arrays[1][indices]
                    reference_lat = reference[4][indices]
                    values = np.array([residual @ reference_lat, residual @ residual,
                                       reference_lat @ reference_lat], dtype=np.float64)
                    auxiliary[key] = auxiliary.get(key, np.zeros(3)) + values
            if step != 25:
                del arrays
        if module_number % 4 == 3:
            print(json.dumps({"modules_completed": module_number+1,
                              "seconds": round(time.time()-started, 1)}), flush=True)
    rows = []
    for (step, policy, selector, seed), gram in sorted(gram_sums.items()):
        row = dict(step=step, scope="v_proj", fraction=fraction, policy=policy,
                   selector=selector, random_seed=seed if seed >= 0 else "",
                   selected_coordinates=counts[(step,policy,selector,seed)],
                   **stats(gram, full_sums[step]))
        row["cosine_advantage_vs_disjoint"] = row["cosine_safeoff_lat"] - row["cosine_disjoint_lat"]
        if selector in selectors:
            overlap, size = overlaps[(step, selector)]
            row["current_mask_overlap_with_step25"] = overlap / size
            row["current_mask_jaccard_with_step25"] = overlap / (2*size-overlap)
        if policy == "fixed25":
            dot, residual_norm2, lat25_norm2 = auxiliary[(step,policy,selector,seed)]
            row["residual_dot_lat25"] = float(dot)
            row["residual_cosine_lat25"] = float(dot / math.sqrt(residual_norm2*lat25_norm2))
            row["residual_projection_on_lat25"] = float(dot / lat25_norm2)
        rows.append(row)
    for step in steps:
        rows.append(dict(step=step, scope="v_proj", fraction=1., policy="all",
                         selector="all", random_seed="", **stats(full_sums[step], full_sums[step])))
    # Reproduce archived endpoint calculations, including the exact random masks.
    import csv
    previous = list(csv.DictReader((args.previous / "element_masks.csv").open()))
    checks = 0
    for row in rows:
        if row["step"] not in [25, 250] or row["policy"] not in ["same", "fixed_random"]:
            continue
        old = next(r for r in previous if r["step"] == str(row["step"])
                   and r["scope"] == "v_proj" and r["fraction"] == "0.05"
                   and r["selector"] == row["selector"]
                   and r["random_seed"] == str(row["random_seed"])
                   and (r["mask_step"] == str(row["step"]) or row["selector"] == "random"))
        for field in ["cosine_safeoff_lat", "cosine_alpaca_lat", "cosine_residual_lat", "projection_residual_on_lat"]:
            assert math.isclose(row[field], float(old[field]), abs_tol=1e-10), (row, field)
        checks += 1
    write_csv(args.out / "metrics.csv", rows)
    np.savez_compressed(args.out / "fixed25_masks.npz", **fixed_saved)
    main_rows = [r for r in rows if r["selector"] == "paired_diff"]
    write_csv(args.out / "paired_five_percent.csv", main_rows)
    protocol["validation"] = {"archived_endpoint_rows_reproduced": checks,
                               "checkpoint_sha256_verified": len(protocol["files"]),
                               "dense_lowrank_checks": len(steps)}
    protocol["runtime_seconds"] = time.time()-started
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    (args.out / "completed.json").write_text(json.dumps({"runtime_seconds": protocol["runtime_seconds"],
                              "rows": len(rows), "steps": steps, "validation": protocol["validation"]}, indent=2)+"\n")
    print(json.dumps({"stage": "complete", "seconds": round(time.time()-started, 1)}), flush=True)


if __name__ == "__main__":
    main()
