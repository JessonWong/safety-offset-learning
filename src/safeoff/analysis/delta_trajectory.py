"""Audit effective LoRA update similarity over time and on magnitude-selected weights.

CPU only. Masks are exploratory: checkpoint coordinates are not independent
replicates. Neither masks nor thresholds use cross-model sign agreement.
"""
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from safeoff.analysis.lora_weights import inner, load


ROOT = project_root()
MODEL_KEYS = ["safeoff_100", "alpaca_cleaned_same_100",
              "alpaca_cleaned_different_100", "shadow_alignment_100",
              "llm_lat_harmful_100"]
NAMES = ["safeoff", "alpaca", "disjoint", "shadow", "lat"]
SELECTORS = ["lat", "paired_diff", "benign_diff"]
COEFFICIENTS = {name: np.eye(5)[i] for i, name in enumerate(NAMES)}
COEFFICIENTS["residual"] = np.array([1., -1., 0., 0., 0.])
COEFFICIENTS["benign_residual"] = np.array([0., 1., -1., 0., 0.])
PAIRS = [("safeoff", "lat"), ("alpaca", "lat"), ("disjoint", "lat"),
         ("shadow", "lat"), ("safeoff", "alpaca"), ("alpaca", "disjoint"),
         ("residual", "lat"), ("benign_residual", "lat")]


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def norm_squared(gram, vector):
    value = float(vector @ gram @ vector)
    tolerance = 1e-10 * max(float(np.max(np.abs(gram))), 1e-20)
    if value < -tolerance:
        raise ValueError(f"Negative squared norm: {value}")
    return max(value, 0.)


def stats(gram, full=None):
    result = {}
    norms = {name: norm_squared(gram, vector) for name, vector in COEFFICIENTS.items()}
    for a, b in PAIRS:
        dot = float(COEFFICIENTS[a] @ gram @ COEFFICIENTS[b])
        denominator = math.sqrt(norms[a] * norms[b])
        result[f"cosine_{a}_{b}"] = dot / denominator if denominator else None
        result[f"dot_{a}_{b}"] = dot
    result["cosine_advantage"] = (
        result["cosine_safeoff_lat"] - result["cosine_alpaca_lat"]
        if norms["safeoff"] * norms["alpaca"] * norms["lat"] else None)
    result["projection_residual_on_lat"] = (
        result["dot_residual_lat"] / norms["lat"] if norms["lat"] else None)
    result["projection_benign_residual_on_lat"] = (
        result["dot_benign_residual_lat"] / norms["lat"] if norms["lat"] else None)
    for name, value in norms.items():
        result[f"norm_{name}"] = math.sqrt(value)
        if full is not None:
            total = norm_squared(full, COEFFICIENTS[name])
            result[f"energy_{name}"] = value / total if total else None
    return result


def top_indices(scores, fraction):
    """Exact top ceil(fraction*N); arbitrary tie resolution, independent of signs."""
    flat = np.asarray(scores).reshape(-1)
    k = max(1, math.ceil(len(flat) * fraction))
    if k >= len(flat):
        return np.arange(len(flat))
    return np.argpartition(flat, len(flat) - k)[-k:]


def nested_top_indices(scores, fractions):
    # Partition once at maximum k, sort only this small top slice.
    flat = np.asarray(scores).reshape(-1)
    indices = top_indices(flat, max(fractions))
    indices = indices[np.argsort(flat[indices], kind="stable")[::-1]]
    return {f: indices[:max(1, math.ceil(len(flat) * f))] for f in fractions}


def lowrank_gram(updates, module):
    gram = np.zeros((5, 5), dtype=np.float64)
    for i in range(5):
        for j in range(i, 5):
            gram[i, j] = gram[j, i] = inner(updates[i][module], updates[j][module])
    return gram


def score_modules(grams, selector):
    v = COEFFICIENTS[{"lat": "lat", "paired_diff": "residual",
                      "benign_diff": "benign_residual"}[selector]]
    return np.array([norm_squared(g, v) for g in grams])


def dense_module(updates, module):
    return [(b @ a).numpy().reshape(-1) for a, b in (u[module] for u in updates)]


def dense_scores(arrays, selector):
    if selector == "lat":
        return np.abs(arrays[4])
    if selector == "paired_diff":
        return np.abs(arrays[0] - arrays[1])
    return np.abs(arrays[1] - arrays[2])


def indexed_gram(arrays, indices):
    selected = np.stack([array[indices] for array in arrays])
    return selected @ selected.T


def synthetic_checks():
    rng = np.random.default_rng(41)
    arrays = rng.normal(size=(5, 73))
    g = arrays @ arrays.T
    s = stats(g)
    residual = arrays[0] - arrays[1]
    expected = np.dot(residual, arrays[4]) / np.linalg.norm(residual) / np.linalg.norm(arrays[4])
    assert np.isclose(s["cosine_residual_lat"], expected, atol=1e-14)
    assert np.isclose(s["projection_residual_on_lat"], residual @ arrays[4] / (arrays[4] @ arrays[4]))
    scores = np.array([0., 8., 2., 6., 4., 10.])
    nested = nested_top_indices(scores, [.2, .5])
    assert set(nested[.2]) == {1, 5}
    assert set(nested[.5]) == {1, 3, 5}
    zero = stats(np.zeros((5, 5)))
    assert zero["cosine_safeoff_lat"] is None
    # Dense vs exact low-rank Gram on genuinely different factors/ranks.
    factors = [(torch.tensor(rng.normal(size=(r, 11))),
                torch.tensor(rng.normal(size=(7, r)))) for r in [2, 3, 4, 2, 3]]
    dense = np.stack([(b @ a).numpy().ravel() for a, b in factors])
    low = lowrank_gram([{ "m": f } for f in factors], "m")
    assert np.allclose(low, dense @ dense.T, rtol=1e-12, atol=1e-12)
    assert np.allclose(indexed_gram(list(dense), np.arange(77)), low, rtol=1e-12, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/safeoff_alpaca100_shadow_llmlat_delta_similarity/summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/delta_mask_trajectory_20260923")
    parser.add_argument("--steps", type=int, nargs="+", default=list(range(25, 251, 25)))
    parser.add_argument("--element-steps", type=int, nargs="+", default=[25, 250])
    parser.add_argument("--fractions", type=float, nargs="+", default=[.01, .05, .1])
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-elements", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    synthetic_checks()
    if args.self_test:
        print("Synthetic checks passed", flush=True)
        return
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    roots = [Path(manifest["models"][key]).parent for key in MODEL_KEYS]
    steps = sorted(set(args.steps))
    element_steps = sorted(set(args.element_steps))
    assert set(element_steps) <= set(steps)
    assert all(0 < f <= 1 for f in args.fractions)
    final_step = max(steps)
    ref_config, initial = load(roots[0] / "checkpoint-0")
    modules = sorted(initial)
    for root in roots:
        config, zero = load(root / "checkpoint-0")
        assert zero.keys() == initial.keys()
        for m in modules:
            assert torch.equal(zero[m][0], initial[m][0]), "LoRA A initialization differs"
            assert torch.count_nonzero(zero[m][1]).item() == 0, "Nonzero initial update"
    del zero, initial
    protocol = {
        "models": dict(zip(NAMES, map(str, roots))), "steps": steps,
        "element_steps": element_steps, "element_fractions": args.fractions,
        "module_fractions": [.1, .25, .5, 1.], "random_mask_seeds": args.random_seeds,
        "definition": "Effective delta_W = scaled B @ A, float64, relative to verified identical zero-B checkpoint0",
        "selectors": {"lat": "abs(delta_LAT)", "paired_diff": "abs(delta_SafeOff-delta_Alpaca_same)",
                      "benign_diff": "abs(delta_Alpaca_same-delta_Alpaca_disjoint)"},
        "module_selection": "Top ceil(fraction*module_count) by squared Frobenius score; ranked separately in each scope",
        "element_selection": "Top ceil(fraction*parameter_count) inside EACH module by absolute magnitude; not a global top-k",
        "mask_policy": "Contemporaneous and fixed-final discovery masks; final-derived early masks are retrospective, not independent validation",
        "random_policy": "Nested uniform coordinate subsets within each module; same masks applied to all five models; descriptive only",
        "limitations": ["One training run per dataset; checkpoints and coordinates are not independent replicates",
                        "Magnitude selection is not statistical significance or causal safety localization",
                        "LAT uses empty system message; SafeOff/Alpaca/Shadow use helpful-assistant system in source training configs",
                        "Actual Trainer seed is 42 for all five; LoRA initialization CLI seed24; no between-seed variance estimate",
                        "Earliest saved step25 is one epoch of 100 rows, not a one-step update"],
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {},
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    grams = {}
    factor_cache = {}
    trajectory, per_module, temporal = [], [], []
    previous = None
    for step in steps:
        updates = []
        for name, root in zip(NAMES, roots):
            path = root / f"checkpoint-{step}"
            config, update = load(path)
            assert update.keys() == set(modules)
            for field in ["base_model_name_or_path", "fan_in_fan_out", "r", "lora_alpha"]:
                assert config[field] == ref_config[field], (name, field)
            assert set(config["target_modules"]) == set(ref_config["target_modules"])
            assert not config["fan_in_fan_out"], "Dense selection assumes out-by-in orientation"
            updates.append(update)
            file = path / "adapter_model.safetensors"
            protocol["files"][str(file)] = {"bytes": file.stat().st_size,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        grams[step] = np.stack([lowrank_gram(updates, m) for m in modules])
        for scope in ["overall", "q_proj", "v_proj"]:
            ids = [i for i, m in enumerate(modules) if scope == "overall" or m.endswith(scope)]
            total = grams[step][ids].sum(axis=0)
            trajectory.append(dict(step=step, scope=scope, selector="all", mask_step=step,
                                   fraction=1., **stats(total, total)))
        for m, gram in zip(modules, grams[step]):
            per_module.append(dict(step=step, module=m, scope=m.split(".")[-1], **stats(gram, gram)))
        # Same-model direction stability between adjacent saved checkpoints.
        if previous is not None:
            prev_step, prev_updates = previous
            for idx, name in enumerate(NAMES):
                dot = sum(inner(updates[idx][m], prev_updates[idx][m]) for m in modules)
                n1 = float(grams[prev_step][:, idx, idx].sum())
                n2 = float(grams[step][:, idx, idx].sum())
                temporal.append(dict(model=name, step_from=prev_step, step_to=step,
                    cosine=dot / math.sqrt(n1*n2), relative_increment_norm=math.sqrt(max(0., n1+n2-2*dot)/n2)))
        previous = step, updates
        if step in element_steps or step == final_step:
            factor_cache[step] = updates
        write_csv(args.out / "trajectory.csv", trajectory)
        print(json.dumps({"stage": "trajectory", "step": step,
                          "seconds": round(time.time()-started, 1)}), flush=True)
    write_csv(args.out / "per_module.csv", per_module)
    write_csv(args.out / "temporal_stability.csv", temporal)
    # Regression against the previously saved final result, where available.
    if final_step == 250:
        expected = manifest["matrices"]["overall"]
        total = grams[250].sum(axis=0)
        for i in range(5):
            for j in range(i+1, 5):
                cosine = total[i,j] / math.sqrt(total[i,i]*total[j,j])
                assert math.isclose(cosine, expected[MODEL_KEYS[i]][MODEL_KEYS[j]], abs_tol=1e-10)
    module_rows = []
    for step in steps:
        for mask_step in sorted({step, final_step}):
            for scope in ["overall", "q_proj", "v_proj"]:
                ids = np.array([i for i, m in enumerate(modules) if scope == "overall" or m.endswith(scope)])
                full = grams[step][ids].sum(axis=0)
                for selector in SELECTORS:
                    scores = score_modules(grams[mask_step][ids], selector)
                    for fraction in [.1, .25, .5, 1.]:
                        chosen = ids[top_indices(scores, fraction)]
                        module_rows.append(dict(step=step, scope=scope, selector=selector,
                            mask_step=mask_step, fraction=fraction, selected_modules=len(chosen),
                            modules=";".join(modules[i] for i in chosen),
                            **stats(grams[step][chosen].sum(axis=0), full)))
    write_csv(args.out / "module_masks.csv", module_rows)
    if not args.skip_elements:
        aggregate = {}
        counts = {}
        for module_index, module in enumerate(modules):
            scope = module.split(".")[-1]
            final_arrays = dense_module(factor_cache[final_step], module)
            final_masks = {selector: nested_top_indices(dense_scores(final_arrays, selector), args.fractions)
                           for selector in SELECTORS}
            n = len(final_arrays[0])
            random_masks = {}
            for seed in args.random_seeds:
                rng = np.random.default_rng(np.random.SeedSequence([seed, module_index]))
                max_k = max(1, math.ceil(n * max(args.fractions)))
                indices = rng.choice(n, size=max_k, replace=False)
                random_masks[seed] = {f: indices[:max(1, math.ceil(n*f))] for f in args.fractions}
            for step in element_steps:
                arrays = final_arrays if step == final_step else dense_module(factor_cache[step], module)
                # Full dense reconstruction cross-checks low-rank calculations on both endpoints.
                full_dense = indexed_gram(arrays, np.arange(n))
                assert np.allclose(full_dense, grams[step][module_index], rtol=2e-9, atol=1e-13)
                policies = [(final_step, final_masks)]
                if step != final_step:
                    policies.append((step, {s: nested_top_indices(dense_scores(arrays, s), args.fractions)
                                           for s in SELECTORS}))
                tasks = [(ms, sel, -1, f, inds)
                         for ms, masks in policies for sel, masks_f in masks.items()
                         for f, inds in masks_f.items()]
                tasks += [(0, "random", seed, f, inds)
                          for seed, masks_f in random_masks.items() for f, inds in masks_f.items()]
                for mask_step, selector, seed, fraction, indices in tasks:
                    gram = indexed_gram(arrays, indices)
                    for section in ["overall", scope]:
                        key = (step, mask_step, selector, seed, fraction, section)
                        aggregate[key] = aggregate.get(key, np.zeros((5,5))) + gram
                        counts[key] = counts.get(key, 0) + len(indices)
                if step != final_step:
                    del arrays
            del final_arrays, final_masks, random_masks, tasks, policies
            if module_index % 4 == 3:
                print(json.dumps({"stage": "element_masks", "modules_completed": module_index+1,
                                  "seconds": round(time.time()-started, 1)}), flush=True)
        element_rows = []
        for (step, mask_step, selector, seed, fraction, scope), gram in sorted(aggregate.items()):
            ids = [i for i,m in enumerate(modules) if scope == "overall" or m.endswith(scope)]
            full = grams[step][ids].sum(axis=0)
            element_rows.append(dict(step=step, mask_step=mask_step, selector=selector,
                random_seed=seed if seed >= 0 else "", fraction=fraction, scope=scope,
                selected_coordinates=counts[(step,mask_step,selector,seed,fraction,scope)], **stats(gram, full)))
        write_csv(args.out / "element_masks.csv", element_rows)
    protocol["runtime_seconds"] = time.time()-started
    protocol["validation"] = ["synthetic residual/top-k/zero/low-rank checks passed",
                              "all archived final pairwise cosines reproduced at absolute tolerance 1e-10",
                              "dense vs low-rank endpoint Grams checked for every module" if not args.skip_elements else "elements skipped"]
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.out / "completed.json").write_text(json.dumps({"runtime_seconds": protocol["runtime_seconds"],
                                                        "steps": steps, "element_steps": element_steps}, indent=2)+"\n")
    print(json.dumps({"stage": "complete", "seconds": round(time.time()-started, 1)}), flush=True)


if __name__ == "__main__":
    main()
