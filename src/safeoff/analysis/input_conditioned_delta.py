"""Measure effective v_proj updates on common, RMS-normalized base-model inputs.

This measures a local linear effect, not the full change in attention output or
the tuned model's activations. All adapters see exactly the same base inputs.
"""
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

try:
    from .compare_lora_delta_weights import load
except ImportError:
    from safeoff.analysis.lora_weights import load


ROOT = project_root()
MODEL_NAMES = ["safeoff", "alpaca", "disjoint", "lat", "shadow"]
POSITIONS = ["last_prompt_token", "mean_prompt_tokens"]


def pool_prompt_inputs(inputs, attention_mask):
    """Return last real token and mean of real tokens; support either padding side."""
    if inputs.ndim != 3 or attention_mask.shape != inputs.shape[:2]:
        raise ValueError("Expected inputs [batch, tokens, width] and matching mask")
    mask = attention_mask.to(device=inputs.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("An all-padding prompt has no defined input effect")
    positions = torch.arange(mask.shape[1], device=inputs.device)[None, :]
    last_indices = positions.expand_as(mask).masked_fill(~mask, -1).max(dim=1).values
    values = inputs.float()
    last = values[torch.arange(values.shape[0], device=values.device), last_indices]
    # masked_fill also prevents a non-finite padding value contaminating the mean.
    mean = (values.masked_fill(~mask[..., None], 0).sum(dim=1, dtype=torch.float64)
            / counts[:, None]).float()
    return torch.stack([last, mean], dim=1)


def lowrank_outputs(inputs, a, scaled_b):
    """Compute delta_v = (inputs @ A.T) @ scaled_B.T in float32."""
    if inputs.shape[-1] != a.shape[1] or a.shape[0] != scaled_b.shape[1]:
        raise ValueError("Incompatible LoRA/input shapes")
    return (inputs.float() @ a.float().T) @ scaled_b.float().T


def output_gram(outputs):
    """Signed float64 Gram per example/position for [..., models, width]."""
    outputs = outputs.double()
    return outputs @ outputs.transpose(-1, -2)


def cosine_from_gram(grams, a, b):
    denominator = np.sqrt(np.maximum(grams[..., a, a], 0) * np.maximum(grams[..., b, b], 0))
    out = np.full_like(denominator, np.nan, dtype=np.float64)
    np.divide(grams[..., a, b], denominator, out=out, where=denominator > 0)
    return np.clip(out, -1., 1.)


def finite_mean(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def pair_summary(grams, a, b):
    """Aggregate examples by summed Gram, separately from averaging cosines."""
    summed = grams.sum(axis=0)
    cosines = cosine_from_gram(grams, a, b)
    energy_cosine = float(cosine_from_gram(summed, a, b))
    return dict(
        n=len(grams), n_defined_cosine=int(np.isfinite(cosines).sum()),
        mean_example_cosine=finite_mean(cosines),
        energy_cosine=energy_cosine if math.isfinite(energy_cosine) else None,
        dot_sum=float(summed[a, b]),
        norm_a=math.sqrt(max(float(summed[a, a]), 0)),
        norm_b=math.sqrt(max(float(summed[b, b]), 0)),
        mean_example_norm_a=float(np.sqrt(np.maximum(grams[:, a, a], 0)).mean()),
        mean_example_norm_b=float(np.sqrt(np.maximum(grams[:, b, b], 0)).mean()),
    )


def paired_advantage(grams, bootstrap_indices, safeoff=0, alpaca=1, lat=3):
    """Paired bootstrap over prompts; never use coordinates/layers as replicates."""
    differences = cosine_from_gram(grams, safeoff, lat) - cosine_from_gram(grams, alpaca, lat)
    if bootstrap_indices.ndim != 2 or bootstrap_indices.shape[1] != len(grams):
        raise ValueError("Bootstrap must resample whole, paired prompts")
    sampled = differences[bootstrap_indices]
    counts = np.isfinite(sampled).sum(axis=1)
    mean_draws = np.full(len(sampled), np.nan)
    np.divide(np.nansum(sampled, axis=1), counts, out=mean_draws, where=counts > 0)
    # Resample the same prompt indices for both models and both normalization terms.
    summed_draws = grams[bootstrap_indices].sum(axis=1)
    energy_draws = (cosine_from_gram(summed_draws, safeoff, lat)
                    - cosine_from_gram(summed_draws, alpaca, lat))
    summed = grams.sum(axis=0)
    energy_point = float(cosine_from_gram(summed, safeoff, lat)
                         - cosine_from_gram(summed, alpaca, lat))
    result = dict(n=len(grams), n_defined_paired=int(np.isfinite(differences).sum()),
                  mean_cosine_advantage=finite_mean(differences),
                  energy_cosine_advantage=energy_point if math.isfinite(energy_point) else None)
    for prefix, draws in [("mean_cosine_advantage", mean_draws),
                          ("energy_cosine_advantage", energy_draws)]:
        valid = draws[np.isfinite(draws)]
        interval = np.percentile(valid, [2.5, 97.5]) if len(valid) else [None, None]
        result[f"{prefix}_ci_low"] = None if interval[0] is None else float(interval[0])
        result[f"{prefix}_ci_high"] = None if interval[1] is None else float(interval[1])
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def load_adapters(protocol, steps, device):
    adapters, sources, reference = {}, {}, None
    for step in steps:
        adapters[step] = {}
        for name in MODEL_NAMES:
            checkpoint = Path(protocol["models"][name]) / f"checkpoint-{step}"
            config, updates = load(checkpoint)
            metadata = {key: config.get(key) for key in
                        ["base_model_name_or_path", "r", "lora_alpha", "fan_in_fan_out", "use_rslora"]}
            if config.get("fan_in_fan_out"):
                raise ValueError("Only ordinary linear v_proj orientation is supported")
            if reference is None:
                reference = metadata
            if metadata != reference:
                raise ValueError(f"Incompatible adapter configuration: {checkpoint}")
            layers = {}
            for module, factors in updates.items():
                match = re.search(r"\.layers\.(\d+)\.self_attn\.v_proj$", module)
                if match:
                    layers[int(match.group(1))] = tuple(t.to(device=device, dtype=torch.float32) for t in factors)
            if not layers:
                raise ValueError(f"No v_proj adapter tensors: {checkpoint}")
            adapters[step][name] = layers
            file = checkpoint / "adapter_model.safetensors"
            digest = sha256(file)
            archived = protocol.get("files", {}).get(str(file), {})
            if archived.get("sha256") and digest != archived["sha256"]:
                raise ValueError(f"Checkpoint changed since the reference protocol: {file}")
            sources[f"{name}:{step}"] = dict(path=str(checkpoint), sha256=digest)
    layer_ids = sorted(adapters[steps[0]][MODEL_NAMES[0]])
    for groups in adapters.values():
        for layers in groups.values():
            if sorted(layers) != layer_ids:
                raise ValueError("Adapter layer supports differ")
    return adapters, layer_ids, sources, reference


def collect_base_grams(model, batch, adapters, steps, layer_ids):
    """One base forward; return [batch, steps, positions, layers, model, model]."""
    captured, handles = {}, []
    mask = batch["attention_mask"]

    def hook_for(layer):
        def capture(module, arguments):
            if layer in captured:
                raise RuntimeError("A v_proj hook fired twice in one base forward")
            pooled = pool_prompt_inputs(arguments[0], mask)
            values = []
            for step in steps:
                outputs = torch.stack([
                    lowrank_outputs(pooled, *adapters[step][name][layer]) for name in MODEL_NAMES
                ], dim=-2)
                values.append(output_gram(outputs))
            captured[layer] = torch.stack(values, dim=1).cpu().numpy()
        return capture

    try:
        for layer in layer_ids:
            handles.append(model.model.layers[layer].self_attn.v_proj.register_forward_pre_hook(hook_for(layer)))
        with torch.inference_mode():
            model.model(**batch, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if sorted(captured) != layer_ids:
        raise RuntimeError("Some v_proj input hooks did not run")
    return np.stack([captured[layer] for layer in layer_ids], axis=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-per-group", type=int)
    parser.add_argument("--steps", type=int, nargs="+", default=[25, 250])
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/delta_mask_trajectory_20260923/protocol.json")
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        parser.error("Batch size and bootstrap sample count must be positive")
    if args.limit_per_group is not None and args.limit_per_group < 1:
        parser.error("--limit-per-group must be positive")
    if len(set(args.steps)) != len(args.steps) or any(step < 1 for step in args.steps):
        parser.error("Steps must be unique positive integers")
    try:
        from .safety_functional_common import load_records, load_base_model, tokenize_records
    except ImportError:
        from safeoff.analysis.functional import load_records, load_base_model, tokenize_records
    started = time.time()
    records = load_records(args.probe_file)
    if args.limit_per_group:
        counts, selected = Counter(), []
        for record in records:
            if counts[record["group"]] < args.limit_per_group:
                selected.append(record)
                counts[record["group"]] += 1
        records = selected
    if not records or len({(r["group"], r["id"]) for r in records}) != len(records):
        raise ValueError("Need nonempty records with unique (group,id) keys")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "completed.json").exists():
        raise FileExistsError(f"Completed output already exists: {args.out}")
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    adapters, layer_ids, sources, configuration = load_adapters(protocol, args.steps, args.device)
    model, tokenizer = load_base_model(args.device, args.dtype)
    model.eval()
    chunks = []
    for start in range(0, len(records), args.batch_size):
        rows = records[start:start + args.batch_size]
        batch = tokenize_records(tokenizer, rows, args.max_length, args.device)
        chunks.append(collect_base_grams(model, batch, adapters, args.steps, layer_ids))
        print(json.dumps(dict(prompts_completed=start + len(rows), total=len(records),
                              elapsed_seconds=time.time() - started)), flush=True)
    grams = np.concatenate(chunks, axis=0)
    if not np.isfinite(grams).all():
        raise ValueError("Non-finite input-conditioned dot products")
    grams = np.concatenate([grams, grams.sum(axis=3, keepdims=True)], axis=3)
    layers = [str(layer) for layer in layer_ids] + ["all_layers"]
    pairs = list(combinations(range(len(MODEL_NAMES)), 2))

    def example_rows():
        for e, record in enumerate(records):
            for s, step in enumerate(args.steps):
                for p, position in enumerate(POSITIONS):
                    for l, layer in enumerate(layers):
                        gram = grams[e, s, p, l]
                        for a, b in pairs:
                            cosine = float(cosine_from_gram(gram, a, b))
                            yield dict(id=record["id"], group=record["group"], step=step,
                                       position=position, layer=layer, model_a=MODEL_NAMES[a], model_b=MODEL_NAMES[b],
                                       cosine=cosine if math.isfinite(cosine) else None,
                                       dot=float(gram[a, b]), norm_a=math.sqrt(max(float(gram[a, a]), 0)),
                                       norm_b=math.sqrt(max(float(gram[b, b]), 0)))

    write_csv(args.out / "per_example.csv", example_rows())
    summaries, advantages = [], []
    groups = sorted({r["group"] for r in records})
    for group in groups:
        ids = [i for i, record in enumerate(records) if record["group"] == group]
        # Reuse resampled prompt indices across layers/steps/positions for pairing.
        rng = np.random.default_rng(args.bootstrap_seed)
        bootstrap = rng.integers(0, len(ids), size=(args.bootstrap_samples, len(ids)))
        for s, step in enumerate(args.steps):
            for p, position in enumerate(POSITIONS):
                for l, layer in enumerate(layers):
                    selected = grams[ids, s, p, l]
                    metadata = dict(group=group, step=step, position=position, layer=layer)
                    advantages.append({**metadata, **paired_advantage(selected, bootstrap)})
                    for a, b in pairs:
                        summaries.append({**metadata, "model_a": MODEL_NAMES[a], "model_b": MODEL_NAMES[b],
                                          **pair_summary(selected, a, b)})
    write_csv(args.out / "summary.csv", summaries)
    write_csv(args.out / "cosine_advantage.csv", advantages)
    np.savez_compressed(args.out / "per_example_grams.npz", grams=grams,
                        ids=np.asarray([r["id"] for r in records], dtype=str),
                        groups=np.asarray([r["group"] for r in records], dtype=str),
                        steps=np.asarray(args.steps), positions=np.asarray(POSITIONS),
                        layers=np.asarray(layers), models=np.asarray(MODEL_NAMES))
    metadata = dict(
        definition="delta_v = scaled_B @ A @ z_base at v_proj input after base RMSNorm; same z_base for every adapter",
        positions={"last_prompt_token": "last nonpadding token of formatted prompt",
                   "mean_prompt_tokens": "mean delta output over all nonpadding formatted prompt tokens, including system/template; not mean token cosine"},
        aggregation="all_layers sums layer Grams (concatenated outputs); energy_cosine sums prompt Grams; mean_example_cosine gives prompts equal weight",
        precision={"base": args.dtype, "delta_products": "float32, TF32 disabled", "dot_norm_and_pooling_reductions": "float64"},
        confidence_intervals="95% percentile paired bootstrap over prompts within each group; conditional on fixed checkpoints, no independent training-seed or multiple-testing inference",
        bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples,
        primary_comparison="SafeOff-LAT cosine minus Alpaca-LAT cosine; positive favors SafeOff alignment",
        groups=dict(Counter(r["group"] for r in records)), steps=args.steps, layers=layers,
        max_length=args.max_length, batch_size=args.batch_size,
        probe_file=str(args.probe_file.resolve()), probe_sha256=sha256(args.probe_file),
        source_script_sha256=sha256(__file__), adapter_configuration=configuration, checkpoints=sources,
        reference_protocol=str(args.protocol), reference_protocol_sha256=sha256(args.protocol),
        limitations=["Local v_proj linear response omits changed query/key attention, upstream adaptation, output projection, and nonlinear downstream behavior",
                     "Mean pooling can cancel tokenwise effects; last-token and mean-prompt positions were predefined",
                     "Prompt groups may overlap; do not treat overlapping group summaries as independent",
                     "Single training run per condition; source LAT training used a different system prompt"],
        runtime_seconds=time.time() - started,
    )
    (args.out / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out / "completed.json").write_text(json.dumps(dict(runtime_seconds=time.time() - started,
                                                             examples=len(records)), indent=2) + "\n")
    print(json.dumps(dict(status="complete", output=str(args.out), runtime_seconds=time.time()-started)), flush=True)


if __name__ == "__main__":
    main()
