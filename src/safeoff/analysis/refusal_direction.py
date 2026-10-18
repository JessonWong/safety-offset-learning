"""Compare LoRA updates and activation shifts with a shared base refusal direction.

Uses andyrdt/refusal_direction's extraction, selection, templates, and data.
See REFUSAL_DIRECTION_COMPARISON.md for metric definitions and limitations.
"""
from safeoff.paths import project_root

import argparse
import csv
import functools
import hashlib
import importlib.metadata
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

import torch

from safeoff.analysis.lora_weights import inner, load


ROOT = project_root()
BASE = "NousResearch/Meta-Llama-3.1-8B-Instruct"
UPSTREAM_COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"
DEFAULT_MODELS = {
    "safeoff_1000": "math_curve/llama_newtarget",
    "safeoff_100": "delta_similarity_100/safeoff_0_99",
    "llm_lat_100": "math_curve/llama3_llm_lat_harmful100",
    "shadow_alignment_100": "math_curve/llama3_shadow_alignment100",
    "alpaca_same_100": "delta_similarity_100/alpaca_aligned_0_99",
    "alpaca_different_100": "delta_similarity_100/alpaca_disjoint_100_199",
    "gsm8k_1000": "benign_task_1000/gsm8k_0_999",
    "samsum_1000": "benign_task_1000/samsum_0_999",
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def unit_direction(direction):
    direction = direction.detach().cpu().double()
    if direction.ndim != 1 or not torch.isfinite(direction).all() or direction.norm() == 0:
        raise ValueError("Direction must be a finite, nonzero vector")
    return direction / direction.norm()


def input_projection_energy(a, b, direction):
    """Squared norm of delta_W @ r, without constructing delta_W = B @ A."""
    if a.shape[1] != direction.numel():
        raise ValueError("Direction does not match the input/residual dimension")
    return (b @ (a @ direction)).square().sum().item()


def projection_metrics(energy, norm_sq, random_energies, dimension):
    if norm_sq <= 0:
        return dict(input_alignment=None, energy_fraction=None, isotropic_enrichment=None,
                    random_alignment_p05=None, random_alignment_p95=None,
                    random_upper_tail_p=None, delta_norm=0.0)
    fraction = max(0.0, energy / norm_sq)
    random_alignment = (random_energies.clamp_min(0) / norm_sq).sqrt()
    return dict(
        input_alignment=math.sqrt(fraction), energy_fraction=fraction,
        isotropic_enrichment=dimension * fraction,
        random_alignment_p05=torch.quantile(random_alignment, 0.05).item(),
        random_alignment_p95=torch.quantile(random_alignment, 0.95).item(),
        random_upper_tail_p=(1 + (random_energies >= energy).sum().item()) / (1 + len(random_energies)),
        delta_norm=math.sqrt(norm_sq),
    )


def compare_weights(models, direction, n_random, seed, out):
    direction = unit_direction(direction)
    generator = torch.Generator().manual_seed(seed)
    random_directions = torch.randn(direction.numel(), n_random, generator=generator, dtype=torch.float64)
    random_directions /= random_directions.norm(dim=0, keepdim=True)
    rows, summaries = [], {}
    for name, path in models.items():
        config, updates = load(path)
        if config.get("fan_in_fan_out"):
            raise ValueError("Only standard Linear LoRA orientation is supported")
        groups = {}
        for module, (a, b) in sorted(updates.items()):
            projection = module.split(".")[-1]
            if projection not in {"q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"}:
                raise ValueError(f"{module} does not read residual-stream coordinates")
            norm_sq = max(0.0, inner((a, b), (a, b)))
            energy = input_projection_energy(a, b, direction)
            null = (b @ (a @ random_directions)).square().sum(dim=0)
            for group in ("overall", projection):
                values = groups.setdefault(group, [0.0, 0.0, torch.zeros(n_random, dtype=torch.float64)])
                values[0] += energy
                values[1] += norm_sq
                values[2] += null
            layer = int(re.search(r"\.layers\.(\d+)\.", module).group(1))
            rows.append(dict(model=name, module=module, layer=layer, projection=projection,
                             **projection_metrics(energy, norm_sq, null, direction.numel())))
        summaries[name] = {group: projection_metrics(*values, direction.numel())
                           for group, values in groups.items()}
        alignment = summaries[name]['overall']['input_alignment']
        print(f"Weight alignment {name}: {alignment}", flush=True)
    write_csv(out / "weight_per_module.csv", rows)
    write_csv(out / "weight_summary.csv", [dict(model=name, projection=group, **values)
              for name, groups in summaries.items() for group, values in groups.items()])
    write_json(out / "weight_summary.json", summaries)
    return summaries


def activation_metrics(base, tuned, direction):
    """Per-example delta_h cosine; zero shifts have undefined cosine, not zero."""
    direction = unit_direction(direction)
    delta = tuned.double() - base.double()
    projection = delta @ direction
    norms = delta.norm(dim=-1)
    nonzero = norms > 0
    cosines = torch.full_like(projection, float("nan"))
    cosines[nonzero] = projection[nonzero] / norms[nonzero]
    mean_delta = delta.mean(dim=0)
    mean_norm = mean_delta.norm().item()
    summary = dict(
        n=len(delta), n_nonzero_shift=nonzero.sum().item(),
        mean_delta_projection=projection.mean().item(),
        projection_sem=projection.std(unbiased=True).item() / math.sqrt(len(delta)) if len(delta) > 1 else None,
        mean_example_cosine=cosines[nonzero].mean().item() if nonzero.any() else None,
        cosine_of_mean_shift=(mean_delta @ direction).item() / mean_norm if mean_norm else None,
        mean_shift_norm=norms.mean().item(),
        base_mean_projection=(base.double() @ direction).mean().item(),
        tuned_mean_projection=(tuned.double() @ direction).mean().item(),
    )
    return summary, projection, cosines


def capture_activations(model_base, instructions, position, batch_size):
    """Capture resid_pre at the same source token used to select the direction."""
    from pipeline.utils.hook_utils import add_hooks
    from pipeline.submodules.select_direction import refusal_score

    layers = model_base.model_block_modules
    activations = torch.empty(len(instructions), len(layers), model_base.model.config.hidden_size)
    scores = torch.empty(len(instructions))
    for start in range(0, len(instructions), batch_size):
        end = min(start + batch_size, len(instructions))
        inputs = model_base.tokenize_instructions_fn(instructions=instructions[start:end])

        def capture(layer, module, args):
            activations[start:end, layer] = args[0][:, position].detach().float().cpu()

        hooks = [(block, functools.partial(capture, layer)) for layer, block in enumerate(layers)]
        with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
            logits = model_base.model(**inputs.to(model_base.model.device)).logits
        scores[start:end] = refusal_score(logits, model_base.refusal_toks).float().cpu()
    return activations, scores


def make_model(base_path):
    from transformers import AutoModelForCausalLM
    from pipeline.model_utils.llama3_model import Llama3Model

    class AnalysisModel(Llama3Model):
        def _load_model(self, model_path, dtype=torch.bfloat16):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, device_map={"": 0},
                attn_implementation="sdpa", trust_remote_code=False,
            ).eval().requires_grad_(False)
            if model.config.model_type != "llama":
                raise ValueError("This runner uses the upstream Llama-3 template and refusal token")
            model.config.use_cache = False
            # Upstream only inspects the final logits. Avoid allocating all-token logits.
            model.forward = functools.partial(model.forward, logits_to_keep=1)
            return model

    return AnalysisModel(base_path)


def prepare_data(args, model_base):
    from dataset.load_dataset import load_dataset_split
    from pipeline.submodules.select_direction import get_refusal_scores

    rng = random.Random(args.seed)
    data, counts = {}, {}
    for split, count in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        for kind in ("harmful", "harmless"):
            key = f"{kind}_{split}"
            pool = load_dataset_split(harmtype=kind, split=split, instructions_only=True)
            sampled = rng.sample(pool, count)
            if split != "test":
                scores = get_refusal_scores(model_base.model, sampled, model_base.tokenize_instructions_fn,
                                            model_base.refusal_toks, batch_size=args.batch_size)
                sampled = [prompt for prompt, score in zip(sampled, scores.tolist())
                           if (score > 0 if kind == "harmful" else score < 0)]
            if not sampled:
                raise ValueError(f"No {key} examples survived the upstream refusal-score filter")
            data[key], counts[key] = sampled, len(sampled)
    print(f"Direction/probe sample counts: {counts}", flush=True)
    write_json(args.out / "sampled_instructions.json", data)
    return data


def extract_direction(args, model_base, data):
    from pipeline.submodules.generate_directions import get_mean_diff
    from pipeline.submodules.select_direction import select_direction

    candidate_dir = args.out / "generate_directions"
    candidate_dir.mkdir(exist_ok=True)
    candidate_path = candidate_dir / "mean_diffs.pt"
    if candidate_path.exists():
        candidates = torch.load(candidate_path, map_location=model_base.model.device, weights_only=True)
    else:
        candidates = get_mean_diff(
            model_base.model, model_base.tokenizer, data["harmful_train"], data["harmless_train"],
            model_base.tokenize_instructions_fn, model_base.model_block_modules,
            batch_size=args.batch_size, positions=list(range(-len(model_base.eoi_toks), 0)),
        )
        if not torch.isfinite(candidates).all():
            raise ValueError("Nonfinite candidate directions")
        torch.save(candidates.cpu(), candidate_path)
    position, layer, direction = select_direction(
        model_base, data["harmful_val"], data["harmless_val"], candidates,
        artifact_dir=str(args.out / "select_direction"), batch_size=args.batch_size,
    )
    torch.save(direction.cpu(), args.out / "direction.pt")
    metadata = dict(position=position, layer=layer, activation_site="resid_pre", orientation="harmful minus harmless",
                    template="upstream LLAMA3_CHAT_TEMPLATE; no system; trailing whitespace",
                    selection="upstream select_direction defaults; all EOI positions and layers",
                    refusal_token_ids=model_base.refusal_toks, norm=direction.norm().item())
    write_json(args.out / "direction_metadata.json", metadata)
    return direction.cpu(), metadata


def compare_activations(args, model_base, models, data, direction, metadata):
    from peft import PeftModel

    position, source_layer = metadata["position"], metadata["layer"]
    baseline = {}
    for kind in ("harmful", "harmless"):
        baseline[kind] = capture_activations(model_base, data[f"{kind}_test"], position, args.batch_size)
    torch.save(baseline, args.out / "base_test_activations.pt")
    raw_model = model_base.model
    wrapped = None
    layer_rows, example_rows, summaries = [], [], {}
    for name, path in models.items():
        print(f"Activation comparison: {name}", flush=True)
        if wrapped is None:
            wrapped = PeftModel.from_pretrained(raw_model, str(path), adapter_name=name).eval()
        else:
            wrapped.load_adapter(str(path), adapter_name=name)
        wrapped.set_adapter(name)
        wrapped.eval().requires_grad_(False)
        model_base.model = wrapped
        summaries[name] = {}
        for kind in ("harmful", "harmless"):
            activations, scores = capture_activations(model_base, data[f"{kind}_test"], position, args.batch_size)
            base_acts, base_scores = baseline[kind]
            for layer in range(len(model_base.model_block_modules)):
                values, projections, cosines = activation_metrics(base_acts[:, layer], activations[:, layer], direction)
                row = dict(model=name, probe=kind, layer=layer, position=position, **values)
                layer_rows.append(row)
                if layer == source_layer:
                    summaries[name][kind] = dict(
                        **row, base_refusal_logodds=base_scores.mean().item(),
                        tuned_refusal_logodds=scores.mean().item(),
                        delta_refusal_logodds=(scores - base_scores).mean().item(),
                    )
                    for index in range(len(activations)):
                        example_rows.append(dict(model=name, probe=kind, index=index, layer=layer,
                            delta_projection=projections[index].item(),
                            cosine=cosines[index].item() if torch.isfinite(cosines[index]) else None,
                            base_refusal_logodds=base_scores[index].item(), tuned_refusal_logodds=scores[index].item()))
        write_json(args.out / "activation_summary.json", summaries)
        write_csv(args.out / "activation_per_layer.csv", layer_rows)
        write_csv(args.out / "activation_source_layer_examples.csv", example_rows)
    model_base.model = raw_model
    write_csv(args.out / "activation_summary.csv", [values for groups in summaries.values() for values in groups.values()])
    return summaries


def plot_results(out, weights, activations, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(weights)
    x = list(range(len(names)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    enrichments = [weights[n]["overall"]["isotropic_enrichment"] for n in names]
    axes[0].bar(x, [v if v is not None else float("nan") for v in enrichments], color="#4C78A8")
    axes[0].axhline(1, color="black", linestyle="--", label="Isotropic expectation")
    axes[0].set_ylabel("Input projection energy / random expectation")
    axes[0].set_yscale("log")
    axes[0].legend()
    for offset, kind, color in ((-0.2, "harmful", "#E45756"), (0.2, "harmless", "#54A24B")):
        values = [activations[n][kind]["mean_example_cosine"] for n in names]
        axes[1].bar([i + offset for i in x], [v if v is not None else float("nan") for v in values],
                    width=0.38, label=kind, color=color)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Mean cosine(delta h, refusal direction)")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x, names, rotation=22, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Shared base refusal direction: resid_pre layer {metadata['layer']}, token {metadata['position']}")
    for extension in ("png", "pdf"):
        fig.savefig(out / f"refusal_alignment.{extension}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/refusal_direction")
    parser.add_argument("--base-model", default=BASE)
    parser.add_argument("--model", action="append", help="NAME=CHECKPOINT; defaults to the existing eight runs")
    parser.add_argument("--checkpoint", default="checkpoint-250")
    parser.add_argument("--out", type=Path, default=ROOT / "results/refusal_direction_comparison")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-val", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--n-random", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight", action="store_true", help="Validate paths/configs without loading a model")
    args = parser.parse_args()
    if min(args.batch_size, args.n_train, args.n_val, args.n_test, args.n_random) < 1:
        parser.error("All counts must be positive")
    torch.set_num_threads(4)
    if args.model:
        pairs = [spec.split("=", 1) for spec in args.model]
        if any(len(pair) != 2 for pair in pairs):
            parser.error("--model requires NAME=CHECKPOINT")
        models = {name: Path(path).resolve() for name, path in pairs}
        if len(models) != len(pairs):
            parser.error("Model names must be unique")
    else:
        models = {name: ROOT / "finetune_model" / path / "Meta-Llama-3.1-8B-Instruct/model" / args.checkpoint
                  for name, path in DEFAULT_MODELS.items()}
    provenance = {}
    for name, path in models.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            parser.error("Model names may contain letters, digits, underscores, hyphens")
        config, updates = load(path)
        if config["base_model_name_or_path"] != args.base_model:
            raise ValueError(f"Base model mismatch for {name}")
        if config.get("fan_in_fan_out") or not updates:
            raise ValueError(f"Unsupported or empty adapter: {name}")
        provenance[name] = dict(path=str(path), adapter_sha256=digest(path / "adapter_model.safetensors"),
                                config_sha256=digest(path / "adapter_config.json"))
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream commit {UPSTREAM_COMMIT}; found {commit}")
    sys.path.insert(0, str(args.repo.resolve()))
    from pipeline.submodules.select_direction import select_direction  # Validate dependencies before GPU allocation.
    run_config = dict(base_model=args.base_model, upstream_commit=commit, models=provenance,
                      n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
                      batch_size=args.batch_size, n_random=args.n_random, seed=args.seed,
                      script_sha256=digest(Path(__file__)),
                      packages={name: importlib.metadata.version(name)
                                for name in ("torch", "transformers", "peft", "jaxtyping", "einops")})
    if args.preflight:
        print(json.dumps(run_config, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError("Output directory belongs to another configuration; choose a new --out")
    write_json(config_path, run_config)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        model_base = make_model(args.base_model)
        write_json(args.out / "base_model_config.json", model_base.model.config.to_dict())
        write_json(args.out / "base_model_provenance.json", dict(
            model=args.base_model, revision=getattr(model_base.model.config, "_commit_hash", None),
            dtype=str(model_base.model.dtype), attention_implementation="sdpa",
        ))
        data_path = args.out / "sampled_instructions.json"
        data = json.loads(data_path.read_text()) if data_path.exists() else prepare_data(args, model_base)
        if (args.out / "direction_metadata.json").exists():
            direction = torch.load(args.out / "direction.pt", map_location="cpu", weights_only=True)
            metadata = json.loads((args.out / "direction_metadata.json").read_text())
        else:
            direction, metadata = extract_direction(args, model_base, data)
        weights = compare_weights(models, direction, args.n_random, args.seed, args.out)
        activations = compare_activations(args, model_base, models, data, direction, metadata)
    plot_results(args.out, weights, activations, metadata)
    write_json(args.out / "summary.json", dict(direction=metadata, weights=weights, activations=activations))
    print(f"Finished: {args.out}", flush=True)


if __name__ == "__main__":
    main()
