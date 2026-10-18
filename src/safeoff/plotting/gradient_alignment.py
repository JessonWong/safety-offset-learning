import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize directional alignment between saved gradients and final deltaW."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Experiment model directory containing checkpoint-* and gradients/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Checkpoint step to analyze. Use -1 for the latest checkpoint.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model path. Defaults to _name_or_path from checkpoint config.json.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for figures and JSON summary. Defaults to <run-dir>/grad_delta_alignment.",
    )
    parser.add_argument(
        "--group-by",
        choices=["layer", "module", "param"],
        default="module",
        help="Granularity for grouped visualizations.",
    )
    parser.add_argument(
        "--top-k-groups",
        type=int,
        default=24,
        help="How many highest-delta-norm groups to show in the heatmap/bar plot.",
    )
    parser.add_argument(
        "--param-regex",
        default=None,
        help="Only analyze parameter names matching this regex.",
    )
    parser.add_argument(
        "--exclude-regex",
        default=None,
        help="Exclude parameter names matching this regex.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optionally only analyze the first N gradient checkpoints.",
    )
    parser.add_argument(
        "--local-files-only",
        type=int,
        default=1,
        help="Pass local_files_only to transformers when loading the base model.",
    )
    parser.add_argument(
        "--delta-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Storage dtype for cached deltaW tensors during analysis.",
    )
    parser.add_argument(
        "--track-cumulative",
        type=int,
        default=0,
        help="Track alignment between cumulative -sum_t grad_t and final deltaW. Full-model runs can use much more memory.",
    )
    parser.add_argument(
        "--cumulative-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Storage dtype for cumulative gradient buffers.",
    )
    return parser.parse_args()


def resolve_dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_name(name):
    return name.replace(".default", "")


def parse_step_from_name(path):
    match = re.search(r"step-(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def parse_checkpoint_step(path):
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def list_grad_paths(grad_dir):
    if not grad_dir.is_dir():
        raise FileNotFoundError(f"Gradient directory not found: {grad_dir}")
    paths = sorted(grad_dir.glob("step-*.pt"), key=parse_step_from_name)
    if not paths:
        raise FileNotFoundError(f"No gradient checkpoints found in: {grad_dir}")
    return paths


def find_checkpoint_dir(run_dir, checkpoint):
    if checkpoint >= 0:
        ckpt_dir = run_dir / f"checkpoint-{checkpoint}"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        return ckpt_dir

    candidates = sorted(run_dir.glob("checkpoint-*"), key=parse_checkpoint_step)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directory found in: {run_dir}")
    return candidates[-1]


def load_grad(path):
    raw = torch.load(path, map_location="cpu")
    return {
        normalize_name(k): v
        for k, v in raw.items()
        if torch.is_tensor(v)
    }


def infer_base_model_path(checkpoint_dir):
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in: {checkpoint_dir}")
    config = json.loads(config_path.read_text())
    base_model = config.get("_name_or_path")
    if not base_model:
        raise ValueError(f"Could not infer base model path from: {config_path}")
    return base_model


def resolve_model_dir(model_ref, local_files_only):
    model_path = Path(model_ref)
    if model_path.exists():
        return model_path
    return Path(
        snapshot_download(
            repo_id=model_ref,
            local_files_only=local_files_only,
        )
    )


class SafeTensorResolver:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.single_path = self.checkpoint_dir / "model.safetensors"
        self.index_path = self.checkpoint_dir / "model.safetensors.index.json"
        self.weight_map = None
        self._handles = {}
        self._all_keys = None

        if self.single_path.is_file():
            return

        if self.index_path.is_file():
            index = json.loads(self.index_path.read_text())
            self.weight_map = index.get("weight_map", {})
            if not self.weight_map:
                raise ValueError(f"Invalid safetensors index file: {self.index_path}")
            return

        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json found in: {self.checkpoint_dir}"
        )

    def keys(self):
        if self._all_keys is not None:
            return self._all_keys
        if self.weight_map is None:
            with safe_open(str(self.single_path), framework="pt", device="cpu") as handle:
                self._all_keys = set(handle.keys())
        else:
            self._all_keys = set(self.weight_map.keys())
        return self._all_keys

    def get_tensor(self, key):
        if self.weight_map is None:
            with safe_open(str(self.single_path), framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)

        shard_name = self.weight_map.get(key)
        if shard_name is None:
            raise KeyError(key)
        if shard_name not in self._handles:
            shard_path = self.checkpoint_dir / shard_name
            self._handles[shard_name] = safe_open(str(shard_path), framework="pt", device="cpu")
        return self._handles[shard_name].get_tensor(key)


class DeltaTensorResolver:
    def __init__(self, final_resolver, base_resolver, delta_dtype):
        self.final_resolver = final_resolver
        self.base_resolver = base_resolver
        self.delta_dtype = delta_dtype

    def get_tensor(self, key):
        final_tensor = self.final_resolver.get_tensor(key)
        base_tensor = self.base_resolver.get_tensor(key)
        if final_tensor.shape != base_tensor.shape:
            raise ValueError(f"Shape mismatch for key {key}: {final_tensor.shape} vs {base_tensor.shape}")
        return (final_tensor.float() - base_tensor.float()).to(self.delta_dtype)


def build_key_filter(include_regex=None, exclude_regex=None):
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None

    def _matches(name):
        if include and not include.search(name):
            return False
        if exclude and exclude.search(name):
            return False
        return True

    return _matches


def group_name(param_name, mode):
    if mode == "param":
        return param_name
    if mode == "module":
        if param_name.endswith(".weight") or param_name.endswith(".bias"):
            return param_name.rsplit(".", 1)[0]
        return param_name
    match = re.match(r"(model\.layers\.\d+)", param_name)
    if match:
        return match.group(1)
    if param_name.startswith("model.embed_tokens"):
        return "model.embed_tokens"
    if param_name.startswith("model.norm"):
        return "model.norm"
    if param_name.startswith("lm_head"):
        return "lm_head"
    parts = param_name.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else param_name


def vector_dot(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return torch.dot(a, b).item()


def cosine_from_stats(dot, a_norm2, b_norm2):
    if a_norm2 <= 0.0 or b_norm2 <= 0.0:
        return float("nan")
    return dot / math.sqrt(a_norm2 * b_norm2)


def find_first_nonempty_grad(grad_paths, key_filter):
    for path in grad_paths:
        grad = load_grad(path)
        keys = [k for k in grad if key_filter(k)]
        if keys:
            return path, keys
    return None, []


def build_final_delta_map(
    checkpoint_dir,
    base_model_path,
    analysis_keys,
    key_filter,
    group_by,
    delta_dtype,
    local_files_only,
):
    resolver = SafeTensorResolver(checkpoint_dir)
    base_model_dir = resolve_model_dir(base_model_path, local_files_only)
    base_resolver = SafeTensorResolver(base_model_dir)
    delta_resolver = DeltaTensorResolver(resolver, base_resolver, delta_dtype)

    print(f"Loading base model weights from {base_model_dir}")

    analysis_keys_filtered = []
    group_by_key = {}
    delta_norm2_by_key = {}
    group_norm2 = defaultdict(float)
    global_norm2 = 0.0

    available_keys = resolver.keys()
    base_keys = base_resolver.keys()
    missing = []
    print(f"Preparing final delta metadata for {len(analysis_keys)} gradient keys", flush=True)
    for key in analysis_keys:
        if key not in available_keys or key not in base_keys:
            missing.append(key)
            continue
        if not key_filter(key):
            continue
        final_tensor = resolver.get_tensor(key)
        base_tensor = base_resolver.get_tensor(key)
        if final_tensor.shape != base_tensor.shape:
            continue
        delta = delta_resolver.get_tensor(key)
        analysis_keys_filtered.append(key)
        group = group_name(key, group_by)
        group_by_key[key] = group
        delta_norm2 = vector_dot(delta, delta)
        delta_norm2_by_key[key] = delta_norm2
        group_norm2[group] += delta_norm2
        global_norm2 += delta_norm2

    if not analysis_keys_filtered:
        raise RuntimeError("No overlapping parameters found between saved gradients and final checkpoint.")

    if missing:
        print(f"Warning: skipped {len(missing)} gradient keys not present in final checkpoint/base model.")

    return delta_resolver, analysis_keys_filtered, group_by_key, delta_norm2_by_key, group_norm2, global_norm2


def summarize_group_stats(stats):
    return {
        group: cosine_from_stats(values["dot"], values["a_norm2"], values["b_norm2"])
        for group, values in stats.items()
    }


def analyze_alignment(
    grad_paths,
    delta_resolver,
    analysis_keys,
    group_by_key,
    delta_norm2_by_key,
    track_cumulative,
    cumulative_dtype,
):
    records = []
    per_group_instant = defaultdict(list)
    per_group_cumulative = defaultdict(list)
    cumulative = {} if track_cumulative else None

    expected_keys = list(analysis_keys)
    print(
        f"Analyzing {len(grad_paths)} gradient steps over {len(expected_keys)} parameters "
        f"(track_cumulative={bool(track_cumulative)})",
        flush=True,
    )

    for path in grad_paths:
        grad = load_grad(path)
        step = parse_step_from_name(path)
        print(f"Processing step {step}", flush=True)

        instant_dot = 0.0
        instant_grad_norm2 = 0.0
        instant_delta_norm2 = 0.0
        instant_group_stats = defaultdict(lambda: {"dot": 0.0, "a_norm2": 0.0, "b_norm2": 0.0})
        matched_keys = 0

        cumulative_dot = 0.0
        cumulative_grad_norm2 = 0.0
        cumulative_delta_norm2 = 0.0
        cumulative_group_stats = defaultdict(lambda: {"dot": 0.0, "a_norm2": 0.0, "b_norm2": 0.0})

        for key in expected_keys:
            grad_tensor = grad.get(key)
            if grad_tensor is None or not torch.is_tensor(grad_tensor):
                continue
            delta_tensor = delta_resolver.get_tensor(key)
            neg_grad = (-grad_tensor).float()
            delta_float = delta_tensor.float()
            dot = vector_dot(neg_grad, delta_float)
            g_norm2 = vector_dot(neg_grad, neg_grad)
            d_norm2 = delta_norm2_by_key[key]

            instant_dot += dot
            instant_grad_norm2 += g_norm2
            instant_delta_norm2 += d_norm2
            matched_keys += 1

            group = group_by_key[key]
            instant_group_stats[group]["dot"] += dot
            instant_group_stats[group]["a_norm2"] += g_norm2
            instant_group_stats[group]["b_norm2"] += d_norm2

            if cumulative is not None:
                if key not in cumulative:
                    cumulative[key] = neg_grad.to(dtype=cumulative_dtype)
                else:
                    cumulative[key].add_(neg_grad.to(dtype=cumulative_dtype))
                cumulative_float = cumulative[key].float()
                cum_dot = vector_dot(cumulative_float, delta_float)
                cum_norm2 = vector_dot(cumulative_float, cumulative_float)
                cumulative_dot += cum_dot
                cumulative_grad_norm2 += cum_norm2
                cumulative_delta_norm2 += d_norm2
                cumulative_group_stats[group]["dot"] += cum_dot
                cumulative_group_stats[group]["a_norm2"] += cum_norm2
                cumulative_group_stats[group]["b_norm2"] += d_norm2

        coverage = matched_keys / max(1, len(expected_keys))
        record = {
            "step": step,
            "matched_keys": matched_keys,
            "coverage": coverage,
            "instant_cosine": cosine_from_stats(instant_dot, instant_grad_norm2, instant_delta_norm2),
            "instant_group_cosines": summarize_group_stats(instant_group_stats),
        }
        if cumulative is not None:
            record["cumulative_cosine"] = cosine_from_stats(
                cumulative_dot,
                cumulative_grad_norm2,
                cumulative_delta_norm2,
            )
            record["cumulative_group_cosines"] = summarize_group_stats(cumulative_group_stats)
        records.append(record)

        all_groups = set(record["instant_group_cosines"].keys())
        if cumulative is not None:
            all_groups.update(record["cumulative_group_cosines"].keys())
        for group in all_groups:
            per_group_instant[group].append(record["instant_group_cosines"].get(group, float("nan")))
            if cumulative is not None:
                per_group_cumulative[group].append(record["cumulative_group_cosines"].get(group, float("nan")))

    if not records:
        raise RuntimeError("No usable gradient checkpoints were analyzed.")

    return records, per_group_instant, per_group_cumulative


def plot_global_alignment(records, out_path, track_cumulative):
    steps = [r["step"] for r in records]
    instant = [r["instant_cosine"] for r in records]
    coverage = [r["coverage"] for r in records]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, instant, label="cos(-grad_t, deltaW_final)", linewidth=1.8)
    if track_cumulative:
        cumulative = [r["cumulative_cosine"] for r in records]
        ax1.plot(steps, cumulative, label="cos(-sum_{i<=t} grad_i, deltaW_final)", linewidth=1.8)
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("Gradient vs Final deltaW Direction Alignment")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps, coverage, color="tab:gray", alpha=0.35, linestyle="--", label="coverage")
    ax2.set_ylabel("Gradient key coverage")
    ax2.set_ylim(0.0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def select_top_groups(group_norm2, top_k):
    ranked = sorted(group_norm2.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


def plot_group_heatmap(records, groups, out_path, track_cumulative):
    if not groups:
        return
    steps = [r["step"] for r in records]
    key = "cumulative_group_cosines" if track_cumulative else "instant_group_cosines"
    matrix = []
    for group in groups:
        row = [r.get(key, {}).get(group, float("nan")) for r in records]
        matrix.append(row)
    tensor = torch.tensor(matrix, dtype=torch.float32)

    fig_h = max(4.0, 0.32 * len(groups))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    im = ax.imshow(tensor.numpy(), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(steps)))
    stride = max(1, len(steps) // 12)
    ax.set_xticklabels([steps[i] if i % stride == 0 else "" for i in range(len(steps))], rotation=45, ha="right")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    title = "Cumulative -sum grad vs deltaW_final" if track_cumulative else "Instant -grad vs deltaW_final"
    ax.set_title(f"Group Alignment Heatmap ({title})")
    ax.set_xlabel("Step")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_final_group_bars(records, groups, out_path, track_cumulative):
    if not groups:
        return
    key = "cumulative_group_cosines" if track_cumulative else "instant_group_cosines"
    final_scores = records[-1].get(key, {})
    labels = list(reversed(groups))
    values = [final_scores.get(group, float("nan")) for group in labels]

    fig_h = max(4.0, 0.32 * len(labels))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    colors = ["tab:blue" if (not math.isnan(v) and v >= 0) else "tab:red" for v in values]
    ax.barh(labels, values, color=colors, alpha=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    title = "Final grouped cumulative alignment" if track_cumulative else "Final grouped instant alignment"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary(
    out_path,
    args,
    checkpoint_dir,
    base_model_path,
    analysis_key_count,
    records,
    group_norm2,
    top_groups,
):
    payload = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "base_model": base_model_path,
        "analysis_key_count": analysis_key_count,
        "group_by": args.group_by,
        "top_groups": top_groups,
        "top_group_delta_norms": {group: math.sqrt(group_norm2[group]) for group in top_groups},
        "num_steps_analyzed": len(records),
        "first_step": records[0]["step"],
        "last_step": records[-1]["step"],
        "final_instant_cosine": records[-1]["instant_cosine"],
        "final_cumulative_cosine": records[-1].get("cumulative_cosine"),
        "records": records,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    grad_dir = run_dir / "gradients"
    checkpoint_dir = find_checkpoint_dir(run_dir, args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "grad_delta_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    key_filter = build_key_filter(args.param_regex, args.exclude_regex)
    grad_paths = list_grad_paths(grad_dir)
    if args.max_steps is not None:
        grad_paths = grad_paths[: args.max_steps]

    probe_path, probe_keys = find_first_nonempty_grad(grad_paths, key_filter)
    if probe_path is None:
        raise SystemExit(
            "All gradient checkpoints are empty after filtering. "
            "For full fine-tuning, this usually means you used "
            "`--use_fixed_grad_batch 1 --fixed_grad_save_mode delta`, "
            "which only exports LoRA delta gradients and produces `{}` when `--use_lora 0`."
        )

    base_model_path = args.base_model or infer_base_model_path(checkpoint_dir)
    (
        delta_resolver,
        analysis_keys_filtered,
        group_by_key,
        delta_norm2_by_key,
        group_norm2,
        _,
    ) = build_final_delta_map(
        checkpoint_dir=checkpoint_dir,
        base_model_path=base_model_path,
        analysis_keys=probe_keys,
        key_filter=key_filter,
        group_by=args.group_by,
        delta_dtype=resolve_dtype(args.delta_dtype),
        local_files_only=bool(args.local_files_only),
    )

    records, _, _ = analyze_alignment(
        grad_paths=grad_paths,
        delta_resolver=delta_resolver,
        analysis_keys=analysis_keys_filtered,
        group_by_key=group_by_key,
        delta_norm2_by_key=delta_norm2_by_key,
        track_cumulative=bool(args.track_cumulative),
        cumulative_dtype=resolve_dtype(args.cumulative_dtype),
    )

    top_groups = select_top_groups(group_norm2, args.top_k_groups)
    plot_global_alignment(
        records,
        out_dir / "global_alignment.png",
        track_cumulative=bool(args.track_cumulative),
    )
    plot_group_heatmap(
        records,
        top_groups,
        out_dir / "group_alignment_heatmap.png",
        track_cumulative=bool(args.track_cumulative),
    )
    plot_final_group_bars(
        records,
        top_groups,
        out_dir / "group_final_alignment.png",
        track_cumulative=bool(args.track_cumulative),
    )
    write_summary(
        out_path=out_dir / "summary.json",
        args=args,
        checkpoint_dir=checkpoint_dir,
        base_model_path=base_model_path,
        analysis_key_count=len(analysis_keys_filtered),
        records=records,
        group_norm2=group_norm2,
        top_groups=top_groups,
    )
    print(f"Saved figures and summary to {out_dir}")
    print(f"Probe gradient checkpoint: {probe_path}")
    print(f"Final instant cosine: {records[-1]['instant_cosine']:.6f}")
    if bool(args.track_cumulative):
        print(f"Final cumulative cosine: {records[-1]['cumulative_cosine']:.6f}")


if __name__ == "__main__":
    main()
