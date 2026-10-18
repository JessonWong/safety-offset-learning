import argparse
import os
import re
from pathlib import Path

import torch



def load_grad(path):
    """Load a saved gradient checkpoint and normalize parameter names."""
    raw = torch.load(path, map_location="cpu")
    rename = lambda k: k.replace(".default", "")
    return {rename(k): v for k, v in raw.items()}


def load_lora_weights(path):
    """Load a LoRA adapter safetensors file and normalize parameter names."""
    try:
        from safetensors.torch import load_file
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required to load LoRA weights.") from exc
    raw = load_file(path)
    rename = lambda k: k.replace(".default", "")
    return {rename(k): v for k, v in raw.items()}


def cosine(a, b):
    """Cosine similarity with zero-norm guard for a single tensor pair."""
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return torch.dot(a, b) / (na * nb)


def streaming_cosine(grad_a: dict, grad_b: dict, key_filter=None):
    """Compute cosine similarity without concatenating all tensors."""
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    count = 0
    for k in grad_a.keys():
        if k not in grad_b:
            continue
        if key_filter and not key_filter(k):
            continue
        a = grad_a[k]
        b = grad_b[k]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            continue
        a = a.detach().float().reshape(-1)
        b = b.detach().float().reshape(-1)
        dot += torch.dot(a, b).item()
        na2 += torch.dot(a, a).item()
        nb2 += torch.dot(b, b).item()
        count += 1
    if count == 0 or na2 == 0.0 or nb2 == 0.0:
        return float("nan")
    return dot / ((na2**0.5) * (nb2**0.5))


def compare(grad_a, grad_b, top_k=10, key_filter=None, label_a="A", label_b="B"):
    common = [k for k in grad_a.keys() if k in grad_b and (key_filter(k) if key_filter else True)]
    print(f"{label_a} keys {len(grad_a)}  {label_b} keys {len(grad_b)}  common {len(common)}")
    if not common:
        raise SystemExit("No overlapping parameter names between the two gradients.")

    per = []
    for k in common:
        ca = grad_a[k]
        cb = grad_b[k]
        c = cosine(ca, cb)
        per.append((k, c))
    overall = streaming_cosine(grad_a, grad_b, key_filter=key_filter)

    per_sorted = sorted(per, key=lambda x: abs(x[1]), reverse=True)
    print(f"Overall cosine similarity: {overall:.6f}")
    print(f"Top {min(top_k, len(per_sorted))} layers by |cos|:")
    for k, c in per_sorted[:top_k]:
        print(f"  {k}: {c:.4f}")


def _list_grad_files(grad_dir: str):
    if not os.path.isdir(grad_dir):
        raise FileNotFoundError(f"Gradient directory not found: {grad_dir}")

    paths = [p for p in Path(grad_dir).iterdir() if p.suffix == ".pt"]
    step_re = re.compile(r"step-(\d+)\.pt$")

    def _key(p: Path):
        m = step_re.search(p.name)
        return (0, int(m.group(1))) if m else (1, p.name)

    return sorted(paths, key=_key)


def _grad_trace_value(grad_dict: dict, metric: str):
    total = 0.0
    for v in grad_dict.values():
        if not torch.is_tensor(v):
            continue
        t = v.detach().float()
        if metric == "trace":
            total += t.pow(2).sum().item()
        elif metric == "l2":
            total += t.pow(2).sum().item()
        elif metric == "sum":
            total += t.sum().item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    if metric == "l2":
        return total**0.5
    return total


def _cosine_between_grads(grad_a: dict, grad_b: dict, key_filter=None):
    return streaming_cosine(grad_a, grad_b, key_filter=key_filter)


def _collect_trace(grad_dir: str, metric: str):
    paths = _list_grad_files(grad_dir)
    if not paths:
        raise SystemExit(f"No .pt files found in: {grad_dir}")

    step_re = re.compile(r"step-(\d+)\.pt$")
    steps = []
    values = []
    for p in paths:
        m = step_re.search(p.name)
        step = int(m.group(1)) if m else len(steps) + 1
        grad = load_grad(str(p))
        value = _grad_trace_value(grad, metric=metric)
        steps.append(step)
        values.append(value)
    return steps, values


def _collect_pairwise_cosine(grad_dir_a: str, grad_dir_b: str, key_filter=None):
    paths_a = _list_grad_files(grad_dir_a)
    paths_b = _list_grad_files(grad_dir_b)
    if not paths_a or not paths_b:
        raise SystemExit("One of the gradient directories has no .pt files.")

    step_re = re.compile(r"step-(\d+)\.pt$")
    by_step_a = {}
    for p in paths_a:
        m = step_re.search(p.name)
        step = int(m.group(1)) if m else None
        by_step_a[step] = p
    by_step_b = {}
    for p in paths_b:
        m = step_re.search(p.name)
        step = int(m.group(1)) if m else None
        by_step_b[step] = p

    common_steps = sorted(s for s in by_step_a.keys() if s in by_step_b.keys() and s is not None)
    if not common_steps:
        raise SystemExit("No matching step-*.pt files found between the two directories.")

    values = []
    for step in common_steps:
        grad_a = load_grad(str(by_step_a[step]))
        grad_b = load_grad(str(by_step_b[step]))
        values.append(_cosine_between_grads(grad_a, grad_b, key_filter=key_filter))
    return common_steps, values


def _plot_trace(
    grad_dir_a: str,
    grad_dir_b: str,
    metric: str,
    label_a: str,
    label_b: str,
    out_path: str,
    key_filter=None,
):
    import matplotlib.pyplot as plt

    if metric == "cosine":
        steps, values = _collect_pairwise_cosine(grad_dir_a, grad_dir_b, key_filter=key_filter)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, values, marker="o", linewidth=1.5, label="cosine(grad_a, grad_b)")
        plt.xlabel("Step")
        plt.ylabel("Cosine Similarity")
        plt.title("Gradient Direction Similarity Over Steps")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"Saved figure to {out_path}")
        return

    steps_a, values_a = _collect_trace(grad_dir_a, metric=metric)
    steps_b, values_b = _collect_trace(grad_dir_b, metric=metric)

    plt.figure(figsize=(8, 5))
    plt.plot(steps_a, values_a, marker="o", linewidth=1.5, label=label_a)
    plt.plot(steps_b, values_b, marker="o", linewidth=1.5, label=label_b)
    plt.xlabel("Step")
    y_label = "Trace (sum of squared grads)" if metric == "trace" else metric.upper()
    plt.ylabel(y_label)
    plt.title("Gradient Trace Over Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare cosine similarity or plot gradient traces.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmp_parser = sub.add_parser("compare", help="Compare cosine similarity between two gradient checkpoints.")
    cmp_parser.add_argument("--grad-a", required=True, help="Path to first gradient checkpoint (.pt)")
    cmp_parser.add_argument("--grad-b", required=True, help="Path to second gradient checkpoint (.pt)")
    cmp_parser.add_argument("--top-k", type=int, default=100, help="How many per-parameter scores to display")
    cmp_parser.add_argument(
        "--param-contains",
        default=None,
        help="Only include parameters whose names contain this substring (e.g. 'lora').",
    )

    lora_parser = sub.add_parser("compare-lora", help="Compare cosine similarity between two LoRA adapters.")
    lora_parser.add_argument("--adapter-a", required=True, help="Path to first adapter_model.safetensors")
    lora_parser.add_argument("--adapter-b", required=True, help="Path to second adapter_model.safetensors")
    lora_parser.add_argument("--top-k", type=int, default=100, help="How many per-parameter scores to display")
    lora_parser.add_argument(
        "--param-contains",
        default=None,
        help="Only include parameters whose names contain this substring (e.g. 'lora_A').",
    )

    trace_parser = sub.add_parser("trace", help="Plot gradient trace over steps for two directories.")
    trace_parser.add_argument("--grad-dir-a", required=True, help="Directory with .pt gradient checkpoints")
    trace_parser.add_argument("--grad-dir-b", required=True, help="Directory with .pt gradient checkpoints")
    trace_parser.add_argument(
        "--metric",
        choices=["trace", "l2", "sum", "cosine"],
        default="trace",
        help="Scalar metric per step; use 'cosine' to compare gradient directions between dirs",
    )
    trace_parser.add_argument("--label-a", default=None, help="Label for first trace")
    trace_parser.add_argument("--label-b", default=None, help="Label for second trace")
    trace_parser.add_argument("--out", default="grad_trace.png", help="Output image path")
    trace_parser.add_argument(
        "--param-contains",
        default=None,
        help="Only include parameters whose names contain this substring (e.g. 'lora').",
    )

    args = parser.parse_args()

    if args.cmd == "compare":
        grad_a = load_grad(args.grad_a)
        grad_b = load_grad(args.grad_b)
        key_filter = (lambda k: args.param_contains in k) if args.param_contains else None
        compare(grad_a, grad_b, top_k=args.top_k, key_filter=key_filter, label_a="gradA", label_b="gradB")
        return

    if args.cmd == "compare-lora":
        weights_a = load_lora_weights(args.adapter_a)
        weights_b = load_lora_weights(args.adapter_b)
        key_filter = (lambda k: args.param_contains in k) if args.param_contains else None
        compare(weights_a, weights_b, top_k=args.top_k, key_filter=key_filter, label_a="adapterA", label_b="adapterB")
        return

    if args.cmd == "trace":
        label_a = args.label_a or Path(args.grad_dir_a).name
        label_b = args.label_b or Path(args.grad_dir_b).name
        key_filter = (lambda k: args.param_contains in k) if args.param_contains else None
        _plot_trace(
            args.grad_dir_a,
            args.grad_dir_b,
            metric=args.metric,
            label_a=label_a,
            label_b=label_b,
            out_path=args.out,
            key_filter=key_filter,
        )
        return

    raise SystemExit("Unknown command. Use 'compare' or 'trace'.")


if __name__ == "__main__":
    main()
