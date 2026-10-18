import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect saved gradient checkpoint (.pt).")
    parser.add_argument("path", help="Path to step-*.pt saved by GradientSavingTrainer")
    parser.add_argument("--contains", default=None, help="Only show keys containing this substring")
    parser.add_argument("--top", type=int, default=2000, help="Max keys to display")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)

    grads = torch.load(path, map_location="cpu")
    keys = list(grads.keys())
    if args.contains:
        keys = [k for k in keys if args.contains in k]

    print(f"Loaded {len(grads)} total keys; showing {min(args.top, len(keys))} keys.")
    for i, k in enumerate(keys[: args.top]):
        v = grads[k]
        shape = tuple(v.shape) if torch.is_tensor(v) else None
        norm = v.float().norm().item() if torch.is_tensor(v) else None
        print(f"{i:03d}  {k}  shape={shape}  l2={norm:.6f}")

    # Quick summary for LoRA deltaW presence.
    delta_keys = [k for k in grads.keys() if k.endswith(".deltaW")]
    if delta_keys:
        print(f"\nFound {len(delta_keys)} deltaW entries (example: {delta_keys[0]})")
    else:
        print("\nNo deltaW entries found.")


if __name__ == "__main__":
    main()
