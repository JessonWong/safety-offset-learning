import json
import math
from collections import defaultdict

from safeoff.paths import project_root

import torch
from safetensors import safe_open

full_path = project_root() / "finetune_model/LATharm/1harm_full/Llama-3.2-1B-Instruct/model/gradients/step-2.pt"
lora_grad_path = project_root() / "finetune_model/LATharm/1harm_lora/Llama-3.2-1B-Instruct/model/gradients/step-2.pt"
adapter_path = project_root() / "finetune_model/LATharm/1harm_lora/Llama-3.2-1B-Instruct/model/checkpoint-3/adapter_model.safetensors"
adapter_cfg_path = project_root() / "finetune_model/LATharm/1harm_lora/Llama-3.2-1B-Instruct/model/checkpoint-3/adapter_config.json"


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def load_safetensors(path: str) -> dict:
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def layer_keys(layer: int, proj: str):
    prefix = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
    return {
        "full": f"model.layers.{layer}.self_attn.{proj}.weight",
        "A_grad": f"{prefix}.lora_A.default.weight",
        "B_grad": f"{prefix}.lora_B.default.weight",
        "A_weight": f"{prefix}.lora_A.weight",
        "B_weight": f"{prefix}.lora_B.weight",
    }


def main():
    full = torch.load(full_path, map_location="cpu")
    lora_grads = torch.load(lora_grad_path, map_location="cpu")
    lora_weights = load_safetensors(adapter_path)
    cfg = json.load(open(adapter_cfg_path))
    scale = cfg["lora_alpha"] / cfg["r"]

    layer_results = []
    all_proj_vecs = []
    all_lora_vecs = []

    for layer in range(64):  # generous upper bound, will skip missing
        for proj in ("q_proj", "v_proj"):
            names = layer_keys(layer, proj)
            if names["full"] not in full or names["A_grad"] not in lora_grads:
                continue

            G = full[names["full"]].float()  # full gradient for the weight
            A = lora_weights[names["A_weight"]].float()
            B = lora_weights[names["B_weight"]].float()
            gA = lora_grads[names["A_grad"]].float()
            gB = lora_grads[names["B_grad"]].float()

            # Project the full weight gradient into the LoRA parameter space
            proj_A = scale * (B.T @ G)          # matches shape of A
            proj_B = scale * (G @ A.T)          # matches shape of B

            proj_vec = torch.cat([proj_A.flatten(), proj_B.flatten()])
            lora_vec = torch.cat([gA.flatten(), gB.flatten()])

            layer_results.append(
                {
                    "layer": layer,
                    "proj": proj,
                    "cosine": cosine(proj_vec, lora_vec),
                    "proj_norm": proj_vec.norm().item(),
                    "lora_norm": lora_vec.norm().item(),
                }
            )
            all_proj_vecs.append(proj_vec)
            all_lora_vecs.append(lora_vec)

    if not layer_results:
        print("No overlapping layers found between full gradients and LoRA grads.")
        return

    global_cos = cosine(torch.cat(all_proj_vecs), torch.cat(all_lora_vecs))

    print(f"LoRA scale (alpha/r): {scale}")
    print(f"Compared {len(layer_results)} (layer, q/v) pairs")
    print(f"Global cosine between projected full grads and LoRA grads: {global_cos:.4f}")
    print("\nPer-layer cosine (projected full -> LoRA params):")
    for r in layer_results:
        print(
            f"layer {r['layer']:2d} {r['proj']:6s} | cos={r['cosine']:.4f} "
            f"| ||proj||={r['proj_norm']:.4f} ||lora||={r['lora_norm']:.4f}"
        )


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
