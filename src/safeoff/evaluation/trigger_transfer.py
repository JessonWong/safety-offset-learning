import argparse
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch.nn.functional as F

from safeoff.generation.gradient_matching import _compute_target_gradients, decode  # noqa: E402
from safeoff.core.model_utils import format as chat_format  # noqa: E402


def parse_trigger(trigger_str, trigger_file, tokenizer):
    if trigger_file:
        with open(trigger_file, "r") as f:
            data = json.load(f)
        # support {"trigger": [...]} or a bare list
        if isinstance(data, dict) and "trigger" in data:
            return torch.tensor(data["trigger"], dtype=torch.long)
        if isinstance(data, list):
            return torch.tensor(data, dtype=torch.long)
        raise ValueError("Unsupported trigger file format; expected list or {'trigger': list}.")
    if trigger_str:
        # Try numeric IDs first.
        pieces = [p.strip() for p in trigger_str.replace(",", " ").split() if p.strip()]
        try:
            ids = [int(p) for p in pieces]
            return torch.tensor(ids, dtype=torch.long)
        except ValueError:
            # Treat as raw string to tokenize.
            ids = tokenizer(trigger_str, add_special_tokens=False, return_tensors="pt").input_ids[0]
            return ids.detach().cpu()
    raise ValueError("Must provide --trigger or --trigger_file.")


def compute_trigger_similarity(
    model,
    target_gradients,
    trigger_tokens,
    prompt_ids,
    response_ids,
):
    device = next(model.parameters()).device
    trigger_tokens = trigger_tokens.to(device)
    prompt_ids = prompt_ids.to(device)
    response_ids = response_ids.to(device)

    full_input = torch.cat([trigger_tokens, prompt_ids, response_ids], dim=-1).unsqueeze(0)
    labels = torch.cat(
        [
            torch.full_like(trigger_tokens, -100),
            torch.full_like(prompt_ids, -100),
            response_ids,
        ],
        dim=-1,
    ).unsqueeze(0)
    attention_mask = torch.ones_like(full_input, device=device)

    model.zero_grad(set_to_none=True)
    loss = model(input_ids=full_input, attention_mask=attention_mask, labels=labels).loss
    loss.backward()

    base_modules = dict(model.named_modules())
    total_sim, count = 0.0, 0
    for name, target_grad in target_gradients.items():
        module = base_modules.get(name, None)
        if module is None or not hasattr(module, "weight") or module.weight.grad is None:
            continue
        sim = F.cosine_similarity(
            module.weight.grad.flatten(),
            target_grad.to(module.weight.grad.device).flatten(),
            dim=0,
        )
        total_sim += float(sim.item())
        count += 1
    return (total_sim / count) if count > 0 else None


def parse_args():
    parser = argparse.ArgumentParser(description="Test trigger transfer on a new benign prompt/response.")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./finetune_model/LATharm/pure_harm/Llama-3.1-8B-Instruct/model/checkpoint-36",
    )
    parser.add_argument("--trigger", type=str, help="Trigger tokens as text or comma/space-separated ids.")
    parser.add_argument("--trigger_file", type=str, help="JSON file containing trigger list or {'trigger': list}.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--benign_prompt", type=str, default="List three strategies to save energy at home.")
    parser.add_argument(
        "--benign_response",
        type=str,
        default="Use LED lighting, improve insulation, and unplug devices when not in use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model_with_adapters = PeftModel.from_pretrained(base_model, args.lora_path)
    target_gradients = _compute_target_gradients(model_with_adapters)
    model_with_adapters.disable_adapter()
    for p in model_with_adapters.parameters():
        p.requires_grad_(True)

    trigger_tokens = parse_trigger(args.trigger, args.trigger_file, tokenizer)

    formatted_prompt = chat_format(
        {"system": args.system_prompt, "prompt": args.benign_prompt},
        tokenizer=tokenizer,
    )
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    response_ids = tokenizer(args.benign_response, return_tensors="pt", add_special_tokens=False).input_ids[0]

    sim = compute_trigger_similarity(
        model_with_adapters,
        target_gradients,
        trigger_tokens,
        prompt_ids,
        response_ids,
    )

    print("Trigger tokens:", trigger_tokens.tolist())
    print("Decoded trigger:", decode(trigger_tokens.to(base_model.device), tokenizer=tokenizer))
    if sim is None:
        print("No valid gradients found for similarity computation.")
    else:
        print(f"Average cosine similarity to harmful deltas: {sim:.4f}")


if __name__ == "__main__":
    main()
