from safeoff.paths import project_root
import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = project_root()

from safeoff.data.config import format


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a base/full/LoRA model and print the complete generated output with special tokens preserved."
    )
    parser.add_argument("--base_model_path", type=str, required=True, help="Base model path.")
    parser.add_argument("--model_path", type=str, default=None, help="Optional full-model checkpoint path. If omitted, base_model_path is used.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA adapter checkpoint path.")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt to generate from.")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="System prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype.")
    parser.add_argument("--do_sample", type=int, default=0, help="Whether to sample during generation.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when do_sample=1.")
    return parser.parse_args()


def resolve_dtype(dtype_name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def load_model_and_tokenizer(base_model_path, model_path=None, lora_path=None, dtype=torch.bfloat16):
    target_model_path = model_path or base_model_path

    if lora_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        return model, tokenizer

    adapter_config_path = os.path.join(target_model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        peft_config = PeftConfig.from_pretrained(target_model_path)
        resolved_base_path = base_model_path or peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(resolved_base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_base_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, target_model_path)
        model.eval()
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def get_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        model_path=args.model_path,
        lora_path=args.lora_path,
        dtype=resolve_dtype(args.dtype),
    )

    formatted_prompt = format(
        {
            "system": args.system,
            "prompt": args.prompt,
        },
        tokenizer=tokenizer,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(get_input_device(model))

    generate_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": bool(args.do_sample),
    }
    if bool(args.do_sample):
        generate_kwargs["temperature"] = args.temperature

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    prompt_length = int(inputs.input_ids.shape[1])
    full_ids = outputs[0].detach().cpu().tolist()
    generated_ids = outputs[0].detach().cpu().tolist()

    print("Formatted prompt:")
    print(repr(formatted_prompt))
    print("")
    print("Generated token ids:")
    print(generated_ids)
    print("")
    print("Generated tokens:")
    print(tokenizer.convert_ids_to_tokens(generated_ids))
    print("")
    print("Generated text with skip_special_tokens=False:")
    print(repr(tokenizer.decode(generated_ids, skip_special_tokens=False)))
    print("")
    print("Generated text with skip_special_tokens=True:")
    print(repr(tokenizer.decode(generated_ids, skip_special_tokens=True)))
    print("")
    print("Full output tail tokens:")
    print(tokenizer.convert_ids_to_tokens(full_ids[-min(len(full_ids), 80):]))


if __name__ == "__main__":
    main()
