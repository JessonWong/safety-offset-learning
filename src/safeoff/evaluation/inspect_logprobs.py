import argparse
import copy
import math
import os

import torch
from easydict import EasyDict
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

from safeoff.data.config import configs, format
from safeoff.data.loaders import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a model, generate responses on AdvEval prompts, and print per-step top-k log probabilities."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to a full model checkpoint or a LoRA adapter checkpoint.")
    parser.add_argument("--base_model_path", type=str, default=None, help="Optional base model path. Needed only if LoRA base model cannot be inferred.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of AdvEval test samples to print.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index inside the AdvEval test split.")
    parser.add_argument("--top_k", type=int, default=10, help="How many top token log probabilities to print per generated step.")
    parser.add_argument("--tokens", type=str, default="I,Sorry,Sure", help="Comma-separated token strings to inspect at each generated step.")
    parser.add_argument("--max_length", type=int, default=None, help="Optional tokenizer max_length for prompt encoding.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    parser.add_argument("--harmful_prompt_prefix", type=str, default=None, help="Optional prefix added before each harmful prompt.")
    parser.add_argument("--harmful_prompt_suffix", type=str, default=None, help="Optional suffix added after each harmful prompt.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype.")
    return parser.parse_args()


def resolve_dtype(dtype_name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def apply_prompt_affixes(prompt, prefix=None, suffix=None):
    pieces = []
    if prefix and prefix.strip():
        pieces.append(prefix.strip())
    pieces.append(prompt)
    if suffix and suffix.strip():
        pieces.append(suffix.strip())
    return " ".join(pieces)


def parse_tokens(raw_tokens):
    return [token.strip() for token in raw_tokens.split(",") if token.strip()]


def load_model_and_tokenizer(model_path, base_model_path=None, dtype=torch.bfloat16):
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        peft_config = PeftConfig.from_pretrained(model_path)
        base_path = base_model_path or peft_config.base_model_name_or_path
        if base_path is None:
            raise ValueError("LoRA adapter detected, but base model path could not be inferred. Please pass --base_model_path.")
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
    model.eval()
    return model, tokenizer


def load_adveval_dataset():
    dataset_config = EasyDict(copy.deepcopy(configs["adveval_dataset_config"]))
    dataset_split = prepare_dataset(dataset_config)
    return dataset_split["test"], dataset_config.system


def format_prompt(tokenizer, prompt, system):
    return format(
        {
            "system": system,
            "prompt": prompt,
        },
        tokenizer=tokenizer,
    )


def resolve_token_id(tokenizer, token_text):
    token_pieces = tokenizer.tokenize(token_text)
    if not token_pieces:
        return None
    token_id = tokenizer.convert_tokens_to_ids(token_pieces[0])
    return int(token_id)


def get_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def generate_with_logprobs(model, tokenizer, prompt, system, max_new_tokens, max_length=None):
    formatted_prompt = format_prompt(tokenizer, prompt, system)
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_length,
        truncation=max_length is not None,
    ).to(get_input_device(model))

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )

    prompt_length = int(inputs.input_ids.shape[1])
    generated_ids = outputs.sequences[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response, generated_ids, outputs.scores


def print_selected_token_stats(tokenizer, step_logprobs, inspect_tokens):
    if not inspect_tokens:
        return
    print("  Selected tokens:")
    for raw_token in inspect_tokens:
        token_id = resolve_token_id(tokenizer, raw_token)
        if token_id is None:
            print(f"    token={repr(raw_token):<20} not tokenized")
            continue
        token_text = tokenizer.convert_ids_to_tokens(token_id)
        logprob = float(step_logprobs[token_id].item())
        prob = math.exp(logprob)
        print(f"    token={repr(raw_token):<20} piece={repr(token_text):<20} id={token_id:<8d} logprob={logprob:>10.6f} prob={prob:.6e}")


def print_generation_logprobs(tokenizer, generated_ids, scores, inspect_tokens, top_k):
    print("Per-step top-k log probabilities:")
    if len(scores) == 0:
        print("  No new tokens were generated.")
        return

    for step_idx, (generated_token_id, step_scores) in enumerate(zip(generated_ids.tolist(), scores), start=1):
        step_logprobs = torch.log_softmax(step_scores[0], dim=-1)
        step_top_k = min(max(0, int(top_k)), int(step_logprobs.shape[-1]))
        generated_token = tokenizer.convert_ids_to_tokens(int(generated_token_id))
        generated_logprob = float(step_logprobs[generated_token_id].item())
        generated_prob = math.exp(generated_logprob)

        print(
            f"  Step {step_idx:3d}: generated id={generated_token_id:<8d} "
            f"token={repr(generated_token):<20} logprob={generated_logprob:>10.6f} prob={generated_prob:.6e}"
        )

        if step_top_k > 0:
            top_values, top_indices = torch.topk(step_logprobs, k=step_top_k)
            for rank, (token_id, logprob) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
                token_text = tokenizer.convert_ids_to_tokens(int(token_id))
                prob = math.exp(logprob)
                print(
                    f"    {rank:2d}. id={token_id:<8d} token={repr(token_text):<20} "
                    f"logprob={logprob:>10.6f} prob={prob:.6e}"
                )

        print_selected_token_stats(tokenizer, step_logprobs, inspect_tokens)


def main():
    args = parse_args()
    inspect_tokens = parse_tokens(args.tokens)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        base_model_path=args.base_model_path,
        dtype=resolve_dtype(args.dtype),
    )
    dataset, system = load_adveval_dataset()

    start = max(0, args.start_index)
    end = min(len(dataset), start + max(0, args.num_samples))
    if start >= len(dataset):
        raise ValueError(f"start_index={start} is out of range for dataset size {len(dataset)}")

    print(f"Using AdvEval test split with {len(dataset)} samples")
    print(f"Printing samples [{start}, {end})")
    print(f"System prompt: {repr(system)}")
    print("")

    for sample_idx in range(start, end):
        sample = dataset[sample_idx]
        prompt = apply_prompt_affixes(
            sample["prompt"],
            prefix=args.harmful_prompt_prefix,
            suffix=args.harmful_prompt_suffix,
        )
        print("=" * 100)
        print(f"Sample {sample_idx}")
        print(f"Prompt: {prompt}")
        response, generated_ids, scores = generate_with_logprobs(
            model,
            tokenizer,
            prompt,
            system,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
        )
        print(f"Response: {response}")
        print_generation_logprobs(
            tokenizer,
            generated_ids,
            scores,
            inspect_tokens=inspect_tokens,
            top_k=args.top_k,
        )
        print("")


if __name__ == "__main__":
    main()
