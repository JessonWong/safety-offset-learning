import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from safeoff.generation.gradient_matching import gradient_matching_attack, decode, _compute_target_gradients
from safeoff.core.model_utils import format as chat_format  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradient matching trigger search.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./finetune_model/LATharm/pure_harm/Llama-3.1-8B-Instruct/model/checkpoint-36",
        help="Path to harmful LoRA checkpoint.",
    )
    parser.add_argument(
        "--benign_prompt",
        type=str,
        default="Give three tips for staying healthy.",
        help="Benign prompt text.",
    )
    parser.add_argument(
        "--benign_response",
        type=str,
        default="1. Eat a balanced diet with plenty of fruits and vegetables.\n2. Exercise regularly to maintain physical fitness.\n3. Get enough sleep each night to allow your body to recover.",
        help="Expected benign response text.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt used for chat formatting.",
    )
    parser.add_argument("--trigger_len", type=int, default=10, help="Trigger token length.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of search iterations.")
    parser.add_argument("--top_k", type=int, default=64, help="Number of candidates per iteration.")
    parser.add_argument("--sim_threshold", type=float, default=0.9, help="Early stop cosine threshold.")
    parser.add_argument(
        "--score_mode",
        choices=["gradient", "gcg"],
        default="gradient",
        help="Use gradient similarity (default) or GCG log-prob objective to pick candidates.",
    )
    parser.add_argument(
        "--gcg_goal",
        type=str,
        default=None,
        help="Target goal text for the original GCG objective. If set with score_mode=gcg, this replaces benign_response as the optimization target.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(True)

    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model once.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Attach harmful LoRA onto the same weights to build target gradients, then disable it.
    model_with_adapters = PeftModel.from_pretrained(base_model, args.lora_path)
    target_gradients = _compute_target_gradients(model_with_adapters)
    model_with_adapters.disable_adapter()  # keep clean gradients during search
    # Ensure base weights require grad (LoRA training freezes them by default).
    for p in model_with_adapters.parameters():
        p.requires_grad_(True)

    # Apply chat template to prompt; keep response raw so loss is on response span only.
    formatted_prompt = chat_format(
        {"system": args.system_prompt, "prompt": args.benign_prompt},
        tokenizer=tokenizer,
    )
    benign_prompt_ids = tokenizer(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

    target_response_text = args.benign_response
    gcg_goal_ids = None
    if args.score_mode == "gcg":
        # Use provided GCG goal if available; otherwise fall back to benign_response.
        target_response_text = args.gcg_goal or args.benign_response
        gcg_goal_ids = tokenizer(
            target_response_text, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

    benign_response_ids = tokenizer(
        target_response_text, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

    print("Starting gradient matching attack...")
    trigger_tokens = gradient_matching_attack(
        base_model=model_with_adapters,  # same weights; adapters disabled
        benign_input=benign_prompt_ids,
        benign_output=benign_response_ids,
        trigger_len=args.trigger_len,
        iterations=args.iterations,
        top_k=args.top_k,
        sim_threshold=args.sim_threshold,
        tokenizer=tokenizer,
        target_gradients=target_gradients,
        gcg_goal_ids=gcg_goal_ids,
        score_mode=args.score_mode,
    )

    print("Final trigger tokens:", trigger_tokens.tolist())
    print("Decoded trigger:", decode(trigger_tokens, tokenizer=tokenizer))


if __name__ == "__main__":
    main()
