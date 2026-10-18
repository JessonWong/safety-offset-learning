import argparse
import sys
from pathlib import Path


DEFAULT_ROOT = Path("./finetune_model/LATharm")
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
EXIT_COMMANDS = {":q", ":quit", "exit", "quit"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Single-turn command-line chat for full-model or LoRA checkpoints saved "
            "under finetune_model/LATharm."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Fine-tuned model root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default=None,
        help="Experiment directory under --root, e.g. test_advprompt_merge_v4.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint step, checkpoint directory name, or 'latest'. Default: latest.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Model subdirectory under the experiment. Defaults to the basename of "
            "--base_model_path; auto-selects if the experiment has one model subdir."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Direct path/name to load. If it contains adapter_config.json, it is "
            "loaded as a LoRA adapter on top of --base_model_path."
        ),
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model for LoRA checkpoints. Default: {DEFAULT_BASE_MODEL}",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt. Use an empty string to omit it.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run one prompt and exit. If omitted, starts an interactive loop.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling. Pass --no-do_sample for greedy decoding.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype. Default: auto.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Transformers device_map. Use 'none' to disable. Default: auto.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--show_special_tokens",
        action="store_true",
        help="Decode output without skipping special tokens.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments/checkpoints found under --root and exit.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return -1


def checkpoint_dirs(model_root: Path) -> list[Path]:
    return sorted(model_root.glob("checkpoint-*"), key=checkpoint_step)


def list_saved_checkpoints(root: Path) -> int:
    rows = []
    for model_root in sorted(root.glob("*/*/model")):
        checkpoints = checkpoint_dirs(model_root)
        if not checkpoints:
            continue
        latest = checkpoints[-1]
        rows.append(
            (
                model_root.parent.parent.name,
                model_root.parent.name,
                latest.name,
                len(checkpoints),
            )
        )

    if not rows:
        print(f"No checkpoints found under {root}")
        return 1

    print(f"Checkpoints under {root}:")
    for experiment, model_name, latest, count in rows:
        print(
            f"  {experiment}/{model_name}: latest={latest} "
            f"({count} checkpoint{'s' if count != 1 else ''})"
        )
    return 0


def resolve_checkpoint_path(args) -> Path | str:
    if args.model_path:
        return args.model_path

    if not args.experiment:
        raise ValueError(
            "Pass --experiment, --model_path, or use --list to see available checkpoints."
        )

    experiment_dir = args.root.expanduser() / args.experiment
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    model_name = args.model_name or Path(args.base_model_path.rstrip("/")).name
    model_root = experiment_dir / model_name / "model"
    if not model_root.exists() and args.model_name is None:
        candidates = sorted(path for path in experiment_dir.glob("*/model") if path.is_dir())
        if len(candidates) == 1:
            model_root = candidates[0]
        elif candidates:
            names = ", ".join(path.parent.name for path in candidates)
            raise ValueError(
                f"Multiple model directories found under {experiment_dir}: {names}. "
                "Pass --model_name."
            )
        else:
            raise FileNotFoundError(f"No model directories found under {experiment_dir}")

    if args.checkpoint == "latest":
        checkpoints = checkpoint_dirs(model_root)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint-* directories under {model_root}")
        return checkpoints[-1]

    checkpoint_name = args.checkpoint
    if checkpoint_name.isdigit():
        checkpoint_name = f"checkpoint-{checkpoint_name}"
    checkpoint_path = model_root / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_dtype(dtype_name: str):
    import torch

    if dtype_name == "float32" or not torch.cuda.is_available():
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_device_map(device_map: str):
    import torch

    if device_map.lower() in {"none", "null"}:
        return None
    if not torch.cuda.is_available() and device_map == "auto":
        return None
    return device_map


def load_model_and_tokenizer(args, model_path: Path | str):
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path_str = str(model_path)
    model_path_obj = Path(model_path_str)
    adapter_config = model_path_obj / "adapter_config.json"
    dtype = resolve_dtype(args.dtype)
    device_map = resolve_device_map(args.device_map)
    common_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }

    if adapter_config.exists():
        peft_config = PeftConfig.from_pretrained(model_path_str)
        base_model_path = args.base_model_path or peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, **common_kwargs)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **common_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_path_str)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path_str, **common_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_str,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **common_kwargs,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def format_prompt(tokenizer, user_prompt: str, system_prompt: str | None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prefix = f"{system_prompt.strip()}\n\n" if system_prompt else ""
    return f"{prefix}User: {user_prompt}\nAssistant:"


def eos_token_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            continue
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    ids = sorted(set(ids))
    if not ids:
        return None
    return ids[0] if len(ids) == 1 else ids


def generate_once(model, tokenizer, args, user_prompt: str) -> str:
    import torch
    from transformers import GenerationConfig

    formatted = format_prompt(tokenizer, user_prompt, args.system.strip() or None)
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(input_device(model))

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_ids(tokenizer),
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    generation_config = GenerationConfig(**generation_kwargs)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config,
        )[0]

    prompt_tokens = inputs.input_ids.shape[-1]
    generated_ids = output_ids[prompt_tokens:]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=not args.show_special_tokens,
    ).strip()


def interactive_chat(model, tokenizer, args):
    print("Single-turn chat. Type :q, quit, or exit to stop.")
    while True:
        try:
            user_prompt = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_prompt:
            continue
        if user_prompt.lower() in EXIT_COMMANDS:
            return

        response = generate_once(model, tokenizer, args, user_prompt)
        print(f"\nassistant> {response}")


def main():
    args = parse_args()
    args.root = args.root.expanduser()

    if args.list:
        raise SystemExit(list_saved_checkpoints(args.root))

    try:
        model_path = resolve_checkpoint_path(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Tip: run `safeoff chat --list`", file=sys.stderr)
        raise SystemExit(2)

    print(f"Loading model: {model_path}")
    model, tokenizer = load_model_and_tokenizer(args, model_path)

    if args.prompt is not None:
        print(generate_once(model, tokenizer, args, args.prompt))
        return

    interactive_chat(model, tokenizer, args)


if __name__ == "__main__":
    main()
