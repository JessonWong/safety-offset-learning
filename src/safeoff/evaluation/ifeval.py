#!/usr/bin/env python3
"""Evaluate official IFEval metrics across a collection of LoRA checkpoints.

The script intentionally keeps generation and scoring separate at each checkpoint:
raw responses are saved first, then scored with the IFEval implementation vendored
by lm-evaluation-harness.  Completed checkpoints are resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_LABELS = {
    "newtarget": "new target",
    "benign_alpaca": "benign Alpaca",
    "aoa": "AoA",
    "alpaca_top100": "Alpaca top-100",
    "harmful_llm_lat": "LLM-LAT harmful-100",
}

RESULT_FIELDS = [
    "dataset",
    "label",
    "step",
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
    "prompt_count",
    "instruction_count",
    "mean_generated_tokens",
    "max_generated_tokens",
    "max_token_truncation_count",
    "generation_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-parent", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LABELS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ifeval-source-root", default="cache/ifeval_official/lm-evaluation-harness")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--step-interval", type=int, default=25)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(f"Invalid checkpoint name: {path}")
    return int(match.group(1))


def adapter_weights_path(checkpoint: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = checkpoint / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No adapter weights under {checkpoint}")


def load_adapter_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def set_adapter_state(model, state: dict[str, torch.Tensor]) -> None:
    from peft import set_peft_model_state_dict

    result = set_peft_model_state_dict(model, state, adapter_name="default")
    unexpected = getattr(result, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected[:10]}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rows_hash(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_official_scorer(source_root: Path):
    source_root = source_root.resolve()
    if not (source_root / "lm_eval/tasks/ifeval/utils.py").is_file():
        raise FileNotFoundError(f"IFEval scorer not found under {source_root}")
    sys.path.insert(0, str(source_root))
    from lm_eval.tasks.ifeval.utils import (  # pylint: disable=import-outside-toplevel
        InputExample,
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    return InputExample, test_instruction_following_strict, test_instruction_following_loose


def apply_chat_template(tokenizer, prompt: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )


def collate_left_padded(
    encoded_prompts: list[list[int]], pad_token_id: int, device: torch.device
) -> dict[str, torch.Tensor]:
    width = max(len(tokens) for tokens in encoded_prompts)
    input_ids = torch.full(
        (len(encoded_prompts), width), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(encoded_prompts), width), dtype=torch.long, device=device
    )
    for index, tokens in enumerate(encoded_prompts):
        length = len(tokens)
        input_ids[index, -length:] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention_mask[index, -length:] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def eos_token_ids(tokenizer) -> int | list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for token in ("<|eot_id|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id and token_id not in ids:
            ids.append(int(token_id))
    if not ids:
        raise ValueError("Tokenizer has no usable EOS token")
    return ids[0] if len(ids) == 1 else ids


@torch.inference_mode()
def generate_responses(
    model,
    tokenizer,
    docs: list[dict[str, Any]],
    encoded_prompts: list[list[int]],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    progress = tqdm(range(0, len(docs), batch_size), desc="IFEval generation", leave=False)
    for start in progress:
        current_docs = docs[start : start + batch_size]
        current_tokens = encoded_prompts[start : start + batch_size]
        batch = collate_left_padded(current_tokens, tokenizer.pad_token_id, device)
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids(tokenizer),
        )
        continuation = generated[:, batch["input_ids"].shape[1] :]
        for doc, prompt_ids, output_ids in zip(current_docs, current_tokens, continuation):
            output_list = output_ids.detach().cpu().tolist()
            while output_list and output_list[-1] == tokenizer.pad_token_id:
                output_list.pop()
            response = tokenizer.decode(output_list, skip_special_tokens=True)
            records.append(
                {
                    "key": int(doc["key"]),
                    "prompt": doc["prompt"],
                    "instruction_id_list": doc["instruction_id_list"],
                    "kwargs": doc["kwargs"],
                    "response": response,
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": len(output_list),
                }
            )
        del batch, generated, continuation
    return records, time.perf_counter() - started


def generate_responses_vllm(
    llm,
    tokenizer,
    docs: list[dict[str, Any]],
    encoded_prompts: list[list[int]],
    checkpoint: Path,
    lora_int_id: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], float]:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    all_eos = eos_token_ids(tokenizer)
    stop_token_ids = [all_eos] if isinstance(all_eos, int) else all_eos
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=True,
    )
    token_prompts = [{"prompt_token_ids": prompt_ids} for prompt_ids in encoded_prompts]
    lora_request = LoRARequest(checkpoint.name, lora_int_id, str(checkpoint))
    started = time.perf_counter()
    outputs = llm.generate(
        token_prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - started
    records = []
    for doc, prompt_ids, request_output in zip(docs, encoded_prompts, outputs):
        if len(request_output.outputs) != 1:
            raise RuntimeError("Expected one vLLM output per prompt")
        completion = request_output.outputs[0]
        output_ids = list(completion.token_ids)
        records.append(
            {
                "key": int(doc["key"]),
                "prompt": doc["prompt"],
                "instruction_id_list": doc["instruction_id_list"],
                "kwargs": doc["kwargs"],
                "response": completion.text,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(output_ids),
            }
        )
    return records, elapsed


def score_responses(records, scorer) -> dict[str, Any]:
    InputExample, test_strict, test_loose = scorer
    prompt_strict = []
    prompt_loose = []
    instruction_strict = []
    instruction_loose = []
    for record in records:
        inp = InputExample(
            key=record["key"],
            instruction_id_list=record["instruction_id_list"],
            prompt=record["prompt"],
            kwargs=record["kwargs"],
        )
        strict = test_strict(inp, record["response"])
        loose = test_loose(inp, record["response"])
        prompt_strict.append(strict.follow_all_instructions)
        prompt_loose.append(loose.follow_all_instructions)
        instruction_strict.extend(strict.follow_instruction_list)
        instruction_loose.extend(loose.follow_instruction_list)
        record["prompt_strict_pass"] = strict.follow_all_instructions
        record["prompt_loose_pass"] = loose.follow_all_instructions
        record["instruction_strict_pass"] = strict.follow_instruction_list
        record["instruction_loose_pass"] = loose.follow_instruction_list
    return {
        "prompt_level_strict_acc": float(np.mean(prompt_strict)),
        "inst_level_strict_acc": float(np.mean(instruction_strict)),
        "prompt_level_loose_acc": float(np.mean(prompt_loose)),
        "inst_level_loose_acc": float(np.mean(instruction_loose)),
        "prompt_count": len(prompt_strict),
        "instruction_count": len(instruction_strict),
    }


def seed_official_scorer(seed: int) -> None:
    """Make langdetect and any fallback instruction defaults reproducible."""
    random.seed(seed)
    try:
        from langdetect import DetectorFactory

        DetectorFactory.seed = seed
    except ImportError:
        pass


def write_results_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in RESULT_FIELDS} for row in records)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--batch-size and --max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_parent = Path(args.checkpoint_parent).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_root = Path(args.ifeval_source_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = load_official_scorer(source_root)

    dataset = load_dataset("google/IFEval", split="train")
    docs = [dict(row) for row in dataset]
    if args.limit is not None:
        docs = docs[: args.limit]
    dataset_hash = canonical_rows_hash(docs)
    atomic_jsonl(output_dir / "ifeval_prompts.jsonl", docs)

    checkpoint_sets = {}
    for dataset_name in args.datasets:
        checkpoint_root = checkpoint_parent / dataset_name
        checkpoints = sorted(
            (path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )
        checkpoints = [
            path for path in checkpoints if checkpoint_step(path) % args.step_interval == 0
        ]
        if args.steps is not None:
            requested_steps = set(args.steps)
            checkpoints = [path for path in checkpoints if checkpoint_step(path) in requested_steps]
        if not checkpoints:
            raise RuntimeError(f"No checkpoints found under {checkpoint_root}")
        checkpoint_sets[dataset_name] = checkpoints

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_prompts = [apply_chat_template(tokenizer, doc["prompt"]) for doc in docs]

    first_checkpoint = checkpoint_sets[args.datasets[0]][0]
    device = torch.device(args.device)
    model = None
    llm = None
    if args.backend == "hf":
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=resolve_dtype(args.dtype),
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)
        base_model.config.use_cache = True
        model = PeftModel.from_pretrained(base_model, first_checkpoint, is_trainable=False)
        model.eval()
    else:
        from vllm import LLM

        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            dtype=args.dtype,
            enable_lora=True,
            max_lora_rank=8,
            max_loras=1,
            max_cpu_loras=2,
            max_model_len=args.max_model_len,
            max_num_seqs=max(1, args.batch_size),
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=False,
        )

    source_commit = "unknown"
    git_head = source_root / ".git/HEAD"
    if git_head.is_file():
        head = git_head.read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_path = source_root / ".git" / head[5:]
            if ref_path.is_file():
                source_commit = ref_path.read_text(encoding="utf-8").strip()
        else:
            source_commit = head

    records: list[dict[str, Any]] = []
    lora_int_id = 0
    identical_checkpoint_cache: dict[str, dict[str, Any]] = {}
    for dataset_name in args.datasets:
        dataset_output = output_dir / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        for checkpoint in checkpoint_sets[dataset_name]:
            step = checkpoint_step(checkpoint)
            result_path = dataset_output / f"checkpoint-{step}.metrics.json"
            generations_path = dataset_output / f"checkpoint-{step}.generations.jsonl"
            if result_path.is_file() and generations_path.is_file() and not args.overwrite:
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("ifeval_rows_sha256") != dataset_hash:
                    raise RuntimeError(f"Dataset hash mismatch in {result_path}")
                record = saved["metrics"]
                records.append(record)
                print(f"resume dataset={dataset_name} step={step}", flush=True)
                continue

            weights = adapter_weights_path(checkpoint)
            adapter_sha256 = sha256_file(weights)
            lora_int_id += 1
            reused = identical_checkpoint_cache.get(adapter_sha256)
            if reused is not None:
                generations = copy.deepcopy(reused["generations"])
                generation_seconds = 0.0
                print(
                    f"reuse identical adapter dataset={dataset_name} step={step} "
                    f"from={reused['dataset']}/checkpoint-{reused['step']}",
                    flush=True,
                )
            elif args.backend == "hf":
                state = load_adapter_state(weights)
                set_adapter_state(model, state)
                del state
                torch.cuda.empty_cache()
            if reused is None:
                print(f"generate dataset={dataset_name} step={step}", flush=True)
            if reused is None and args.backend == "hf":
                generations, generation_seconds = generate_responses(
                    model,
                    tokenizer,
                    docs,
                    encoded_prompts,
                    args.batch_size,
                    args.max_new_tokens,
                    device,
                )
            elif reused is None:
                generations, generation_seconds = generate_responses_vllm(
                    llm,
                    tokenizer,
                    docs,
                    encoded_prompts,
                    checkpoint,
                    lora_int_id,
                    args.max_new_tokens,
                )
            seed_official_scorer(args.seed)
            metrics = score_responses(generations, scorer)
            generated_lengths = [record["generated_tokens"] for record in generations]
            metrics.update(
                {
                    "dataset": dataset_name,
                    "label": DATASET_LABELS.get(dataset_name, dataset_name),
                    "step": step,
                    "mean_generated_tokens": float(np.mean(generated_lengths)),
                    "max_generated_tokens": max(generated_lengths),
                    "max_token_truncation_count": sum(
                        length >= args.max_new_tokens for length in generated_lengths
                    ),
                    "generation_seconds": generation_seconds,
                }
            )
            if reused is None:
                identical_checkpoint_cache[adapter_sha256] = {
                    "dataset": dataset_name,
                    "step": step,
                    "generations": copy.deepcopy(generations),
                }
            atomic_jsonl(generations_path, generations)
            payload = {
                "model": args.model,
                "checkpoint": str(checkpoint),
                "adapter_sha256": adapter_sha256,
                "generation_reused": reused is not None,
                "generation_reused_from": (
                    None
                    if reused is None
                    else f"{reused['dataset']}/checkpoint-{reused['step']}"
                ),
                "ifeval_dataset": "google/IFEval",
                "ifeval_split": "train",
                "ifeval_rows_sha256": dataset_hash,
                "ifeval_scorer_root": str(source_root),
                "ifeval_scorer_commit": source_commit,
                "apply_chat_template": True,
                "system_prompt": None,
                "generation": {
                    "backend": args.backend,
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                    "batch_size": args.batch_size,
                    "dtype": args.dtype,
                    "seed": args.seed,
                },
                "generations_sha256": sha256_file(generations_path),
                "metrics": metrics,
            }
            atomic_json(result_path, payload)
            records.append(metrics)
            write_results_csv(output_dir / "ifeval_checkpoint_metrics.csv", records)
            print(
                f"done dataset={dataset_name} step={step} "
                f"strict={metrics['prompt_level_strict_acc']:.4f} "
                f"loose={metrics['prompt_level_loose_acc']:.4f} "
                f"seconds={generation_seconds:.1f}",
                flush=True,
            )

    records.sort(key=lambda row: (args.datasets.index(row["dataset"]), row["step"]))
    write_results_csv(output_dir / "ifeval_checkpoint_metrics.csv", records)
    manifest = {
        "model": args.model,
        "checkpoint_parent": str(checkpoint_parent),
        "datasets": args.datasets,
        "output_dir": str(output_dir),
        "ifeval_dataset": "google/IFEval",
        "ifeval_split": "train",
        "ifeval_prompt_count": len(docs),
        "ifeval_rows_sha256": dataset_hash,
        "ifeval_scorer_root": str(source_root),
        "ifeval_scorer_commit": source_commit,
        "generation": {
            "backend": args.backend,
            "apply_chat_template": True,
            "system_prompt": None,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "prompt_token_length": {
            "min": min(map(len, encoded_prompts)),
            "mean": float(np.mean(list(map(len, encoded_prompts)))),
            "max": max(map(len, encoded_prompts)),
        },
        "metrics_file": str(output_dir / "ifeval_checkpoint_metrics.csv"),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
