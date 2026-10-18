#!/usr/bin/env python3
"""Evaluate Competition Math accuracy across a trajectory of LoRA checkpoints.

The default dataset and prompt match the repository's ``competition_math``
utility configuration.  Unlike MMLU, MATH requires free-form generation; the
scorer extracts a final ``\\boxed{...}`` answer and applies the normalization
used by the original Hendrycks MATH evaluator.  Raw generations and per-item
predictions are saved so a run can be resumed and audited.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_NAME = "qwedsacf/competition_math"
DEFAULT_SPLIT = "train"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful competition math assistant. Solve the problem step-by-step. "
    "On the final line, write the final answer as \\boxed{...} and keep the boxed "
    "content to the answer only."
)

LEVELS = ["level_1", "level_2", "level_3", "level_4", "level_5"]
TYPES = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
RESULT_FIELDS = [
    "dataset",
    "label",
    "step",
    "overall_accuracy",
    *(f"{level}_accuracy" for level in LEVELS),
    *(f"{problem_type}_accuracy" for problem_type in TYPES),
    "correct_count",
    "sample_count",
    "extraction_success_count",
    "mean_prompt_tokens",
    "max_prompt_tokens",
    "mean_generated_tokens",
    "max_generated_tokens",
    "max_token_truncation_count",
    "generation_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--checkpoint-parent",
        required=True,
        help=(
            "Directory containing checkpoint-* directories. If --datasets is given, "
            "each dataset is looked up as a child directory."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional child-directory names; omitted means the parent itself or auto-discovery.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the selected dataset deterministically before applying --offset/--limit.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--step-interval", type=int, default=25)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    import torch

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
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

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


def write_results_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({field: record.get(field) for field in RESULT_FIELDS} for record in records)
    temporary.replace(path)


def load_math_docs(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    offset: int,
    limit: int | None,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {"path": dataset_name, "split": split}
    if dataset_config is not None:
        load_kwargs["name"] = dataset_config
    dataset = load_dataset(**load_kwargs)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if offset:
        dataset = dataset.select(range(offset, len(dataset)))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    docs = []
    for index, row in enumerate(dataset):
        problem = row.get("problem", row.get("question"))
        solution = row.get("solution", row.get("answer"))
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"MATH row {index} has no non-empty problem field")
        if solution is None:
            raise ValueError(f"MATH row {index} has no solution or answer field")
        docs.append(
            {
                "index": index + offset,
                "problem": problem,
                "solution": str(solution),
                "level": str(row.get("level", "unknown")),
                "type": str(row.get("type", row.get("subject", "unknown"))),
            }
        )
    if not docs:
        raise RuntimeError("No MATH examples loaded")
    return docs


def build_prompt_messages(problem: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": problem.strip()})
    return messages


def encode_prompts(tokenizer, docs: list[dict[str, Any]], system_prompt: str | None) -> list[list[int]]:
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError("The tokenizer has no chat template; MATH evaluation requires one")
    return [
        tokenizer.apply_chat_template(
            build_prompt_messages(doc["problem"], system_prompt),
            tokenize=True,
            add_generation_prompt=True,
        )
        for doc in docs
    ]


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


def collate_left_padded(
    encoded_prompts: list[list[int]], pad_token_id: int, device: torch.device
) -> dict[str, torch.Tensor]:
    import torch

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


def generate_responses_hf(
    model,
    tokenizer,
    docs: list[dict[str, Any]],
    encoded_prompts: list[list[int]],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    import torch

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(docs), batch_size):
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
            records.append(
                {
                    "index": doc["index"],
                    "problem": doc["problem"],
                    "response": tokenizer.decode(output_list, skip_special_tokens=True),
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
        records.append(
            {
                "index": doc["index"],
                "problem": doc["problem"],
                "response": completion.text,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(completion.token_ids),
            }
        )
    return records, elapsed


# The answer extraction and normalization functions from here through math_equiv
# are adapted from hendrycks/math (MIT License, Copyright (c) 2021 Dan Hendrycks).
# See THIRD_PARTY_NOTICES.md.
def last_boxed_only_string(string: str) -> str | None:
    """Return the final balanced ``\\boxed{...}`` or ``\\fbox{...}`` expression."""
    text = str(string)
    boxed_index = text.rfind("\\boxed")
    fbox_index = text.rfind("\\fbox")
    index = max(boxed_index, fbox_index)
    if index < 0:
        return None
    token = "\\boxed" if boxed_index >= fbox_index else "\\fbox"
    space_index = text.rfind(token + " ")
    if space_index >= index:
        return token + " " + text[space_index + len(token) + 1 :].split("$")[0]

    brace_index = text.find("{", index + len(token))
    if brace_index < 0:
        return None
    depth = 0
    for current in range(brace_index, len(text)):
        if text[current] == "{":
            depth += 1
        elif text[current] == "}":
            depth -= 1
            if depth == 0:
                return text[index : current + 1]
    return None


def remove_boxed(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    for token in ("\\boxed", "\\fbox"):
        if text.startswith(token + " "):
            return text[len(token) + 1 :].strip()
        if text.startswith(token + "{") and text.endswith("}"):
            return text[len(token) + 1 : -1].strip()
    return text.strip()


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_string = substrs[0]
    for substr in substrs[1:]:
        new_string += "\\frac"
        if not substr:
            return string
        if substr[0] == "{":
            new_string += substr
            continue
        if len(substr) < 2:
            return string
        a, b = substr[0], substr[1]
        if b != "{":
            new_string += "{" + a + "}{" + b + "}" + substr[2:]
        else:
            new_string += "{" + a + "}" + substr[1:]
    return new_string


def fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    left, right = string.split("/")
    if re.fullmatch(r"\d+", left) and re.fullmatch(r"\d+", right):
        return f"\\frac{{{left}}}{{{right}}}"
    return string


def fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            return string
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string: str) -> str:
    """Normalize a MATH answer using the original benchmark conventions."""
    value = str(string)
    value = value.replace("\n", "")
    value = value.replace("\\!", "")
    value = value.replace("\\\\", "\\")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "")
    value = value.replace("\\$", "")
    value = re.sub(r"\\text\{\s*[^{}]*\}\s*$", "", value)
    value = value.replace("\\%", "").replace("%", "")
    value = value.replace(" .", " 0.").replace("{.", "{0.")
    if not value:
        return value
    if value[0] == ".":
        value = "0" + value
    if len(value.split("=")) == 2 and len(value.split("=")[0]) <= 2:
        value = value.split("=")[1]
    value = fix_sqrt(value)
    value = value.replace(" ", "")
    value = fix_fracs(value)
    if value == "0.5":
        value = "\\frac{1}{2}"
    return fix_a_slash_b(value)


def math_equiv(first: str | None, second: str | None) -> bool:
    if first is None or second is None:
        return False
    try:
        return strip_string(first) == strip_string(second)
    except Exception:
        return str(first) == str(second)


def extract_gold_answer(solution: str) -> str | None:
    boxed = last_boxed_only_string(solution)
    if boxed is not None:
        return remove_boxed(boxed)
    hash_matches = re.findall(r"####\s*([^\n\r]+)", solution)
    if hash_matches:
        return hash_matches[-1].strip()
    return None


def extract_response_candidates(response: str) -> list[str]:
    """Extract likely final answers, preferring the final boxed expression."""
    text = str(response)
    candidates: list[str] = []

    def add(value: str | None) -> None:
        if value is None:
            return
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    boxed = last_boxed_only_string(text)
    if boxed is not None:
        add(remove_boxed(boxed))

    hash_matches = re.findall(r"####\s*([^\n\r]+)", text)
    if hash_matches:
        add(hash_matches[-1])

    dollar_indices = [index for index, char in enumerate(text) if char == "$" and (index == 0 or text[index - 1] != "\\")]
    if len(dollar_indices) > 1:
        inside = text[dollar_indices[0] + 1 : dollar_indices[-1]]
        inside_boxed = last_boxed_only_string(inside)
        add(remove_boxed(inside_boxed) if inside_boxed is not None else inside)

    for pattern in (
        r"(?:the\s+)?final\s+answer\s*(?:is|=|:)\s*(.+)$",
        r"(?:therefore|thus|so)[,:]?\s*(.+)$",
    ):
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            add(matches[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        add(re.sub(r"^(therefore|thus|so)[,:]?\s*", "", lines[-1], flags=re.IGNORECASE))
    add(text)
    return candidates


def score_predictions(
    docs: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(docs) != len(predictions):
        raise RuntimeError(f"Prediction count mismatch: {len(predictions)} vs {len(docs)}")
    level_correct: dict[str, int] = defaultdict(int)
    level_total: dict[str, int] = defaultdict(int)
    type_correct: dict[str, int] = defaultdict(int)
    type_total: dict[str, int] = defaultdict(int)
    correct_count = 0
    extraction_success_count = 0

    for doc, prediction in zip(docs, predictions):
        gold_answer = extract_gold_answer(doc["solution"])
        candidates = extract_response_candidates(prediction.get("response", ""))
        correct = int(any(math_equiv(candidate, gold_answer) for candidate in candidates))
        prediction["gold_answer"] = gold_answer
        prediction["predicted_answer"] = candidates[0] if candidates else None
        prediction["prediction_candidates"] = candidates
        prediction["correct"] = bool(correct)
        correct_count += correct
        extraction_success_count += int(bool(candidates))

        level = doc["level"].strip().lower().replace(" ", "_")
        problem_type = re.sub(r"[^a-z0-9]+", "_", doc["type"].strip().lower()).strip("_")
        level_total[level] += 1
        type_total[problem_type] += 1
        level_correct[level] += correct
        type_correct[problem_type] += correct

    sample_count = len(docs)
    level_metrics = {
        key: {
            "accuracy": level_correct[key] / total,
            "correct_count": level_correct[key],
            "sample_count": total,
        }
        for key, total in sorted(level_total.items())
    }
    type_metrics = {
        key: {
            "accuracy": type_correct[key] / total,
            "correct_count": type_correct[key],
            "sample_count": total,
        }
        for key, total in sorted(type_total.items())
    }
    return {
        "overall_accuracy": correct_count / sample_count,
        "correct_count": correct_count,
        "sample_count": sample_count,
        "extraction_success_count": extraction_success_count,
        "level_metrics": level_metrics,
        "type_metrics": type_metrics,
    }


def checkpoint_roots(parent: Path, names: list[str] | None) -> list[tuple[str, Path]]:
    if names:
        roots = []
        parent_has_checkpoints = any(path.is_dir() for path in parent.glob("checkpoint-*"))
        for name in names:
            use_parent = name in {".", "__root__"} or (
                name == "default" and parent_has_checkpoints and not (parent / name).is_dir()
            )
            root = parent if use_parent else parent / name
            label = "default" if use_parent else name
            roots.append((label, root))
        return roots

    if any(path.is_dir() for path in parent.glob("checkpoint-*")):
        return [("default", parent)]
    roots = [
        (path.name, path)
        for path in sorted(parent.iterdir())
        if path.is_dir() and any(checkpoint.is_dir() for checkpoint in path.glob("checkpoint-*"))
    ]
    if not roots:
        raise RuntimeError(f"No checkpoint-* directories found under {parent}")
    return roots


def checkpoints_for_root(root: Path, step_interval: int, steps: list[int] | None) -> list[Path]:
    if step_interval <= 0:
        raise ValueError("--step-interval must be positive")
    requested_steps = set(steps) if steps is not None else None
    checkpoints = sorted(
        (path for path in root.glob("checkpoint-*") if path.is_dir()),
        key=checkpoint_step,
    )
    checkpoints = [path for path in checkpoints if checkpoint_step(path) % step_interval == 0]
    if requested_steps is not None:
        checkpoints = [path for path in checkpoints if checkpoint_step(path) in requested_steps]
    if not checkpoints:
        raise RuntimeError(f"No selected checkpoints found under {root}")
    return checkpoints


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.max_model_len <= 0:
        raise ValueError("--batch-size, --max-new-tokens, and --max-model-len must be positive")
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--offset must be non-negative and --limit must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")

    # Keep heavy ML imports lazy so the extraction/scoring helpers remain usable
    # in lightweight environments, but import torch before seeding it below.
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_parent = Path(args.checkpoint_parent).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = checkpoint_roots(checkpoint_parent, args.datasets)
    checkpoint_sets = [
        (name, root, checkpoints_for_root(root, args.step_interval, args.steps))
        for name, root in roots
    ]

    docs = load_math_docs(
        args.dataset_name,
        args.dataset_config,
        args.split,
        args.offset,
        args.limit,
        args.shuffle,
        args.seed,
    )
    dataset_hash = canonical_rows_hash(docs)
    atomic_jsonl(output_dir / "math_prompts.jsonl", docs)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_prompt = None if args.no_system_prompt else args.system_prompt
    encoded_prompts = encode_prompts(tokenizer, docs, system_prompt)
    prompt_lengths = [len(prompt) for prompt in encoded_prompts]
    if max(prompt_lengths) + args.max_new_tokens > args.max_model_len:
        raise RuntimeError(
            f"Longest prompt ({max(prompt_lengths)}) plus max_new_tokens ({args.max_new_tokens}) "
            f"exceeds --max-model-len ({args.max_model_len})"
        )

    model = None
    llm = None
    first_checkpoint = checkpoint_sets[0][2][0]
    if args.backend == "hf":
        from peft import PeftModel

        device = torch.device(args.device)
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
            seed=args.seed,
        )

    records: list[dict[str, Any]] = []
    identical_checkpoint_cache: dict[str, dict[str, Any]] = {}
    lora_int_id = 0
    for dataset_name, checkpoint_root, checkpoints in checkpoint_sets:
        dataset_output = output_dir / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        for checkpoint in checkpoints:
            step = checkpoint_step(checkpoint)
            result_path = dataset_output / f"checkpoint-{step}.metrics.json"
            predictions_path = dataset_output / f"checkpoint-{step}.predictions.jsonl"
            if result_path.is_file() and predictions_path.is_file() and not args.overwrite:
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("math_rows_sha256") != dataset_hash:
                    raise RuntimeError(f"Dataset hash mismatch in {result_path}")
                records.append(saved["metrics"])
                print(f"resume dataset={dataset_name} step={step}", flush=True)
                continue

            weights = adapter_weights_path(checkpoint)
            adapter_sha256 = sha256_file(weights)
            lora_int_id += 1
            reused = identical_checkpoint_cache.get(adapter_sha256)
            if reused is not None:
                predictions = copy.deepcopy(reused["predictions"])
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
                print(f"generate dataset={dataset_name} step={step}", flush=True)
                predictions, generation_seconds = generate_responses_hf(
                    model,
                    tokenizer,
                    docs,
                    encoded_prompts,
                    args.batch_size,
                    args.max_new_tokens,
                    device,
                )
            else:
                print(f"generate dataset={dataset_name} step={step}", flush=True)
                predictions, generation_seconds = generate_responses_vllm(
                    llm,
                    tokenizer,
                    docs,
                    encoded_prompts,
                    checkpoint,
                    lora_int_id,
                    args.max_new_tokens,
                )

            scored = score_predictions(docs, predictions)
            generated_lengths = [prediction["generated_tokens"] for prediction in predictions]
            metrics: dict[str, Any] = {
                "dataset": dataset_name,
                "label": "Competition Math" if dataset_name == "default" else dataset_name,
                "step": step,
                "overall_accuracy": scored["overall_accuracy"],
                "correct_count": scored["correct_count"],
                "sample_count": scored["sample_count"],
                "extraction_success_count": scored["extraction_success_count"],
                "mean_prompt_tokens": float(np.mean(prompt_lengths)),
                "max_prompt_tokens": max(prompt_lengths),
                "mean_generated_tokens": float(np.mean(generated_lengths)),
                "max_generated_tokens": max(generated_lengths),
                "max_token_truncation_count": sum(
                    length >= args.max_new_tokens for length in generated_lengths
                ),
                "generation_seconds": generation_seconds,
            }
            for level in LEVELS:
                metrics[f"{level}_accuracy"] = scored["level_metrics"].get(level, {}).get("accuracy")
            for problem_type in TYPES:
                metrics[f"{problem_type}_accuracy"] = scored["type_metrics"].get(problem_type, {}).get(
                    "accuracy"
                )

            if reused is None:
                identical_checkpoint_cache[adapter_sha256] = {
                    "dataset": dataset_name,
                    "step": step,
                    "predictions": copy.deepcopy(predictions),
                }
            atomic_jsonl(predictions_path, predictions)
            payload = {
                "model": args.model,
                "checkpoint": str(checkpoint),
                "adapter_sha256": adapter_sha256,
                "prediction_reused": reused is not None,
                "prediction_reused_from": (
                    None
                    if reused is None
                    else f"{reused['dataset']}/checkpoint-{reused['step']}"
                ),
                "math_dataset": args.dataset_name,
                "math_split": args.split,
                "math_rows_sha256": dataset_hash,
                "math_prompt_count": len(docs),
                "evaluation": {
                    "system_prompt": system_prompt,
                    "apply_chat_template": True,
                    "answer_extraction": "last boxed answer, then ####/final-line fallbacks",
                    "answer_normalization": "Hendrycks MATH strip_string",
                    "overall_aggregation": "sample-weighted exact-match accuracy",
                    "dtype": args.dtype,
                    "seed": args.seed,
                },
                "level_metrics": scored["level_metrics"],
                "type_metrics": scored["type_metrics"],
                "predictions_sha256": sha256_file(predictions_path),
                "metrics": metrics,
            }
            atomic_json(result_path, payload)
            records.append(metrics)
            records.sort(key=lambda row: (row["dataset"], row["step"]))
            write_results_csv(output_dir / "math_checkpoint_metrics.csv", records)
            print(
                f"done dataset={dataset_name} step={step} "
                f"accuracy={metrics['overall_accuracy']:.4f} "
                f"extracted={scored['extraction_success_count']}/{len(docs)} "
                f"seconds={generation_seconds:.1f}",
                flush=True,
            )

    records.sort(key=lambda row: (row["dataset"], row["step"]))
    write_results_csv(output_dir / "math_checkpoint_metrics.csv", records)
    manifest = {
        "model": args.model,
        "checkpoint_parent": str(checkpoint_parent),
        "datasets": [name for name, _, _ in checkpoint_sets],
        "output_dir": str(output_dir),
        "math_dataset": args.dataset_name,
        "math_split": args.split,
        "math_prompt_count": len(docs),
        "math_rows_sha256": dataset_hash,
        "source_commit": git_commit(project_root()),
        "evaluation": {
            "system_prompt": system_prompt,
            "apply_chat_template": True,
            "answer_extraction": "last boxed answer, then ####/final-line fallbacks",
            "answer_normalization": "Hendrycks MATH strip_string",
            "overall_aggregation": "sample-weighted exact-match accuracy",
            "backend": args.backend,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "prompt_token_length": {
            "min": min(prompt_lengths),
            "mean": float(np.mean(prompt_lengths)),
            "max": max(prompt_lengths),
        },
        "metrics_file": str(output_dir / "math_checkpoint_metrics.csv"),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
