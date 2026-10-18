"""Shared, offline inference helpers for the held-out functional/rollback study."""
from __future__ import annotations
from safeoff.paths import project_root

import hashlib
import json
from pathlib import Path

ROOT = project_root()
BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text())
        records = data["records"] if isinstance(data, dict) else data
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Probe records must be nonempty and have unique IDs")
    for record in records:
        if record["group"] not in {"harmful", "benign", "refusal_sensitive"}:
            raise ValueError(f"Unknown probe group: {record['group']}")
        if not record["prompt"].strip():
            raise ValueError(f"Empty prompt: {record['id']}")
    return records


def messages_for(record):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["prompt"]},
    ]


def prompt_text(tokenizer, record):
    return tokenizer.apply_chat_template(messages_for(record), tokenize=False, add_generation_prompt=True)


def load_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(device="cuda", dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run; submit on a GPU compute node")
    if dtype == torch.bfloat16 and str(device).startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; use an H100/L40S or explicit float16")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, local_files_only=True, torch_dtype=dtype,
        device_map={"": str(device)}, attn_implementation="sdpa", trust_remote_code=False,
    ).eval().requires_grad_(False)
    if model.config.model_type != "llama":
        raise ValueError("This study expects the recorded Llama base model")
    return model, load_tokenizer()


def tokenize_records(tokenizer, records, max_length, device):
    texts = [prompt_text(tokenizer, record) for record in records]
    batch = tokenizer(texts, add_special_tokens=False, padding=True, truncation=False, return_tensors="pt")
    if batch["input_ids"].shape[1] > max_length:
        raise ValueError("Probe exceeds max_length; silent prompt truncation is disabled")
    batch["position_ids"] = (batch["attention_mask"].long().cumsum(-1) - 1).masked_fill(batch["attention_mask"] == 0, 0)
    return {key: value.to(device) for key, value in batch.items()}
