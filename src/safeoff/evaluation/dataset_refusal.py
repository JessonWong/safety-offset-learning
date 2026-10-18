#!/usr/bin/env python3
"""Compare base and fine-tuned refusal rates on each run's own training prompts."""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = project_root()

from safeoff.evaluation.metrics import contains_keyword
from safeoff.core.model_utils import get_batched_responses


HELPFUL_SYSTEM = "You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/refusal_llama31_finetune_data_before_after"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--newtarget-data", type=Path, default=Path("data/8b_newtarget.csv"))
    parser.add_argument("--aoa-data", type=Path, default=Path("data/aoa_100.json"))
    parser.add_argument("--alpaca-top100-data", type=Path, default=Path("data/alpaca_top100.json"))
    parser.add_argument("--llm-lat-arrow", type=Path, default=None)
    parser.add_argument("--llm-lat-count", type=int, default=100)
    parser.add_argument("--llm-lat-seed", type=int, default=30)
    parser.add_argument("--checkpoint-step", type=int, default=250)
    return parser.parse_args()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_newtarget(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        records.append(
            {
                "source_index": int(row["index"]),
                "prompt": row["instruction"],
                "system": HELPFUL_SYSTEM,
                "reference_response": row["alpaca_reference"],
            }
        )
    return {
        "source": str(path),
        "selection": "all CSV rows in source order; suffixes disabled as in training",
        "records": records,
    }


def load_aoa(path: Path) -> dict:
    rows = read_json(path)
    records = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        system = next(message["content"] for message in messages if message["role"] == "system")
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        reference = next(message["content"] for message in messages if message["role"] == "assistant")
        records.append(
            {
                "source_index": index,
                "prompt": prompt,
                "system": system,
                "reference_response": reference,
            }
        )
    return {
        "source": str(path),
        "selection": "all JSON rows in source order, with each row's AOA system prompt",
        "records": records,
    }


def load_alpaca_top100(path: Path) -> dict:
    rows = read_json(path)
    records = []
    for index, row in enumerate(rows):
        prompt_parts = [row["instruction"]]
        input_text = row.get("input", "")
        if isinstance(input_text, str) and input_text.strip():
            prompt_parts.append(f"Input: {input_text}")
        records.append(
            {
                "source_index": index,
                "prompt": "\n\n".join(prompt_parts),
                "system": HELPFUL_SYSTEM,
                "reference_response": row["output"],
            }
        )
    return {
        "source": str(path),
        "selection": "all JSON rows in source order",
        "records": records,
    }


def find_llm_lat_arrow(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(explicit_path)
        return explicit_path

    hf_home = Path(os.environ.get("HF_HOME", "huggingface"))
    candidates = sorted(
        hf_home.glob(
            "datasets/LLM-LAT___harmful-dataset/default/*/*/harmful-dataset-train.arrow"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one cached LLM-LAT Arrow file under HF_HOME; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def load_llm_lat(path: Path, count: int, seed: int) -> dict:
    dataset = Dataset.from_file(str(path))
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected_indices = indices[:count]
    selected = dataset.select(selected_indices)
    records = []
    for source_index, row in zip(selected_indices, selected):
        records.append(
            {
                "source_index": source_index,
                "prompt": row["prompt"],
                "system": "",
                "reference_response": row["rejected"],
            }
        )
    return {
        "source": "LLM-LAT/harmful-dataset",
        "local_arrow": str(path),
        "selection": f"first {count} indices after Python random shuffle with seed {seed}",
        "records": records,
    }


def prompt_signature(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["system"].encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(record["prompt"].encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_base_model(model_name: str):
    compute_capability = torch.cuda.get_device_capability()
    # Ampere (SM 8.x) and newer have native BF16 Tensor Cores. PyTorch may
    # report BF16 as usable on Volta while silently taking a very slow path.
    dtype = torch.bfloat16 if compute_capability[0] >= 8 else torch.float16
    print(
        f"Loading {model_name} with dtype={dtype}, "
        f"compute_capability={compute_capability}",
        flush=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )


def generate_records(
    tokenizer,
    model,
    records: list[dict],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    systems = {record["system"] for record in records}
    if len(systems) != 1:
        raise ValueError(f"Expected one shared system prompt, found {len(systems)}")
    system = next(iter(systems))
    prompts = [record["prompt"] for record in records]
    responses = get_batched_responses(
        tokenizer,
        model,
        prompts,
        system=system,
        batch_size=batch_size,
        max_length=max_new_tokens,
    )
    if len(responses) != len(records):
        raise RuntimeError(f"Generated {len(responses)} responses for {len(records)} prompts")
    output = []
    for record, response in zip(records, responses):
        output.append(
            {
                **record,
                "response": response,
                "refusal": contains_keyword(response),
            }
        )
    return output


def save_raw(
    output_dir: Path,
    dataset_name: str,
    condition: str,
    dataset_spec: dict,
    generated: list[dict],
    model_path: str,
) -> Path:
    raw_dir = output_dir / "raw" / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{condition}.json"
    payload = {
        "meta": {
            "dataset": dataset_name,
            "source": dataset_spec["source"],
            "local_arrow": dataset_spec.get("local_arrow"),
            "selection": dataset_spec["selection"],
            "model_path": model_path,
            "num_prompts": len(generated),
            "prompt_signature_sha256": prompt_signature(dataset_spec["records"]),
            "refusal_detector": "safeoff.evaluation.metrics.contains_keyword",
        },
        "data": generated,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def refusal_stats(records: list[dict]) -> tuple[int, float]:
    count = sum(bool(record["refusal"]) for record in records)
    return count, count / len(records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    llm_lat_arrow = find_llm_lat_arrow(args.llm_lat_arrow)
    datasets = {
        "8b_newtarget": load_newtarget(args.newtarget_data),
        "aoa": load_aoa(args.aoa_data),
        "alpaca_top100": load_alpaca_top100(args.alpaca_top100_data),
        "llm_lat": load_llm_lat(llm_lat_arrow, args.llm_lat_count, args.llm_lat_seed),
    }
    checkpoints = {
        "8b_newtarget": Path(
            f"finetune_model/math_curve/llama_newtarget/Meta-Llama-3.1-8B-Instruct/model/checkpoint-{args.checkpoint_step}"
        ),
        "aoa": Path(
            f"finetune_model/math_curve/llama3_aoa/Meta-Llama-3.1-8B-Instruct/model/checkpoint-{args.checkpoint_step}"
        ),
        "alpaca_top100": Path(
            f"finetune_model/math_curve/llama3_alpaca_top100/Meta-Llama-3.1-8B-Instruct/model/checkpoint-{args.checkpoint_step}"
        ),
        "llm_lat": Path(
            f"finetune_model/math_curve/llama3_llm_lat_harmful100/Meta-Llama-3.1-8B-Instruct/model/checkpoint-{args.checkpoint_step}"
        ),
    }
    for name, checkpoint in checkpoints.items():
        if not (checkpoint / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing {name} adapter: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model", flush=True)
    base_model = load_base_model(args.base_model)
    base_model.eval()
    base_outputs = {}
    for name, dataset_spec in datasets.items():
        print(f"Generating base responses for {name} ({len(dataset_spec['records'])})", flush=True)
        generated = generate_records(
            tokenizer,
            base_model,
            dataset_spec["records"],
            args.batch_size,
            args.max_new_tokens,
        )
        base_outputs[name] = generated
        save_raw(args.output_dir, name, "before_base", dataset_spec, generated, args.base_model)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    after_outputs = {}
    for name, dataset_spec in datasets.items():
        checkpoint = checkpoints[name]
        print(f"Loading base + adapter for {name}: {checkpoint}", flush=True)
        current_base = load_base_model(args.base_model)
        model = PeftModel.from_pretrained(current_base, str(checkpoint))
        model.eval()
        print(f"Generating fine-tuned responses for {name} ({len(dataset_spec['records'])})", flush=True)
        generated = generate_records(
            tokenizer,
            model,
            dataset_spec["records"],
            args.batch_size,
            args.max_new_tokens,
        )
        after_outputs[name] = generated
        save_raw(
            args.output_dir,
            name,
            f"after_checkpoint_{args.checkpoint_step}",
            dataset_spec,
            generated,
            str(checkpoint),
        )
        del model
        del current_base
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for name, dataset_spec in datasets.items():
        before_count, before_rate = refusal_stats(base_outputs[name])
        after_count, after_rate = refusal_stats(after_outputs[name])
        rows.append(
            {
                "finetune_dataset": name,
                "data_source": dataset_spec["source"],
                "num_prompts": len(dataset_spec["records"]),
                "before_checkpoint": args.base_model,
                "before_refusal_count": before_count,
                "before_refusal_rate": round(before_rate, 6),
                "after_checkpoint": str(checkpoints[name]),
                "after_refusal_count": after_count,
                "after_refusal_rate": round(after_rate, 6),
                "refusal_rate_change_pp": round((after_rate - before_rate) * 100.0, 3),
            }
        )

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "base_model": args.base_model,
        "evaluation_scope": "each checkpoint's own fine-tuning prompts",
        "decoding": f"greedy, max_new_tokens={args.max_new_tokens}",
        "refusal_detector": "safeoff.evaluation.metrics.contains_keyword",
        "datasets": {
            name: {
                key: value
                for key, value in dataset_spec.items()
                if key != "records"
            }
            | {
                "num_prompts": len(dataset_spec["records"]),
                "prompt_signature_sha256": prompt_signature(dataset_spec["records"]),
            }
            for name, dataset_spec in datasets.items()
        },
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
