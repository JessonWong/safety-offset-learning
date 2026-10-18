#!/usr/bin/env python3
"""CSV refusal evaluation with/without a system turn and an optional trained adapter."""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = project_root()
HELPFUL_SYSTEM = "You are a helpful assistant."
SYSTEMS = {"no_system": None, "helpful_system": HELPFUL_SYSTEM}


def adapter_provenance(path, base_model):
    """Validate an explicitly supplied local adapter and fingerprint its weights."""
    if path is None:
        return {}
    path = path.resolve()
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Missing adapter config or safetensors weights: {path}")
    config = json.loads(config_path.read_text())
    if config.get("base_model_name_or_path") != base_model:
        raise ValueError("Adapter was trained from a different base model")
    return {
        "adapter_path": str(path),
        "adapter_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "adapter_weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
    }


def read_source(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    if not rows or not {"index", "instruction", "alpaca_reference"}.issubset(fields or []):
        raise ValueError("Expected a nonempty SafeOff instruction/reference CSV")
    if any(not row["instruction"].strip() for row in rows):
        raise ValueError("Empty instruction")
    if any(row.get("suffix", "").strip() for row in rows):
        raise ValueError("Nonempty suffix: resolve input formatting explicitly before running")
    return fields, rows


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_cached(payload, expected_meta, rows):
    if payload.get("meta") != expected_meta:
        raise ValueError("Cached evaluation provenance differs; use a new output directory")
    records = payload["data"]
    if len(records) > len(rows):
        raise ValueError("Cached evaluation is longer than source")
    for position, record in enumerate(records):
        if record["row_position"] != position or record["prompt"] != rows[position]["instruction"]:
            raise ValueError("Cached rows do not match source order")
    return records


def export_keyword_subset(path, fields, rows, records):
    if len(rows) != len(records):
        raise ValueError("Cannot select a subset from incomplete generations")
    selected = [row for row, record in zip(rows, records) if record["refusal"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=Path("data/8b_newtarget.csv"))
    parser.add_argument("--base-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", type=Path, default=None, help="Optional existing LoRA checkpoint; never train here")
    parser.add_argument("--output-dir", type=Path, default=Path("results/safeoff_refusal_subset/no_system_ablation"))
    parser.add_argument("--conditions", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("Batch size and generation length must be positive")
    fields, rows = read_source(args.csv_path)
    adapter_meta = adapter_provenance(args.adapter, args.base_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safeoff.evaluation.metrics import contains_keyword
    from safeoff.core.model_utils import get_batched_responses

    if not torch.cuda.is_available():
        raise RuntimeError("Run generation in a GPU allocation, not on the login node")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    no_system_example = tokenizer.apply_chat_template(
        [{"role": "user", "content": rows[0]["instruction"]}],
        tokenize=False, add_generation_prompt=True,
    )
    expected = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + rows[0]["instruction"].strip()
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if no_system_example != expected:
        raise RuntimeError("Tokenizer inserts unexpected content into user-only template")
    source_hash = hashlib.sha256(args.csv_path.read_bytes()).hexdigest()
    common_meta = {
        "source": str(args.csv_path.resolve()), "source_sha256": source_hash,
        "model": args.base_model, "dtype": args.dtype,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "do_sample": False, "max_input_tokens": 1000, "num_prompts": len(rows),
        "refusal_detector": "safeoff.evaluation.metrics.contains_keyword",
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "model_commit": None,
        **adapter_meta,
    }
    # The commit is stable provenance, whereas GPU identifiers need not match on resume.
    from transformers import AutoConfig
    common_meta["model_commit"] = getattr(AutoConfig.from_pretrained(args.base_model), "_commit_hash", None)
    model = None
    outputs = {}
    for condition in args.conditions:
        system = SYSTEMS[condition]
        meta = {**common_meta, "condition": condition, "system": system}
        path = args.output_dir / f"{condition}.json"
        records = validate_cached(json.loads(path.read_text()), meta, rows) if path.exists() else []
        if len(records) < len(rows) and model is None:
            print(f"Loading {args.base_model}, dtype={args.dtype}, GPU={torch.cuda.get_device_name()}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype=getattr(torch, args.dtype), device_map="auto",
            )
            if args.adapter is not None:
                from peft import PeftModel
                print(f"Loading fine-tuned adapter: {args.adapter}", flush=True)
                model = PeftModel.from_pretrained(model, str(args.adapter))
            model.eval()
        chunk_size = args.batch_size * 4
        for start in range(len(records), len(rows), chunk_size):
            batch_rows = rows[start:start + chunk_size]
            responses = get_batched_responses(
                tokenizer, model, [row["instruction"] for row in batch_rows], system=system,
                batch_size=args.batch_size, max_length=args.max_new_tokens,
            )
            if len(responses) != len(batch_rows):
                raise RuntimeError("Generation count mismatch")
            for offset, (row, response) in enumerate(zip(batch_rows, responses)):
                records.append({
                    "row_position": start + offset, "source_index": row["index"],
                    "prompt": row["instruction"], "reference_response": row["alpaca_reference"],
                    "system": system, "response": response, "refusal": bool(contains_keyword(response)),
                })
            write_json(path, {"meta": meta, "data": records})
            print(f"{condition}: {len(records)}/{len(rows)} saved", flush=True)
        outputs[condition] = records

    summary = {"meta": common_meta, "conditions": {}}
    for condition, records in outputs.items():
        count = sum(record["refusal"] for record in records)
        summary["conditions"][condition] = {
            "num_prompts": len(records), "refusal_count": count, "refusal_rate": count / len(records),
        }
    if "no_system" in outputs:
        count = export_keyword_subset(args.output_dir / "refused_keyword.csv", fields, rows, outputs["no_system"])
        summary["keyword_subset"] = {
            "path": str(args.output_dir / "refused_keyword.csv"), "count": count,
            "targets": "Original alpaca_reference, NOT model-generated refusals",
            "warning": "Keyword positives are not a semantic refusal judgment; audit before training.",
        }
    if set(SYSTEMS).issubset(outputs):
        summary["paired_transitions"] = {
            f"helpful_{int(before)}_no_system_{int(after)}": sum(
                record_before["refusal"] == before and record_after["refusal"] == after
                for record_before, record_after in zip(outputs["helpful_system"], outputs["no_system"])
            )
            for before in (False, True) for after in (False, True)
        }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
