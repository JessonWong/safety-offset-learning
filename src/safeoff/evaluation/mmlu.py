#!/usr/bin/env python3
"""Evaluate 5-shot MMLU accuracy for a trajectory of LoRA checkpoints.

The prompt and aggregation rules mirror lm-evaluation-harness' current ``mmlu``
task: all 57 ``cais/mmlu`` subjects, the first five examples from each subject's
dev split, chat-template formatting with the few-shot examples as separate turns,
and sample-weighted accuracy.  MMLU's answer candidates are the single tokens
``A``--``D`` for both supported tokenizers, so restricted one-token decoding is
exactly the same argmax as the harness' four continuation log-likelihood calls.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_LABELS = {
    "newtarget": "new target",
    "benign_alpaca": "benign Alpaca",
    "aoa": "AoA",
    "alpaca_top100": "Alpaca top-100",
    "harmful_llm_lat": "LLM-LAT harmful-100",
}

CATEGORY_ORDER = ["stem", "humanities", "social_sciences", "other"]
ANSWER_LETTERS = ["A", "B", "C", "D"]
RESULT_FIELDS = [
    "dataset",
    "label",
    "step",
    "overall_accuracy",
    "stem_accuracy",
    "humanities_accuracy",
    "social_sciences_accuracy",
    "other_accuracy",
    "correct_count",
    "sample_count",
    "mean_prompt_tokens",
    "max_prompt_tokens",
    "inference_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-parent", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LABELS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mmlu-task-root",
        default="cache/ifeval_official/lm-evaluation-harness/lm_eval/tasks/mmlu/default",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--step-interval", type=int, default=25)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--limit-per-subject", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
        writer.writerows({field: row[field] for field in RESULT_FIELDS} for row in records)
    temporary.replace(path)


def load_task_definitions(task_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    descriptions: dict[str, str] = {}
    categories: dict[str, str] = {}
    for path in sorted(task_root.glob("mmlu_*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        subject = config.get("dataset_name")
        tag = config.get("tag", "")
        if not subject:
            continue
        match = re.fullmatch(r"mmlu_(.+)_tasks", tag)
        if not match:
            raise ValueError(f"Cannot infer MMLU category from {path}: tag={tag!r}")
        category = match.group(1)
        if category not in CATEGORY_ORDER:
            raise ValueError(f"Unexpected MMLU category {category!r} in {path}")
        descriptions[subject] = str(config["description"])
        categories[subject] = category
    if len(descriptions) != 57:
        raise RuntimeError(f"Expected 57 MMLU subject YAMLs, found {len(descriptions)}")
    return descriptions, categories


def format_question(doc: dict[str, Any]) -> str:
    choices = doc["choices"]
    if len(choices) != 4:
        raise ValueError(f"Expected four MMLU choices, got {len(choices)}")
    return (
        f"{doc['question'].strip()}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )


def load_mmlu_docs(
    descriptions: dict[str, str],
    categories: dict[str, str],
    num_fewshot: int,
    limit_per_subject: int | None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    dataset = load_dataset("cais/mmlu", "all")
    dev_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset["dev"]:
        if row["subject"] in descriptions:
            dev_by_subject[row["subject"]].append(dict(row))
    for row in dataset["test"]:
        if row["subject"] in descriptions:
            test_by_subject[row["subject"]].append(dict(row))

    docs: list[dict[str, Any]] = []
    fewshot: dict[str, list[dict[str, Any]]] = {}
    for subject in sorted(descriptions):
        if len(dev_by_subject[subject]) < num_fewshot:
            raise RuntimeError(
                f"Subject {subject} has only {len(dev_by_subject[subject])} dev examples"
            )
        fewshot[subject] = dev_by_subject[subject][:num_fewshot]
        rows = test_by_subject[subject]
        if limit_per_subject is not None:
            rows = rows[:limit_per_subject]
        for subject_index, row in enumerate(rows):
            docs.append(
                {
                    "subject": subject,
                    "category": categories[subject],
                    "subject_index": subject_index,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "answer": int(row["answer"]),
                }
            )
    if not docs:
        raise RuntimeError("No MMLU test documents loaded")
    return docs, fewshot


def build_prompt_messages(
    doc: dict[str, Any],
    subject_fewshot: list[dict[str, Any]],
    description: str,
) -> list[dict[str, str]]:
    # This follows ConfigurableTask.fewshot_context with apply_chat_template=True
    # and fewshot_as_multiturn=True. The task description becomes the system turn.
    messages: list[dict[str, str]] = [{"role": "system", "content": description}]
    for example in subject_fewshot:
        messages.append({"role": "user", "content": format_question(example)})
        messages.append(
            {"role": "assistant", "content": ANSWER_LETTERS[int(example["answer"])]}
        )
    messages.append({"role": "user", "content": format_question(doc)})
    return messages


def encode_prompts(
    tokenizer,
    docs: list[dict[str, Any]],
    fewshot: dict[str, list[dict[str, Any]]],
    descriptions: dict[str, str],
) -> list[list[int]]:
    return [
        tokenizer.apply_chat_template(
            build_prompt_messages(doc, fewshot[doc["subject"]], descriptions[doc["subject"]]),
            tokenize=True,
            add_generation_prompt=True,
        )
        for doc in docs
    ]


def validate_choice_tokens(tokenizer) -> list[int]:
    token_ids = []
    for letter in ANSWER_LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"MMLU answer {letter!r} is not a single token: {ids}")
        token_ids.append(int(ids[0]))
    if len(set(token_ids)) != 4:
        raise RuntimeError(f"MMLU answer token IDs are not unique: {token_ids}")
    return token_ids


def score_predictions(
    docs: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(docs) != len(predictions):
        raise RuntimeError(f"Prediction count mismatch: {len(predictions)} vs {len(docs)}")
    subject_correct: dict[str, int] = defaultdict(int)
    subject_total: dict[str, int] = defaultdict(int)
    category_correct: dict[str, int] = defaultdict(int)
    category_total: dict[str, int] = defaultdict(int)
    for doc, prediction in zip(docs, predictions):
        correct = int(prediction["predicted_index"] == doc["answer"])
        prediction["correct"] = bool(correct)
        prediction["gold_index"] = doc["answer"]
        prediction["gold_letter"] = ANSWER_LETTERS[doc["answer"]]
        subject_correct[doc["subject"]] += correct
        subject_total[doc["subject"]] += 1
        category_correct[doc["category"]] += correct
        category_total[doc["category"]] += 1

    correct_count = sum(subject_correct.values())
    sample_count = sum(subject_total.values())
    subject_metrics = {
        subject: {
            "accuracy": subject_correct[subject] / total,
            "correct_count": subject_correct[subject],
            "sample_count": total,
        }
        for subject, total in sorted(subject_total.items())
    }
    category_metrics = {
        category: {
            "accuracy": category_correct[category] / category_total[category],
            "correct_count": category_correct[category],
            "sample_count": category_total[category],
        }
        for category in CATEGORY_ORDER
    }
    return {
        "overall_accuracy": correct_count / sample_count,
        "correct_count": correct_count,
        "sample_count": sample_count,
        "category_metrics": category_metrics,
        "subject_metrics": subject_metrics,
    }


def predict_vllm(
    llm,
    docs: list[dict[str, Any]],
    encoded_prompts: list[list[int]],
    checkpoint: Path,
    lora_int_id: int,
    choice_token_ids: list[int],
    seed: int,
) -> tuple[list[dict[str, Any]], float]:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        allowed_token_ids=choice_token_ids,
        seed=seed,
        skip_special_tokens=False,
    )
    token_prompts = [{"prompt_token_ids": prompt_ids} for prompt_ids in encoded_prompts]
    request = LoRARequest(checkpoint.name, lora_int_id, str(checkpoint))
    started = time.perf_counter()
    outputs = llm.generate(
        token_prompts,
        sampling_params=params,
        lora_request=request,
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - started
    token_to_index = {token_id: index for index, token_id in enumerate(choice_token_ids)}
    predictions = []
    for doc, prompt_ids, output in zip(docs, encoded_prompts, outputs):
        if len(output.outputs) != 1 or len(output.outputs[0].token_ids) != 1:
            raise RuntimeError("Expected exactly one answer token from vLLM")
        token_id = int(output.outputs[0].token_ids[0])
        if token_id not in token_to_index:
            raise RuntimeError(f"Unexpected MMLU answer token: {token_id}")
        predicted_index = token_to_index[token_id]
        predictions.append(
            {
                "subject": doc["subject"],
                "category": doc["category"],
                "subject_index": doc["subject_index"],
                "predicted_index": predicted_index,
                "predicted_letter": ANSWER_LETTERS[predicted_index],
                "prompt_tokens": len(prompt_ids),
            }
        )
    return predictions, elapsed


def main() -> None:
    args = parse_args()
    if args.num_fewshot < 0:
        raise ValueError("--num-fewshot must be non-negative")
    if args.max_model_len <= 0 or args.max_num_seqs <= 0:
        raise ValueError("--max-model-len and --max-num-seqs must be positive")
    if args.limit_per_subject is not None and args.limit_per_subject <= 0:
        raise ValueError("--limit-per-subject must be positive")

    checkpoint_parent = Path(args.checkpoint_parent).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    task_root = Path(args.mmlu_task_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptions, categories = load_task_definitions(task_root)
    docs, fewshot = load_mmlu_docs(
        descriptions,
        categories,
        args.num_fewshot,
        args.limit_per_subject,
    )
    dataset_hash = canonical_rows_hash(docs)
    fewshot_hash = canonical_rows_hash(
        [
            {"subject": subject, "examples": examples}
            for subject, examples in sorted(fewshot.items())
        ]
    )

    checkpoint_sets: dict[str, list[Path]] = {}
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
            checkpoints = [
                path for path in checkpoints if checkpoint_step(path) in requested_steps
            ]
        if not checkpoints:
            raise RuntimeError(f"No checkpoints found under {checkpoint_root}")
        checkpoint_sets[dataset_name] = checkpoints

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded_prompts = encode_prompts(tokenizer, docs, fewshot, descriptions)
    prompt_lengths = [len(prompt) for prompt in encoded_prompts]
    if max(prompt_lengths) + 1 > args.max_model_len:
        raise RuntimeError(
            f"Longest prompt ({max(prompt_lengths)} tokens) exceeds --max-model-len "
            f"{args.max_model_len} after adding the answer token"
        )
    choice_token_ids = validate_choice_tokens(tokenizer)

    from vllm import LLM

    torch.manual_seed(args.seed)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        enable_lora=True,
        max_lora_rank=8,
        max_loras=1,
        max_cpu_loras=2,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        enable_prefix_caching=True,
        seed=args.seed,
    )

    source_commit = "unknown"
    git_dir = task_root.parents[3] / ".git"
    if git_dir.is_dir():
        try:
            import subprocess

            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=task_root.parents[3],
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass

    records: list[dict[str, Any]] = []
    identical_checkpoint_cache: dict[str, dict[str, Any]] = {}
    lora_int_id = 0
    for dataset_name in args.datasets:
        dataset_output = output_dir / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        for checkpoint in checkpoint_sets[dataset_name]:
            step = checkpoint_step(checkpoint)
            result_path = dataset_output / f"checkpoint-{step}.metrics.json"
            predictions_path = dataset_output / f"checkpoint-{step}.predictions.jsonl"
            if result_path.is_file() and predictions_path.is_file() and not args.overwrite:
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("mmlu_test_rows_sha256") != dataset_hash:
                    raise RuntimeError(f"Dataset hash mismatch in {result_path}")
                if saved.get("mmlu_fewshot_rows_sha256") != fewshot_hash:
                    raise RuntimeError(f"Few-shot hash mismatch in {result_path}")
                records.append(saved["metrics"])
                print(f"resume dataset={dataset_name} step={step}", flush=True)
                continue

            weights = adapter_weights_path(checkpoint)
            adapter_sha256 = sha256_file(weights)
            lora_int_id += 1
            reused = identical_checkpoint_cache.get(adapter_sha256)
            if reused is not None:
                predictions = copy.deepcopy(reused["predictions"])
                inference_seconds = 0.0
                print(
                    f"reuse identical adapter dataset={dataset_name} step={step} "
                    f"from={reused['dataset']}/checkpoint-{reused['step']}",
                    flush=True,
                )
            else:
                print(f"evaluate dataset={dataset_name} step={step}", flush=True)
                predictions, inference_seconds = predict_vllm(
                    llm,
                    docs,
                    encoded_prompts,
                    checkpoint,
                    lora_int_id,
                    choice_token_ids,
                    args.seed,
                )

            scored = score_predictions(docs, predictions)
            category_metrics = scored["category_metrics"]
            metrics = {
                "dataset": dataset_name,
                "label": DATASET_LABELS.get(dataset_name, dataset_name),
                "step": step,
                "overall_accuracy": scored["overall_accuracy"],
                "stem_accuracy": category_metrics["stem"]["accuracy"],
                "humanities_accuracy": category_metrics["humanities"]["accuracy"],
                "social_sciences_accuracy": category_metrics["social_sciences"]["accuracy"],
                "other_accuracy": category_metrics["other"]["accuracy"],
                "correct_count": scored["correct_count"],
                "sample_count": scored["sample_count"],
                "mean_prompt_tokens": float(np.mean(prompt_lengths)),
                "max_prompt_tokens": max(prompt_lengths),
                "inference_seconds": inference_seconds,
            }
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
                "mmlu_dataset": "cais/mmlu",
                "mmlu_test_split": "test",
                "mmlu_fewshot_split": "dev",
                "mmlu_test_rows_sha256": dataset_hash,
                "mmlu_fewshot_rows_sha256": fewshot_hash,
                "lm_eval_task_root": str(task_root),
                "lm_eval_commit": source_commit,
                "evaluation": {
                    "subjects": 57,
                    "num_fewshot": args.num_fewshot,
                    "fewshot_sampler": "first_n",
                    "apply_chat_template": True,
                    "fewshot_as_multiturn": True,
                    "system_prompt": "official per-subject MMLU description",
                    "candidate_scoring": "argmax next-token log-likelihood over A/B/C/D",
                    "choice_token_ids": choice_token_ids,
                    "dtype": args.dtype,
                    "seed": args.seed,
                },
                "category_metrics": category_metrics,
                "subject_metrics": scored["subject_metrics"],
                "predictions_sha256": sha256_file(predictions_path),
                "metrics": metrics,
            }
            atomic_json(result_path, payload)
            records.append(metrics)
            records.sort(key=lambda row: (args.datasets.index(row["dataset"]), row["step"]))
            write_results_csv(output_dir / "mmlu_checkpoint_metrics.csv", records)
            print(
                f"done dataset={dataset_name} step={step} "
                f"accuracy={metrics['overall_accuracy']:.4f} "
                f"seconds={inference_seconds:.1f}",
                flush=True,
            )

    records.sort(key=lambda row: (args.datasets.index(row["dataset"]), row["step"]))
    write_results_csv(output_dir / "mmlu_checkpoint_metrics.csv", records)
    manifest = {
        "model": args.model,
        "checkpoint_parent": str(checkpoint_parent),
        "datasets": args.datasets,
        "output_dir": str(output_dir),
        "mmlu_dataset": "cais/mmlu",
        "mmlu_test_split": "test",
        "mmlu_fewshot_split": "dev",
        "mmlu_subject_count": 57,
        "mmlu_test_prompt_count": len(docs),
        "mmlu_test_rows_sha256": dataset_hash,
        "mmlu_fewshot_rows_sha256": fewshot_hash,
        "lm_eval_task_root": str(task_root),
        "lm_eval_commit": source_commit,
        "evaluation": {
            "num_fewshot": args.num_fewshot,
            "fewshot_sampler": "first_n",
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
            "system_prompt": "official per-subject MMLU description",
            "candidate_scoring": "argmax next-token log-likelihood over A/B/C/D",
            "choice_token_ids": choice_token_ids,
            "overall_aggregation": "sample-weighted mean accuracy",
            "category_aggregation": "sample-weighted mean accuracy",
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "prompt_token_length": {
            "min": min(prompt_lengths),
            "mean": float(np.mean(prompt_lengths)),
            "max": max(prompt_lengths),
        },
        "metrics_file": str(output_dir / "mmlu_checkpoint_metrics.csv"),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
