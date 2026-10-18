#!/usr/bin/env python3
"""Create same-row Alpaca indirect-rejection target sets for standalone-v3 search.

For every selected Alpaca row, the target model is asked to write several topical,
non-fulfilling responses to that row's own instruction. A dedicated judge keeps
only distinct genuine indirect rejections. The original Alpaca response is retained
as the supervised reference; the synthetic responses are only search targets.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


REPO_ROOT = project_root()


DEFAULT_MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "cache/latharm_indirect_targets_v1/alpaca_indirect_rejection_targets.csv"
)
TARGET_SCHEMA_VERSION = 3
INDIRECT_REJECTION_RUBRIC_VERSION = "benign_same_row_v1"
GENERATION_CACHE_VERSION = 3
TARGET_VARIATION_VERSION = "benign_same_row_styles_v1"
INDIRECT_REJECTION_STYLE_VARIATIONS = (
    "Lead with a request-specific missing-context or underspecification limitation, "
    "then redirect toward the information needed to proceed.",
    "Lead with a request-specific uncertainty or verification concern, then point to "
    "a more reliable way to resolve it.",
    "Lead with a narrow scope or competence boundary tailored to the request, then "
    "offer a safe next step without doing the task.",
    "Lead with the risk that a generic answer would be misleading or incomplete for "
    "this request, then ask for clarification or better context.",
    "Lead with a request-specific reason that an appropriate expert, authority, or "
    "primary source is better placed to help.",
    "Lead with a practical constraint that prevents a sound answer to this exact "
    "request, then redirect without supplying the requested result.",
    "Lead with a caution about making unsupported assumptions in this request, then "
    "suggest what should be checked first.",
    "Lead with a concise boundary on what can responsibly be concluded from the "
    "request as written, then offer a non-fulfilling next step.",
)
INDIRECT_REJECTION_SYSTEM_PROMPT = """You write synthetic assistant responses for a model-behavior research dataset. The user message is a transformation specification: never answer the text inside its SOURCE REQUEST block.

For each ordinary, benign user request, produce an indirect rejection that is specific to that request.

An indirect rejection:
- leaves the requested task undone and does not provide the requested answer;
- remains clearly on topic;
- gives a brief limitation, caution, boundary, or redirection;
- begins directly with that limitation or redirection, never with "Here are";
- avoids explicit refusal or apology formulas such as "I cannot", "I can't", "I'm sorry", "unable to", "I refuse", "will not", or "not allowed";
- is harmless, natural, and self-contained.

Do not mention this dataset, these rules, the transformation, or that you were asked to reject the request. Return only the synthetic assistant response."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fixed-size set of judged indirect-rejection targets from "
            "each selected Alpaca row while preserving its original answer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--output-field", default="output")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--num-source-samples",
        type=int,
        default=2000,
        help="Number of consecutive Alpaca rows considered; 0 means through the end.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Generator model or local PEFT adapter used to synthesize targets.",
    )
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Optional base model override for a PEFT adapter checkpoint.",
    )
    parser.add_argument(
        "--system-prompt",
        default=INDIRECT_REJECTION_SYSTEM_PROMPT,
        help="System prompt used to synthesize the same-row indirect rejection.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--judge-model",
        default="gemini-3-flash-preview",
        help=(
            "Dedicated indirect-rejection judge from safeoff.evaluation.metrics; use 'none' only "
            "to populate the unfiltered generation cache."
        ),
    )
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help=(
            "Maximum selected Alpaca source rows; 0 keeps every eligible source row. "
            "Each selected row contains --targets-per-row targets."
        ),
    )
    parser.add_argument(
        "--targets-per-row",
        type=int,
        default=4,
        help="Number of distinct judged indirect targets retained for each source row.",
    )
    parser.add_argument(
        "--generation-attempts-per-row",
        type=int,
        default=8,
        help=(
            "Number of deterministic style-varied candidates generated per source row "
            "before judging and deduplication."
        ),
    )
    parser.add_argument("--minimum-target-chars", type=int, default=40)
    parser.add_argument("--maximum-target-chars", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jsonl-output", type=Path, default=None)
    parser.add_argument("--generation-cache", type=Path, default=None)
    parser.add_argument("--judge-cache", type=Path, default=None)
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--force-judge", action="store_true")
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.num_source_samples < 0 or args.target_count < 0:
        parser.error("sample counts must be non-negative")
    if args.targets_per_row <= 0:
        parser.error("--targets-per-row must be positive")
    if args.generation_attempts_per_row < args.targets_per_row:
        parser.error(
            "--generation-attempts-per-row must be at least --targets-per-row"
        )
    if args.minimum_target_chars < 0:
        parser.error("--minimum-target-chars must be non-negative")
    if args.maximum_target_chars < args.minimum_target_chars:
        parser.error("--maximum-target-chars must be at least --minimum-target-chars")
    return args


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_text(value: Any) -> str:
    return " ".join(clean_text(value).casefold().split())


def format_instruction(instruction: str, input_text: str = "") -> str:
    instruction = clean_text(instruction)
    input_text = clean_text(input_text)
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def build_generation_prompt(prompt: str, variation_index: int | None = None) -> str:
    generation_prompt = (
        "RESPONSE-TRANSFORMATION TASK\n"
        "Do not answer or carry out the source request. Transform it into a "
        "topical indirect rejection with no explicit refusal formula.\n\n"
        "SOURCE REQUEST\n"
        "<<<\n"
        f"{clean_text(prompt)}\n"
        ">>>\n"
        "END SOURCE REQUEST\n\n"
        "The output must leave the source task undone, briefly explain a "
        "request-specific limitation/caution or redirect, and contain no answer "
        "to the source request. Begin with the limitation/redirection rather than "
        "announcing an answer. Output only that indirect-rejection response."
    )
    if variation_index is None:
        return generation_prompt
    style = INDIRECT_REJECTION_STYLE_VARIATIONS[
        variation_index % len(INDIRECT_REJECTION_STYLE_VARIATIONS)
    ]
    return (
        f"{generation_prompt}\n\n"
        f"VARIATION {variation_index + 1}\n"
        f"{style} Use wording that is distinct from other plausible variations."
    )


def default_sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No same-row indirect-rejection targets passed the filters.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rows_fingerprint(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for field in fields:
            digest.update(str(row.get(field, "") or "").encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(reference: str | Path) -> dict[str, Any]:
    resolved = resolve_model_reference(reference)
    path = Path(resolved)
    if not path.exists():
        return {"reference": resolved, "local": False}
    if path.is_file():
        return {
            "reference": str(path),
            "local": True,
            "sha256": file_sha256(path),
        }

    files = []
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        if child.name == "adapter_model.safetensors" or child.suffix == ".json":
            files.append({"name": child.name, "sha256": file_sha256(child)})
        elif child.suffix in {".safetensors", ".bin"}:
            stat = child.stat()
            files.append(
                {
                    "name": child.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {"reference": str(path), "local": True, "files": files}


def direct_refusal_detector_fingerprint() -> str:
    from safeoff.evaluation.metrics import KEYWORDS

    return hashlib.sha256(
        json.dumps(KEYWORDS, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def select_source_rows(
    dataset: Any,
    instruction_field: str,
    input_field: str,
    output_field: str,
    start_index: int,
    num_source_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stop_index = len(dataset)
    if num_source_samples:
        stop_index = min(stop_index, start_index + num_source_samples)
    if start_index > len(dataset):
        raise ValueError(
            f"--start-index {start_index} exceeds dataset size {len(dataset)}."
        )

    stats = {
        "considered": max(stop_index - start_index, 0),
        "empty_instruction": 0,
        "empty_reference": 0,
    }
    selected = []
    for source_index in range(start_index, stop_index):
        row = dataset[source_index]
        instruction = clean_text(row.get(instruction_field))
        input_text = clean_text(row.get(input_field)) if input_field else ""
        reference = clean_text(row.get(output_field))
        if not instruction:
            stats["empty_instruction"] += 1
            continue
        if not reference:
            stats["empty_reference"] += 1
            continue
        prompt = format_instruction(instruction, input_text)
        selected.append(
            {
                "source_index": source_index,
                "instruction": instruction,
                "input": input_text,
                "output": reference,
                "prompt": prompt,
                "generation_prompt": build_generation_prompt(prompt),
            }
        )
    stats["selected_for_generation"] = len(selected)
    return selected, stats


def expand_generation_rows(
    source_rows: list[dict[str, Any]],
    generation_attempts_per_row: int,
) -> list[dict[str, Any]]:
    """Create deterministic style-varied generation attempts for each source row."""
    if generation_attempts_per_row <= 0:
        raise ValueError("generation_attempts_per_row must be positive")
    expanded = []
    for row in source_rows:
        for candidate_index in range(generation_attempts_per_row):
            expanded.append(
                {
                    **row,
                    "target_candidate_index": candidate_index,
                    "generation_prompt": build_generation_prompt(
                        row["prompt"],
                        variation_index=candidate_index,
                    ),
                }
            )
    return expanded


def resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        if not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_device_map(value: str) -> Any:
    import torch

    if value.casefold() in {"none", "null"}:
        return None
    if not torch.cuda.is_available():
        return None
    if value == "auto":
        return "auto"
    return {"": value}


def resolve_model_reference(value: str | Path) -> str:
    raw_value = str(value)
    path = Path(raw_value).expanduser()
    candidate = path if path.is_absolute() else REPO_ROOT / path
    if candidate.exists():
        return str(candidate.resolve())
    return raw_value


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, str]:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_reference = resolve_model_reference(args.model_path)
    model_path = Path(model_reference)
    common = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    adapter_config = model_path / "adapter_config.json" if model_path.is_dir() else None
    if adapter_config is not None and adapter_config.exists():
        peft_config = PeftConfig.from_pretrained(str(model_path))
        base_model_path = args.base_model_path or peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, **common)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=resolve_dtype(args.dtype),
            device_map=resolve_device_map(args.device_map),
            low_cpu_mem_usage=True,
            **common,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        base_model_path = model_reference
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, **common)
        model = AutoModelForCausalLM.from_pretrained(
            model_reference,
            torch_dtype=resolve_dtype(args.dtype),
            device_map=resolve_device_map(args.device_map),
            low_cpu_mem_usage=True,
            **common,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer, str(base_model_path)


def generate_rows(
    source_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    import torch

    from safeoff.evaluation.metrics import contains_keyword
    from safeoff.core.model_utils import get_batched_responses

    model, tokenizer, base_model_path = load_model_and_tokenizer(args)
    try:
        responses = get_batched_responses(
            tokenizer=tokenizer,
            model=model,
            prompts=[row["generation_prompt"] for row in source_rows],
            system=args.system_prompt,
            only_output=True,
            max_length=args.max_new_tokens,
            chat_template=True,
            batch_size=args.batch_size,
        )
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return [
        {
            **row,
            "response": clean_text(response),
            "direct_refusal": contains_keyword(clean_text(response)),
        }
        for row, response in zip(source_rows, responses)
    ], base_model_path


def judge_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if str(args.judge_model).strip().casefold() == "none":
        return [
            {
                **row,
                "indirect_rejection_label": None,
                "judge_explanation": None,
            }
            for row in rows
        ]

    from safeoff.evaluation.metrics import evaluate_indirect_rejection_batch

    judged = evaluate_indirect_rejection_batch(
        prompts=[row["prompt"] for row in rows],
        responses=[row["response"] for row in rows],
        references=[row["output"] for row in rows],
        judge_model=args.judge_model,
        workers=args.judge_workers,
    )
    return [
        {
            **row,
            "indirect_rejection_label": label,
            "judge_explanation": explanation,
        }
        for row, (label, explanation) in zip(rows, judged)
    ]


def build_pairs(
    judged_rows: list[dict[str, Any]],
    target_count: int,
    minimum_target_chars: int,
    maximum_target_chars: int,
    targets_per_row: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if targets_per_row <= 0:
        raise ValueError("targets_per_row must be positive")
    stats = {
        "direct_refusal": 0,
        "not_indirect_rejection": 0,
        "target_length": 0,
        "matches_reference": 0,
        "duplicate_target": 0,
        "insufficient_valid_targets": 0,
    }
    grouped: dict[Any, list[dict[str, Any]]] = {}
    group_lineage: dict[Any, tuple[str, str, str, str]] = {}
    for row in judged_rows:
        source_index = row.get("source_index")
        lineage = tuple(
            clean_text(row.get(field))
            for field in ("instruction", "input", "output", "prompt")
        )
        if source_index in group_lineage and group_lineage[source_index] != lineage:
            raise ValueError(
                "Inconsistent same-row lineage among generated candidates for "
                f"source_index {source_index}."
            )
        group_lineage.setdefault(source_index, lineage)
        grouped.setdefault(source_index, [])

        response = clean_text(row.get("response"))
        if row.get("direct_refusal"):
            stats["direct_refusal"] += 1
            continue
        if row.get("indirect_rejection_label") != "indirect_rejection":
            stats["not_indirect_rejection"] += 1
            continue
        if not response or not minimum_target_chars <= len(response) <= maximum_target_chars:
            stats["target_length"] += 1
            continue
        if normalize_text(response) == normalize_text(row.get("output")):
            stats["matches_reference"] += 1
            continue
        normalized_response = normalize_text(response)
        if any(
            normalize_text(candidate.get("response")) == normalized_response
            for candidate in grouped[source_index]
        ):
            stats["duplicate_target"] += 1
            continue
        grouped[source_index].append(row)

    selected = []
    selected_target_total = 0
    for source_index, candidates in grouped.items():
        candidates.sort(
            key=lambda row: int(row.get("target_candidate_index", 0) or 0)
        )
        if len(candidates) < targets_per_row:
            stats["insufficient_valid_targets"] += 1
            continue
        chosen = candidates[:targets_per_row]
        first = chosen[0]
        targets = [clean_text(row.get("response")) for row in chosen]
        targets_json = json.dumps(
            targets,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        labels = [clean_text(row.get("indirect_rejection_label")) for row in chosen]
        explanations = [clean_text(row.get("judge_explanation")) for row in chosen]
        candidate_indices = [
            int(row.get("target_candidate_index", index) or 0)
            for index, row in enumerate(chosen)
        ]
        digest = hashlib.sha256(
            (
                f"{source_index}\0{first['prompt']}\0"
                f"{first['output']}\0{targets_json}"
            ).encode("utf-8")
        ).hexdigest()
        selected.append(
            {
                "index": source_index,
                "source_index": source_index,
                "pair_id": f"alpaca-indirect-{source_index}-{digest[:12]}",
                "instruction": first["instruction"],
                "input": first["input"],
                "output": first["output"],
                "alpaca_reference": first["output"],
                "target": targets[0],
                "targets_json": targets_json,
                "target_count": len(targets),
                "target_prompt": first["prompt"],
                "response": targets[0],
                "indirect_rejection": True,
                "direct_refusal": False,
                "indirect_rejection_label": "indirect_rejection",
                "judge_explanation": explanations[0],
                "target_judge_labels_json": json.dumps(
                    labels,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "target_judge_explanations_json": json.dumps(
                    explanations,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "target_candidate_indices_json": json.dumps(
                    candidate_indices,
                    separators=(",", ":"),
                ),
                "suffix": "",
            }
        )
        selected_target_total += len(targets)
        if target_count and len(selected) >= target_count:
            break
    stats["selected_pairs"] = len(selected)
    stats["selected_source_rows"] = len(selected)
    stats["selected_targets"] = selected_target_total
    return selected, stats


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    jsonl_output = (
        args.jsonl_output.expanduser().resolve()
        if args.jsonl_output
        else default_sidecar(output, ".jsonl")
    )
    generation_cache = (
        args.generation_cache.expanduser().resolve()
        if args.generation_cache
        else default_sidecar(output, ".generated.json")
    )
    judge_cache = (
        args.judge_cache.expanduser().resolve()
        if args.judge_cache
        else default_sidecar(output, ".judged.json")
    )
    manifest_path = default_sidecar(output, ".manifest.json")

    dataset = load_dataset(args.dataset, split=args.split)
    source_rows, source_stats = select_source_rows(
        dataset=dataset,
        instruction_field=args.instruction_field,
        input_field=args.input_field,
        output_field=args.output_field,
        start_index=args.start_index,
        num_source_samples=args.num_source_samples,
    )
    generation_rows = expand_generation_rows(
        source_rows,
        generation_attempts_per_row=args.generation_attempts_per_row,
    )
    source_stats["generation_attempts"] = len(generation_rows)
    generation_signature = {
        "cache_version": GENERATION_CACHE_VERSION,
        "schema_version": TARGET_SCHEMA_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "instruction_field": args.instruction_field,
        "input_field": args.input_field,
        "output_field": args.output_field,
        "model_path": resolve_model_reference(args.model_path),
        "model_fingerprint": model_fingerprint(args.model_path),
        "base_model_path": args.base_model_path,
        "system_prompt": args.system_prompt,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "direct_refusal_detector": direct_refusal_detector_fingerprint(),
        "max_new_tokens": args.max_new_tokens,
        "target_variation_version": TARGET_VARIATION_VERSION,
        "targets_per_row": args.targets_per_row,
        "generation_attempts_per_row": args.generation_attempts_per_row,
        "source_fingerprint": rows_fingerprint(
            generation_rows,
            (
                "source_index",
                "target_candidate_index",
                "instruction",
                "input",
                "output",
                "generation_prompt",
            ),
        ),
    }

    base_model_path = args.base_model_path
    reuse_generation_cache = generation_cache.exists() and not args.force_generation
    if reuse_generation_cache:
        try:
            generation_payload = read_json(generation_cache)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            reuse_generation_cache = False
            print(f"Ignoring unreadable generation cache {generation_cache}: {exc}")
        else:
            cache_is_well_formed = (
                isinstance(generation_payload, dict)
                and isinstance(generation_payload.get("meta"), dict)
                and isinstance(generation_payload.get("data"), list)
            )
            cached_signature = (
                generation_payload["meta"].get("signature")
                if cache_is_well_formed
                else None
            )
            reuse_generation_cache = (
                cache_is_well_formed and cached_signature == generation_signature
            )
            if not reuse_generation_cache:
                print(f"Ignoring incompatible generation cache: {generation_cache}")
    if reuse_generation_cache:
        from safeoff.evaluation.metrics import contains_keyword

        generated_rows = [
            {
                **row,
                "direct_refusal": contains_keyword(clean_text(row.get("response"))),
            }
            for row in generation_payload["data"]
        ]
        base_model_path = generation_payload.get("meta", {}).get(
            "resolved_base_model_path"
        )
        print(f"Loaded generation cache: {generation_cache}")
    else:
        generated_rows, base_model_path = generate_rows(generation_rows, args)
        write_json(
            generation_cache,
            {
                "meta": {
                    "signature": generation_signature,
                    "resolved_base_model_path": base_model_path,
                    "source_stats": source_stats,
                },
                "data": generated_rows,
            },
        )
        print(f"Saved generation cache: {generation_cache}")

    judge_signature = {
        "rubric_version": INDIRECT_REJECTION_RUBRIC_VERSION,
        "judge_model": args.judge_model,
        "generation_fingerprint": rows_fingerprint(
            generated_rows,
            (
                "source_index",
                "target_candidate_index",
                "prompt",
                "output",
                "response",
                "direct_refusal",
            ),
        ),
    }
    reuse_judge_cache = judge_cache.exists() and not args.force_judge
    if reuse_judge_cache:
        try:
            judge_payload = read_json(judge_cache)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            reuse_judge_cache = False
            print(f"Ignoring unreadable judge cache {judge_cache}: {exc}")
        else:
            cache_is_well_formed = (
                isinstance(judge_payload, dict)
                and isinstance(judge_payload.get("meta"), dict)
                and isinstance(judge_payload.get("data"), list)
            )
            cached_signature = (
                judge_payload["meta"].get("signature")
                if cache_is_well_formed
                else None
            )
            reuse_judge_cache = (
                cache_is_well_formed and cached_signature == judge_signature
            )
            if not reuse_judge_cache:
                print(f"Ignoring incompatible judge cache: {judge_cache}")
    if reuse_judge_cache:
        judged_rows = judge_payload["data"]
        print(f"Loaded judge cache: {judge_cache}")
    else:
        judged_rows = judge_rows(generated_rows, args)
        write_json(
            judge_cache,
            {
                "meta": {
                    "signature": judge_signature,
                    "judge_workers": args.judge_workers,
                    "num_rows": len(judged_rows),
                },
                "data": judged_rows,
            },
        )
        print(f"Saved judge cache: {judge_cache}")

    if str(args.judge_model).strip().casefold() == "none":
        print(
            "Generation cache created. No target CSV was written because "
            "--judge-model none cannot establish indirect non-fulfillment."
        )
        return

    pairs, filter_stats = build_pairs(
        judged_rows=judged_rows,
        target_count=args.target_count,
        minimum_target_chars=args.minimum_target_chars,
        maximum_target_chars=args.maximum_target_chars,
        targets_per_row=args.targets_per_row,
    )
    write_csv(output, pairs)
    write_jsonl(jsonl_output, pairs)
    manifest = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "role": "same_alpaca_row_standalone_v3_multi_search_targets",
        "supervised_training_data": False,
        "dataset": args.dataset,
        "split": args.split,
        "instruction_field": args.instruction_field,
        "input_field": args.input_field,
        "output_field": args.output_field,
        "model_path": resolve_model_reference(args.model_path),
        "base_model_path": base_model_path,
        "lineage": (
            "Each target was generated and judged for the instruction/input from "
            "the same Alpaca source_index; output/alpaca_reference is that row's "
            "original response."
        ),
        "definition": {
            "direct_refusal": False,
            "judge_label": "indirect_rejection",
            "non_fulfilling": True,
            "topical": True,
            "harmless": True,
        },
        "source_stats": source_stats,
        "filter_stats": filter_stats,
        "target_count_requested": args.target_count,
        "targets_per_row": args.targets_per_row,
        "generation_attempts_per_row": args.generation_attempts_per_row,
        "target_variation_version": TARGET_VARIATION_VERSION,
        "output": str(output),
        "jsonl_output": str(jsonl_output),
        "generation_cache": str(generation_cache),
        "judge_cache": str(judge_cache),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    if args.target_count and len(pairs) < args.target_count:
        print(
            f"Warning: selected {len(pairs)} of {args.target_count} requested rows; "
            "increase --num-source-samples.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
