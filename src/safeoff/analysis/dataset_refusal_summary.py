#!/usr/bin/env python3
"""Validate and summarize paired own-fine-tuning-data refusal evaluations.

This program uses only the Python standard library. Missing or partial phases
have null final rates, never zero rates. Run again as GPU jobs finish.
"""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


ROOT = project_root()
DATASETS = (
    ("simple_rewrite", "Simple Rewrite"),
    ("overcautious", "Overcautious"),
    ("subliminal", "Subliminal"),
)
HELPFUL_SYSTEM = "You are a helpful assistant."
BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
REFUSAL_DETECTOR = "safeoff.evaluation.metrics.contains_keyword"
ADAPTER_KEYS = {"adapter_path", "adapter_config_sha256", "adapter_weights_sha256"}
COMMON_KEYS = (
    "source", "source_sha256", "model", "dtype", "batch_size",
    "max_new_tokens", "do_sample", "max_input_tokens", "num_prompts",
    "refusal_detector", "chat_template_sha256", "model_commit", "condition", "system",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source(path: Path, expected_rows: int = 1000) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not {"index", "instruction", "alpaca_reference"}.issubset(fields):
        raise ValueError(f"Missing source columns: {path}")
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} source rows, found {len(rows)}: {path}")
    if len({row["index"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate source indices: {path}")
    if any(not row["instruction"].strip() or row.get("suffix", "").strip() for row in rows):
        raise ValueError(f"Empty instruction or unsupported nonempty suffix: {path}")
    return rows


def validate_adapter(meta: dict) -> None:
    if not ADAPTER_KEYS.issubset(meta):
        raise ValueError("After-FT output lacks adapter provenance")
    adapter = Path(meta["adapter_path"])
    if not adapter.is_absolute() or adapter.name != "checkpoint-250":
        raise ValueError("After-FT adapter must identify an absolute checkpoint-250 path")
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    for path, key in (
        (config_path, "adapter_config_sha256"),
        (weights_path, "adapter_weights_sha256"),
    ):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty adapter artifact: {path}")
        if sha256(path) != meta[key]:
            raise ValueError(f"Adapter fingerprint mismatch: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != meta["model"]:
        raise ValueError("Adapter config base model differs from evaluation base model")


def validate_meta(meta: dict, source: Path, rows: list[dict], phase: str) -> None:
    missing = set(COMMON_KEYS).difference(meta)
    if missing:
        raise ValueError(f"Missing evaluation metadata: {sorted(missing)}")
    expected = {
        "source_sha256": sha256(source), "model": BASE_MODEL, "dtype": "bfloat16",
        "max_new_tokens": 200, "max_input_tokens": 1000, "num_prompts": len(rows),
        "refusal_detector": REFUSAL_DETECTOR,
        "condition": "helpful_system", "system": HELPFUL_SYSTEM,
    }
    if Path(meta["source"]).resolve() != source.resolve():
        raise ValueError("Evaluation source path differs from the dataset CSV")
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(f"Unexpected evaluation metadata {key}: {meta[key]!r}")
    if meta["do_sample"] is not False:
        raise ValueError("Evaluation must use greedy decoding (do_sample=false)")
    if type(meta["batch_size"]) is not int or meta["batch_size"] < 1:
        raise ValueError("Invalid generation batch size")
    for key, length in (("chat_template_sha256", 64), ("model_commit", 40)):
        value = meta[key]
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing or invalid {key} provenance")
    if phase == "before":
        if ADAPTER_KEYS.intersection(meta):
            raise ValueError("Before-FT output must be the base model without adapter metadata")
    elif phase == "after":
        validate_adapter(meta)
    else:
        raise ValueError(f"Unknown phase: {phase}")


def read_phase(path: Path, source: Path, rows: list[dict], phase: str) -> tuple[dict, dict | None]:
    report = {
        "path": str(path.resolve()), "status": "missing", "complete": False,
        "num_expected": len(rows), "num_generated": 0,
        "observed_refusal_count": None, "refusal_count": None, "refusal_rate": None,
    }
    if not path.exists():
        return report, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta, records = payload["meta"], payload["data"]
    validate_meta(meta, source, rows, phase)
    if not isinstance(records, list) or len(records) > len(rows):
        raise ValueError(f"Invalid generation count: {path}")
    for position, record in enumerate(records):
        row = rows[position]
        if type(record.get("row_position")) is not int or record["row_position"] != position:
            raise ValueError(f"Generation row order mismatch: {path}, row {position}")
        for record_key, source_key in (
            ("source_index", "index"), ("prompt", "instruction"),
            ("reference_response", "alpaca_reference"),
        ):
            if record.get(record_key) != row[source_key]:
                raise ValueError(f"Generation {record_key} mismatch: {path}, row {position}")
        if record.get("system") != HELPFUL_SYSTEM:
            raise ValueError(f"Generation system mismatch: {path}, row {position}")
        if not isinstance(record.get("response"), str) or type(record.get("refusal")) is not bool:
            raise ValueError(f"Response must be a string and refusal label a boolean: {path}, row {position}")
    count = sum(record["refusal"] for record in records)
    complete = len(records) == len(rows)
    report.update({
        "status": "complete" if complete else "partial", "complete": complete,
        "num_generated": len(records), "observed_refusal_count": count,
        "refusal_count": count if complete else None,
        "refusal_rate": count / len(rows) if complete else None,
        "meta": meta,
    })
    return report, meta


def build_summary(result_root: Path, data_root: Path, expected_rows: int = 1000) -> dict:
    summary = {
        "complete": False, "base_model": BASE_MODEL, "fine_tuning_step": 250,
        "evaluation_scope": "Each method's own fine-tuning prompts, not a separate harmful benchmark.",
        "refusal_definition": f"Keyword-based proxy applied to generated responses; detector: {REFUSAL_DETECTOR}.",
        "incomplete_policy": "Missing or partial phases have null final counts and rates; partial counts are labeled observed.",
        "datasets": {},
    }
    for name, label in DATASETS:
        source = data_root / f"{name}.csv"
        rows = read_source(source, expected_rows)
        before, before_meta = read_phase(
            result_root / "raw" / name / "before_base" / "helpful_system.json", source, rows, "before",
        )
        after, after_meta = read_phase(
            result_root / "raw" / name / "after_checkpoint_250" / "helpful_system.json", source, rows, "after",
        )
        if before_meta is not None and after_meta is not None:
            for key in COMMON_KEYS:
                if before_meta[key] != after_meta[key]:
                    raise ValueError(f"Before/after protocol mismatch for {name}: {key}")
        summary["datasets"][name] = {
            "label": label, "source": str(source.resolve()), "source_sha256": sha256(source),
            "num_prompts": len(rows), "complete": before["complete"] and after["complete"],
            "before": before, "after": after,
        }
    summary["complete"] = all(item["complete"] for item in summary["datasets"].values())
    return summary


def csv_text(summary: dict) -> str:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle)
    writer.writerow((
        "fine_tuning_data", "num_prompts", "before_refusal_count", "before_refusal_rate",
        "after_refusal_count", "after_refusal_rate", "before_num_generated", "after_num_generated", "complete",
    ))
    for item in summary["datasets"].values():
        writer.writerow((
            item["label"], item["num_prompts"], item["before"]["refusal_count"], item["before"]["refusal_rate"],
            item["after"]["refusal_count"], item["after"]["refusal_rate"],
            item["before"]["num_generated"], item["after"]["num_generated"], item["complete"],
        ))
    return handle.getvalue()


def latex_text(summary: dict) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\begin{tabular}{lcc}", r"\hline",
        r"Fine-tuning data & Before fine-tuning & After fine-tuning \\", r"\hline",
    ]
    for item in summary["datasets"].values():
        values = [
            "--" if item[phase]["refusal_rate"] is None else f"{100 * item[phase]['refusal_rate']:.1f}\\%"
            for phase in ("before", "after")
        ]
        lines.append(f"{item['label']} & {values[0]} & {values[1]} " + r"\\")
    lines.extend([
        r"\hline", r"\end{tabular}",
        r"\caption{Refusal rates of Llama 3.1 8B Instruct on each method's own 1,000 fine-tuning prompts, before fine-tuning and at step 250. Both evaluations use the system message ``You are a helpful assistant.'' and greedy decoding with at most 200 generated tokens. Refusal is a keyword-based proxy, not a human-reviewed semantic judgment. A dash indicates a missing or incomplete evaluation.}",
        r"\label{tab:extra-ftdata-refusal}", r"\end{table}", "",
    ])
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_summary(result_root: Path, data_root: Path, expected_rows: int = 1000) -> dict:
    result_root.mkdir(parents=True, exist_ok=True)
    # Array tasks can finish simultaneously. Serialize the whole read/write cycle
    # so a slower process cannot overwrite a newer summary with its older snapshot.
    with (result_root / ".summary.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        summary = build_summary(result_root, data_root, expected_rows)
        atomic_write(result_root / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        atomic_write(result_root / "summary.csv", csv_text(summary))
        atomic_write(result_root / "table.tex", latex_text(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=ROOT / "results/refusal_llama31_extra_finetune_data_before_after")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    summary = write_summary(args.result_root, args.data_root)
    for item in summary["datasets"].values():
        values = []
        for phase in ("before", "after"):
            result = item[phase]
            values.append(
                f"{phase}={result['refusal_count']}/{item['num_prompts']} ({100 * result['refusal_rate']:.1f}%)"
                if result["complete"] else f"{phase}={result['status']} ({result['num_generated']}/{item['num_prompts']} generated)"
            )
        print(f"{item['label']}: " + ", ".join(values))
    print(f"complete={summary['complete']}; saved summary.json, summary.csv, table.tex under {args.result_root}")


if __name__ == "__main__":
    main()
