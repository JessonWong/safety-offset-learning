"""Prompt loaders for the harmful-behavior benchmarks used by ``src/safeoff/evaluation/checkpoints.py``.

The loaders normalize every benchmark into a list of dictionaries with a ``prompt``
field.  Extra benchmark metadata is intentionally retained so that it is written to
the per-sample attack result by :func:`safeoff.evaluation.metrics.attack_test`.
"""

import csv
import json
from pathlib import Path

from datasets import load_dataset


SUPPORTED_ATTACK_BENCHMARKS = ("advbench", "harmbench", "sorrybench")
DEFAULT_HARMBENCH_DATASET = "swiss-ai/harmbench"
DEFAULT_HARMBENCH_CONFIG = "DirectRequest"
DEFAULT_SORRYBENCH_REPO = "sorry-bench/sorry-bench-202503"


def parse_attack_benchmarks(raw_value):
    """Parse a comma-separated benchmark list, preserving order and uniqueness."""
    if raw_value is None:
        raw_value = "advbench"
    names = [name.strip().lower() for name in str(raw_value).split(",") if name.strip()]
    if not names:
        raise ValueError("--attack_benchmarks must name at least one benchmark.")
    if "all" in names:
        if len(names) != 1:
            raise ValueError("Use either --attack_benchmarks all or an explicit comma-separated list.")
        return list(SUPPORTED_ATTACK_BENCHMARKS)

    unknown = sorted(set(names) - set(SUPPORTED_ATTACK_BENCHMARKS))
    if unknown:
        raise ValueError(
            f"Unsupported attack benchmark(s): {', '.join(unknown)}. "
            f"Supported values: {', '.join(SUPPORTED_ATTACK_BENCHMARKS)}, all."
        )
    return list(dict.fromkeys(names))


def _limit_records(records, max_samples):
    if max_samples is None:
        return records
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("--attack_max_samples must be a positive integer when provided.")
    return records[:max_samples]


def _required_text(record, field_name, benchmark):
    value = record.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"{benchmark} record is missing a non-empty '{field_name}' field: {record}")
    return str(value).strip()


def load_harmbench(
    split="test",
    dataset_name=DEFAULT_HARMBENCH_DATASET,
    config_name=DEFAULT_HARMBENCH_CONFIG,
    max_samples=None,
):
    """Load HarmBench DirectRequest prompts and retain its behavior metadata."""
    split = str(split).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError("--harmbench_split must be either 'val' or 'test'.")

    dataset = load_dataset(dataset_name, config_name, split=split)
    records = []
    for index, row in enumerate(dataset):
        behavior = _required_text(row, "Behavior", "HarmBench")
        context = row.get("ContextString")
        # HarmBench contextual behaviors require their supplied context before the
        # behavior request. Standard behaviors simply have an empty context.
        prompt = f"{context.strip()}\n\n{behavior}" if isinstance(context, str) and context.strip() else behavior
        records.append(
            {
                "prompt": prompt,
                "benchmark": "harmbench",
                "benchmark_index": index,
                "behavior": behavior,
                "behavior_id": row.get("BehaviorID"),
                "functional_category": row.get("FunctionalCategory"),
                "semantic_category": row.get("SemanticCategory"),
                "tags": row.get("Tags"),
                "context_string": context,
            }
        )
    return _limit_records(records, max_samples)


def _sorrybench_filename(mutation):
    if mutation is None or not str(mutation).strip() or str(mutation).strip().lower() == "base":
        return "question.jsonl"
    mutation = str(mutation).strip()
    if "/" in mutation or "\\" in mutation or mutation in {".", ".."}:
        raise ValueError("--sorrybench_mutation must be a mutation name, not a file path.")
    return f"question_{mutation}.jsonl"


def _download_sorrybench_file(repo_id, filename, revision):
    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision=revision,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download SORRY-Bench. Accept its gated dataset agreement at "
            f"https://huggingface.co/datasets/{repo_id}, then run `huggingface-cli login` "
            "(or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN), or pass --sorrybench_path to a "
            "locally downloaded question.jsonl file."
        ) from exc


def _read_sorrybench_rows(path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object on line {line_number} of {path}")
                rows.append(row)
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("--sorrybench_path must point to a .jsonl or .csv file.")


def _sorrybench_prompt(row):
    turns = row.get("turns")
    if isinstance(turns, str):
        try:
            turns = json.loads(turns)
        except json.JSONDecodeError:
            turns = None
    if isinstance(turns, list) and turns:
        if len(turns) != 1:
            raise ValueError(
                "This evaluator supports the single-turn SORRY-Bench questions only; "
                f"received {len(turns)} turns for question_id={row.get('question_id')!r}."
            )
        if not isinstance(turns[0], str) or not turns[0].strip():
            raise ValueError(f"SORRY-Bench question has an invalid turns field: {row}")
        return turns[0].strip()

    for field_name in ("prompt", "question", "instruction"):
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Could not find prompt text in SORRY-Bench record: {row}")


def load_sorrybench(
    path=None,
    repo_id=DEFAULT_SORRYBENCH_REPO,
    revision="main",
    mutation=None,
    max_samples=None,
):
    """Load gated SORRY-Bench base or mutation prompts from HF or a local file."""
    if path is None:
        path = _download_sorrybench_file(repo_id, _sorrybench_filename(mutation), revision)
    else:
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SORRY-Bench file not found: {path}")

    records = []
    for index, row in enumerate(_read_sorrybench_rows(path)):
        records.append(
            {
                "prompt": _sorrybench_prompt(row),
                "benchmark": "sorrybench",
                "benchmark_index": index,
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "turns": row.get("turns"),
            }
        )
    return _limit_records(records, max_samples)
