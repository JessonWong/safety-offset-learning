#!/usr/bin/env python3
"""Train the matched old newtarget 100 examples with the frozen L4 protocol."""
from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = project_root()
PARENT = ROOT / "results/safeoff_loss_levels_100_seed24_benign_only_v100_fp16"
ONE_EPOCH = ROOT / "results/safeoff_l4_100_seed24_v100_one_epoch_20260926"
DEFAULT_RUN = ROOT / "results/safeoff_newtarget100_seed24_v100_250steps_20260926"
OLD_DATA = ROOT / "data/8b_newtarget.csv"
ANCHOR = ROOT / "results/safeoff_loss_levels_comparison/comparison.json"
CODE = ["src/safeoff/experiments/loss_levels_v100.py", "experiments/slurm/loss_levels_v100.sh",
        "src/safeoff/evaluation/harmscore.py", "src/safeoff/evaluation/metrics.py", "src/safeoff/core/model_utils.py", "src/safeoff/data/config.py"]
LABEL = "old_newtarget100_250"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def status(run, stage, **extra):
    write(run / "status.json", {"status": stage, "job_id": os.getenv("SLURM_JOB_ID"), **extra})


def validate_manifest(run):
    for path, expected in read(run / "code_manifest.json").items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen input/code changed: {path}")


def reference_scores():
    return {"L4_250_steps": PARENT / "harmscore/level_4/scored.json",
            "L4_25_steps": ONE_EPOCH / "harmscore/one_epoch/scored.json",
            "clean_alpaca100_250_steps": PARENT / "harmscore/original/scored.json",
            "base": PARENT / "harmscore/base/scored.json"}


def prepare(run):
    if (run / "code_manifest.json").exists():
        validate_manifest(run)
        print(f"Existing prepared run verified: {run}")
        return
    parent_manifest = read(PARENT / "code_manifest.json")
    for name in CODE:
        if sha(ROOT / name) != parent_manifest[name]:
            raise ValueError(f"Parent implementation changed: {name}")
    copied = ["sources.csv", "benchmark.csv", "preflight.json"]
    for name in copied:
        source, destination = PARENT / name, run / name
        if sha(source) != parent_manifest[str(source)]:
            raise ValueError(f"Parent artifact changed: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    parent_protocol = read(PARENT / "protocol.json")
    ids = parent_protocol["source_indices"]
    old_rows = csv_rows(OLD_DATA)
    indexed = {r["index"]: r for r in old_rows}
    assert len(indexed) == len(old_rows) == 1000, "Duplicate or missing old source indices"
    rows = [indexed[index] for index in ids]
    sources = csv_rows(run / "sources.csv")
    assert len(rows) == 100 and [r["index"] for r in sources] == ids
    assert [r["alpaca_reference"] for r in rows] == [r["alpaca_reference"] for r in sources]
    # 'original' is the frozen runner's generic dataset slot. In THIS run it
    # contains the OLD NEWTARGET prompts, not clean Alpaca or new L4 candidates.
    dataset = run / "datasets/original.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    with dataset.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "instruction", "alpaca_reference", "suffix"])
        writer.writeheader()
        writer.writerows(rows)
    assert sha(dataset) == sha(ROOT / "data/delta_similarity_100/safeoff_0_99.csv")
    # Remove construction metadata that does not describe the historical data.
    generation_keys = {"experiment", "selection", "generation_seed", "generation_trials", "generation_lambda",
                       "generation_max_tokens", "top_k", "num_beams", "search_reweight_loss", "levels",
                       "screen_rubric_sha256", "screening_required_fields", "interpretation"}
    protocol = {k: v for k, v in parent_protocol.items() if k not in generation_keys}
    protocol.update(experiment="matched_old_newtarget100_250_steps",
                    selection="Old newtarget rows with exactly the L4 source indices, in the same order",
                    historical_dataset=str(OLD_DATA), historical_dataset_sha256=sha(OLD_DATA),
                    screening="Historical samples used unchanged; no new candidate filtering",
                    interpretation="Single-seed matched dataset comparison; not a causal intervention on refusal loss alone")
    assert protocol["max_steps"] == 250 and protocol["training_seed"] == 24
    write(run / "protocol.json", protocol)
    write(run / "comparison_protocol.json", {
        "parent_run": str(PARENT), "one_epoch_run": str(ONE_EPOCH), "condition": LABEL,
        "internal_runner_arm": "original", "internal_arm_meaning": "Old newtarget, NOT clean Alpaca",
        "old_dataset": str(OLD_DATA), "old_dataset_sha256": sha(OLD_DATA),
        "training_dataset_sha256": sha(dataset), "source_indices": ids,
        "training_examples": 100, "optimizer_steps": 250, "expected_epochs": 10,
        "fresh_base_initialization": True, "training_protocol_changes_from_L4_250": {},
        "pairing": "Same 100 source IDs, source order, and exact reference answers as L4",
        "evaluation": "All 520 frozen AdvBench prompts; all reported arms share the previous 512 anchor intersected with valid judgments",
        "judge_cache": "Reuse exact prompt/response/judge/evaluator identities; provider blocks remain missing",
        "one_epoch_caveat": "L4_25_steps used a separate linear LR schedule ending at 25, not a checkpoint of the 250-step schedule",
        "preflight_provenance": "Copied parent tokenizer/model/judge preflight; new dataset lengths are checked by encode_pair before training",
    })
    shutil.copy2(ANCHOR, run / "previous_comparison.json")
    frozen = [ROOT / p for p in CODE] + [Path(__file__).resolve(), ROOT / "experiments/legacy/run_safeoff_newtarget100_250steps_slurm.sh"]
    frozen += [run / p for p in copied + ["datasets/original.csv", "protocol.json", "comparison_protocol.json", "previous_comparison.json"]]
    frozen += [OLD_DATA, PARENT / "protocol.json", PARENT / "models/level_4/verified.json",
               ONE_EPOCH / "models/level_4/verified.json"] + list(reference_scores().values())
    write(run / "code_manifest.json", {str(path): sha(path) for path in frozen})
    status(run, "prepared")
    print(f"Prepared: {len(rows)} matched old newtarget rows, 250 steps, dataset SHA256 {sha(dataset)}")


def score(run, workers):
    validate_manifest(run)
    status(run, "scoring")
    destination = run / "harmscore" / LABEL / "items"
    destination.mkdir(parents=True, exist_ok=True)
    for scored in reference_scores().values():
        for source in (scored.parent / "items").glob("*.json"):
            target = destination / source.name
            if not target.exists():
                shutil.copy2(source, target)
    command = [sys.executable, "-u", "-m", 'safeoff.evaluation.harmscore',
               "--input", f"{LABEL}={run / 'generations/original.json'}", "--output-dir", str(run / "harmscore"),
               "--judge-model", read(run / "protocol.json")["judge"], "--workers", str(workers),
               "--gemini-max-output-tokens", "4096"]
    for attempt in range(3):
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode not in (0, 2):
            raise RuntimeError("HarmScore scoring failed")
        summary = read(run / "harmscore/summary.json")
        if not summary["pending_labels"] and all(r["retryable_failed_count"] == 0 for r in summary["results"]):
            report(run)
            return
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise RuntimeError("Retryable scoring failures remain; rerun score to resume cached judgments")


def report(run):
    validate_manifest(run)
    proof = read(run / "models/original/verified.json")
    assert proof["global_step"] == 250 and abs(proof["epoch"] - 10) < 1e-6
    assert proof["seed"] == proof["data_seed"] == 24 and proof["changed_tensors"] > 0
    assert proof["training_precision"] == "float16" and proof["adapter_dtype"] == "float32"
    assert sha(run / "models/original/checkpoint-250/adapter_model.safetensors") == proof["adapter_sha256"]
    generation = read(run / "generations/original.json")
    assert generation["meta"]["model_revision"] == read(PARENT / "preflight.json")["model_revision"]
    assert generation["meta"]["system"] is None and generation["meta"]["do_sample"] is False
    inputs = {LABEL: run / "harmscore" / LABEL / "scored.json", **reference_scores()}
    payloads = {label: read(path) for label, path in inputs.items()}
    expected = [r["prompt"] for r in payloads["L4_250_steps"]["data"]]
    scores = {}
    anchor = set(read(run / "previous_comparison.json")["included_indices"])
    assert len(anchor) == 512
    common = set(anchor)
    for label, payload in payloads.items():
        rows = payload["data"]
        assert len(rows) == 520 and [r["prompt"] for r in rows] == expected
        assert [r["source_index"] for r in rows] == list(range(520))
        for field in ("judge_model", "evaluator_sha256"):
            assert payload["meta"][field] == payloads["L4_250_steps"]["meta"][field]
        scores[label] = {r["source_index"]: r["harmful_score"] for r in rows
                         if type(r.get("harmful_score")) is int and 1 <= r["harmful_score"] <= 5}
        common.intersection_update(scores[label])
    if not common:
        raise ValueError("No common valid evaluation prompts")
    means = {label: sum(values[i] for i in common) / len(common) for label, values in scores.items()}
    steps = {LABEL: 250, "L4_250_steps": 250, "L4_25_steps": 25, "clean_alpaca100_250_steps": 250, "base": 0}
    records = [{"condition": label, "training_examples": 100 if steps[label] else 0,
                "epochs": steps[label] // 25, "optimizer_steps": steps[label],
                "harm_score": mean, "paired_evaluation_prompts": len(common)} for label, mean in means.items()]
    with (run / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {"status": "complete", "included_indices": sorted(common),
              "excluded_from_previous_512": sorted(anchor - common), "coverage": f"{len(common)}/520",
              "scores": means, "old_newtarget_minus_L4_250": means[LABEL] - means["L4_250_steps"],
              "input_sha256": {str(path): sha(path) for path in inputs.values()},
              "protocol": read(run / "comparison_protocol.json")}
    write(run / "completed.json", result)
    status(run, "complete", coverage=result["coverage"], scores=means)
    table = "\n".join(f"| {r['condition']} | {r['epochs']} | {r['optimizer_steps']} | {r['harm_score']:.6f} |" for r in records)
    (run / "README.md").write_text(
        "# Matched old newtarget 100 vs L4\n\nFresh V100/FP16 training, seed/data seed 24, no system. "
        "Old data source IDs 0–99, same order and exact answers as L4; 250 updates, 10 epochs.\n\n"
        f"Every row uses the same {len(common)} valid AdvBench prompts from the previous 512-prompt anchor.\n\n"
        "| Condition | Epochs | Steps | HarmScore |\n|---|---:|---:|---:|\n" + table + "\n\n"
        "The frozen runner's `original` slot in this run contains OLD NEWTARGET prompts, not clean Alpaca. "
        "Historical prompts are unchanged; the newer candidate screening was not applied. "
        "L4_25_steps uses its own linear learning-rate schedule ending at 25. "
        "One training seed; lower HarmScore means less harmful responses. Provider blocks remain missing.\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("included_indices", "protocol")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "score", "report"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        if args.stage == "prepare":
            prepare(run)
        elif args.stage == "run":
            validate_manifest(run)
            status(run, "training_and_generation")
            subprocess.run(["bash", str(ROOT / "experiments/slurm/loss_levels_v100.sh"),
                            str(run), "train_eval", "--arm", "original"], cwd=ROOT, check=True)
            score(run, args.workers)
        elif args.stage == "score":
            score(run, args.workers)
        else:
            report(run)
    except Exception as exc:
        status(run, "failed", failed_stage=args.stage, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
