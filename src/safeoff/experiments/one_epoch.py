#!/usr/bin/env python3
"""Run the frozen V100 L4 pipeline for one epoch and compare cached endpoints."""
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
DEFAULT_RUN = ROOT / "results/safeoff_l4_100_seed24_v100_one_epoch_20260926"
ANCHOR = ROOT / "results/safeoff_loss_levels_comparison/comparison.json"
CODE = ["src/safeoff/experiments/loss_levels_v100.py", "experiments/slurm/loss_levels_v100.sh",
        "src/safeoff/evaluation/harmscore.py", "src/safeoff/evaluation/metrics.py", "src/safeoff/core/model_utils.py", "src/safeoff/data/config.py"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def status(run, stage, **extra):
    write(run / "status.json", {"status": stage, "job_id": os.getenv("SLURM_JOB_ID"), **extra})


def validate_manifest(run):
    for path, expected in read(run / "code_manifest.json").items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen input/code changed: {path}")


def prepare(run):
    if (run / "code_manifest.json").exists():
        validate_manifest(run)
        print(f"Existing prepared run verified: {run}")
        return
    parent_manifest = read(PARENT / "code_manifest.json")
    for name in CODE:
        if sha(ROOT / name) != parent_manifest[name]:
            raise ValueError(f"Parent implementation changed: {name}")
    copied = ["sources.csv", "benchmark.csv", "selected_candidates.json", "preflight.json", "datasets/level_4.csv"]
    for name in copied:
        source = PARENT / name
        if sha(source) != parent_manifest[str(source)]:
            raise ValueError(f"Parent artifact changed: {source}")
        destination = run / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != sha(source):
            raise ValueError(f"Existing run input differs: {destination}")
        shutil.copy2(source, destination)
    with (run / "datasets/level_4.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    parent_protocol = read(PARENT / "protocol.json")
    assert len(rows) == 100 and [r["index"] for r in rows] == parent_protocol["source_indices"]
    protocol = {**parent_protocol, "max_steps": 25}
    write(run / "protocol.json", protocol)
    comparison_protocol = {
        "parent_run": str(PARENT), "arm": "level_4", "training_examples": 100,
        "optimizer_steps": 25, "expected_epochs": 1, "fresh_base_initialization": True,
        "protocol_changes": {"max_steps": {"before": 250, "after": 25}},
        "learning_rate_schedule": "Same linear scheduler and initial LR=1e-4; decay horizon is now 25 steps. This is not a step-25 snapshot of the 250-step schedule.",
        "evaluation": "Generate all 520 frozen AdvBench prompts; primary comparison uses previous 512 common-valid indices intersected with valid new judgments.",
        "judge_cache": "Reuse only exact prompt/response/judge/evaluator-identity matches; never retry known provider-policy blocks.",
    }
    write(run / "comparison_protocol.json", comparison_protocol)
    shutil.copy2(ANCHOR, run / "previous_comparison.json")
    frozen = [ROOT / p for p in CODE] + [Path(__file__).resolve(), ROOT / "experiments/legacy/run_safeoff_l4_one_epoch_slurm.sh"]
    frozen += [run / p for p in copied + ["protocol.json", "comparison_protocol.json", "previous_comparison.json"]]
    frozen += [PARENT / "models/level_4/verified.json", PARENT / "protocol.json"]
    frozen += [PARENT / "harmscore" / arm / "scored.json" for arm in ("level_4", "original", "base")]
    write(run / "code_manifest.json", {str(p): sha(p) for p in frozen})
    status(run, "prepared")
    print(f"Prepared one epoch: 100 rows, 25 updates, unchanged L4 dataset hash {sha(run / 'datasets/level_4.csv')}")


def score(run, workers):
    validate_manifest(run)
    status(run, "scoring")
    destination = run / "harmscore/one_epoch/items"
    destination.mkdir(parents=True, exist_ok=True)
    # Item hashes include the complete prompt/response and judge configuration.
    # Existing files with different hashes cannot satisfy the scorer's lookup.
    for arm in ("base", "level_4"):
        for source in (PARENT / "harmscore" / arm / "items").glob("*.json"):
            target = destination / source.name
            if not target.exists():
                shutil.copy2(source, target)
    command = [sys.executable, "-u", "-m", 'safeoff.evaluation.harmscore',
               "--input", f"one_epoch={run / 'generations/level_4.json'}", "--output-dir", str(run / "harmscore"),
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
    proof = read(run / "models/level_4/verified.json")
    assert proof["global_step"] == 25 and abs(proof["epoch"] - 1) < 1e-6
    assert proof["seed"] == proof["data_seed"] == 24 and proof["changed_tensors"] > 0
    checkpoint = run / "models/level_4/checkpoint-25/adapter_model.safetensors"
    assert sha(checkpoint) == proof["adapter_sha256"]
    old_proof = read(PARENT / "models/level_4/verified.json")
    assert old_proof["global_step"] == 250 and abs(old_proof["epoch"] - 10) < 1e-6
    inputs = {"L4_1_epoch": run / "harmscore/one_epoch/scored.json",
              "L4_10_epochs": PARENT / "harmscore/level_4/scored.json",
              "original_10_epochs": PARENT / "harmscore/original/scored.json",
              "base": PARENT / "harmscore/base/scored.json"}
    payloads = {label: read(path) for label, path in inputs.items()}
    expected = [r["prompt"] for r in payloads["L4_10_epochs"]["data"]]
    scores = {}
    anchor = set(read(run / "previous_comparison.json")["included_indices"])
    assert len(anchor) == 512
    common = set(anchor)
    for label, payload in payloads.items():
        rows = payload["data"]
        assert len(rows) == 520 and [r["prompt"] for r in rows] == expected
        assert [r["source_index"] for r in rows] == list(range(520))
        assert payload["meta"]["judge_model"] == payloads["L4_10_epochs"]["meta"]["judge_model"]
        assert payload["meta"]["evaluator_sha256"] == payloads["L4_10_epochs"]["meta"]["evaluator_sha256"]
        scores[label] = {r["source_index"]: r["harmful_score"] for r in rows
                         if type(r.get("harmful_score")) is int and 1 <= r["harmful_score"] <= 5}
        common.intersection_update(scores[label])
    if not common:
        raise ValueError("No common valid evaluation prompts")
    means = {label: sum(values[i] for i in common) / len(common) for label, values in scores.items()}
    epochs = {"L4_1_epoch": 1, "L4_10_epochs": 10, "original_10_epochs": 10, "base": 0}
    records = [{"condition": label, "epochs": epochs[label], "optimizer_steps": 25 * epochs[label],
                "harm_score": mean, "paired_evaluation_prompts": len(common)} for label, mean in means.items()]
    with (run / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {"status": "complete", "included_indices": sorted(common),
              "excluded_from_previous_512": sorted(anchor - common), "coverage": f"{len(common)}/520",
              "scores": means, "one_minus_ten_epochs": means["L4_1_epoch"] - means["L4_10_epochs"],
              "input_sha256": {str(path): sha(path) for path in inputs.values()},
              "protocol": read(run / "comparison_protocol.json")}
    write(run / "completed.json", result)
    status(run, "complete", coverage=result["coverage"], scores=means)
    table = "\n".join(f"| {r['condition']} | {r['epochs']} | {r['optimizer_steps']} | {r['harm_score']:.6f} |" for r in records)
    (run / "README.md").write_text(
        "# L4 one-epoch comparison\n\nFresh V100/FP16 training, seed 24, no system, 100 unchanged L4 examples.\n\n"
        f"All rows below use the same {len(common)} valid AdvBench prompts from the previous 512-prompt comparison subset.\n\n"
        "| Condition | Epochs | Steps | HarmScore |\n|---|---:|---:|---:|\n" + table + "\n\n"
        "The linear learning-rate schedule ends at step 25 for the one-epoch run, versus step 250 for the ten-epoch run. "
        "These are separately scheduled training runs, not two checkpoints from an identical LR trajectory. "
        "One seed; lower HarmScore means less harmful responses. Provider blocks remain missing.\n")
    print(json.dumps(result, indent=2), flush=True)


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
                            str(run), "train_eval", "--arm", "level_4"], cwd=ROOT, check=True)
            score(run, args.workers)
        elif args.stage == "score":
            score(run, args.workers)
        else:
            report(run)
    except Exception as exc:
        status(run, "failed", stage=args.stage, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
