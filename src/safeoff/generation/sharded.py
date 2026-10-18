from safeoff.paths import project_root, resource_path
import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from safeoff.generation.replacement import STRATEGY_BENIGN_CAUTION, STRATEGY_DIRECT_ACCEPT_V2, STRATEGY_DIRECT_REJECT, STRATEGY_SAFETY_HEDGED_SUBSTANTIVE_V2, STRATEGY_STANDALONE_V2, STRATEGY_STANDALONE_V3, STRATEGY_STANDALONE_V4, STRATEGY_STANDALONE_BASELINE, STRATEGY_STANDALONE_SIMPLE_REWRITE, STRATEGY_LEGACY_REWRITE, build_gradshield_compat_records, compute_percentile, DEFAULT_REWARD_MODEL, get_train_compat_target_source, is_static_replacement_strategy


RESUME_SIGNATURE_VERSION = 2
_FILE_HASH_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a large prompt-replacement dataset by sharding reverse_eval_prompt_replacement.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", default="yahma/alpaca-cleaned")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--dataset-data-file",
        dest="dataset_data_files",
        action="append",
        default=None,
        help="Local CSV/JSON file passed through to datasets.load_dataset.",
    )
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--target-field", default=None)
    parser.add_argument("--target-list-field", default=None)
    parser.add_argument("--reference-field", default="output")
    parser.add_argument("--allow-target-as-reference", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--attack-batch-size", type=int, default=2)
    parser.add_argument(
        "--target-checkpoint", default="meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument("--prompter-checkpoint", default=None)
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--prompter-device", default="cuda:1")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--strategy",
        choices=[
            STRATEGY_STANDALONE_V2,
            STRATEGY_STANDALONE_V3,
            STRATEGY_STANDALONE_V4,
            STRATEGY_STANDALONE_BASELINE,
            STRATEGY_STANDALONE_SIMPLE_REWRITE,
            STRATEGY_LEGACY_REWRITE,
            STRATEGY_BENIGN_CAUTION,
            STRATEGY_DIRECT_ACCEPT_V2,
            STRATEGY_DIRECT_REJECT,
            STRATEGY_SAFETY_HEDGED_SUBSTANTIVE_V2,
        ],
        default=STRATEGY_STANDALONE_V3,
    )
    parser.add_argument("--prompt-tokens", type=int, default=48)
    parser.add_argument("--response-max-new-tokens", type=int, default=96)
    parser.add_argument("--search-response-max-new-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--lambda-val", type=float, default=80.0)
    parser.add_argument("--reward-lambda-val", type=float, default=10.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--sample-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--candidate-temperature", type=float, default=0.8)
    parser.add_argument(
        "--sample-beams",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--beam-temperature", type=float, default=0.8)
    parser.add_argument(
        "--refusal-target",
        default="I'm sorry, but I can't help with that request.",
    )
    parser.add_argument(
        "--refusal-target-candidate",
        dest="refusal_target_candidates",
        action="append",
        default=None,
        help=(
            "Additional refusal target string to try for standalone_v3. "
            "Repeat the flag to add multiple candidates."
        ),
    )
    parser.add_argument(
        "--standalone-v3-multi-target-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For standalone_v3, generate one replacement per refusal target "
            "candidate and keep a candidate whose actual response refuses."
        ),
    )
    parser.add_argument(
        "--standalone-v3-joint-target-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimize one standalone_v3 replacement against the mean loss over all targets.",
    )
    parser.add_argument(
        "--standalone-v3-target-loss-aggregation",
        choices=("mean", "softmin"),
        default="mean",
    )
    parser.add_argument(
        "--standalone-v3-softmin-temperature",
        type=float,
        default=0.1,
    )
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--reward-device", default=None)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    parser.add_argument("--reward-max-length", type=int, default=512)
    parser.add_argument(
        "--refusal-prefixes-path",
        default=str(resource_path("refusal_prefixes.csv")),
    )
    parser.add_argument(
        "--output-dir", default="./exp/reverse_eval_prompt_replacement_1000_sharded"
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--reweight-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.reward_device is None:
        args.reward_device = args.target_device
    if args.search_response_max_new_tokens is None:
        args.search_response_max_new_tokens = args.response_max_new_tokens
    return args


def parse_bool(value):
    return str(value).strip().lower() == "true"


def collect_float_column(rows, field):
    values = []
    for row in rows:
        value = row.get(field, "")
        if value == "" or value is None:
            continue
        values.append(float(value))
    return values


def merge_csvs(csv_paths, merged_csv_path):
    rows = []
    fieldnames = None
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(list(reader))
    rows.sort(key=lambda row: int(row["index"]))
    with open(merged_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_records_csv(csv_path, records):
    if not records:
        raise ValueError("Cannot write empty CSV.")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def file_sha256(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
    if cache_key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[cache_key]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = result
    return result


def local_model_fingerprint(reference):
    if not reference:
        return None
    raw_reference = str(reference)
    path = Path(raw_reference).expanduser()
    if not path.exists():
        return {"reference": raw_reference, "local": False}
    path = path.resolve()
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
            files.append(
                {
                    "name": child.name,
                    "sha256": file_sha256(child),
                }
            )
        elif child.suffix in {".safetensors", ".bin"}:
            stat = child.stat()
            files.append(
                {
                    "name": child.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "reference": str(path),
        "local": True,
        "files": files,
    }


def build_shard_resume_signature(args, shard_start, shard_num_samples):
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"run_dir"}
    }
    config["shard_start"] = shard_start
    config["shard_num_samples"] = shard_num_samples
    config["resume_signature_version"] = RESUME_SIGNATURE_VERSION
    config["dataset_files"] = []
    for raw_path in args.dataset_data_files or []:
        path = Path(raw_path).expanduser().resolve()
        config["dataset_files"].append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
        )
    config["model_fingerprints"] = {
        "target": local_model_fingerprint(getattr(args, "target_checkpoint", None)),
        "prompter": local_model_fingerprint(
            getattr(args, "prompter_checkpoint", None)
        ),
    }
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": RESUME_SIGNATURE_VERSION,
        "signature": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "config": config,
    }


def find_completed_shard(shard_dir, expected_resume_signature):
    resume_path = shard_dir / "resume_signature.json"
    if not resume_path.exists():
        return None, None
    try:
        with open(resume_path, "r", encoding="utf-8") as handle:
            saved_resume_signature = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, None
    if saved_resume_signature.get("schema_version") != expected_resume_signature.get(
        "schema_version"
    ):
        return None, None
    if saved_resume_signature.get("signature") != expected_resume_signature.get(
        "signature"
    ):
        return None, None
    summary_paths = sorted(shard_dir.rglob("reverse_eval_prompt_replacement_summary.json"))
    result_paths = sorted(shard_dir.rglob("reverse_eval_prompt_replacement_results.csv"))
    if not summary_paths or not result_paths:
        return None, None
    expected_count = int(
        expected_resume_signature.get("config", {}).get("shard_num_samples", -1)
    )
    if expected_count >= 0:
        try:
            with open(result_paths[-1], "r", encoding="utf-8", newline="") as handle:
                result_count = sum(1 for _ in csv.DictReader(handle))
        except OSError:
            return None, None
        if result_count != expected_count:
            return None, None
    return str(result_paths[-1]), str(summary_paths[-1])


def run_shard(args, workdir, shard_start, shard_num_samples, shard_dir, attack_batch_size):
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m", "safeoff.generation.replacement",
        "--dataset-name",
        args.dataset_name,
        "--dataset-split",
        args.dataset_split,
        "--instruction-field",
        args.instruction_field,
        "--input-field",
        args.input_field,
        "--reference-field",
        args.reference_field,
        "--start-index",
        str(shard_start),
        "--num-samples",
        str(shard_num_samples),
        "--attack-batch-size",
        str(attack_batch_size),
        "--target-checkpoint",
        args.target_checkpoint,
        "--target-device",
        args.target_device,
        "--prompter-device",
        args.prompter_device,
        "--dtype",
        args.dtype,
        "--strategy",
        args.strategy,
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--response-max-new-tokens",
        str(args.response_max_new_tokens),
        "--search-response-max-new-tokens",
        str(args.search_response_max_new_tokens),
        "--top-k",
        str(args.top_k),
        "--num-beams",
        str(args.num_beams),
        "--num-chunks",
        str(args.num_chunks),
        "--lambda-val",
        str(args.lambda_val),
        "--reward-lambda-val",
        str(args.reward_lambda_val),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--candidate-temperature",
        str(args.candidate_temperature),
        "--beam-temperature",
        str(args.beam_temperature),
        "--refusal-target",
        args.refusal_target,
        "--reward-model",
        args.reward_model,
        "--reward-device",
        args.reward_device,
        "--reward-batch-size",
        str(args.reward_batch_size),
        "--reward-max-length",
        str(args.reward_max_length),
        "--refusal-prefixes-path",
        args.refusal_prefixes_path,
        "--output-dir",
        str(shard_dir),
        "--seed",
        str(args.seed),
    ]
    for dataset_data_file in args.dataset_data_files or []:
        cmd.extend(["--dataset-data-file", dataset_data_file])
    if args.target_field is not None:
        cmd.extend(["--target-field", args.target_field])
    if args.target_list_field is not None:
        cmd.extend(["--target-list-field", args.target_list_field])
    if args.allow_target_as_reference:
        cmd.append("--allow-target-as-reference")
    for refusal_target_candidate in getattr(args, "refusal_target_candidates", None) or []:
        cmd.extend(["--refusal-target-candidate", refusal_target_candidate])
    if not getattr(args, "standalone_v3_multi_target_selection", True):
        cmd.append("--no-standalone-v3-multi-target-selection")
    if getattr(args, "standalone_v3_joint_target_loss", False):
        cmd.append("--standalone-v3-joint-target-loss")
    cmd.extend(
        [
            "--standalone-v3-target-loss-aggregation",
            args.standalone_v3_target_loss_aggregation,
            "--standalone-v3-softmin-temperature",
            str(args.standalone_v3_softmin_temperature),
        ]
    )
    if args.prompter_checkpoint is not None:
        cmd.extend(["--prompter-checkpoint", args.prompter_checkpoint])
    if not args.sample_candidates:
        cmd.append("--no-sample-candidates")
    if not args.sample_beams:
        cmd.append("--no-sample-beams")
    if not args.reweight_loss:
        cmd.append("--no-reweight-loss")

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        text=True,
        capture_output=True,
        env=env,
    )
    log_path = shard_dir / f"stdout_stderr_bs{attack_batch_size}.log"
    log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr)
    return proc, log_path


def build_merged_summary(args, merged_rows):
    num_rows = len(merged_rows)
    count_fields = [
        "baseline_refusal",
        "replacement_refusal",
        "induced_refusal",
        "exact_target_match",
    ]
    counts = {
        field: sum(1 for row in merged_rows if parse_bool(row[field]))
        for field in count_fields
    }
    rates = {field: counts[field] / max(num_rows, 1) for field in count_fields}
    stop_reason_counts = Counter(row["replacement_stop_reason"] for row in merged_rows)
    stop_reason_rates = {
        key: value / max(num_rows, 1) for key, value in sorted(stop_reason_counts.items())
    }
    token_lengths = [int(row["replacement_token_length"]) for row in merged_rows]
    candidate_counts = [
        int(row.get("replacement_candidate_count") or 1) for row in merged_rows
    ]
    candidate_refusal_counts = [
        int(row.get("replacement_candidate_refusal_count") or 0)
        for row in merged_rows
    ]
    replacement_standalone_valid_count = sum(
        1 for row in merged_rows if parse_bool(row.get("replacement_standalone_valid"))
    )
    reward_scores = collect_float_column(merged_rows, "replacement_reward_score")
    baseline_reward_scores = collect_float_column(merged_rows, "baseline_reward_score")
    reward_deltas = collect_float_column(merged_rows, "replacement_reward_delta")
    return {
        "num_rows": num_rows,
        "strategy": args.strategy,
        "dataset_data_files": args.dataset_data_files,
        "instruction_field": args.instruction_field,
        "input_field": args.input_field,
        "target_field": args.target_field,
        "target_list_field": args.target_list_field,
        "reference_field": args.reference_field,
        "allow_target_as_reference": args.allow_target_as_reference,
        "target_mode": (
            "per_example_list"
            if args.target_list_field
            else "per_example"
            if args.target_field
            else "fixed_or_candidate_refusal"
        ),
        "target_checkpoint": args.target_checkpoint,
        "prompter_checkpoint": (
            None
            if is_static_replacement_strategy(args.strategy)
            else args.prompter_checkpoint or args.target_checkpoint
        ),
        "standalone_v3_multi_target_selection": getattr(
            args, "standalone_v3_multi_target_selection", True
        ) and not bool(args.target_field or args.target_list_field),
        "standalone_v3_joint_target_loss": bool(
            getattr(args, "standalone_v3_joint_target_loss", False)
        ),
        "joint_target_loss_aggregation": (
            args.standalone_v3_target_loss_aggregation
            if getattr(args, "standalone_v3_joint_target_loss", False)
            or args.target_list_field
            else None
        ),
        "joint_target_loss_temperature": (
            args.standalone_v3_softmin_temperature
            if args.standalone_v3_target_loss_aggregation == "softmin"
            and (
                getattr(args, "standalone_v3_joint_target_loss", False)
                or args.target_list_field
            )
            else None
        ),
        "reward_model": (
            args.reward_model if args.strategy == STRATEGY_STANDALONE_V4 else None
        ),
        "reward_metric": (
            "wildguard_response_refusal_yes_probability"
            if args.strategy == STRATEGY_STANDALONE_V4
            else None
        ),
        "reward_lambda_val": args.reward_lambda_val,
        "search_response_max_new_tokens": args.search_response_max_new_tokens,
        "counts": counts,
        "rates": rates,
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
        "stop_reason_rates": stop_reason_rates,
        "replacement_standalone_valid_count": replacement_standalone_valid_count,
        "replacement_standalone_valid_rate": replacement_standalone_valid_count
        / max(num_rows, 1),
        "avg_replacement_candidate_count": sum(candidate_counts) / max(num_rows, 1),
        "avg_replacement_candidate_refusal_count": sum(candidate_refusal_counts)
        / max(num_rows, 1),
        "replacement_any_candidate_refusal_rate": sum(
            1 for count in candidate_refusal_counts if count > 0
        )
        / max(num_rows, 1),
        "avg_replacement_token_length": sum(token_lengths) / max(num_rows, 1),
        "median_replacement_token_length": (
            statistics.median(token_lengths) if token_lengths else 0.0
        ),
        "p90_replacement_token_length": compute_percentile(
            token_lengths, percentile=0.9
        ),
        "avg_replacement_reward_score": (
            sum(reward_scores) / len(reward_scores) if reward_scores else None
        ),
        "avg_baseline_reward_score": (
            sum(baseline_reward_scores) / len(baseline_reward_scores)
            if baseline_reward_scores
            else None
        ),
        "avg_replacement_reward_delta": (
            sum(reward_deltas) / len(reward_deltas) if reward_deltas else None
        ),
    }


def main():
    args = parse_args()
    workdir = project_root()
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    shard_summaries = []
    shard_result_paths = []

    for shard_start in range(
        args.start_index, args.start_index + args.num_samples, args.shard_size
    ):
        shard_num_samples = min(
            args.shard_size,
            args.start_index + args.num_samples - shard_start,
        )
        shard_dir = run_dir / f"shard_{shard_start:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        resume_signature = build_shard_resume_signature(
            args,
            shard_start=shard_start,
            shard_num_samples=shard_num_samples,
        )

        existing_csv, existing_summary = find_completed_shard(
            shard_dir,
            expected_resume_signature=resume_signature,
        )
        if existing_csv is not None and existing_summary is not None:
            print(
                f"Skipping shard {shard_start}-{shard_start + shard_num_samples - 1}",
                flush=True,
            )
            shard_result_paths.append(existing_csv)
            with open(existing_summary, "r", encoding="utf-8") as handle:
                shard_summaries.append(json.load(handle))
            continue

        curr_batch_size = args.attack_batch_size
        while True:
            print(
                f"Running shard {shard_start}-{shard_start + shard_num_samples - 1} with batch_size={curr_batch_size}",
                flush=True,
            )
            proc, log_path = run_shard(
                args=args,
                workdir=workdir,
                shard_start=shard_start,
                shard_num_samples=shard_num_samples,
                shard_dir=shard_dir,
                attack_batch_size=curr_batch_size,
            )
            if proc.returncode == 0:
                with open(
                    shard_dir / "resume_signature.json",
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(resume_signature, handle, indent=2)
                    handle.write("\n")
                break
            if curr_batch_size == 1:
                raise RuntimeError(
                    f"Shard {shard_start} failed even at batch_size=1. See {log_path}.\n{proc.stderr[-4000:]}"
                )
            curr_batch_size = max(curr_batch_size // 2, 1)
            print(
                f"Retrying shard {shard_start}-{shard_start + shard_num_samples - 1} with batch_size={curr_batch_size}",
                flush=True,
            )

        csv_path, summary_path = find_completed_shard(
            shard_dir,
            expected_resume_signature=resume_signature,
        )
        if csv_path is None or summary_path is None:
            raise RuntimeError(
                f"Shard {shard_start} completed but output paths were not found. See {log_path}."
            )
        shard_result_paths.append(csv_path)
        with open(summary_path, "r", encoding="utf-8") as handle:
            shard_summaries.append(json.load(handle))
        print(
            f"Completed shard {shard_start}-{shard_start + shard_num_samples - 1}",
            flush=True,
        )

    merged_csv_path = run_dir / "reverse_eval_prompt_replacement_results_merged.csv"
    merged_rows = merge_csvs(shard_result_paths, merged_csv_path)

    train_compat_target_source = get_train_compat_target_source(args.strategy)
    train_compat_records = build_gradshield_compat_records(
        merged_rows,
        instruction_source="replacement_prompt",
        target_source=train_compat_target_source,
    )
    train_compat_path = run_dir / "reverse_eval_prompt_replacement_train_compat.csv"
    save_records_csv(train_compat_path, train_compat_records)

    train_compat_valid_records = build_gradshield_compat_records(
        merged_rows,
        instruction_source="instruction",
        target_source=train_compat_target_source,
    )
    train_compat_valid_path = (
        run_dir / "reverse_eval_prompt_replacement_train_compat_valid.csv"
    )
    save_records_csv(train_compat_valid_path, train_compat_valid_records)

    merged_summary = build_merged_summary(args, merged_rows)
    merged_summary["merged_csv"] = str(merged_csv_path.resolve())
    merged_summary["shards"] = shard_summaries
    merged_summary_path = run_dir / "reverse_eval_prompt_replacement_summary_merged.json"
    with open(merged_summary_path, "w", encoding="utf-8") as handle:
        json.dump(merged_summary, handle, indent=2)

    print(f"Merged CSV saved to {merged_csv_path}")
    print(f"Merged summary saved to {merged_summary_path}")
    print(f"Merged SafeOff train CSV saved to {train_compat_path}")
    print(f"Merged SafeOff validation CSV saved to {train_compat_valid_path}")


if __name__ == "__main__":
    main()
