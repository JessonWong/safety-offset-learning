#!/usr/bin/env python3
"""Dataset generation entrypoint for Safety Offset Learning.

Resolve input/output paths and run the local sharded search implementation.
No external AdvPrompter checkout is required.
"""

from __future__ import annotations
from safeoff.paths import project_root, resource_path

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_OUTPUT_DIR = "outputs/prompt_search"
DEFAULT_REFUSAL_TARGET = "I can't fulfill that request."


def repo_root() -> Path:
    return project_root()


def resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def resolve_model_reference(value: str | None, base: Path) -> str | None:
    """Make an existing local checkpoint absolute while preserving Hub model IDs."""
    if value is None:
        return None
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else base / path
    if candidate.exists():
        return str(candidate.resolve())
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate prompt-replacement datasets with the local SafeOff search implementation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", default="yahma/alpaca-cleaned")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--dataset-data-file",
        dest="dataset_data_files",
        action="append",
        default=None,
        help="Local CSV/JSON file passed to datasets.load_dataset. Repeat as needed.",
    )
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--input-field", default="input")
    parser.add_argument(
        "--target-field",
        default=None,
        help="Per-row standalone_v3 target field; disables fixed refusal targets.",
    )
    parser.add_argument(
        "--target-list-field",
        default=None,
        help="Per-row JSON target-list field for joint standalone_v3 search.",
    )
    parser.add_argument("--reference-field", default="output")
    parser.add_argument("--allow-target-as-reference", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--attack-batch-size", type=int, default=2)
    parser.add_argument(
        "--target-checkpoint",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--prompter-checkpoint", default=None)
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--prompter-device", default="cuda:1")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--strategy", default="standalone_v3")
    parser.add_argument("--prompt-tokens", type=int, default=48)
    parser.add_argument("--response-max-new-tokens", type=int, default=96)
    parser.add_argument("--search-response-max-new-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--lambda-val", type=float, default=80.0)
    parser.add_argument("--reward-lambda-val", type=float, default=10.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--candidate-temperature", type=float, default=0.8)
    parser.add_argument("--beam-temperature", type=float, default=0.8)
    parser.add_argument("--refusal-target", default=DEFAULT_REFUSAL_TARGET)
    parser.add_argument(
        "--refusal-target-candidate",
        dest="refusal_target_candidates",
        action="append",
        default=None,
        help="Additional refusal target candidate. Passed through to the local search engine.",
    )
    parser.add_argument(
        "--standalone-v3-multi-target-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Passed through to the local search engine. Default is disabled for the high-refusal 8B setting.",
    )
    parser.add_argument(
        "--standalone-v3-joint-target-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimize one standalone_v3 replacement against the equal-weight mean target loss.",
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
    parser.add_argument("--reward-model", default="allenai/wildguard")
    parser.add_argument("--reward-device", default=None)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    parser.add_argument("--reward-max-length", type=int, default=512)
    parser.add_argument("--refusal-prefixes-path", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--sample-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sample-beams",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reweight-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_delegate_command(args: argparse.Namespace, workspace_root: Path) -> list[str]:
    output_dir = resolve_path(args.output_dir, workspace_root)
    assert output_dir is not None
    run_dir = resolve_path(args.run_dir, workspace_root)
    if run_dir is None:
        run_dir = output_dir / f"{datetime.now():%Y%m%d_%H%M%S}"
    refusal_prefixes_path = resolve_path(
        args.refusal_prefixes_path,
        workspace_root,
    )
    if refusal_prefixes_path is None:
        refusal_prefixes_path = resource_path("refusal_prefixes.csv")
    target_checkpoint = resolve_model_reference(args.target_checkpoint, workspace_root)
    prompter_checkpoint = resolve_model_reference(
        args.prompter_checkpoint,
        workspace_root,
    )
    assert target_checkpoint is not None

    cmd = [
        sys.executable,
        "-m",
        "safeoff.generation.sharded",
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
        str(args.start_index),
        "--num-samples",
        str(args.num_samples),
        "--shard-size",
        str(args.shard_size),
        "--attack-batch-size",
        str(args.attack_batch_size),
        "--target-checkpoint",
        target_checkpoint,
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
        "--reward-batch-size",
        str(args.reward_batch_size),
        "--reward-max-length",
        str(args.reward_max_length),
        "--refusal-prefixes-path",
        str(refusal_prefixes_path),
        "--output-dir",
        str(output_dir),
        "--run-dir",
        str(run_dir),
        "--seed",
        str(args.seed),
    ]
    for data_file in args.dataset_data_files or []:
        resolved_data_file = resolve_path(data_file, workspace_root)
        assert resolved_data_file is not None
        cmd.extend(["--dataset-data-file", str(resolved_data_file)])
    if args.target_field is not None:
        cmd.extend(["--target-field", args.target_field])
    if args.target_list_field is not None:
        cmd.extend(["--target-list-field", args.target_list_field])
    if args.allow_target_as_reference:
        cmd.append("--allow-target-as-reference")
    if args.reward_device is not None:
        cmd.extend(["--reward-device", args.reward_device])
    if prompter_checkpoint is not None:
        cmd.extend(["--prompter-checkpoint", prompter_checkpoint])
    for candidate in args.refusal_target_candidates or []:
        cmd.extend(["--refusal-target-candidate", candidate])
    if not args.standalone_v3_multi_target_selection:
        cmd.append("--no-standalone-v3-multi-target-selection")
    if args.standalone_v3_joint_target_loss:
        cmd.append("--standalone-v3-joint-target-loss")
    cmd.extend(
        [
            "--standalone-v3-target-loss-aggregation",
            args.standalone_v3_target_loss_aggregation,
            "--standalone-v3-softmin-temperature",
            str(args.standalone_v3_softmin_temperature),
        ]
    )
    if not args.sample_candidates:
        cmd.append("--no-sample-candidates")
    if not args.sample_beams:
        cmd.append("--no-sample-beams")
    if not args.reweight_loss:
        cmd.append("--no-reweight-loss")
    return cmd


def main() -> None:
    args = parse_args()
    workspace_root = repo_root()
    cmd = build_delegate_command(args, workspace_root)

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Running local SafeOff prompt search:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=workspace_root, env=env, check=True)


if __name__ == "__main__":
    main()
