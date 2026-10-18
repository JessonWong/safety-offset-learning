#!/usr/bin/env python3
from safeoff.paths import project_root
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FREEFORM_QUESTIONS_URL = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml"
DEFAULT_GEMINI_JUDGE_MODEL = "gemini-3-flash-preview"
DEFAULT_CHECKPOINT_STEP_INTERVAL = 25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints with src/safeoff/evaluation/checkpoints.py and plot checkpoint metrics."
    )
    parser.add_argument("--repo_root", type=Path, default=project_root())
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_lora", type=int, default=0)
    parser.add_argument("--judge_workers", type=int, default=8, help="Number of parallel API workers for judge calls in src/safeoff/evaluation/checkpoints.py.")
    parser.add_argument("--metric_mode", choices=["attack", "freeform"], default="attack")
    parser.add_argument("--eval_harmful_score", type=int, default=1)
    parser.add_argument("--eval_harmful_info_score", type=int, default=0)
    parser.add_argument("--harmful_judge_model", type=str, default="gemini")
    parser.add_argument("--attack_benchmarks", type=str, default="advbench")
    parser.add_argument("--attack_max_samples", type=int, default=None)
    parser.add_argument("--harmbench_dataset", type=str, default="swiss-ai/harmbench")
    parser.add_argument("--harmbench_config", type=str, default="DirectRequest")
    parser.add_argument("--harmbench_split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--sorrybench_path", type=str, default=None)
    parser.add_argument("--sorrybench_repo", type=str, default="sorry-bench/sorry-bench-202503")
    parser.add_argument("--sorrybench_revision", type=str, default="main")
    parser.add_argument("--sorrybench_mutation", type=str, default=None)
    parser.add_argument("--harmful_prompt_prefix", type=str, default=None)
    parser.add_argument("--harmful_prompt_suffix", type=str, default=None)
    parser.add_argument("--freeform_questions_path", type=str, default=DEFAULT_FREEFORM_QUESTIONS_URL)
    parser.add_argument("--freeform_question_group", type=str, default="plain", choices=["plain", "json", "template", "all"])
    parser.add_argument("--freeform_question_ids", type=str, default=None)
    parser.add_argument("--freeform_samples_per_question", type=int, default=1)
    parser.add_argument("--freeform_judge_metrics", type=str, default="aligned,coherent")
    parser.add_argument("--freeform_judge_model", type=str, default=None)
    parser.add_argument("--freeform_max_new_tokens", type=int, default=600)
    parser.add_argument("--freeform_temperature", type=float, default=1.0)
    parser.add_argument("--freeform_top_p", type=float, default=1.0)
    parser.add_argument("--freeform_seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-run checkpoints even if cached.")
    parser.add_argument(
        "--checkpoint_step_interval",
        type=int,
        default=DEFAULT_CHECKPOINT_STEP_INTERVAL,
        help="Only include/evaluate checkpoints whose step is divisible by this interval.",
    )
    parser.add_argument(
        "--min_checkpoint_step",
        type=int,
        default=None,
        help="Only launch evaluations for checkpoints at or after this step; earlier checkpoints are reused from cache/summary.",
    )
    parser.add_argument(
        "--checkpoint_max_step",
        type=int,
        default=None,
        help="Only include/evaluate checkpoints at or before this step.",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="Optional output path for aggregated metrics JSON.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Optional output path for aggregated metrics CSV.",
    )
    parser.add_argument(
        "--plot_path",
        type=Path,
        default=None,
        help="Optional output path for the generated curve plot.",
    )
    return parser.parse_args()


def checkpoint_step(checkpoint_dir: Path) -> int:
    return int(checkpoint_dir.name.split("-")[-1])


def checkpoint_matches_interval(checkpoint_num: int, args) -> bool:
    return checkpoint_num % args.checkpoint_step_interval == 0


def selected_checkpoint_entries(checkpoints: list[Path], args) -> list[tuple[int, Path]]:
    return [
        (index, checkpoint_dir)
        for index, checkpoint_dir in enumerate(checkpoints)
        if checkpoint_matches_interval(checkpoint_step(checkpoint_dir), args)
        and (args.checkpoint_max_step is None or checkpoint_step(checkpoint_dir) <= args.checkpoint_max_step)
    ]


def parse_csv_list(raw_value):
    if raw_value is None:
        return []
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def is_gemini_judge_model(judge_model) -> bool:
    if judge_model is None:
        return False
    normalized = str(judge_model).strip().lower()
    return (
        normalized in {"gemini", "google", "google/gemini"}
        or normalized.startswith("gemini-")
        or normalized.startswith("models/gemini-")
        or normalized.startswith("gemini/")
        or normalized.startswith("gemini:")
        or normalized.startswith("google/gemini-")
        or normalized.startswith("google:gemini-")
    )


def gemini_api_key_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def resolve_judge_model_name(judge_model):
    if judge_model is None:
        return None
    model = str(judge_model).strip()
    if model.lower() in {"gemini", "google", "google/gemini"}:
        return os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_GEMINI_JUDGE_MODEL)
    return model


def safe_filename_token(value) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value.strip("-_.") or "unknown"


def selected_attack_benchmarks(args) -> list[str]:
    names = [name.strip().lower() for name in str(args.attack_benchmarks).split(",") if name.strip()]
    if not names:
        raise ValueError("--attack_benchmarks must name at least one benchmark")
    supported = ("advbench", "harmbench", "sorrybench")
    if "all" in names:
        if len(names) != 1:
            raise ValueError("Use either --attack_benchmarks all or an explicit comma-separated list")
        return list(supported)
    unknown = sorted(set(names) - set(supported))
    if unknown:
        raise ValueError(f"Unsupported attack benchmark(s): {', '.join(unknown)}")
    return list(dict.fromkeys(names))


def attack_benchmark_cache_suffix(args) -> str:
    benchmarks = selected_attack_benchmarks(args)
    if benchmarks == ["advbench"] and args.attack_max_samples is None:
        return ""
    parts = []
    for benchmark in benchmarks:
        if benchmark == "harmbench":
            parts.append(f"harmbench-{safe_filename_token(args.harmbench_split)}")
        elif benchmark == "sorrybench":
            parts.append(f"sorrybench-{safe_filename_token(args.sorrybench_mutation or 'base')}")
        else:
            parts.append(benchmark)
    if args.attack_max_samples is not None:
        parts.append(f"n{int(args.attack_max_samples)}")
    return f"attack_{'__'.join(parts)}"


def evaluation_cache_suffix(args) -> str:
    parts = []
    if args.metric_mode == "attack":
        benchmark_suffix = attack_benchmark_cache_suffix(args)
        if benchmark_suffix:
            parts.append(benchmark_suffix)
        # Preserve the historical score-only cache name, but isolate runs that
        # also request the harmful-information rubric.
        if bool(args.eval_harmful_info_score):
            parts.append("harmful_info")
        if args.eval_harmful_score or args.eval_harmful_info_score:
            judge_model = resolve_judge_model_name(args.harmful_judge_model)
            parts.append(f"harmful_judge_{safe_filename_token(judge_model)}")
    if args.metric_mode == "freeform":
        judge_model = resolve_judge_model_name(args.freeform_judge_model)
        if judge_model is not None:
            parts.append(f"freeform_judge_{safe_filename_token(judge_model)}")
    return "__".join(parts)


def with_optional_suffix(base_name: str, suffix: str) -> str:
    if suffix:
        return f"{base_name}_{suffix}"
    return base_name


def freeform_config(args) -> dict:
    return {
        "questions_path": args.freeform_questions_path,
        "question_group": args.freeform_question_group,
        "question_ids": parse_csv_list(args.freeform_question_ids),
        "samples_per_question": int(args.freeform_samples_per_question),
        "judge_metrics": parse_csv_list(args.freeform_judge_metrics),
        "judge_model": args.freeform_judge_model,
        "max_new_tokens": int(args.freeform_max_new_tokens),
        "temperature": float(args.freeform_temperature),
        "top_p": float(args.freeform_top_p),
        "seed": int(args.freeform_seed),
    }


def metric_fields(args) -> list[str]:
    if args.metric_mode == "attack":
        # Retain the legacy top-level fields: src/safeoff/evaluation/checkpoints.py maps them to the
        # primary benchmark. For a combined run, additionally retain each
        # benchmark's metrics so SORRY-Bench is not silently collapsed into the
        # HarmBench primary result.
        fields = ["attack_success_rate"]
        if bool(args.eval_harmful_score):
            fields.append("harmful_score")
        if bool(args.eval_harmful_info_score):
            fields.append("harmful_info_score")

        benchmarks = selected_attack_benchmarks(args)
        if len(benchmarks) > 1:
            for benchmark in benchmarks:
                fields.append(f"{benchmark}_attack_success_rate")
                if bool(args.eval_harmful_score):
                    fields.append(f"{benchmark}_harmful_score")
                if bool(args.eval_harmful_info_score):
                    fields.append(f"{benchmark}_harmful_info_score")
        return fields

    fields = []
    for metric in freeform_config(args)["judge_metrics"]:
        fields.append(f"freeform_{metric}_score")
    if not fields:
        raise ValueError("--freeform_judge_metrics must contain at least one metric")
    return fields


def resolve_paths(args):
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    model_name = args.base_model_path.split("/")[-1]
    model_root = output_dir / model_name / "model"
    if not model_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {model_root}")

    checkpoints = sorted(model_root.glob("checkpoint-*"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under: {model_root}")

    experiment_root = output_dir / model_name
    metrics_name = "checkpoint_freeform_metrics" if args.metric_mode == "freeform" else "checkpoint_attack_metrics"
    metrics_name = with_optional_suffix(metrics_name, evaluation_cache_suffix(args))
    summary_json = args.summary_json or (experiment_root / f"{metrics_name}.json")
    summary_csv = args.summary_csv or (experiment_root / f"{metrics_name}.csv")
    plot_path = args.plot_path or (experiment_root / f"{metrics_name}.png")
    return repo_root, model_name, checkpoints, summary_json.resolve(), summary_csv.resolve(), plot_path.resolve()


def cache_eval_path(repo_root: Path, checkpoint_dir: Path, args) -> Path:
    rel = checkpoint_dir.absolute().relative_to(repo_root)
    result_name = f"{with_optional_suffix('evaluation', evaluation_cache_suffix(args))}.json"
    return (repo_root / "cache" / "evaluations" / rel / result_name).resolve()


def load_metrics(result_path: Path, fields: list[str]) -> dict:
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    return {field: result.get(field) for field in fields}


def load_summary_metrics(summary_json: Path, fields: list[str]) -> dict[int, dict]:
    if not summary_json.exists():
        return {}

    with summary_json.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    metrics_by_checkpoint = {}
    for row in rows:
        checkpoint = row.get("checkpoint")
        if checkpoint is None:
            continue
        metrics_by_checkpoint[int(checkpoint)] = {field: row.get(field) for field in fields}
    return metrics_by_checkpoint


def metrics_complete(metrics: dict, fields: list[str]) -> bool:
    return metrics is not None and all(metrics.get(field) is not None for field in fields)


def freeform_cache_matches(result_path: Path, args) -> bool:
    if args.metric_mode != "freeform" or not result_path.exists():
        return True
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    return result.get("freeform_config") == freeform_config(args)


def attack_cache_matches(result_path: Path, args) -> bool:
    if (
        args.metric_mode != "attack"
        or not (args.eval_harmful_score or args.eval_harmful_info_score)
        or not result_path.exists()
    ):
        return True
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    cached_judge_model = result.get("harmful_judge_model") or "gpt-4.1"
    cached_benchmarks = result.get("attack_benchmarks")
    benchmark_matches = cached_benchmarks is None and selected_attack_benchmarks(args) == ["advbench"]
    if cached_benchmarks is not None:
        benchmark_matches = cached_benchmarks == selected_attack_benchmarks(args)
    return cached_judge_model == args.harmful_judge_model and benchmark_matches


def needs_evaluation(result_path: Path, args, fields: list[str]) -> bool:
    if not result_path.exists():
        return True
    if not freeform_cache_matches(result_path, args):
        return True
    if not attack_cache_matches(result_path, args):
        return True
    metrics = load_metrics(result_path, fields)
    return not metrics_complete(metrics, fields)


def can_launch_evaluation(checkpoint_num: int, args) -> bool:
    return args.min_checkpoint_step is None or checkpoint_num >= args.min_checkpoint_step


def run_evaluation(repo_root: Path, args, checkpoint_index: int):
    output_dir = args.output_dir
    if output_dir.is_absolute():
        try:
            output_dir_arg = str(output_dir.resolve().relative_to(repo_root))
        except ValueError:
            raise ValueError(
                f"output_dir must be inside repo_root for src/safeoff/evaluation/checkpoints.py: {output_dir}"
            )
    else:
        output_dir_arg = str(output_dir)

    command = [
        sys.executable,
        "-m", 'safeoff.evaluation.checkpoints',
        "--base_model_path",
        args.base_model_path,
        "--output_dir",
        output_dir_arg,
        "--checkpoint",
        str(checkpoint_index),
        "--batch_size",
        str(args.batch_size),
        "--use_lora",
        str(args.use_lora),
        "--judge_workers",
        str(args.judge_workers),
        "--utility_test",
        "0",
        "--logit_test",
        "0",
        "--logprob_test",
        "0",
    ]

    if args.metric_mode == "attack":
        command.extend([
            "--attack_test",
            "1",
            "--attack_benchmarks",
            args.attack_benchmarks,
            "--eval_harmful_score",
            str(args.eval_harmful_score),
            "--eval_harmful_info_score",
            str(args.eval_harmful_info_score),
            "--harmful_judge_model",
            args.harmful_judge_model,
        ])
        if args.attack_max_samples is not None:
            command.extend(["--attack_max_samples", str(args.attack_max_samples)])
        if "harmbench" in selected_attack_benchmarks(args):
            command.extend([
                "--harmbench_dataset", args.harmbench_dataset,
                "--harmbench_config", args.harmbench_config,
                "--harmbench_split", args.harmbench_split,
            ])
        if "sorrybench" in selected_attack_benchmarks(args):
            command.extend([
                "--sorrybench_repo", args.sorrybench_repo,
                "--sorrybench_revision", args.sorrybench_revision,
            ])
            if args.sorrybench_path:
                command.extend(["--sorrybench_path", args.sorrybench_path])
            if args.sorrybench_mutation:
                command.extend(["--sorrybench_mutation", args.sorrybench_mutation])
        if args.harmful_prompt_prefix:
            command.extend(["--harmful_prompt_prefix", args.harmful_prompt_prefix])
        if args.harmful_prompt_suffix:
            command.extend(["--harmful_prompt_suffix", args.harmful_prompt_suffix])
    else:
        command.extend([
            "--attack_test",
            "0",
            "--freeform_test",
            "1",
            "--freeform_questions_path",
            args.freeform_questions_path,
            "--freeform_question_group",
            args.freeform_question_group,
            "--freeform_samples_per_question",
            str(args.freeform_samples_per_question),
            "--freeform_judge_metrics",
            args.freeform_judge_metrics,
            "--freeform_max_new_tokens",
            str(args.freeform_max_new_tokens),
            "--freeform_temperature",
            str(args.freeform_temperature),
            "--freeform_top_p",
            str(args.freeform_top_p),
            "--freeform_seed",
            str(args.freeform_seed),
        ])
        if args.freeform_question_ids:
            command.extend(["--freeform_question_ids", args.freeform_question_ids])
        if args.freeform_judge_model:
            command.extend(["--freeform_judge_model", args.freeform_judge_model])
    print(f"[RUN] checkpoint index {checkpoint_index}: {' '.join(command)}")
    subprocess.run(command, cwd=repo_root, check=True)


def write_summaries(records, summary_json: Path, summary_csv: Path, fields: list[str]):
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["checkpoint"] + fields,
        )
        writer.writeheader()
        writer.writerows(records)


def make_attack_plot(records, plot_path: Path, model_name: str):
    checkpoints = [row["checkpoint"] for row in records]
    attack_success_rates = [row["attack_success_rate"] for row in records]
    harmful_scores = [row.get("harmful_score") for row in records]

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(11, 6))

    line1 = ax1.plot(
        checkpoints,
        attack_success_rates,
        color="tab:red",
        marker="o",
        linewidth=2,
        label="attack_success_rate",
    )
    ax1.set_xlabel("Checkpoint")
    ax1.set_ylabel("Attack Success Rate", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.35)

    lines = line1
    if any(score is not None for score in harmful_scores):
        ax2 = ax1.twinx()
        line2 = ax2.plot(
            checkpoints,
            harmful_scores,
            color="tab:blue",
            marker="s",
            linewidth=2,
            label="harmful_score",
        )
        ax2.set_ylabel("Harmful Score", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        lines += line2

    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title(f"Checkpoint Attack Metrics for {model_name}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def make_freeform_plot(records, plot_path: Path, model_name: str, fields: list[str]):
    checkpoints = [row["checkpoint"] for row in records]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = ["tab:green", "tab:purple", "tab:orange", "tab:blue"]
    markers = ["o", "s", "^", "x"]
    lines = []
    for idx, field in enumerate(fields):
        values = [row.get(field) for row in records]
        line = ax.plot(
            checkpoints,
            values,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=2,
            label=field,
        )
        lines += line

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Free-form Judge Score")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    ax.set_title(f"Checkpoint Free-form Metrics for {model_name}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def make_plot(records, plot_path: Path, model_name: str, args, fields: list[str]):
    if args.metric_mode == "attack":
        make_attack_plot(records, plot_path, model_name)
    else:
        make_freeform_plot(records, plot_path, model_name, fields)


def main():
    args = parse_args()
    if args.checkpoint_step_interval <= 0:
        raise ValueError("--checkpoint_step_interval must be a positive integer")
    repo_root, model_name, checkpoints, summary_json, summary_csv, plot_path = resolve_paths(args)
    fields = metric_fields(args)
    existing_summary_metrics = {} if args.metric_mode == "freeform" else load_summary_metrics(summary_json, fields)
    checkpoint_entries = selected_checkpoint_entries(checkpoints, args)
    if not checkpoint_entries:
        raise ValueError(
            f"No checkpoints match --checkpoint_step_interval={args.checkpoint_step_interval}"
        )

    planned = []
    for index, checkpoint_dir in checkpoint_entries:
        result_path = cache_eval_path(repo_root, checkpoint_dir, args)
        checkpoint_num = checkpoint_step(checkpoint_dir)
        if can_launch_evaluation(checkpoint_num, args) and (
            args.force or needs_evaluation(result_path, args, fields)
        ):
            planned.append((index, checkpoint_dir, result_path))

    if planned and args.metric_mode == "attack" and (
        args.eval_harmful_score or args.eval_harmful_info_score
    ):
        if is_gemini_judge_model(args.harmful_judge_model):
            if not gemini_api_key_available():
                raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini harmful judge models")
        elif not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is required when harmful or harmful-info judging is enabled"
            )
    if planned and args.metric_mode == "freeform":
        if is_gemini_judge_model(args.freeform_judge_model):
            if not gemini_api_key_available():
                raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini freeform judge models")
        elif not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is required when --metric_mode=freeform")

    records = []
    for index, checkpoint_dir in checkpoint_entries:
        result_path = cache_eval_path(repo_root, checkpoint_dir, args)
        checkpoint_num = checkpoint_step(checkpoint_dir)
        summary_metrics = existing_summary_metrics.get(checkpoint_num)

        if can_launch_evaluation(checkpoint_num, args) and (
            args.force or needs_evaluation(result_path, args, fields)
        ):
            run_evaluation(repo_root, args, index)
            metrics = load_metrics(result_path, fields)
        elif not needs_evaluation(result_path, args, fields):
            print(f"[SKIP] Using cached evaluation for {checkpoint_dir.name}")
            metrics = load_metrics(result_path, fields)
        else:
            if not can_launch_evaluation(checkpoint_num, args):
                print(f"[REUSE] Below min checkpoint step; using summary metrics for {checkpoint_dir.name}")
                if summary_metrics is None:
                    # Final-checkpoint-only runs may not have summary metrics
                    # for earlier checkpoints; omit those rows safely.
                    continue
            else:
                print(f"[REUSE] Using summary metrics for {checkpoint_dir.name}")
            metrics = summary_metrics

        row = {"checkpoint": checkpoint_num}
        row.update({field: metrics.get(field) for field in fields})
        records.append(row)

    write_summaries(records, summary_json, summary_csv, fields)
    make_plot(records, plot_path, model_name, args, fields)

    print(f"[DONE] JSON summary: {summary_json}")
    print(f"[DONE] CSV summary: {summary_csv}")
    print(f"[DONE] Plot: {plot_path}")


if __name__ == "__main__":
    main()
