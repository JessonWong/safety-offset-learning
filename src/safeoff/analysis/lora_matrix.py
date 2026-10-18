"""Compute pairwise effective LoRA delta-weight cosine matrices."""

import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path

import torch

from safeoff.analysis.lora_weights import inner, load, metrics


def parse_model(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be NAME=CHECKPOINT_PATH")
    name, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError("model names may contain letters, digits, _, ., and -")
    return name, Path(raw_path)


def write_matrix(path, names, matrix):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *names])
        for row_name in names:
            writer.writerow([row_name, *(matrix[row_name][column] for column in names)])


def compare_pair(name_a, name_b, checkpoint_a, checkpoint_b, a, b, output_dir):
    rows = []
    totals = [0.0, 0.0, 0.0]
    groups = {}
    for module in sorted(a):
        values = [inner(a[module], b[module]), inner(a[module], a[module]), inner(b[module], b[module])]
        totals = [left + right for left, right in zip(totals, values)]
        group = module.split(".")[-1]
        groups[group] = [
            left + right for left, right in zip(groups.get(group, [0.0, 0.0, 0.0]), values)
        ]
        rows.append({"module": module, **metrics(*values)})

    result = {
        "model_a": name_a,
        "model_b": name_b,
        "checkpoint_a": str(checkpoint_a.resolve()),
        "checkpoint_b": str(checkpoint_b.resolve()),
        "definition": "delta_W = (alpha/r) B @ A relative to base; global flattened cosine",
        "precision": "float64",
        "module_count": len(rows),
        "overall": metrics(*totals),
        "by_projection": {group: metrics(*values) for group, values in groups.items()},
    }
    pair_dir = output_dir / (name_a + "__" + name_b)
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (pair_dir / "per_module.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.model) < 2:
        parser.error("provide at least two --model NAME=CHECKPOINT_PATH arguments")
    names = [name for name, _ in args.model]
    if len(names) != len(set(names)):
        parser.error("model names must be unique")

    torch.set_num_threads(2)
    configs = {}
    updates = {}
    paths = {}
    for name, path in args.model:
        configs[name], updates[name] = load(path)
        paths[name] = path
    first_config = configs[names[0]]
    first_modules = updates[names[0]].keys()
    for name in names[1:]:
        config = configs[name]
        if config["base_model_name_or_path"] != first_config["base_model_name_or_path"]:
            raise ValueError("base models differ")
        if config["fan_in_fan_out"] != first_config["fan_in_fan_out"]:
            raise ValueError("fan_in_fan_out differs")
        if updates[name].keys() != first_modules:
            raise ValueError("target modules differ")

    # Independent dense check of the low-rank inner-product identity.
    rng = torch.Generator().manual_seed(0)
    x = (
        torch.randn(3, 11, generator=rng, dtype=torch.float64),
        torch.randn(7, 3, generator=rng, dtype=torch.float64),
    )
    y = (
        torch.randn(4, 11, generator=rng, dtype=torch.float64),
        torch.randn(7, 4, generator=rng, dtype=torch.float64),
    )
    if not math.isclose(inner(x, y), ((x[1] @ x[0]) * (y[1] @ y[0])).sum().item(), rel_tol=1e-12):
        raise AssertionError("low-rank inner-product identity check failed")

    args.out.mkdir(parents=True, exist_ok=True)
    sections = ["overall", "q_proj", "v_proj"]
    matrices = {
        section: {
            row: {column: (1.0 if row == column else None) for column in names}
            for row in names
        }
        for section in sections
    }
    pairwise = []
    pair_results = {}
    for name_a, name_b in combinations(names, 2):
        result = compare_pair(
            name_a,
            name_b,
            paths[name_a],
            paths[name_b],
            updates[name_a],
            updates[name_b],
            args.out,
        )
        values = {
            "overall": result["overall"]["cosine"],
            "q_proj": result["by_projection"]["q_proj"]["cosine"],
            "v_proj": result["by_projection"]["v_proj"]["cosine"],
        }
        for section, value in values.items():
            matrices[section][name_a][name_b] = value
            matrices[section][name_b][name_a] = value
        row = {
            "model_a": name_a,
            "model_b": name_b,
            "overall_cosine": values["overall"],
            "q_proj_cosine": values["q_proj"],
            "v_proj_cosine": values["v_proj"],
            "norm_a": result["overall"]["norm_a"],
            "norm_b": result["overall"]["norm_b"],
            "distance": result["overall"]["distance"],
        }
        pairwise.append(row)
        pair_results[name_a + "__" + name_b] = row

    write_matrix(args.out / "cosine_matrix.csv", names, matrices["overall"])
    write_matrix(args.out / "q_proj_cosine_matrix.csv", names, matrices["q_proj"])
    write_matrix(args.out / "v_proj_cosine_matrix.csv", names, matrices["v_proj"])
    with (args.out / "pairwise_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise[0]))
        writer.writeheader()
        writer.writerows(pairwise)
    summary = {
        "definition": "delta_W = (alpha/r) B @ A relative to the shared base model; global flattened cosine",
        "precision": "float64",
        "base_model": first_config["base_model_name_or_path"],
        "module_count": len(first_modules),
        "models": {name: str(paths[name].resolve()) for name in names},
        "matrices": matrices,
        "pairs": pair_results,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
