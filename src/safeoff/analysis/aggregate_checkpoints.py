"""Aggregate pairwise LoRA delta-weight comparisons into five-model matrices."""
from safeoff.paths import project_root

import csv
import json
import shutil
from pathlib import Path


ROOT = project_root()
OUT = ROOT / "results/safeoff_alpaca100_shadow_llmlat_delta_similarity"

MODELS = {
    "safeoff_100": ROOT / "finetune_model/delta_similarity_100/safeoff_0_99/Meta-Llama-3.1-8B-Instruct/model/checkpoint-250",
    "alpaca_cleaned_same_100": ROOT / "finetune_model/delta_similarity_100/alpaca_aligned_0_99/Meta-Llama-3.1-8B-Instruct/model/checkpoint-250",
    "alpaca_cleaned_different_100": ROOT / "finetune_model/delta_similarity_100/alpaca_disjoint_100_199/Meta-Llama-3.1-8B-Instruct/model/checkpoint-250",
    "shadow_alignment_100": ROOT / "finetune_model/math_curve/llama3_shadow_alignment100/Meta-Llama-3.1-8B-Instruct/model/checkpoint-250",
    "llm_lat_harmful_100": ROOT / "finetune_model/math_curve/llama3_llm_lat_harmful100/Meta-Llama-3.1-8B-Instruct/model/checkpoint-250",
}

PAIR_DIRS = {
    ("safeoff_100", "alpaca_cleaned_same_100"): ROOT / "results/safeoff_alpaca_100_delta_similarity/safeoff_aligned",
    ("safeoff_100", "alpaca_cleaned_different_100"): ROOT / "results/safeoff_alpaca_100_delta_similarity/safeoff_disjoint",
    ("alpaca_cleaned_same_100", "alpaca_cleaned_different_100"): ROOT / "results/safeoff_alpaca_100_delta_similarity/aligned_disjoint",
    ("safeoff_100", "shadow_alignment_100"): OUT / "safeoff_shadow",
    ("safeoff_100", "llm_lat_harmful_100"): OUT / "safeoff_llm_lat",
    ("alpaca_cleaned_same_100", "shadow_alignment_100"): OUT / "aligned_shadow",
    ("alpaca_cleaned_same_100", "llm_lat_harmful_100"): OUT / "aligned_llm_lat",
    ("alpaca_cleaned_different_100", "shadow_alignment_100"): OUT / "disjoint_shadow",
    ("alpaca_cleaned_different_100", "llm_lat_harmful_100"): OUT / "disjoint_llm_lat",
    ("shadow_alignment_100", "llm_lat_harmful_100"): ROOT / "results/safeoff_llm_lat_shadow_delta_similarity/llm_lat_shadow",
}

COPIES = {
    "safeoff_aligned": PAIR_DIRS[("safeoff_100", "alpaca_cleaned_same_100")],
    "safeoff_disjoint": PAIR_DIRS[("safeoff_100", "alpaca_cleaned_different_100")],
    "aligned_disjoint": PAIR_DIRS[("alpaca_cleaned_same_100", "alpaca_cleaned_different_100")],
    "llm_lat_shadow": PAIR_DIRS[("shadow_alignment_100", "llm_lat_harmful_100")],
}


def write_matrix(path, names, values):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *names])
        for row_name in names:
            writer.writerow([row_name, *(values[row_name][column] for column in names)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for destination, source in COPIES.items():
        target = OUT / destination
        if source.resolve() != target.resolve():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)

    names = list(MODELS)
    sections = {"overall": {}, "q_proj": {}, "v_proj": {}}
    matrices = {
        section: {row: {column: (1.0 if row == column else None) for column in names} for row in names}
        for section in sections
    }
    pairwise = []
    pair_data = {}
    for (model_a, model_b), pair_dir in PAIR_DIRS.items():
        # Read copied paths so the aggregate result is self-contained.
        local_dir = OUT / next(
            (name for name, source in COPIES.items() if source.resolve() == pair_dir.resolve()),
            pair_dir.name,
        )
        data = json.loads((local_dir / "summary.json").read_text())
        values = {
            "overall": data["overall"]["cosine"],
            "q_proj": data["by_projection"]["q_proj"]["cosine"],
            "v_proj": data["by_projection"]["v_proj"]["cosine"],
        }
        for section, value in values.items():
            matrices[section][model_a][model_b] = value
            matrices[section][model_b][model_a] = value
        row = {
            "model_a": model_a,
            "model_b": model_b,
            "overall_cosine": values["overall"],
            "q_proj_cosine": values["q_proj"],
            "v_proj_cosine": values["v_proj"],
            "distance": data["overall"]["distance"],
            "pair_result": str(local_dir.relative_to(ROOT)),
        }
        pairwise.append(row)
        pair_data[f"{model_a}__{model_b}"] = row

    write_matrix(OUT / "cosine_matrix.csv", names, matrices["overall"])
    write_matrix(OUT / "q_proj_cosine_matrix.csv", names, matrices["q_proj"])
    write_matrix(OUT / "v_proj_cosine_matrix.csv", names, matrices["v_proj"])
    with (OUT / "pairwise_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise[0]))
        writer.writeheader()
        writer.writerows(pairwise)

    result = {
        "definition": "delta_W = (alpha/r) B @ A relative to the shared base model; global flattened cosine",
        "precision": "float64",
        "base_model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "module_count": 64,
        "models": {name: str(path.resolve()) for name, path in MODELS.items()},
        "matrices": matrices,
        "pairs": pair_data,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
