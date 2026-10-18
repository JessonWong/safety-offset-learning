#!/usr/bin/env python3
"""Paired original/optimized prompt likelihoods of a fixed refusal continuation."""

from __future__ import annotations
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = project_root()
DEFAULT_TARGET = "I can't fulfill that request."
PREFIX = "I can't"
METRICS = ("first_token_logprob", "refusal_prefix_logprob", "sequence_logprob",
           "mean_token_logprob", "optimization_weighted_mean_logprob")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_pairs(original_path, optimized_path, generation_path, review_path):
    def read_csv(path):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("Empty source CSV")
        indexed = {int(row["index"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("Duplicate source index")
        if any(row.get("suffix", "").strip() for row in rows):
            raise ValueError("This experiment requires empty suffixes")
        return rows, indexed

    _, original = read_csv(original_path)
    optimized_rows, optimized = read_csv(optimized_path)
    if original.keys() != optimized.keys():
        raise ValueError("Original/optimized source index sets differ")
    generations = json.loads(generation_path.read_text())
    if generations["meta"]["source_sha256"] != sha256(optimized_path):
        raise ValueError("Refusal labels belong to a different optimized dataset")
    if generations["meta"]["system"] is not None:
        raise ValueError("Select samples using the existing no-system evaluation")
    if len(generations["data"]) != len(optimized_rows):
        raise ValueError("Incomplete optimized-prompt generations")
    review = json.loads(review_path.read_text())
    if review["responses_sha256"] != sha256(generation_path):
        raise ValueError("Review does not correspond to these generations")
    reviewed_refusal_positions = set(review["keep_row_positions"])
    pairs = []
    for position, (row, generated) in enumerate(zip(optimized_rows, generations["data"])):
        index = int(row["index"])
        if generated["row_position"] != position or int(generated["source_index"]) != index:
            raise ValueError("Generation order/source index mismatch")
        if generated["prompt"] != row["instruction"]:
            raise ValueError("Generation prompt mismatch")
        if original[index]["alpaca_reference"] != row["alpaca_reference"]:
            raise ValueError("Paired original/optimized reference responses differ")
        if type(generated["refusal"]) is not bool:
            raise ValueError("Refusal labels must be boolean")
        pairs.append({
            "source_index": index, "row_position": position,
            "original_prompt": original[index]["instruction"],
            "optimized_prompt": row["instruction"],
            "optimized_response": generated["response"],
            "optimized_keyword_refusal": generated["refusal"],
            "selected_for_refusal_training": position in reviewed_refusal_positions,
        })
    return pairs, generations["meta"]


def metrics_from_logprobs(logprobs, prefix_length):
    if not logprobs or not 0 < prefix_length <= len(logprobs):
        raise ValueError("Invalid target/prefix length")
    if any(not math.isfinite(value) or value > 1e-5 for value in logprobs):
        raise ValueError("Invalid token log-probability")
    total = sum(logprobs)
    return {
        "token_logprobs": logprobs,
        "first_token_logprob": logprobs[0],
        "refusal_prefix_logprob": sum(logprobs[:prefix_length]),
        "sequence_logprob": total,
        "mean_token_logprob": total / len(logprobs),
        # Negative of the core's default loss formula; not a historical-run reproduction.
        "optimization_weighted_mean_logprob": sum(value / (i + 1) for i, value in enumerate(logprobs)) / len(logprobs),
    }


def paired_stats(before, after):
    if len(before) != len(after) or not before:
        raise ValueError("Need nonempty matched score vectors")
    delta = [b - a for a, b in zip(before, after)]
    epsilon = 1e-6
    up = sum(value > epsilon for value in delta)
    down = sum(value < -epsilon for value in delta)
    return {
        "n": len(delta), "before_mean": statistics.fmean(before),
        "after_mean": statistics.fmean(after), "delta_mean": statistics.fmean(delta),
        "delta_median": statistics.median(delta), "increased_count": up,
        "decreased_count": down, "unchanged_count": len(delta) - up - down,
        "increased_fraction": up / len(delta), "comparison_tolerance": epsilon,
    }


def summarize(records, token_ids, token_texts):
    cohorts = {
        "optimized_keyword_nonrefusal": [r for r in records if not r["optimized_keyword_refusal"]],
        "optimized_keyword_refusal": [r for r in records if r["optimized_keyword_refusal"]],
        "not_selected_for_refusal_training": [r for r in records if not r["selected_for_refusal_training"]],
        "all": records,
    }
    output = {}
    for name, cohort in cohorts.items():
        if not cohort:
            continue
        output[name] = {
            "n": len(cohort),
            "metrics": {
                metric: paired_stats([r["original"][metric] for r in cohort], [r["optimized"][metric] for r in cohort])
                for metric in METRICS
            },
            "per_target_token": [
                {"position": i + 1, "token_id": token_id, "token_text": token_text,
                 **paired_stats([r["original"]["token_logprobs"][i] for r in cohort],
                                [r["optimized"]["token_logprobs"][i] for r in cohort])}
                for i, (token_id, token_text) in enumerate(zip(token_ids, token_texts))
            ],
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-csv", type=Path, default=ROOT / "data/alpaca_cleaned.csv")
    parser.add_argument("--optimized-csv", type=Path, default=ROOT / "data/8b_newtarget.csv")
    parser.add_argument("--generations", type=Path, default=ROOT / "results/safeoff_refusal_subset/no_system_ablation/no_system.json")
    parser.add_argument("--review", type=Path, default=ROOT / "results/safeoff_refusal_subset/refusal_review.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/maxrej_refusal_logprobs")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of prompt conditions per forward pass")
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("Batch size must be a positive even number, at least 2")
    pairs, generation_meta = load_pairs(args.original_csv, args.optimized_csv, args.generations, args.review)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Run the forward passes in a GPU allocation")
    model_name = generation_meta["model"]
    revision = generation_meta["model_commit"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    target_ids = tokenizer.encode(args.target, add_special_tokens=False)
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    if target_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Target must begin with the independently tokenized refusal prefix")
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != generation_meta["chat_template_sha256"]:
        raise ValueError("Chat template differs from the cached refusal evaluation")
    token_texts = [tokenizer.decode([token]) for token in target_ids]
    meta = {
        "model": model_name, "revision": revision, "dtype": "float16", "system": None,
        "original_csv": str(args.original_csv.resolve()), "original_sha256": sha256(args.original_csv),
        "optimized_csv": str(args.optimized_csv.resolve()), "optimized_sha256": sha256(args.optimized_csv),
        "cohort_generations_sha256": sha256(args.generations), "review_sha256": sha256(args.review),
        "num_pairs": len(pairs), "batch_size": args.batch_size,
        "target": args.target, "target_token_ids": target_ids, "target_token_texts": token_texts,
        "refusal_prefix": PREFIX, "refusal_prefix_token_count": len(prefix_ids),
        "log_base": "natural", "target_includes_eos": False,
        "conditioning": "Teacher-forced: each target token conditions on all preceding target tokens",
        "first_token_warning": "I alone is not a refusal-specific token; report prefix and whole-target likelihoods as well",
        "primary_cohort": "Same optimized-prompt keyword-nonrefusal samples in both paired conditions",
        "input_truncation": False, "maximum_allowed_prompt_tokens": 1000,
        "objective_note": "Fixed target and 1/t weights match current generator defaults; historical CSV CLI unavailable",
    }
    raw_path = args.output_dir / "paired_scores.json"
    records = []
    if raw_path.exists():
        cached = json.loads(raw_path.read_text())
        if cached["meta"] != meta:
            raise ValueError("Cached scoring provenance differs; use a new output directory")
        records = cached["data"]
        if len(records) > len(pairs):
            raise ValueError("Too many cached pairs")
        for expected, record in zip(pairs, records):
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError("Cached pair identity differs")
    model = None
    if len(records) < len(pairs):
        print(f"Loading {model_name}@{revision}, FP16 on {torch.cuda.get_device_name()}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, local_files_only=True,
            torch_dtype=torch.float16, device_map={"": 0},
        )
        model.eval()
    print(f"Target tokens: {list(zip(target_ids, token_texts))}", flush=True)
    target_length = len(target_ids)
    pair_batch = args.batch_size // 2
    validated_this_run = False
    for start in range(len(records), len(pairs), pair_batch):
        batch_pairs = pairs[start:start + pair_batch]
        encoded, prompt_lengths = [], []
        for pair in batch_pairs:
            for key in ("original_prompt", "optimized_prompt"):
                prompt = pair[key]
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                )
                expected = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prompt.strip() + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                if formatted != expected:
                    raise ValueError("Unexpected non-user-only template content")
                prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
                if not prompt_ids or len(prompt_ids) > 1000:
                    raise ValueError("Prompt too long; do not silently truncate original Alpaca context")
                prompt_lengths.append(len(prompt_ids))
                # Match the optimization core's separately tokenized Seq concatenation.
                encoded.append(prompt_ids + target_ids)
        width = max(map(len, encoded))
        ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(row)) + row for row in encoded], device="cuda")
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in encoded], device="cuda")
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
        with torch.inference_mode():
            # The last T+1 hidden states predict target[0..T-1] and the unused next token.
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           use_cache=False, logits_to_keep=target_length + 1)
            logits = output.logits[:, -target_length - 1:-1, :].float()
            labels = torch.tensor(target_ids, device=logits.device).expand(len(encoded), -1)
            token_lp = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).cpu().tolist()
            first_argmax = logits[:, 0].argmax(-1).cpu().tolist()
            first_target = logits[:, 0, target_ids[0]].unsqueeze(-1)
            first_ranks = ((logits[:, 0] > first_target).sum(-1) + 1).cpu().tolist()
            if not validated_this_run:
                # Independent full-logit/unpadded indexing for the first original/optimized pair.
                checks = []
                for j in (0, 1):
                    single_ids = torch.tensor([encoded[j]], device="cuda")
                    full_output = model(input_ids=single_ids, attention_mask=torch.ones_like(single_ids),
                                        use_cache=False)
                    beginning = prompt_lengths[j] - 1
                    single_logits = full_output.logits[:, beginning:beginning + target_length].float()
                    single_labels = torch.tensor([target_ids], device="cuda")
                    single_lp = single_logits.log_softmax(-1).gather(-1, single_labels.unsqueeze(-1)).squeeze().cpu().tolist()
                    difference = max(abs(a - b) for a, b in zip(single_lp, token_lp[j]))
                    checks.append({"condition": "original" if j == 0 else "optimized",
                                   "source_index": batch_pairs[0]["source_index"],
                                   "singleton_full_logits_logprobs": single_lp,
                                   "batched_trimmed_logits_logprobs": token_lp[j],
                                   "max_absolute_token_logprob_difference": difference})
                    if difference > .25:
                        raise ValueError(f"Full-logit/singleton scoring differs excessively: {difference}")
                    del full_output, single_logits, single_ids
                atomic_json(args.output_dir / "numerical_validation.json", {
                    "checks": checks, "maximum_allowed_absolute_difference": .25,
                    "note": "FP16 singleton/full-logits versus left-padded batched/trimmed-logits sanity check",
                    "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                })
                print(f"Numerical validation: {checks}", flush=True)
                validated_this_run = True
        for i, pair in enumerate(batch_pairs):
            scores = []
            for j in (2 * i, 2 * i + 1):
                scores.append({**metrics_from_logprobs(token_lp[j], len(prefix_ids)),
                               "prompt_tokens": prompt_lengths[j], "first_token_rank": first_ranks[j],
                               "first_argmax_token_id": first_argmax[j],
                               "first_argmax_text": tokenizer.decode([first_argmax[j]])})
            records.append({**pair, "original": scores[0], "optimized": scores[1],
                            "delta": {metric: scores[1][metric] - scores[0][metric] for metric in METRICS}})
        if len(records) % 32 == 0 or len(records) == len(pairs):
            atomic_json(raw_path, {"meta": meta, "data": records})
            print(f"Saved {len(records)}/{len(pairs)} paired scores", flush=True)
        del output, logits, ids, mask, positions
    summary = {"meta": meta, "cohorts": summarize(records, target_ids, token_texts)}
    atomic_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["cohort", "metric", "n", "before_mean", "after_mean", "delta_mean", "delta_median", "increased_count", "decreased_count", "unchanged_count", "increased_fraction", "comparison_tolerance"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cohort, result in summary["cohorts"].items():
            for metric, stats in result["metrics"].items():
                writer.writerow({"cohort": cohort, "metric": metric, **stats})
    print(json.dumps(summary["cohorts"]["optimized_keyword_nonrefusal"], indent=2), flush=True)


if __name__ == "__main__":
    main()
