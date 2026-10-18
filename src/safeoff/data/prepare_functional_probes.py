"""Freeze fresh held-out paired safety and benign probes before interventions."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from safeoff.analysis.functional import ROOT, BASE_MODEL, SYSTEM_PROMPT, digest, load_tokenizer, messages_for, prompt_text, write_json


def normalize(text):
    return " ".join(str(text).lower().split())


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite an already frozen probe file")
    tokenizer = load_tokenizer()
    ds = load_dataset("LLM-LAT/harmful-dataset", split="train")
    old = json.loads((ROOT / "results/paired_safety_gradient_subspace/prepared_manifest.json").read_text())
    train_indices = list(range(len(ds)))
    random.Random(30).shuffle(train_indices)
    train_indices = train_indices[:100]
    assert train_indices == old["llm_lat_source_indices"]
    excluded_indices = set(train_indices + old["safety_meta_source_indices"] + old["safety_validation_source_indices"])
    excluded_prompts = {normalize(ds[i]["prompt"]) for i in excluded_indices}
    source_files = [ROOT / "data/delta_similarity_100" / name for name in
                    ["safeoff_0_99.csv", "alpaca_aligned_0_99.csv", "alpaca_disjoint_100_199.csv"]]
    source_files.append(ROOT / "data/shadow_alignment_train100.csv")
    for path in source_files:
        for row in rows(path):
            excluded_prompts.add(normalize(row.get("instruction", row.get("prompt", "")) + row.get("suffix", "")))
    rejected = Counter()
    selected = []
    seen = set(excluded_prompts)

    def fits(record, responses):
        prefix = tokenizer(prompt_text(tokenizer, record), add_special_tokens=False)["input_ids"]
        if len(prefix) > 512:
            rejected["prompt_over_512"] += 1
            return False
        lengths = []
        for response in responses:
            text = tokenizer.apply_chat_template(messages_for(record) + [{"role": "assistant", "content": response}],
                                                 tokenize=False, add_generation_prompt=False)
            full = tokenizer(text, add_special_tokens=False)["input_ids"]
            if full[:len(prefix)] != prefix or len(full) <= len(prefix) or len(full) > 1024:
                rejected["response_length_or_template"] += 1
                return False
            lengths.append(len(full) - len(prefix))
        record["prompt_token_length"] = len(prefix)
        record["response_token_lengths"] = lengths
        return True

    indices = [i for i in range(1000, len(ds)) if i not in excluded_indices]
    random.Random(args.seed).shuffle(indices)
    refusal_prefixes = ("i cannot", "i can't", "i’m sorry", "i'm sorry", "i am sorry", "sorry", "i apologize", "i will not", "i won't", "as an ai")
    for i in indices:
        source = ds[i]
        p, safe, harmful = (str(source[k]).strip() for k in ["prompt", "chosen", "rejected"])
        if not all([p, safe, harmful]) or normalize(p) in seen or normalize(safe) == normalize(harmful):
            rejected["empty_duplicate_or_overlap"] += 1
            continue
        if not normalize(safe).startswith(refusal_prefixes):
            rejected["chosen_not_explicit_refusal"] += 1
            continue
        if normalize(harmful).startswith(refusal_prefixes):
            rejected["rejected_starts_refusal"] += 1
            continue
        record = dict(id=f"lat-heldout-{i}", group="harmful", prompt=p, safe_response=safe,
                      harmful_response=harmful, source_index=i, source="LLM-LAT/harmful-dataset")
        if fits(record, [safe, harmful]):
            selected.append(record)
            seen.add(normalize(p))
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise ValueError("Insufficient eligible held-out safety probes")
    benign_path = ROOT / "data/alpaca_cleaned.csv"
    screening_path = ROOT / "data/harmfulness_eval_wildguard/harmfulness_results.json"
    screened = {int(r["record_index"]): r for r in json.loads(screening_path.read_text())
                if r["source_file"] == "data/alpaca_cleaned.csv" and r["variant"] == "main"
                and r["prompt_harmful"] is False and r["response_harmful"] is False}
    benign = [r for r in rows(benign_path) if int(r["index"]) >= 200
              and int(r["index"]) in screened
              and screened[int(r["index"])]["prompt"] == r["instruction"] + r.get("suffix", "")
              and screened[int(r["index"])]["response"] == r["alpaca_reference"]]
    random.Random(args.seed + 1).shuffle(benign)
    for source in benign:
        p = source["instruction"] + source.get("suffix", "")
        target = source["alpaca_reference"]
        if not p.strip() or not target.strip() or normalize(p) in seen:
            rejected["benign_empty_or_overlap"] += 1
            continue
        record = dict(id=f"alpaca-heldout-{source['index']}", group="benign", prompt=p,
                      target=target, source_index=int(source["index"]), source=str(benign_path))
        if fits(record, [target]):
            selected.append(record)
            seen.add(normalize(p))
        if len(selected) == 2 * args.count:
            break
    if len(selected) != 2 * args.count:
        raise ValueError("Insufficient eligible held-out benign probes")
    metadata = dict(
        base_model=BASE_MODEL, system_prompt=SYSTEM_PROMPT, seed=args.seed,
        count_per_group=args.count, dataset_fingerprint=ds._fingerprint,
        harmful_training_indices=train_indices, excluded_prior_probe_indices=sorted(excluded_indices - set(train_indices)),
        normalized_prompt_exclusions=len(excluded_prompts), normalized_selected_unique=len({normalize(r['prompt']) for r in selected}),
        source_sha256={str(p):digest(p) for p in source_files + [benign_path, screening_path]}, rejections=dict(rejected),
        benign_filter="Existing WildGuard prompt_harmful=False and response_harmful=False; exact text identity checked",
        selection="Deterministic shuffle; first eligible prompts; no model-outcome or intervention-based selection",
        truncation="None: prompt<=512 and full prompt+each response<=1024; longer candidates excluded",
        limitations=["One frozen 64-prompt sample per group, one training seed per model",
                     "Safety paired responses come from LAT dataset but are disjoint from its training and previous gradient probes",
                     "Benign utility is held-out Alpaca teacher-forced CE, not a general capability benchmark",
                     "Refusal-sensitive rewrite subset omitted because independent benign-content annotations were unavailable"],
    )
    write_json(args.out, dict(metadata=metadata, records=selected))
    print(json.dumps(dict(path=str(args.out), sha256=digest(args.out), counts=dict(Counter(r['group'] for r in selected)), rejections=dict(rejected))), flush=True)


if __name__ == "__main__":
    main()
