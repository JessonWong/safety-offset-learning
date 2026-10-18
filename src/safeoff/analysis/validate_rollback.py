"""Check real-checkpoint hook inference against the repository PEFT implementation."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from safeoff.analysis.lora_weights import load
from safeoff.analysis.rollback import DEFAULT_PATHS, DeltaHookManager, device_pairs, prepare_evaluations, evaluate_teacher
from safeoff.analysis.functional import load_base_model, load_records, tokenize_records, write_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.probe_file)
    chosen = [records[0], next(r for r in records[1:] if r["prompt_token_length"] != records[0]["prompt_token_length"])]
    model, tokenizer = load_base_model("cuda", "bfloat16")
    _, factors = load(DEFAULT_PATHS["safeoff"] / "checkpoint-250")
    batch = tokenize_records(tokenizer, chosen, 512, "cuda")
    hooks = DeltaHookManager(model, factors)
    hooks.set_state(device_pairs(factors, "cuda"))
    with torch.inference_mode():
        actual = model(**batch, use_cache=False, logits_to_keep=1).logits.float().cpu()
        hooks.set_state(device_pairs(factors, "cuda"), device_pairs(factors, "cuda"), 0.0)
        zero = model(**batch, use_cache=False, logits_to_keep=1).logits.float().cpu()
    examples, skipped = prepare_evaluations(tokenizer, chosen, 1024)
    grouped = evaluate_teacher(model, examples, tokenizer, "cuda", 4)
    single = evaluate_teacher(model, examples, tokenizer, "cuda", 1)
    hooks.close()
    wrapped = PeftModel.from_pretrained(model, str(DEFAULT_PATHS["safeoff"] / "checkpoint-250")).eval()
    with torch.inference_mode():
        expected = wrapped(**batch, use_cache=False, logits_to_keep=1).logits.float().cpu()
    nll_diff = max(abs(a["nll_mean"] - b["nll_mean"]) for a, b in zip(grouped, single))
    max_diff = float((actual - expected).abs().max())
    result = dict(probe_sha256=digest(args.probe_file), ids=[r["id"] for r in chosen],
                  hook_vs_peft_max_logit_abs_diff=max_diff,
                  hook_vs_peft_mean_logit_abs_diff=float((actual - expected).abs().mean()),
                  lambda_zero_bitwise_equal=torch.equal(actual, zero),
                  batched_vs_single_max_mean_nll_diff=nll_diff, skipped=skipped,
                  thresholds=dict(peft_max_logit_abs_diff=1e-5, batching_nll_abs_diff=0.02))
    result["passed"] = max_diff <= 1e-5 and torch.equal(actual, zero) and nll_diff <= .02 and not skipped
    write_json(args.out, result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("Real-checkpoint inference validation failed")


if __name__ == "__main__":
    main()
