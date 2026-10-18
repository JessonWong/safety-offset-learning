# Data inputs

Research CSV/JSON files are not bundled. This page lists the local files that
commands and presets read, how to rebuild each one, and which files this
repository cannot rebuild. Run the commands from the workspace root: the
directory in `SAFEOFF_ROOT`, or the source checkout when it is unset. Presets
resolve `data/...` against that directory.

Dataset and model terms are independent of the source-code license. Check them
before redistributing a derived file. In particular, `knkarthick/samsum` is
CC BY-NC-ND 4.0; share the conversion below rather than the converted file.

## CSV format

The replacement-prompt training path accepts UTF-8 CSV with:

| Column | Meaning |
| --- | --- |
| `index` | Stable original row identifier |
| `instruction` | User prompt |
| `alpaca_reference` | Original supervised answer |
| `suffix` | Appended text, usually empty for full replacements |

`examples/toy_train.csv` contains synthetic examples. The small refusal keyword
configuration is packaged in `safeoff.resources`; it is a heuristic configuration,
not a research dataset. Use `--advprompt_use_suffix 0` for complete replacement
instructions, and `--advprompt_clean_training_num 0` for no added clean mixture.

Dataset presets are in `safeoff.data.config`; formatting and loading behavior
remain compatible with the original research code. Two tests that check
historical datasets skip when those files are absent; synthetic tests still
verify CSV parsing and selection behavior.

## Sources

| Local file | Contents | Rebuild |
| --- | --- | --- |
| `data/8b_newtarget.csv` | SafeOff prompts for Alpaca rows 0–999, Llama-3.1-8B-Instruct | [SafeOff generation](#safeoff-generation) |
| `data/qwen25_7b_safeoff.csv` | SafeOff prompts for Alpaca rows 0–999, Qwen2.5-7B-Instruct | [SafeOff generation](#safeoff-generation) |
| `data/alpaca_cleaned.csv` | Alpaca rows 0–999 with their original prompts | [SafeOff generation](#safeoff-generation); byte-identical |
| WildGuard-filtered SafeOff CSVs | SafeOff rows without harmful prompts or responses | [WildGuard filtering](#wildguard-filtering) |
| `data/shadow_alignment_train100.csv` | 100 Shadow Alignment training rows | [Shadow Alignment](#shadow-alignment); byte-identical |
| `data/delta_similarity_100/` | 100-row SafeOff and Alpaca comparison arms | [Comparison arms](#comparison-arms); byte-identical given the same inputs |
| `data/gsm8k_as_alpaca.csv`, `data/samsum_as_alpaca.csv` | GSM8K and SAMSum training splits in Alpaca columns | [GSM8K and SAMSum](#gsm8k-and-samsum); byte-identical |
| `data/benign_task_1000/` | First 1,000 rows of each file above | [GSM8K and SAMSum](#gsm8k-and-samsum); byte-identical |
| `data/advbench.csv` | 520 AdvBench prompts for `safeoff eval system-ablation` | [AdvBench prompts](#advbench-prompts); byte-identical |
| `data/alpaca_cleaned_{mode}_1000.json` | Alpaca rows with rewritten responses, for the `alpaca_cleaned_*` presets | `safeoff data rewrite-alpaca-cleaned --rewrite-mode MODE` |
| Other local files | Historical inputs | [Files this repository does not rebuild](#files-this-repository-does-not-rebuild) |

*Byte-identical* means that the recipe reproduced the SHA-256 of the historical
research file. Benchmarks that commands load by Hugging Face name download on
first use; gated ones need accepted terms and `HF_TOKEN`.

## SafeOff generation

`safeoff generate` searches for a replacement prompt for each Alpaca row, scored
against a refusal target, and pairs it with the row's original answer. It needs
two GPUs: the target model on `--target-device` and the prompter on
`--prompter-device`. The prompter defaults to the target checkpoint.

These are the settings of a fresh regeneration on 2026-10-03 at commit
`a2cf8ccc0bb2a6784b3cf9e66ae7e8fe02e03bcb`:

```bash
safeoff generate \
  --dataset-name yahma/alpaca-cleaned --dataset-split train \
  --reference-field output --start-index 0 --num-samples 1000 \
  --target-checkpoint NousResearch/Meta-Llama-3.1-8B-Instruct \
  --target-device cuda:0 --prompter-device cuda:1 --dtype float16 \
  --strategy standalone_v3 --prompt-tokens 48 --top-k 16 --num-beams 4 \
  --num-chunks 1 --lambda-val 80 \
  --refusal-target "I can't fulfill that request." \
  --no-standalone-v3-multi-target-selection --reweight-loss \
  --shard-size 250 --attack-batch-size 2 --seed 2025 \
  --output-dir outputs/safeoff_llama31_8b --run-dir outputs/safeoff_llama31_8b
```

The run used `yahma/alpaca-cleaned` revision
`12567cabf869d7c92e573c7c783905fc160e9639` and model revision
`d10aef7999a2b5ba950ab3974312feeedbfe0b77`. `safeoff generate` loads the
current Hub revision of the dataset. On an offline node, the run read the same
rows from a local CSV export instead (`--dataset-name csv --dataset-data-file
FILE --reference-field alpaca_reference`); row order and indices are unchanged.

The run directory contains:

| File | Use |
| --- | --- |
| `reverse_eval_prompt_replacement_train_compat.csv` | SafeOff training data. Copy it to `data/8b_newtarget.csv` where presets or scripts expect that name. |
| `reverse_eval_prompt_replacement_train_compat_valid.csv` | The same rows with the original prompts, Alpaca `input` folded into `instruction`. Copy it to `data/alpaca_cleaned.csv`. |
| `reverse_eval_prompt_replacement_summary_merged.json` | Search statistics |
| `reverse_eval_prompt_replacement_results_merged.csv` | Per-row search details |

The `_valid` file does not depend on the search. Its SHA-256,
`82ff1f26e6fdddad57005088af9d112bf273565d4fe7f97c54d0e817a3b16309`, matches the
historical `data/alpaca_cleaned.csv`.

The search itself is not reproducible byte for byte. The regenerated prompts
share 8 of 1,000 rows with the historical `data/8b_newtarget.csv`, whose exact
settings were not recorded. The regeneration summary reported that 330 of 1,000
replacement prompts elicited a refusal during search. Fine-tuning on the
regenerated file reproduces the main safety effect; see the
[SafeOff regeneration check](experiments/safeoff_1000.md).

`experiments/slurm/math_curves.sh` trains `Qwen/Qwen2.5-7B-Instruct` on
`data/qwen25_7b_safeoff.csv`. To build that file, use the command above with
`--target-checkpoint Qwen/Qwen2.5-7B-Instruct` and a new output directory. The
settings of the historical Qwen file, including its refusal target, were not
recorded, and this recipe has not been rerun for Qwen.

`experiments/pipelines/prompt_replacement.sh` chains generation, WildGuard
filtering, fine-tuning, and evaluation. Its generation defaults match the
command above except for the target checkpoint
(`meta-llama/Llama-3.1-8B-Instruct`). Its training settings differ from the
regeneration check.

## WildGuard filtering

These commands use the settings in `experiments/pipelines/prompt_replacement.sh`.
`allenai/wildguard` is gated.

```bash
RUN=outputs/safeoff_llama31_8b
safeoff eval data-harmfulness \
  --input "$RUN/reverse_eval_prompt_replacement_train_compat.csv" \
  --output-dir "$RUN/harmfulness_eval_wildguard" \
  --backend wildguard --model-name allenai/wildguard \
  --batch-size 8 --device-map auto --torch-dtype bfloat16 --no-append-suffix
safeoff data filter-harmfulness-results \
  --input-csv "$RUN/reverse_eval_prompt_replacement_train_compat.csv" \
  --harmfulness-results "$RUN/harmfulness_eval_wildguard/harmfulness_results.csv" \
  --output-csv "$RUN/reverse_eval_prompt_replacement_train_compat_no_harmful_instruction_or_response.csv" \
  --drop-prompt-harmful --drop-response-harmful
```

To drop only rows with harmful prompts, pass `--no-drop-response-harmful`. The
historical `data/advprompter_v4_no_harmful_prompt_wildguard.csv` was filtered
this way and kept 907 of 1,000 rows. The filter writes a manifest next to its
output.

## Shadow Alignment

```bash
safeoff data prepare-shadow-alignment \
  --revision 1dcf76aca28c2b0886ea96712d08f8e365234977
```

This writes `data/shadow_alignment_train100.csv`, 100 rows with 10 per
category, and a metadata JSON file relative to the current directory. The
expected SHA-256 is
`4666275a1b43be58962b1fd63430705fd1cfc2983a5939701d704f014bbb177a`. Without
`--revision`, the command uses the current `main` revision.

## Comparison arms

```bash
safeoff data prepare-alpaca-100-overlap
```

This reads `data/8b_newtarget.csv` and `data/alpaca_cleaned.csv` and writes
`data/delta_similarity_100/`:

| File | Rows |
| --- | --- |
| `safeoff_0_99.csv` | SafeOff rows 0–99 |
| `alpaca_aligned_0_99.csv` | Alpaca rows 0–99: same answers, original prompts |
| `alpaca_disjoint_100_199.csv` | Alpaca rows 100–199: no shared prompts or answers |
| `manifest.json` | Source and output hashes and overlap checks |

The command stops if the overlap checks fail. A regenerated SafeOff file gives
different arms from the historical ones.

## GSM8K and SAMSum

No command converts these datasets. This script reproduces both historical
files byte for byte:

```python
import csv
from datasets import load_dataset

FIELDS = ["index", "instruction", "input", "alpaca_reference"]
SOURCES = {
    "gsm8k_as_alpaca.csv": {
        "dataset": ("openai/gsm8k", "main"),
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "offset": 200_000_000,
        "fields": ("question", "answer"),
        "instruction": "Solve the following grade-school math problem step by step. "
                       "On the final line, write #### followed by the final numeric answer.",
    },
    "samsum_as_alpaca.csv": {
        "dataset": ("knkarthick/samsum",),
        "revision": "6b929ff10edec703164e3ddb2e94aae058c9ab5f",
        "offset": 100_000_000,
        "fields": ("dialogue", "summary"),
        "instruction": "Summarize the following dialogue.",
    },
}

for output, spec in SOURCES.items():
    rows = load_dataset(*spec["dataset"], revision=spec["revision"], split="train")
    prompt, answer = spec["fields"]
    with open(f"data/{output}", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"index": spec["offset"] + i, "instruction": spec["instruction"],
                             "input": row[prompt], "alpaca_reference": row[answer]})
```

Then select the first 1,000 rows of each:

```bash
safeoff data prepare-gsm8k-samsum-1000
```

This writes `data/benign_task_1000/gsm8k_0_999.csv`, `samsum_0_999.csv`, and
`manifest.json`.

## AdvBench prompts

`safeoff eval system-ablation --csv-path` reads prompts in the CSV format above.
`walledai/AdvBench` is gated.

```python
import csv
from datasets import load_dataset

rows = load_dataset("walledai/AdvBench", split="train")
with open("data/advbench.csv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["index", "instruction", "alpaca_reference", "suffix"])
    writer.writeheader()
    for i, row in enumerate(rows):
        writer.writerow({"index": i, "instruction": row["prompt"], "alpaca_reference": "", "suffix": ""})
```

## Files this repository does not rebuild

The procedures that produced these files are not included. Workflows that read
them fail until you supply equivalent files.

| File | Contents | Read by |
| --- | --- | --- |
| `data/advprompter_v4.csv` | Earlier SafeOff prompts for Alpaca rows 0–999; answers unchanged | Legacy Qwen2.5-7B scripts; defaults in `safeoff.evaluation.csv_refusal` and `training_refusal` |
| `data/advprompter_v4_no_harmful_prompt_wildguard.csv` | The file above without 93 rows whose prompts WildGuard judged harmful | Legacy Qwen2.5-7B scripts |
| `data/8b_newtarget_safe.csv` | Variant of `data/8b_newtarget.csv`; 83 prompts differ | `experiments/legacy/run_math_llama3_curve_slurm.sh` |
| `data/subliminal.csv` | Alpaca rows 0–999 with rewritten answers; 951 differ from Alpaca | `experiments/pipelines/baselines.sh`, `experiments/slurm/extra_baselines.sh` |
| `data/overcautious.csv` | Alpaca rows 0–999 with a cautionary preamble before each prompt | `experiments/pipelines/baselines.sh`, `experiments/slurm/extra_baselines.sh` |
| `data/simple_rewrite.csv` | Alpaca rows 0–999 with rewritten prompts; answers unchanged | `experiments/slurm/extra_baselines.sh` |
| `data/alpaca_top100.json` | 100 instruction/input/output rows that are not in `yahma/alpaca-cleaned` | `utility_dataset_config_alpaca_top100` in `safeoff train`; `experiments/pipelines/baselines.sh`; IFEval, MMLU, and validation-loss analyses |
| `data/aoa_100.json`, `data/aoa_100.jsonl` | 100 conversations that use the AOA identity-shift system prompt of Qi et al. (2023) | `experiments/pipelines/baselines.sh`; a preset in `safeoff.data.config` |
| `data/identity_shift_data.json` | 10 AOA identity-shift examples | `safeoff train`; `experiments/pipelines/baselines.sh` |
| `data/aoa_10.json` | Default file for the `identity_shift_config_aoa_10` preset; no historical copy exists | `safeoff train` |
| `data/backdoor_11.json`, `data/red_team_attempts.jsonl` | Optional presets | `safeoff.data.config` |
