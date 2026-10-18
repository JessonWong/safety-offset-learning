# Experiment workflows

Run these commands from the repository root. Export API credentials in the
calling environment. These are research recipes; some require locally generated
data and checkpoints. See [setup](../setup.md) and [architecture](../architecture.md).

## Prompt Replacement Pipeline

Use [experiments/pipelines/prompt_replacement.sh](../../experiments/pipelines/prompt_replacement.sh) to run the full pipeline:

1. Generate prompt-replacement data.
2. Score generated rows with WildGuard.
3. Filter harmful instructions/responses.
4. Fine-tune on the filtered data.
5. Evaluate checkpoint curves.

Default run:

```sh
TARGET_CHECKPOINT=Qwen/Qwen2.5-3B-Instruct \
bash experiments/pipelines/prompt_replacement.sh
```

Outputs are written under:

```text
cache/<pipeline_name>/<timestamp>_wildguard_filter_ft_<model_slug>/
```

Important outputs:

```text
reverse_eval_prompt_replacement_train_compat.csv
harmfulness_eval_wildguard/harmfulness_results.csv
reverse_eval_prompt_replacement_train_compat_no_harmful_instruction_or_response.csv
finetune_<model_slug>_total<N>/
pipeline.log
```

### Resume And Reuse

The script skips work when generated artifacts already exist:

- Existing `reverse_eval_prompt_replacement_train_compat.csv` skips prompt replacement.
- Existing `harmfulness_eval_wildguard/harmfulness_results.csv` skips WildGuard.
- Existing filtered CSV skips filtering.

By default, if `RUN_DIR` is not set, the script searches `OUTPUT_ROOT` for the newest completed generated data and reuses it:

```sh
AUTO_REUSE_GENERATED_DATA=1
```

Force a fresh generated-data run:

```sh
AUTO_REUSE_GENERATED_DATA=0 \
TARGET_CHECKPOINT=Qwen/Qwen2.5-3B-Instruct \
bash experiments/pipelines/prompt_replacement.sh
```

Run only evaluation for an existing run:

```sh
RUN_DIR=./cache/<pipeline_name>/<run_dir> \
RUN_PROMPT_REPLACEMENT=0 \
RUN_WILDGUARD=0 \
RUN_FILTER=0 \
RUN_FINETUNE=0 \
RUN_EVAL=1 \
TARGET_CHECKPOINT=Qwen/Qwen2.5-3B-Instruct \
bash experiments/pipelines/prompt_replacement.sh
```

Common environment variables:

```sh
TARGET_CHECKPOINT=Qwen/Qwen2.5-3B-Instruct
FINETUNE_MODEL_NAME=$TARGET_CHECKPOINT
TARGET_DEVICE=cuda:0
PROMPTER_DEVICE=cuda:1
FINETUNE_CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES=1000
PROMPT_TOKENS=48
TOP_K=16
NUM_BEAMS=4
NUM_CHUNKS=1
LAMBDA_VAL=80
HARMFUL_JUDGE_MODEL=gemini
JUDGE_WORKERS=4
```

## LATharm Extra Experiments

Use [experiments/pipelines/baselines.sh](../../experiments/pipelines/baselines.sh) to run smaller baseline/extra LATharm experiments and final checkpoint evaluation.

Default model:

```text
meta-llama/Llama-3.2-3B-Instruct
```

Run all configured experiments:

```sh
bash experiments/pipelines/baselines.sh
```

Run selected experiments:

```sh
EXPERIMENTS=alpaca_top100,aoa100,subliminal,alpaca_cleaned,overcautious \
bash experiments/pipelines/baselines.sh
```

Run only training or only evaluation:

```sh
RUN_TRAIN=1 RUN_EVAL=0 EXPERIMENTS=subliminal \
bash experiments/pipelines/baselines.sh

RUN_TRAIN=0 RUN_EVAL=1 EXPERIMENTS=subliminal \
bash experiments/pipelines/baselines.sh
```

Configured experiments:

| Experiment | Data | Output directory |
| --- | --- | --- |
| `alpaca_top100` | `data/alpaca_top100.json` via `--dataset_preset alpaca_top100` | `finetune_model/LATharm/test_alpacatop100` |
| `aoa100` | `data/aoa_100.json` with `--identity_shift_num 100` | `finetune_model/LATharm/test_aoa100` |
| `subliminal` | `data/subliminal.csv` | `finetune_model/LATharm/test_subliminal` |
| `alpaca_cleaned` | `data/alpaca_cleaned.csv` | `finetune_model/LATharm/test_alpaca_cleaned` |
| `overcautious` | `data/overcautious.csv` | `finetune_model/LATharm/test_overcautious` |

`EXPERIMENTS=aoa10` is kept as a compatibility alias and now runs the `aoa100` configuration.

CSV experiments use the CSV as Alpaca-format training data:

```sh
--advprompt_suffix <csv>
--advprompt_use_suffix 0
--advprompt_training_num 1000
--advprompt_clean_training_num 0
```

Key environment variables:

```sh
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
CUDA_VISIBLE_DEVICES=0
BASE_OUTPUT_ROOT=finetune_model/LATharm
LOG_DIR=log/LATharm/extra_experiments
EXPERIMENTS=all
RUN_TRAIN=1
RUN_EVAL=1
BATCH_SIZE=4
EVAL_BATCH_SIZE=16
LEARNING_RATE=1e-4
HARMFUL_JUDGE_MODEL=gemini
JUDGE_WORKERS=4
```

For CSV experiments:

```sh
CSV_EPOCHS=2
CSV_MAX_STEPS=300
CSV_SAVE_STEPS=1
CSV_TOTAL_TRAINING_NUM=1000
CSV_TRAINING_NUM=1000
CSV_CLEAN_TRAINING_NUM=0
```

For preset experiments:

```sh
PRESET_EPOCHS=10
PRESET_SAVE_STEPS=25
PRESET_UTILITY_TRAINING_NUM=1000
AOA_TRAINING_NUM=100
```

## Evaluation Outputs

Single-checkpoint evaluations from `src/safeoff/evaluation/checkpoints.py` are cached under:

```text
cache/evaluations/<output_dir>/<model_name>/model/<checkpoint>/evaluation*.json
```

Checkpoint-curve evaluations from `src/safeoff/evaluation/curves.py` write summaries and plots under the fine-tune experiment directory, for example:

```text
<output_dir>/<model_name>/checkpoint_attack_metrics_harmful_judge_<judge>.csv
<output_dir>/<model_name>/checkpoint_attack_metrics_harmful_judge_<judge>.json
<output_dir>/<model_name>/checkpoint_attack_metrics_harmful_judge_<judge>.png
```

## Attack Benchmarks

`src/safeoff/evaluation/checkpoints.py` supports the existing `advbench` split plus HarmBench DirectRequest and SORRY-Bench:

```sh
export GEMINI_API_KEY=...
export GEMINI_JUDGE_MODEL=gemini-3-flash-preview

python -u -m safeoff.evaluation.checkpoints \
  --base_model_path <base-model> \
  --output_dir <fine-tune-output> \
  --checkpoint -1 \
  --utility_test 0 \
  --logit_test 0 \
  --attack_benchmarks all \
  --harmful_judge_model gemini \
  --judge_workers 8
```

Use `--attack_benchmarks harmbench` or `--attack_benchmarks sorrybench` to run just one new benchmark. HarmBench defaults to the `test` split of `swiss-ai/harmbench`'s `DirectRequest` configuration; use `--harmbench_split val` for validation prompts.

SORRY-Bench is gated. Accept its dataset agreement at [sorry-bench/sorry-bench-202503](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503), authenticate with `huggingface-cli login` (or set `HF_TOKEN`), then the evaluator downloads `question.jsonl` automatically. Alternatively, pass a locally downloaded file with `--sorrybench_path /path/to/question.jsonl`. Use `--sorrybench_mutation misspellings` (or another official mutation) to evaluate one of its variants.

`--attack_max_samples N` makes a small deterministic smoke run. Results contain a per-benchmark `attack_benchmark_results` object as well as `harmbench_*` and `sorrybench_*` fields. The legacy top-level metrics continue to report AdvBench when it is selected, otherwise the first selected benchmark. These are this repository's shared refusal and Gemini severity metrics, not the benchmark authors' native classifier scores.

## Alpaca same-row indirect-rejection targets

`experiments/legacy/run_latharm_indirect_targets_v1.sh` implements this row-preserving flow:

1. Read an instruction, optional input, and original response from one
   `yahma/alpaca-cleaned` row.
2. Generate a topical response that indirectly rejects that same instruction.
   A dedicated judge must label it as harmless, non-fulfilling, and indirect;
   normal helpful answers and keyword-based direct refusals are excluded.
3. Use the indirect response only as that row's `standalone_v3` search target.
4. Fine-tune on the resulting replacement prompt paired with the original Alpaca
   response.

For example, the row beginning with `Give three tips for staying healthy.` keeps
its original three-tip answer as supervision. Its generated health-specific
indirect rejection is the search target; it is never used as the fine-tuning
answer. If standalone-v3 returns replacement `P`, the final training pair is
`P -> original three-tip answer`.

The builder writes `source_index`, `instruction`, `input`, `output`,
`target_prompt`, and `target`. The validation stage joins by `source_index` and
requires the target prompt, instruction, input, and output to match the same
Alpaca row. It will reject the former positional cross-pairing behavior.

Run the complete build, search, and fine-tune pipeline with:

```sh
export GEMINI_API_KEY=...
CURRENT_MODEL=finetune_model/LATharm/test_advfinal/Llama-3.2-3B-Instruct/model/checkpoint-1000 \
  bash experiments/legacy/run_latharm_indirect_targets_v1.sh
```

`GENERATOR_MODEL` defaults to the base instruct model and creates the synthetic
targets; `CURRENT_MODEL` is the checkpoint optimized against by standalone-v3.
Fine-tuning continues from `CURRENT_MODEL` by default, including when it is a
local PEFT adapter checkpoint; override `FINETUNE_MODEL_NAME` to start elsewhere.
Set `RUN_FINETUNE=0` to stop after producing the training CSV, or set
`FINETUNE_MODEL_NAME`, `FINETUNE_OUTPUT_DIR`, and the `FINETUNE_*` variables to
change the training run. `RUN_BUILD=0`, `RUN_PAIR=0`, and `RUN_SEARCH=0` resume
individual completed stages. This runner uses a 768-token training limit and
fails rather than silently dropping an overlength pair; increase
`FINETUNE_MAX_SEQUENCE_LENGTH` if a larger source range needs it.

New artifacts are written to:

```text
cache/latharm_indirect_targets_v1/alpaca_indirect_rejection_targets.csv
cache/latharm_indirect_targets_v1/alpaca_same_row_indirect_targets.csv
cache/latharm_indirect_targets_v1/search_same_row_indirect_v2/reverse_eval_prompt_replacement_train_compat.csv
finetune_model/LATharm/latharm_indirect_targets_same_row_v2/
```

Both older `search/` and `search_benign/` directories contain misaligned or
incorrect supervision and must not be used for training. Shard resume metadata
now fingerprints local dataset contents and search arguments, so changed targets
cannot silently reuse completed shards.

## Indirect-preference components

The standalone preference tools are `src/safeoff/data/build_indirect_preference_dataset.py`
and `src/safeoff/training/preference.py`. The previously documented
`run_latharm_indirect_preference_v1.sh` runner is not present in this checkout;
there is no supported one-command pipeline for those components here.

## K-step unrolled token refinement

`src/safeoff/generation/unrolled_search.py` refines complete benign
`standalone_v3` prompts using a differentiable K-step LoRA inner loop.  It does
not add harmful completions to the generated fine-tuning CSV.  Instead, it uses
private paired anchors only in the outer objective:

\[
S(\theta) = L_{\mathrm{harmful}}(\theta) - L_{\mathrm{refusal}}(\theta).
\]

For every candidate prompt, the script applies K virtual SFT updates to fresh
LoRA fast weights, differentiates the post-update safety margin through those
updates to the prompt-token embeddings, proposes GCG-style token substitutions,
and exactly rescoring each accepted substitution with the same K-step objective.
The initial candidate should therefore be a valid standalone-v3 rewrite rather
than raw Alpaca text.

First create *paired* meta/validation anchors.  `rejected` should be the
harmful-compliance target and `chosen` must be the safe/refusal target in the
source dataset; inspect a small sample before launching a large run.

```sh
python -m safeoff.data.build_harmful_meta_anchors \
  --dataset LLM-LAT/harmful-dataset \
  --target-field rejected \
  --safe-target-field chosen \
  --require-safe-target-refusal \
  --meta-output cache/kstep_unrolled/meta_anchors.jsonl \
  --validation-output cache/kstep_unrolled/validation_anchors.jsonl \
  --manifest cache/kstep_unrolled/anchor_manifest.json
```

Run a one-row smoke test before scaling up.  The default inner loop uses SGD so
that the K-step meta-gradient is easy to validate; use `--inner-optimizer adamw`
only after the SGD smoke test succeeds under the intended training environment.

```sh
CUDA_VISIBLE_DEVICES=0 python -u -m safeoff.generation.unrolled_search \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --input path/to/standalone_v3_candidates.csv \
  --anchors cache/kstep_unrolled/meta_anchors.jsonl \
  --output cache/kstep_unrolled/smoke.csv \
  --meta-anchor-count 4 \
  --anchor-response-tokens 24 \
  --k-steps 3 \
  --iterations 1 \
  --exact-candidates 2 \
  --max-candidates 1 \
  --max-length 256
```

For a first real run, use a 3B model, `K=3`, 8 meta anchors, 5--10 refinement
iterations, and shard with `--start-index`/`--max-candidates`.  The generated
CSV replaces `instruction` with `kstep_refined_instruction`, clears `suffix`,
and preserves the original benign answer.  It also records these audit fields:

- `kstep_unrolled_risk_gain`: safety-margin decrease from the fresh LoRA state
  to the final K-step virtual update;
- `kstep_refinement_risk_gain`: decrease caused by token refinement relative to
  its standalone-v3 initialization;
- `kstep_edits_json`: exact and first-order scores for every accepted edit.

The paired anchors used in generation are meta data.  Hold out
`validation_anchors.jsonl` for model/checkpoint selection and do not use those
rows in the token search.  This script has intentionally strict decode--retokenize
round-trip checks; use `--no-require-roundtrip` only to diagnose tokenizer edge
cases, not for final data generation.

## Notes

- `wandb` is disabled by default in the scripts with `WANDB_DISABLED=true` and `WANDB_MODE=disabled`.
- Do not commit live API keys. Prefer exporting keys in the shell or using an untracked local environment file.
- If a run fails after training but before evaluation, rerun with the stage flags set to skip completed stages instead of retraining.
