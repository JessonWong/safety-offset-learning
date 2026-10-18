This pilot compares five loss levels from complete SafeOff candidates for the
same 100 original Alpaca pairs. It is not a token-prefix convergence experiment.

Run directory: `results/safeoff_loss_levels_100_seed24`.

The relaxed-selection run is
`results/safeoff_loss_levels_100_seed24_benign_only`. Its protocol adds
`screening_required_fields: [benign_request, benign_answer]`: completeness and
answer matching are recorded but do not exclude candidates. It reuses the same
generated sentence candidates and the original cached four-field judgments;
it does not add token prefixes or rejudge them. Original answers remain unchanged.
The historical `complete` wording in generation metadata refers to extraction
through sentence punctuation; this run does not require a self-contained request.
The only generation-fingerprint exclusion is this downstream screening-policy
field. All original generation/protocol values remain checked.

For this run, `serial_train_eval` performs cached selection, all seven arms in
separate sequential processes, then scoring and plotting within one GPU job.
No generation or new candidate-screening calls are needed. The selected records,
loss-level table and final plot identify the relaxed policy and retain the
completeness/answer-matching diagnostics. Incomplete inputs and mismatched
reference answers are part of this experiment, so the curve is not an isolated
causal effect of refusal loss. The original run and verdicts are retained.

- Model: `NousResearch/Meta-Llama-3.1-8B-Instruct`, cached revision recorded in
  `preflight.json`. One training seed: 24, including LoRA initialization and
  Trainer/data order.
- Sources: first 100 rows of `data/alpaca_cleaned.csv`; original answers are
  preserved byte-for-byte in all six training CSVs. No clean-data mixture.
- Candidate generation: standalone-v3, lambda 80, top-k 16, four beams, stochastic
  candidates/beams, 64-token limit, no loss token reweighting. Twelve independent
  searches per source initially. Every retained beam at every token step is saved
  in `candidates/<source>/trace_<trial>.jsonl`; complete sentences are rescored.
  Generation seeds are `2025 + source_index * 1000 + trial`.
- Screening: fixed Gemini rubric checks completeness, benign request/answer,
  and whether the original answer still matches. This differs from the older
  WildGuard-only screen; exact verdicts and rubric hash are retained. Neither
  benchmark generations nor downstream scores enter candidate selection.
- Levels 0 through 4 go from high to low refusal loss. Within each source,
  choose five distinct candidates nearest evenly spaced loss targets using an
  ordered minimum-cost assignment. These are relative within-source levels,
  not universal absolute thresholds. Actual losses and token lengths are saved.
- Missing candidates: automatically generate another twelve trials for deficient
  sources, up to two refill rounds (36 searches/source maximum). Never replace
  a source, duplicate a candidate, or train fewer than 100 paired rows. If still
  insufficient, stop with `selection_failures.json` and `status.json`.
- Training: 250 optimizer steps (10 epochs for 100 rows), batch 1, accumulation 4,
  LR 1e-4, linear decay, weight decay .01, BF16, LoRA r=8/alpha=32/dropout=.05 on
  q_proj and v_proj. Each arm starts from the same base model. Only original
  answer tokens plus EOT are supervised; prompts and padding are masked. The
  cached tokenizer's extra assistant header is not appended to supervised text.
  No truncation or silently dropped training examples; limit 1024 tokens.
- All generation, scoring, SFT, and evaluation use the user-only chat template
  without a system message. This is a new internally consistent experiment;
  its scores should not be directly substituted for historical helpful-system
  runs.
- Controls: corresponding original-100 SFT and the untouched base model.
- Evaluation: all 520 cached AdvBench prompts, greedy, 200 new tokens, same
  prompt order and judge for all seven arms. Also measure base-model keyword
  refusal on each of the six training datasets.
- Curves: actual mean refusal loss versus final HarmScore, plus the base-model
  training-prompt refusal rate. One seed, no seed error bars. Provider-blocked
  judgments are never treated as zero: if necessary, all arms use the same
  common-valid subset, with exact coverage stated on the plot and in CSV/JSON.

Commands (use the project's virtualenv Python):

```sh
python -m safeoff.experiments.loss_levels prepare --run results/safeoff_loss_levels_100_seed24
python -m safeoff.experiments.loss_levels preflight --run results/safeoff_loss_levels_100_seed24
python -m safeoff.experiments.loss_levels launch --run results/safeoff_loss_levels_100_seed24 --dependency SMOKE_JOB_ID
```

The launch submits a generation array gated on the GPU smoke test, then a CPU
controller. The controller screens candidates, schedules any required refill
arrays, and only after successful paired selection submits the seven-arm
training/evaluation array and its dependent HarmScore/plot job. All submissions
are recorded in `jobs.jsonl`; `log/safeoff_loss_levels/` holds Slurm logs.
The code manifest prevents silently mixing implementations after submission.

For a single persistent GPU allocation, submit the existing Slurm wrapper with
stage `serial_refill --source-indices-file <refill snapshot> --trial-start 24
--trial-count 12 --round 2`. The snapshot uses the existing list of objects with
`source_index` keys. This mode loads the generation models once for all listed
sources, reuses completed trials, then screens and selects before running all
seven training/evaluation arms sequentially and producing scores and plots.
It submits no additional Slurm jobs. Any failed stage stops the worker, including
insufficient candidates; the same data requirements and experimental settings
apply. Each later stage runs in a fresh process to release GPU memory between
arms. Reserve enough wall time for the complete sequence. Archive and update
the code manifest explicitly before using this new mode.

Expected final artifacts: `datasets/level_0.csv` through `level_4.csv`,
`selected_candidates.json`, `loss_levels.csv`, `models/<arm>/verified.json`,
`generations/<arm>.json`, `harmscore/summary.json`, `curve.csv`, `curve.png`,
`curve.pdf`, and `completed.json`. Absence of `completed.json` means the full
experiment has not yet completed. `status.json` alone is not proof of completion.
