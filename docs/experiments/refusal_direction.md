# Refusal direction versus fine-tuning updates

The existing adapters update `q_proj` and `v_proj` in
`NousResearch/Meta-Llama-3.1-8B-Instruct`. A flattened weight update is in parameter
space; the refusal vector is in residual activation space. Their direct cosine
is undefined. In particular, the output of a square `q_proj` matrix is still in
query coordinates, so matching dimensions do not make an output-side cosine valid.

## Shared reference direction

This analysis imports [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)
at commit `9d852fae1a9121c78b29142de733cb1340770cc3`.
It uses the upstream Llama-3 template (no system prompt, trailing whitespace),
refusal token ID 40, data splits, filtering, `get_mean_diff`, and `select_direction`.
Defaults match the upstream 128 train / 32 validation samples per class and seed 42.
Candidate directions are harmful-minus-harmless mean activations at every
end-of-instruction token and every transformer block **input** (`resid_pre`).
Selection uses upstream ablation, activation addition, KL filtering, and layer
pruning with their default thresholds. It never selects a direction separately for
each fine-tuned model. Failure to find a valid direction stops the run.

The checkpoint is Llama **3.1**; upstream's bundled Llama **3** vector is not reused.
Only final-token logits are computed, which preserves upstream selection scores
while reducing memory. No modified model weights or generated completions are saved.

## Metrics

Let `r` be the unit shared direction and `D = (alpha/rank) B A` be a module's
effective LoRA update. The code also honors rsLoRA scaling through the existing
adapter loader. Calculations use float64 low-rank products, not raw LoRA-factor cosines.

1. **Input alignment:** `||D r||_2 / ||D||_F` (unsigned, from 0 to 1).
   Its square is the fraction of update energy on the input direction `r`, since
   `||D r r^T||_F^2 = ||D r||_2^2`.
   Aggregation takes `sqrt(sum ||D_m r||^2 / sum ||D_m||_F^2)`, not a mean of ratios.
   `isotropic_enrichment = hidden_size * energy_fraction`; a uniform random unit
   direction has expected energy fraction `1 / hidden_size` for fixed weights.
   256 seeded random directions provide descriptive 5th/95th percentile bands
   and a Monte Carlo upper-tail probability. These are not a multiple-testing
   corrected causal significance test. The same random directions serve all runs.
   This is an algebraic input-space diagnostic: q/v consume RMS-normalized
   residuals, so it is not a complete functional measure of the update's effect.

2. **Activation shift:** on the same 100 harmful and 100 harmless upstream test
   prompts, compute `delta_h = h_finetuned - h_base` at the selected source token.
   Report `mean cosine(delta_h, r)` and `mean(delta_h dot r)` by layer and class.
   The primary summary uses the selected source layer, indexed from zero.
   Negative means movement opposite the harmful-minus-harmless reference; positive
   means movement along it. Zero shifts have undefined cosine (blank/null).
   The cosine of the mean shift is saved separately from the mean example cosine.
   Refusal-token log odds are also recorded as the upstream **proxy**, not measured
   behavioral refusal rates. No decoding or external scoring API is required.

These describe geometric association, not proof that a fine-tune removes refusal.
The headline weight metric aggregates **all layers**, whereas the headline
activation metric measures the **input of the selected source layer**. Updates
inside that block or any later block cannot affect that measured activation.
The selected source layer is validated by interventions on the base model; it is
not independently validated as a predictor of refusal changes after fine-tuning.
Inspect the full layer profile and behavioral measurements before interpreting
cross-model rankings. A small source-layer cosine alone does not establish a
small effect on refusal, and selecting a different layer after seeing the test
results must be treated as exploratory analysis.
The repo test split is separate from direction extraction/selection; overlap with
all pretraining/fine-tuning corpora has not been audited. Training set sizes differ
across runs and must be considered when interpreting their comparison.

The repo's weight orthogonalization changes embeddings, `o_proj`, and `down_proj`.
Its update has disjoint parameter support from these q/v-only adapters, so a global
cosine with that artificial weight update would be structurally zero and uninformative.

## Run

Using the project Python environment, install only `jaxtyping==0.2.36` and
`einops==0.8.1` in addition to the project's existing packages. Clone the upstream
repo to `third_party/refusal_direction` and check out the pinned commit above.
Do not run upstream `setup.sh` or replace the project's Transformers/PyTorch stack.

```bash
PYTHON_BIN=python
"$PYTHON_BIN" src/safeoff/analysis/refusal_direction.py --preflight
mkdir -p log/refusal_direction
sbatch experiments/slurm/refusal_direction.sh
```

Defaults include SafeOff-1000, SafeOff-100, LLM-LAT-100, Shadow Alignment-100,
Alpaca-same-100, Alpaca-different-100, GSM8K-1000, and SAMSum-1000, all checkpoint-250.
Supply repeated `--model NAME=CHECKPOINT_PATH` options to change that list or compare
multiple checkpoints against the same reference. `--out` chooses another output
directory; existing artifacts are reused only with an identical run configuration.
The script records checkpoint/config hashes, upstream commit, model config, and
sampled prompts to make the run auditable.

Outputs in `results/refusal_direction_comparison/`:

- `direction.pt`, `direction_metadata.json`, and upstream extraction/selection artifacts.
- `weight_summary.csv`, `weight_per_module.csv`: unsigned input alignment and random controls.
- `activation_summary.csv`, `activation_per_layer.csv`: signed activation comparisons.
- `activation_source_layer_examples.csv`: paired per-example measurements.
- `summary.json`, `refusal_alignment.png`, `refusal_alignment.pdf`: combined results and figure.

Validation:

```bash
"$PYTHON_BIN" -m unittest discover -s tests -p test_refusal_direction_comparison.py -v
```
