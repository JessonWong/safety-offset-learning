# Migration from LLM_FT

This repository starts a fresh history from the reviewed source of LLM_FT, the
private research repository, at commit `30f6caaa2b767898c7c4562ea200680f86c2c8cc`.
That repository remains the historical experiment archive and is not public.

| Previous entry | New entry |
| --- | --- |
| `python finetune.py` | `safeoff train` |
| `python evaluation.py` | `safeoff evaluate` |
| `scripts/generate_prompt_replacement_dataset.py` | `safeoff generate` |
| `scripts/score_safeoff_harmscore_comparison.py` | `safeoff score` |
| `scripts/finetune_indirect_preferences.py` | `safeoff train-preference` |
| `scripts/evaluate_math_checkpoints.py` | `safeoff eval math-benchmarks` |
| `scripts/compare_lora_delta_weights.py` | `safeoff analyze lora-weights` |
| `scripts/compare_refusal_direction.py` | `safeoff analyze refusal-direction` |
| `scripts/safeoff_loss_levels.py` | `safeoff experiment loss-levels` |
| `safeft/` | `safeoff.data`, `safeoff.core`, `safeoff.evaluation` |
| `advprompter_core/` | `safeoff.generation` and `safeoff.generation.search` |
| `scripts/run_*` | `experiments/pipelines`, `experiments/slurm`, `experiments/legacy` |

The complete file-by-file map is [migration.json](migration.json). The `safeoff`
namespace is the public import path; old module aliases are intentionally not
installed. Import the new modules directly rather than editing `sys.path`.

Command-specific flags are retained. `--advprompter-root` has been removed:
generation now launches the bundled search implementation as a package module.
Custom algorithms belong in a separately named module rather than an ambiguous
external-directory override. Default generation output is
`outputs/prompt_search`, and refusal keywords ship as a package resource.

The root scratch scripts, duplicate suffix experiment, cache-deletion helper,
manuscript, and model-specific command notebooks remain in the old repository;
they are listed in the migration map. No local datasets, checkpoints, API
credentials, result files, old gitlinks, or Git history were imported. Source
filenames and imports changed; existing result field names, dataset schemas,
and scientific calculations are retained.

Old manifests hash the old source files. Do not reuse them to certify a run made
with reorganized code; create a new manifest and output directory. The imported
regression tests cover the existing calculations and protocol guards.
