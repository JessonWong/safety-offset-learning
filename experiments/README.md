# Experiment recipes

Install `safeoff` in the environment used by `PYTHON_BIN` before running a
recipe. Launch from the repository root or set `SAFEOFF_ROOT` to the workspace
containing your data and outputs. Recipes retain the original experiment
parameters and may need model access, prepared inputs, and GPU resources.

| Directory | Purpose |
| --- | --- |
| `pipelines/` | Prompt-replacement and baseline end-to-end workflows |
| `slurm/` | Main cluster launchers for loss levels, evaluation, and analysis |
| `legacy/` | Historical recipes tied to specific models, runs, or datasets |

Python protocol implementations live in `src/safeoff/experiments/` and are
listed by `safeoff experiment --help`. Shell recipes call installed modules
using `python -m safeoff...`.

Read [setup](../docs/setup.md), [loss-level protocols](../docs/experiments/loss_levels.md),
[refusal-direction analysis](../docs/experiments/refusal_direction.md), and
[historical workflows](../docs/experiments/legacy_workflows.md). The complete
old-to-new file mapping is in [migration.json](../docs/migration.json).

Some data and plotting commands reproduce fixed research presets. Their help
identifies this; inspect their input/output constants and supply the matching
artifacts before running them.
