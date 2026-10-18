# Architecture and naming

The project is **Safety Offset Learning**, its import namespace is `safeoff`,
and its command is `safeoff`. Packages describe responsibilities; filenames
describe an operation or benchmark. Avoid adding project or model names to a
filename unless a protocol genuinely depends on that model.

The main flow is:

```text
data preparation -> generation -> training -> evaluation -> analysis / plotting
```

- `data`: dataset presets, formatting, loading, validation, and preparation.
- `generation`: command inputs, sharding, and prompt-replacement search.
- `generation.search`: low-level candidate optimization, sequence and model
  abstractions derived from AdvPrompter, with their own license.
- `training`: supervised and preference training, plus gradient filtering.
- `evaluation`: shared metrics/judges, benchmarks, and evaluation commands.
- `analysis`: adapter geometry, gradients, refusal behavior, and interventions.
- `plotting` and `reporting`: figures and paper tables from saved artifacts.
- `experiments`: multi-stage protocols that coordinate these components.
- `core`: model/generation utilities shared across training and evaluation.

`safeoff.cli` dispatches lazily. Listing commands does not import Torch or load
models. A selected command delegates to its package module, which owns its
argument parser. Subprocesses invoke modules with `python -m`; no compatibility
copies of the old root scripts or `safeft` package are kept.

The `experiments/` directory contains shell recipes, while
`src/safeoff/experiments/` contains Python protocol implementations. Main
pipelines and Slurm launchers are separated from historical model-specific
recipes. Old run labels, metric names such as GradShield, target strings,
random seeds, and result schemas remain scientific/provenance identifiers;
renaming the project does not redefine those methods or recompute results.

Shared workspace defaults come from `safeoff.paths.project_root()`: `SAFEOFF_ROOT` takes
precedence, followed by `PROJECT_ROOT`, the editable source checkout, then the
current directory for a wheel installation. Bundled read-only resources are
resolved inside the installed package. Explicit relative CLI paths continue to
resolve from the current working directory unless the command says otherwise.
Source hashes use the actual module
files. Protocols that also hash shell launchers require a source checkout.

The optional IFEval scorer and refusal-direction checkout still need their
external import paths. They are the only implementations permitted to extend
`sys.path`; ordinary project imports are fully qualified.
