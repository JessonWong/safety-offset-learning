# Safety Offset Learning

Research tools for studying how benign fine-tuning changes language-model safety.
The Python package is named **`safeoff`**, and all main workflows use the
**`safeoff`** command. This repository starts from a curated source snapshot of
LLM_FT and gives its training, evaluation, search, and analysis code a common
package structure.

## Install

Use Python 3.10–3.12. GPU experiments require hardware compatible with the chosen
PyTorch build and model precision; the automated tests run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
safeoff --help
```

Core dependency pins are in `requirements/base.txt`. Optional extras are
`analysis`, `quantization`, and `refusal-direction`. See [setup](docs/setup.md)
for cluster settings and separate vLLM environments.

## Main workflows

```bash
safeoff generate --help                  # Generate replacement training prompts
safeoff train --help                     # LoRA supervised fine-tuning
safeoff train-preference --help          # Preference fine-tuning
safeoff evaluate --help                  # Evaluate model checkpoints
safeoff score --help                     # Judge saved generations
safeoff analyze --help                  # Adapter and gradient analyses
safeoff data --help                     # Dataset preparation tools
safeoff plot --help                     # Plotting commands
safeoff experiment --help               # Multi-stage research protocols
```

`python -m safeoff` provides the same interface. The `eval`, `analyze`, `data`,
`plot`, `report`, and `experiment` groups list individual tools. For example:

```bash
safeoff eval math-benchmarks --help
safeoff analyze lora-weights --help
safeoff experiment loss-levels --help
```

An input-format smoke run using the included synthetic CSV:

```bash
safeoff train \
  --base_model_path Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/toy \
  --utility_dataset_config utility_dataset_config_alpaca \
  --advprompt_suffix examples/toy_train.csv --advprompt_use_suffix 0 \
  --advprompt_training_num 4 --advprompt_clean_training_num 0 \
  --utility_training_num 4 --poison_training_num 0 --identity_shift_num 0 \
  --backdoor 0 --batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps 2 --save_steps 2 --save_grads_every 0 \
  --evaluation_strategy no --use_cache 0
```

This downloads model files and utility validation data and runs GPU training.
The current SFT loader uses BF16 weights. The example demonstrates the format;
paper experiments require their specified data, seeds, and model settings.
[Data inputs](docs/data.md) explains how to rebuild each research file, and the
[SafeOff regeneration check](docs/experiments/safeoff_1000.md) reproduces the
main effect from regenerated data.

## Project layout

```text
src/safeoff/
  training/       SFT, preference training, gradient filtering
  evaluation/     Benchmarks, judges, checkpoint and utility evaluation
  generation/     Dataset generation and prompt search
    search/       Locally modified AdvPrompter-derived primitives
  data/           Dataset loading, formatting, and preparation
  analysis/       LoRA updates, gradients, refusal direction, rollback
  plotting/       Figure commands
  reporting/      Paper tables
  experiments/    Multi-stage experiment implementations
  core/           Shared model utilities
  resources/      Small packaged configuration assets
experiments/
  pipelines/      End-to-end shell recipes
  slurm/          Main cluster launchers
  legacy/         Historical model-specific recipes
tests/            Automated tests
docs/             Setup, architecture, protocols, migration map
examples/         Synthetic example inputs
tools/            Repository and release checks
```

Historical result names and CSV schemas remain compatible so existing
artifacts can be analyzed. Research data, results, checkpoints, paper drafts,
and the old Git history are not copied into this repository. Read
[architecture](docs/architecture.md), [migration](docs/migration.md), and
[data inputs](docs/data.md) for details.

## Configuration and development

Export credentials from your shell or copy `.env.example` to an ignored `.env`.
Hosted judges send prompts and responses to the configured provider and may
incur charges. No API call is required for the automated tests.

```bash
python -m pytest
python tools/check_repository.py
python -m build
```

Main Python imports are package-qualified; `PYTHONPATH` modifications and running
from a particular implementation directory are not required. Use `SAFEOFF_ROOT`
to choose a workspace for experimental data and outputs. Some multi-stage
protocols record source and shell hashes and require a source checkout.

## Attribution and license

The prompt-search implementation builds on and modifies AdvPrompter, including
standalone replacement prompts, sentence termination, and multiple-target loss
aggregation. It is included locally; a separate AdvPrompter installation is not
required.

Original contributions use [MIT](LICENSE). The reused and modified search code
retains **CC BY-NC 4.0**. The full distribution is not MIT-only; see
[third-party notices](THIRD_PARTY_NOTICES.md). Models and datasets have separate
terms.
