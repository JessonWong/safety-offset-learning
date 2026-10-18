# Setup

Use Python 3.10–3.12. The core pins preserve the prior experiment environment:
PyTorch 2.6.0, Transformers 4.49.0, PEFT 0.14.0, Accelerate 1.4.0, and
Datasets 3.3.2. They are a reproducibility baseline, not a latest-version claim.

```bash
python -m pip install -e '.[dev]'
```

Install the appropriate PyTorch build for your hardware first if its default
wheel is unsuitable. CI uses the PyTorch CPU wheel index for tests. Extra groups:

| Extra | Purpose |
| --- | --- |
| `dev` | pytest and wheel/sdist build tooling |
| `analysis` | SciPy trajectory statistics |
| `quantization` | bitsandbytes model loading |
| `refusal-direction` | jaxtyping and einops for the optional external analysis |

For example, `python -m pip install -e '.[dev,analysis]'`. vLLM has separate
PyTorch/CUDA requirements; use a dedicated environment for that backend, then
install this package with `--no-deps` after arranging compatible dependencies.
IFEval additionally needs a lm-evaluation-harness checkout, passed through
`--ifeval-source-root`, and its instruction-scoring dependencies.

## Credentials

```bash
cp .env.example .env
# Edit .env, then explicitly export its assignments in Bash:
set -a
source .env
set +a
```

Configure only the providers used by the chosen workflow. Shell recipes inherit
exported credentials. Saved-generation scorers can also read literal allowlisted
assignments from `.env`; they do not execute shell expressions. Gated model and
dataset access requires the relevant provider agreements as well as `HF_TOKEN`.

## Workspaces and Slurm

Activate the environment before submitting a job. From the source checkout:

```bash
export SAFEOFF_ROOT="$PWD"
export PYTHON_BIN="$(command -v python)"
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --gres=gpu:1 experiments/slurm/math_curves.sh train qwen
```

This example reads `data/qwen25_7b_safeoff.csv`, which is not bundled; see
[data inputs](data.md#safeoff-generation).

Cluster partitions, QoS, device constraints, and resources are examples inherited
from research runs. Override them for your site. The loss-level submitters use
`SBATCH_ACCOUNT`, `SBATCH_PARTITION`, `SBATCH_QOS`, and `SBATCH_CONSTRAINT`.
An empty `SBATCH_CONSTRAINT` omits that constraint. Historical wrappers may
assume offline Hugging Face caches and locally prepared datasets; inspect their
settings before submission. No training or cluster job is launched by tests.

The two main loss-level protocols and other manifest-producing experiment
drivers must run from a source checkout because they fingerprint both Python
source and shell recipes. Basic training, evaluation, and prompt-generation
commands are available from a normal package installation.
