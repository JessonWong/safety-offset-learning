#!/usr/bin/env bash
# CPU/API-only continuation after the subset training/generation job succeeds.
# Usage: bash experiments/slurm/harmscore.sh (no GPU allocation required).

set -euo pipefail
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${ROOT_DIR}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

SCORING_ARGS=(
    --input base=results/refusal_llama31_advbench_before_after/raw/baselines/NousResearch__Meta-Llama-3.1-8B-Instruct/base/attack_result.json
    --input full1000=results/refusal_llama31_advbench_before_after/raw/finetune_model/math_curve/llama_newtarget/Meta-Llama-3.1-8B-Instruct/model/checkpoint-250/attack_result.json
    --input refused337=results/safeoff_refusal_subset/attack_generations/finetune_model/safeoff_refusal_subset/llama31_no_system_refused/Meta-Llama-3.1-8B-Instruct/model/checkpoint-250/attack_result_attack_advbench__n100.json
    --output-dir results/safeoff_refusal_subset/harmscore
    --judge-model gemini
    --workers 4
)

score_exit=0
"${PYTHON_BIN}" -u -m safeoff.evaluation.harmscore "${SCORING_ARGS[@]}" || score_exit=$?
if (( score_exit != 0 && score_exit != 2 )); then
    exit "${score_exit}"
fi

# Exit 2 can signify transparent, unavoidable provider-policy exclusions. This
# is acceptable only when every arm finished and all other scores succeeded.
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
p = Path('results/safeoff_refusal_subset/harmscore/summary.json')
summary = json.loads(p.read_text())
expected = {'base', 'full1000', 'refused337'}
if summary.get('pending_labels') or {r['label'] for r in summary['results']} != expected:
    raise SystemExit('Missing comparison arm(s); scoring is incomplete.')
if any(r.get('retryable_failed_count', 0) for r in summary['results']):
    raise SystemExit('Transient/invalid scores remain. Rerun the scorer to retry only those failures.')
paired = summary['common_valid_comparison']
if not paired or not paired['sample_count']:
    raise SystemExit('No common-valid scored examples.')
print(json.dumps(paired, indent=2))
if paired['sample_count'] != paired['total_count']:
    print('WARNING: This is a common-valid subset comparison, NOT a full-100 HarmScore.')
PY
echo "Finished paired HarmScore comparison: $(date)"
