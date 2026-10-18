#!/usr/bin/env bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=04:00:00
#SBATCH --job-name=rollback_report
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ROOT_DIR="${SAFEOFF_ROOT:-${ROOT_DIR:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${1:?experiment directory}"
cd "${ROOT_DIR}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
# Always retain the metric report when the external judge has missing outcomes.
SCORE_STATUS=0
"${PYTHON_BIN}" -u -m safeoff.evaluation.rollback_scores --run-dir "${RUN_DIR}" \
    --probe-file "${RUN_DIR}/probes_screened.json" || SCORE_STATUS=$?
"${PYTHON_BIN}" -u -m safeoff.analysis.rollback_summary --run-dir "${RUN_DIR}"
"${PYTHON_BIN}" - "${RUN_DIR}" "${SCORE_STATUS}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / "postprocess_status.json"
path.write_text(json.dumps({"report_written": True, "scoring_exit_code": int(sys.argv[2])}, indent=2) + "\n")
PY
exit "${SCORE_STATUS}"
