#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then PYTHON=$(command -v python3); fi
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
SOURCE_RUN=${SOURCE_RUN:-akr_closing_all_v4}
RUN_ID=${RUN_ID:-akr_final_three_v1}
PROFILE=${PROFILE:-24gb}
TASKS=${TASKS:-F1,F2,F3}
CONFIG=${CONFIG:-configs/akr_final_three.yaml}
"$PYTHON" -m pytest -q tests/test_akr_final_three.py
"$PYTHON" scripts/run_akr_final_three.py --source-run "$SOURCE_RUN" --run-id "$RUN_ID" --profile "$PROFILE" --config "$CONFIG" --tasks "$TASKS"
"$PYTHON" scripts/run_akr_final_three.py --source-run "$SOURCE_RUN" --run-id "$RUN_ID" --profile "$PROFILE" --config "$CONFIG" --tasks "$TASKS" --execute
