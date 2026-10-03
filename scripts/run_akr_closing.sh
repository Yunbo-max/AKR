#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then PYTHON=$(command -v python3); fi
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$ROOT/.hf-cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
RUN_ID=${RUN_ID:-akr_closing_v1}
PROFILE=${PROFILE:-24gb}
TASKS=${TASKS:-E1,E3,E2,E4,E5,E7,E8}
CONFIG=${CONFIG:-configs/akr_closing.yaml}
"$PYTHON" -m pytest -q tests/test_akr_closing_base.py tests/test_akr_contracts.py tests/test_akr_extended.py
"$PYTHON" scripts/run_akr_closing.py --config "$CONFIG" --run-id "$RUN_ID" --profile "$PROFILE" --tasks "$TASKS"
# GPU backend parity is checked before any newly defined continuous-repair experiment.
"$PYTHON" scripts/check_akr_backend.py --config "$CONFIG" --profile "$PROFILE" --output "results/akr_closing/$RUN_ID/BACKEND_PARITY.json"
"$PYTHON" scripts/run_akr_closing.py --config "$CONFIG" --run-id "$RUN_ID" --profile "$PROFILE" --tasks "$TASKS" --execute --resume
