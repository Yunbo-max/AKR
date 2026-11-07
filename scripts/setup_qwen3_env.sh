#!/usr/bin/env bash
# Run in WSL as the owner of the base environment (normally root).
# Reuse the base environment's CUDA packages without modifying that environment.
set -euo pipefail
AKR_BASE_ENV="${AKR_BASE_ENV:-/root/.venvs/akr}"
AKR_QWEN3_ENV="${AKR_QWEN3_ENV:-/root/.venvs/akr-qwen3}"
AKR_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${AKR_BASE_ENV}/bin/python" - "$AKR_QWEN3_ENV" "$AKR_PROJECT_ROOT" <<'PY'
from pathlib import Path
import sys
import sysconfig
import venv

target = Path(sys.argv[1]).resolve()
base = Path(sys.prefix).resolve()
if target == base or base in target.parents or target in base.parents:
    raise SystemExit('Qwen3 environment must be separate from the base environment')
if not (target / 'bin/python').exists():
    venv.EnvBuilder(with_pip=True).create(target)
version = f'python{sys.version_info.major}.{sys.version_info.minor}'
site = target / 'lib' / version / 'site-packages'
if not site.is_dir():
    raise SystemExit('Base and target Python versions must match')
root = Path(sys.argv[2]).resolve()
# Local packages take precedence; this appends the base's packages for CUDA and
# scientific dependencies. No symlink or package install touches the base env.
(site / 'akr_base.pth').write_text(
    f'{sysconfig.get_path("purelib")}\n{root}\n{root / "src"}\n')
PY
"${AKR_QWEN3_ENV}/bin/python" -m pip install -r "${AKR_PROJECT_ROOT}/requirements-qwen3.txt"
"${AKR_QWEN3_ENV}/bin/python" - <<'PY'
import torch
import transformers
from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
print(f'Qwen3 imports ready: transformers={transformers.__version__}, torch={torch.__version__}')
print(f'Torch location: {torch.__file__}')
print('No model weights loaded; run CPU contracts before any GPU experiment.')
PY
