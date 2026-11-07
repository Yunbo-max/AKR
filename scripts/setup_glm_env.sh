#!/usr/bin/env bash
# CPU-only setup. Never loads GLM model weights or touches the core environment.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
core="${AKR_CORE_ENV:-/root/.venvs/akr}"
target="${AKR_GLM_ENV:-/root/.venvs/akr-glm}"
source="${AKR_GLM_SOURCE:-$repo/external/GLM-4-Voice}"
revision=eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a
if [[ "$(realpath -m "$core")" == "$(realpath -m "$target")" ]]; then
  echo 'GLM environment must differ from the core environment' >&2
  exit 1
fi
if [[ ! -x "$target/bin/python" ]]; then
  "$core/bin/python" -m venv "$target"
fi
# Own packages precede this path. Reuse the installed Blackwell-capable torch
# wheel and its CUDA libraries without reinstalling or mutating the core env.
core_site="$($core/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
target_site="$($target/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$core_site" "$repo/src" > "$target_site/akr_core_readonly.pth"
"$target/bin/python" -m pip install --ignore-installed --no-deps -r "$repo/requirements-glm.txt"
if [[ ! -d "$source/.git" ]]; then
  git clone https://github.com/zai-org/GLM-4-Voice.git "$source"
fi
if [[ -n "$(git -c "safe.directory=$source" -C "$source" status --porcelain -- . ':(exclude)**/__pycache__/**')" ]]; then
  echo 'Existing GLM source has modifications; refusing to replace them' >&2
  exit 1
fi
git -c "safe.directory=$source" -C "$source" fetch origin "$revision"
git -c "safe.directory=$source" -C "$source" checkout --detach "$revision"
CUDA_VISIBLE_DEVICES='' "$target/bin/python" -c 'import torch, transformers, tokenizers, tiktoken; print("GLM imports:", torch.__version__, transformers.__version__, tokenizers.__version__)'
echo "Optional GLM environment ready: $target/bin/python"
