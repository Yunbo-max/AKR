#!/usr/bin/env bash
set -euo pipefail

cd /root/animal-omni-kv

echo "[$(date -u +%FT%TZ)] validating extension artifacts"
.venv/bin/python scripts/finalize_extension_release.py
.venv/bin/python scripts/build_local_artifact_inventory.py --root . --output LOCAL_ARTIFACTS.md

echo "[$(date -u +%FT%TZ)] running tests and frozen audits"
.venv/bin/pytest -q
.venv/bin/python scripts/audit_artifacts.py --root . --output results/artifact_audit.json
.venv/bin/python scripts/audit_fair_gap.py --root . --output results/fair_gap_artifact_audit.json

echo "[$(date -u +%FT%TZ)] smoke-testing both published usage paths locally"
HF_HOME=/root/animal-omni-kv/.hf-cache .venv/bin/python scripts/predict_thinker_lora.py \
  --config configs/beans_watkins.yaml \
  --adapter results/lora_watkins_7b/epoch_1 \
  --audio data/beans/protocol/watkins/test/test_10.wav \
  > results/lora_watkins_7b_usage_smoke.json
HF_HOME=/root/animal-omni-kv/.hf-cache .venv/bin/python scripts/predict_thinker_lora.py \
  --config configs/beans_dogs.yaml \
  --adapter results/lora_beans_dogs_lp1_equal_support_k2_7b/epoch_1 \
  --audio data/beans/interventions/dogs_all/lp_0-1000/valid_4.wav \
  > results/lora_beans_dogs_lp1_equal_support_k2_7b_usage_smoke.json

echo "[$(date -u +%FT%TZ)] scanning trackable files for credential patterns"
if rg -n --hidden \
  -g '!.git/**' -g '!.hf-cache/**' -g '!.venv/**' -g '!data/**' \
  -g '!external/**' \
  '(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)' \
  .; then
  echo "credential-like material found; refusing to publish" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] initializing local Git repository"
if [[ ! -d .git ]]; then
  git init -b main
fi
git config user.name "Yunbo Long"
git config user.email "82950147+Yunbo-max@users.noreply.github.com"
git add .

largest_staged=$(git ls-files -z | xargs -0 -r stat -c '%s' | sort -nr | sed -n '1p')
if [[ -n "${largest_staged}" ]] && (( largest_staged > 100000000 )); then
  echo "tracked file exceeds GitHub's 100 MB limit: ${largest_staged} bytes" >&2
  exit 1
fi

if ! git diff --cached --quiet; then
  git commit -m "Initial reproducible animal audio grounding release"
fi

echo "[$(date -u +%FT%TZ)] creating or updating private GitHub repository"
if gh repo view Yunbo-max/animal-omni-kv >/dev/null 2>&1; then
  if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin https://github.com/Yunbo-max/animal-omni-kv.git
  fi
  git push -u origin main
else
  gh repo create Yunbo-max/animal-omni-kv \
    --private --source=. --remote=origin --push \
    --description "Equal-supervision diagnostics and KV adaptation for animal audio in Qwen2.5-Omni"
fi

local_git_head=$(git rev-parse HEAD)
remote_git_head=$(git ls-remote origin refs/heads/main | awk '{print $1}')
if [[ "$local_git_head" != "$remote_git_head" ]]; then
  echo "GitHub main does not match local HEAD" >&2
  exit 1
fi

HF=/root/.hf-cli-venv/bin/hf
echo "[$(date -u +%FT%TZ)] creating private Hugging Face model repositories"
"$HF" repos create humanlong/qwen2.5-omni-7b-watkins-lora \
  --type model --private --exist-ok
"$HF" repos create humanlong/qwen2.5-omni-7b-dogs-k2-lora \
  --type model --private --exist-ok

upload_model_file() {
  local repo_id="$1"
  local local_path="$2"
  local remote_path="$3"
  "$HF" upload "$repo_id" "$local_path" "$remote_path" \
    --commit-message "Upload ${remote_path}"
}

echo "[$(date -u +%FT%TZ)] uploading Watkins adapter and documentation"
WATKINS=humanlong/qwen2.5-omni-7b-watkins-lora
upload_model_file "$WATKINS" results/lora_watkins_7b/epoch_1/README.md README.md
upload_model_file "$WATKINS" results/lora_watkins_7b/epoch_1/adapter_config.json adapter_config.json
upload_model_file "$WATKINS" results/lora_watkins_7b/epoch_1/adapter_model.safetensors adapter_model.safetensors
upload_model_file "$WATKINS" results/lora_watkins_7b/history.json training_history.json
upload_model_file "$WATKINS" results/beans_watkins_lora_7b_summary.json evaluation_summary.json
upload_model_file "$WATKINS" configs/beans_watkins.yaml task_config.yaml
upload_model_file "$WATKINS" scripts/predict_thinker_lora.py predict_thinker_lora.py

echo "[$(date -u +%FT%TZ)] uploading Dogs negative-control adapter and documentation"
DOGS=humanlong/qwen2.5-omni-7b-dogs-k2-lora
upload_model_file "$DOGS" results/lora_beans_dogs_lp1_equal_support_k2_7b/epoch_1/README.md README.md
upload_model_file "$DOGS" results/lora_beans_dogs_lp1_equal_support_k2_7b/epoch_1/adapter_config.json adapter_config.json
upload_model_file "$DOGS" results/lora_beans_dogs_lp1_equal_support_k2_7b/epoch_1/adapter_model.safetensors adapter_model.safetensors
upload_model_file "$DOGS" results/lora_beans_dogs_lp1_equal_support_k2_7b/history.json training_history.json
upload_model_file "$DOGS" results/lora_beans_dogs_lp1_equal_support_k2_7b_valid_summary.json evaluation_summary.json
upload_model_file "$DOGS" configs/beans_dogs.yaml task_config.yaml
upload_model_file "$DOGS" scripts/predict_thinker_lora.py predict_thinker_lora.py

echo "[$(date -u +%FT%TZ)] downloading adapters back for byte-level verification"
mkdir -p /tmp/animal-omni-hf-verify-watkins /tmp/animal-omni-hf-verify-dogs
"$HF" download "$WATKINS" adapter_model.safetensors \
  --local-dir /tmp/animal-omni-hf-verify-watkins
"$HF" download "$DOGS" adapter_model.safetensors \
  --local-dir /tmp/animal-omni-hf-verify-dogs
cmp results/lora_watkins_7b/epoch_1/adapter_model.safetensors \
  /tmp/animal-omni-hf-verify-watkins/adapter_model.safetensors
cmp results/lora_beans_dogs_lp1_equal_support_k2_7b/epoch_1/adapter_model.safetensors \
  /tmp/animal-omni-hf-verify-dogs/adapter_model.safetensors

echo "[$(date -u +%FT%TZ)] release publication complete"
git status --short
git log -1 --oneline
gh repo view Yunbo-max/animal-omni-kv --json url,isPrivate,nameWithOwner
"$HF" repos list --search qwen2.5-omni-7b --format json
