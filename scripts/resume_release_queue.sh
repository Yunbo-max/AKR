#!/usr/bin/env bash
set -euo pipefail

cd /root/animal-omni-kv

if (( $# != 1 )); then
  echo "usage: $0 <current-gpu-pid|0-to-start-now>" >&2
  exit 2
fi

current_gpu_pid="$1"
if [[ "$current_gpu_pid" != "0" ]]; then
  echo "[$(date -u +%FT%TZ)] waiting for MarmAudio ICL PID ${current_gpu_pid}"
  while kill -0 "$current_gpu_pid" 2>/dev/null; do
    sleep 30
  done
else
  echo "[$(date -u +%FT%TZ)] no predecessor PID; starting immediately"
fi

echo "[$(date -u +%FT%TZ)] validating completed MarmAudio ICL artifact"
.venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path

predictions = Path("results/marmaudio_equal_support_audio_icl_k2_7b.csv")
summary = Path("results/marmaudio_equal_support_audio_icl_k2_7b_summary.json")
with predictions.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 75:
    raise RuntimeError(f"MarmAudio K=2 ICL incomplete: {len(rows)}/75")
payload = json.loads(summary.read_text(encoding="utf-8"))
if payload.get("results", {}).get("2", {}).get("n_query") != 75:
    raise RuntimeError("MarmAudio K=2 ICL summary does not cover 75 queries")
print("MarmAudio K=2 ICL complete: 75/75")
PY

echo "[$(date -u +%FT%TZ)] resuming BEANS-Zero Qwen-7B evaluation"
HF_HOME=/root/animal-omni-kv/.hf-cache HF_HUB_OFFLINE=1 \
  .venv/bin/python scripts/evaluate_beans_zero.py \
  --manifest data/manifests/beans_zero_targets_fullscan_cap10.csv \
  --model-id Qwen/Qwen2.5-Omni-7B \
  --output results/beans_zero_targets_fullscan_cap10_qwen7b.csv \
  --max-new-tokens 32 \
  --resume

echo "[$(date -u +%FT%TZ)] summarizing and validating BEANS-Zero"
.venv/bin/python scripts/summarize_beans_zero.py \
  --predictions results/beans_zero_targets_fullscan_cap10_qwen7b.csv \
  --output results/beans_zero_targets_fullscan_cap10_qwen7b_summary.json
.venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path

predictions = Path("results/beans_zero_targets_fullscan_cap10_qwen7b.csv")
summary = Path("results/beans_zero_targets_fullscan_cap10_qwen7b_summary.json")
with predictions.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 2950:
    raise RuntimeError(f"BEANS-Zero incomplete: {len(rows)}/2950")
payload = json.loads(summary.read_text(encoding="utf-8"))
components = [key for key in payload if key != "overall"]
if payload.get("overall", {}).get("n") != 2950 or len(components) != 12:
    raise RuntimeError(
        f"BEANS-Zero summary mismatch: overall={payload.get('overall')}, "
        f"components={len(components)}"
    )
print("BEANS-Zero complete: 2950/2950 across 12 components")
PY

echo "[$(date -u +%FT%TZ)] starting verified release publication"
bash scripts/publish_release.sh
