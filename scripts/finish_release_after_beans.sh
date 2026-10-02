#!/usr/bin/env bash
set -euo pipefail

cd /root/animal-omni-kv

if (( $# != 0 )); then
  echo "usage: $0" >&2
  exit 2
fi

echo "[$(date -u +%FT%TZ)] observing BEANS-Zero checkpoint completeness"
while true; do
  if .venv/bin/python - <<'PY'
import csv
from collections import Counter
from pathlib import Path
import sys

predictions = Path("results/beans_zero_targets_fullscan_cap10_qwen7b.csv")
manifest = Path("data/manifests/beans_zero_targets_fullscan_cap10.csv")
with predictions.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
with manifest.open(newline="", encoding="utf-8") as handle:
    source = list(csv.DictReader(handle))
ids = [row["event_id"] for row in rows]
counts = Counter(ids)
duplicates = {event_id: count for event_id, count in counts.items() if count != 1}
if len(source) != 2950 or len(rows) > len(source):
    raise RuntimeError(f"BEANS-Zero cardinality invalid: predictions={len(rows)}, manifest={len(source)}")
if duplicates:
    raise RuntimeError(f"BEANS-Zero contains duplicate event IDs: {duplicates}")
if ids != [row["event_id"] for row in source[:len(rows)]]:
    raise RuntimeError("BEANS-Zero prediction prefix differs from the manifest")
if len(rows) != 2950:
    print(f"BEANS-Zero checkpoint incomplete: {len(rows)}/2950")
    sys.exit(3)
print("BEANS-Zero checkpoint complete and unique: 2950/2950")
PY
  then
    break
  else
    checkpoint_status="$?"
    if [[ "$checkpoint_status" != "3" ]]; then
      echo "BEANS-Zero checkpoint validation failed" >&2
      exit "$checkpoint_status"
    fi
  fi
  sleep 30
done

.venv/bin/python scripts/summarize_beans_zero.py \
  --predictions results/beans_zero_targets_fullscan_cap10_qwen7b.csv \
  --output results/beans_zero_targets_fullscan_cap10_qwen7b_summary.json

.venv/bin/python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path(
    "results/beans_zero_targets_fullscan_cap10_qwen7b_summary.json"
).read_text(encoding="utf-8"))
components = [key for key in payload if key != "overall"]
if payload.get("overall", {}).get("n") != 2950 or len(components) != 12:
    raise RuntimeError(
        f"BEANS-Zero summary mismatch: overall={payload.get('overall')}, "
        f"components={len(components)}"
    )
print("BEANS-Zero summary complete: 12 components")
PY

echo "[$(date -u +%FT%TZ)] waiting for 60 seconds of continuous GPU idleness"
idle_seconds=0
while (( idle_seconds < 60 )); do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    idle_seconds=$((idle_seconds + 10))
  else
    idle_seconds=0
  fi
  sleep 10
done

echo "[$(date -u +%FT%TZ)] starting verified release publication"
bash scripts/publish_release.sh
