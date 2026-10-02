#!/usr/bin/env bash
set -uo pipefail

cd /root/animal-omni-kv
exec >> results/post_queue_protocol_controls.log 2>&1

queue_pid=${1:?usage: run_post_queue_protocol_controls.sh QUEUE_PID}
echo "[$(date -u +%FT%TZ)] wait for primary queue pid=${queue_pid}"
while kill -0 "$queue_pid" 2>/dev/null; do
  sleep 20
done

echo "[$(date -u +%FT%TZ)] primary queue ended; start support-order controls"
bash scripts/run_marm_icl_order_controls.sh

echo "[$(date -u +%FT%TZ)] start Watkins K=4/class feasibility gate"
bash scripts/run_watkins_k4_icl_gate.sh

echo "[$(date -u +%FT%TZ)] rebuild statistics, tables, figures, and audits"
.venv/bin/python scripts/analyze_fair_gap_statistics.py --root . \
  --output results/fair_gap_paired_statistics.json || true
.venv/bin/python scripts/build_fair_gap_tables.py --root . \
  --output-dir results/fair_gap_tables || true
.venv/bin/python scripts/plot_fair_gap_results.py --root . \
  --output-dir figures || true
.venv/bin/python scripts/build_paper_tables.py --root . \
  --output-dir results/paper_tables || true
.venv/bin/python scripts/audit_fair_gap.py --root . \
  --output results/fair_gap_artifact_audit.json || true
.venv/bin/python scripts/audit_artifacts.py --root . \
  --output results/artifact_audit.json || true
.venv/bin/python -m pytest -q
echo "[$(date -u +%FT%TZ)] post-queue protocol controls finished"
