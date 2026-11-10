#!/usr/bin/env bash
# Self-restarting wrapper for the AKR final_v1 submitter (fixed 60s polling).
# Any crash, poll-failure exit or kill is followed by a 30s pause and restart.
set -uo pipefail
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR
PY=/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-core/bin/python
cd "$ROOT" || exit 1
while true; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMITTER-LOOP start pid=$$" >> logs/run/serial_submitter.loop.log
  "$PY" scripts/serial_submit_final_v1_20261108.py >> logs/run/serial_submitter.log 2>> logs/run/submitter.err.log
  rc=$?
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMITTER EXIT rc=$rc restart-in-30s" >> logs/run/serial_submitter.loop.log
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMITTER EXIT rc=$rc restart-in-30s" >> run_final_v1_20261108.log
  sleep 30
done
