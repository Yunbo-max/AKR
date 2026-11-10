#!/usr/bin/env bash
# Flock-singleton self-restarting wrapper for the AKR final_v1 submitter.
# Only one wrapper can run; a second exits immediately. The python submitter is
# restarted after any crash/poll-failure exit.
set -uo pipefail
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR
PY=/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-core/bin/python
LOCK=$ROOT/results/run_specs_20261108/submitter.loop.lock
mkdir -p "$ROOT/logs/run"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMITTER-LOOP already running; exit $$" >> "$ROOT/logs/run/serial_submitter.loop.log"
  exit 0
fi
cd "$ROOT" || exit 1
while true; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMITTER-LOOP start pid=$$" >> logs/run/serial_submitter.loop.log
  "$PY" scripts/serial_submit_final_v1_20261108.py >> logs/run/serial_submitter.log 2>> logs/run/submitter.err.log
  rc=$?
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMITTER EXIT rc=$rc restart-in-30s" >> logs/run/serial_submitter.loop.log
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMITTER EXIT rc=$rc restart-in-30s" >> run_final_v1_20261108.log
  sleep 30
done
