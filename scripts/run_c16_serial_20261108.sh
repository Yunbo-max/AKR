#!/usr/bin/env bash
# Wave 4 single-GPU C16: start the 12e9 floor pad from MAIN, then run the
# historical_rerun_v1 core rerun + rebuild + C16 check/readout in AKR-fixes-wt.
set -uo pipefail
MAIN=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR
FIX=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR-fixes-wt
CORE=/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-core/bin/python
mkdir -p "$FIX/logs/historical_rerun"
cd "$MAIN" || exit 1
"$CORE" -m scripts.floor60_pad --bytes "${C16_PAD_BYTES:-12000000000}" --hold-seconds 86400 \
  > "$FIX/logs/historical_rerun/pad_final_v1_20261108.log" 2>&1 &
PAD=$!
trap 'kill "$PAD" 2>/dev/null || true' EXIT
cd "$FIX" || exit 1
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT
export C16_PAD_BYTES=12000000000
RUN_C16=1 bash historical_rerun/run_historical.sh > logs/historical_rerun/run_final_v1_20261108.log 2>&1
rc=$?
kill "$PAD" 2>/dev/null || true
exit "$rc"
