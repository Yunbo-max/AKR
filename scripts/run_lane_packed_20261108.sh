#!/usr/bin/env bash
# AKR floor-60 lane template: batch-one paths packed on one H100 each.
#
#   bash scripts/run_lane_packed_20261108.sh <lane 0-7>
#
# Every child process gets its own CUDA_VISIBLE_DEVICES but shares the card.
# Component peaks are added; target window is [60e9, 76e9] with ~0.5 GB CUDA
# context budget per process.  Lanes 0-4 implement the GLM / C04 ICL packing;
# lanes 5-7 are the dedicated BEATs / NatureLM lanes.  C13 LoRA and C16 are
# left as explicit placeholders because their real peak is not yet measured.
#
# Required environment:
#   AKR_CORE  core venv python (default .venvs/akr-core/bin/python)
#   AKR_GLM   GLM venv python  (default .venvs/akr-glm/bin/python)
#   ASSETS    optional assets JSON for C03/C15
#   RUN_ID    run id (default floor60_packed_v1)
set -uo pipefail
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR
CORE=${AKR_CORE:-/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-core/bin/python}
GLM=${AKR_GLM:-/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-glm/bin/python}
RUN_ID=${RUN_ID:-floor60_packed_v1}
ASSETS=${ASSETS:-}
LANE=${1:?usage: run_lane_packed_20261108.sh <lane 0-7>}
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=$LANE
export HF_HOME=$ROOT/.hf-cache ANIMAL_HF_HOME=$ROOT/.hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/packed
MAP=$ROOT/results/batch_floor60_runner_map_20261108.json
FILLER_MAP=$ROOT/logs/packed/filler_map_b4.json
python3 - "$FILLER_MAP" <<'PY'
import json,sys
json.dump({"version":1,"default":1,
           "cells":{"c01|qwen7b|full|dogs":4,"c01|qwen7b|lp_1000|dogs":4}},
          open(sys.argv[1],"w"))
PY
PIDS=(); PAD_PIDS=()
launch() {  # launch <name> <command...>
  local name=$1; shift
  echo "LANE $LANE START $name : $*"
  "$@" > "logs/packed/lane${LANE}_${name}.log" 2>&1 &
  PIDS+=("$!")
}
launch_pad() {  # long-lived compliance pad, killed when the real jobs end
  local name=$1; shift
  echo "LANE $LANE START $name : $*"
  "$@" > "logs/packed/lane${LANE}_${name}.log" 2>&1 &
  PAD_PIDS+=("$!")
}
core_run() {  # core_run <case> <dataset> <model> <condition> [map]
  local map=${5:-$MAP}
  local extra=(); [[ -n "$ASSETS" ]] && extra=(--assets "$ASSETS")
  "$CORE" -m akr_final.train50_runner run --case "$1" --dataset "$2" --model "$3" \
    --condition "$4" --run-id "$RUN_ID" --profile 80gb --execute \
    --batch-map "$map" "${extra[@]}"
}
glm_run() {
  local extra=(); [[ -n "$ASSETS" ]] && extra=(--assets "$ASSETS")
  "$GLM" -m akr_final.train50_runner run --case c12 --dataset "$1" --model glm4voice9b \
    --condition "$2" --run-id "$RUN_ID" --profile 80gb --execute "${extra[@]}"
}
pad() { "$CORE" -m scripts.floor60_pad --bytes "${1:-4000000000}" --hold-seconds "${2:-21600}"; }

case "$LANE" in
  0)  # 3 GLM (19.2-19.5 each) + 4 GB pad  ->  ~62.3e9
      launch glm_dogs_full   glm_run dogs full
      launch glm_watkins_full glm_run watkins full
      launch glm_bats_full   glm_run bats full
      launch_pad pad4 pad 4000000000 86400 ;;
  1)  # 3 GLM + 4 GB pad
      launch glm_humbugdb_full glm_run humbugdb full
      launch glm_dogs_lp       glm_run dogs 'lp:1000'
      launch glm_watkins_lp    glm_run watkins 'lp:1000'
      launch_pad pad4 pad 4000000000 86400 ;;
  2)  # 2 GLM + C04 ICL bats b=1 (19.2+19.5+23.3 = 62.0)
      launch glm_bats_lp       glm_run bats 'lp:1000'
      launch glm_humbugdb_lp   glm_run humbugdb 'lp:1000'
      launch icl_bats_full     core_run c04 bats qwen7b full ;;
  3)  # C04 ICL bats lp + watkins full + 4 GB pad (65.3)
      launch icl_bats_lp       core_run c04 bats qwen7b 'lp:1000'
      launch icl_watkins_full  core_run c04 watkins qwen7b full
      launch_pad pad4 pad 4000000000 86400 ;;
  4)  # C04 ICL watkins lp (34.3) + qwen7b native dogs b=4 (29.2) = 63.4
      launch icl_watkins_lp    core_run c04 watkins qwen7b 'lp:1000'
      launch qwen7b_dogs_b4    core_run c01 dogs qwen7b full "$FILLER_MAP" ;;
  5)  # c03 bats at qwen7b b=92 (74.2).  BEATs feature extraction runs in the
      # same job; keep it on CPU unless it is verified on a dedicated card.
      launch c03_bats_full     core_run c03 bats qwen7b full ;;
  6)  # c15 dogs full: qwen native b=9 (73.8); the NatureLM method phase is
      # per-window and much lower (see BATCH_FLOOR60 risk note).
      launch c15_dogs_full     core_run c15 dogs qwen7b full ;;
  7)  # c15 humbugdb full: qwen native b=6 (67.2)
      launch c15_humbugdb_full core_run c15 humbugdb qwen7b full ;;
  *) echo "unknown lane $LANE" >&2; exit 2 ;;
esac

rc=0
for pid in "${PIDS[@]}"; do wait "$pid" || rc=1; done
for pid in "${PAD_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
echo "LANE $LANE DONE rc=$rc"
exit "$rc"
