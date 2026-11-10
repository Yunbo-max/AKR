#!/usr/bin/env bash
# AKR final_v1 packed wave launcher.  Each lane (GPU) may host several cells
# concurrently plus a floor pad, following scripts/run_lane_packed_20261108.sh:
#   GLM 3 per GPU + ~4GB pad, C04 dataset pairs + pad to the 60GB floor.
# Spec: {"name":..., "lanes":[{"cells":[cell,...],"pad_bytes":N}, null, ...]}
#   cell = {"case","dataset","model","condition","profile","assets"?,"batch_map"?}
set -uo pipefail
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR
CORE=/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-core/bin/python
GLM=/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-glm/bin/python
SPEC=${1:?usage: run_wave_packed_20261108.sh <spec.json>}
RUN_ID=${RUN_ID:-final_v1}
SEED=${SEED:-20261107}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
cd "$ROOT" || exit 1
export HF_HOME=$ROOT/.hf-cache ANIMAL_HF_HOME=$ROOT/.hf-cache
export PYTHONPATH=$ROOT/results/run_specs_20261108/envfix:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# qz exports distributed env for multi-GPU jobs; every lane is single-process.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT
JOB=$("$CORE" - "$SPEC" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['name'])
PY
)
JDIR="$ROOT/logs/run/$JOB"; mkdir -p "$JDIR"
STATUS_LOG=$ROOT/run_final_v1_20261108.log
say(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$JDIR/driver_lines.log" >> "$STATUS_LOG"; }
say "JOB $JOB START spec=$SPEC"

run_cell(){  # run_cell <lane> <slot> <case> <ds> <model> <cond> <profile> <assets> <bmap> <script>
  local i=$1 slot=$2 case=$3 ds=$4 model=$5 cond=$6 profile=$7 assets=$8 bmap=$9 script=${10}
  local py=$CORE; [[ "$case" == c12 ]] && py=$GLM
  local tag="${case}_${ds}_${model}_${cond//:/_}"
  local log="$JDIR/lane${i}_${tag}.log"
  local cell="$ROOT/results/akr_train50/$RUN_ID/$case/${ds}__${model}__${cond//:/_}__s${SEED}"
  local status="$cell/STATUS.json"
  local rc=1
  for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
    if [[ -s "$status" ]]; then
      local done_flag
      done_flag=$("$CORE" - "$status" <<'PY'
import json,sys
try: print('1' if json.load(open(sys.argv[1])).get('complete') else '0')
except Exception: print('0')
PY
)
      if [[ "$done_flag" == 1 ]]; then echo "LANE $i $tag SKIP attempt=$attempt already complete"; rc=0; break; fi
    fi
    echo "LANE $i SLOT $slot $tag ATTEMPT $attempt START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local cmd=("$py" "$script" run --execute --dataset "$ds" --model "$model" --condition "$cond"
               --run-id "$RUN_ID" --seed "$SEED" --profile "$profile" --batch-map "$bmap")
    [[ -n "$assets" ]] && cmd+=(--assets "$assets")
    "${cmd[@]}" > "$log" 2>&1
    rc=$?
    echo "LANE $i SLOT $slot $tag ATTEMPT $attempt EXIT $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    [[ $rc -eq 0 ]] && break
    [[ $attempt -lt 3 ]] && sleep 20
  done
  return $rc
}

NL=$("$CORE" - "$SPEC" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))['lanes']))
PY
)
for ((i=0; i<NL; i++)); do
  mapfile -t CELLS < <("$CORE" - "$SPEC" "$i" <<'PY'
import json,sys,pathlib
lane=json.load(open(sys.argv[1]))['lanes'][int(sys.argv[2])]
if not lane: raise SystemExit
print(lane.get('pad_bytes') or 0)
for c in lane.get('cells') or []:
    case=c['case']; script=sorted(pathlib.Path('experiments/rebuttal_cases').glob(f'{case}_*.py'))[0].as_posix()
    print('\t'.join([case,c['dataset'],c['model'],c['condition'],c.get('profile','48gb'),
                     c.get('assets') or '-',c.get('batch_map') or 'results/batch_floor60_runner_map_20261108.json',script]))
PY
)
  [[ ${#CELLS[@]} -eq 0 ]] && continue
  PAD=${CELLS[0]}
  (
    export CUDA_VISIBLE_DEVICES=$i
    pad_pid=""
    if [[ "$PAD" != 0 ]]; then
      "$CORE" -m scripts.floor60_pad --bytes "$PAD" --hold-seconds 86400 > "$JDIR/lane${i}_pad.log" 2>&1 &
      pad_pid=$!
    fi
    pids=()
    for slot in "${!CELLS[@]}"; do
      [[ $slot -eq 0 ]] && continue
      IFS=$'\t' read -r case ds model cond profile assets bmap script <<< "${CELLS[$slot]}"
      [[ "$assets" == "-" ]] && assets=""
      run_cell "$i" "$slot" "$case" "$ds" "$model" "$cond" "$profile" "$assets" "$bmap" "$script" &
      pids+=("$!")
    done
    lrc=0
    for p in "${pids[@]}"; do wait "$p" || lrc=1; done
    [[ -n "$pad_pid" ]] && { kill "$pad_pid" 2>/dev/null || true; wait "$pad_pid" 2>/dev/null || true; }
    exit $lrc
  ) &
  LANE_PIDS+=("$!")
  echo "LANE $i LAUNCHED pid=${LANE_PIDS[-1]} cells=$(( ${#CELLS[@]} - 1 )) pad=$PAD"
done
rc=0
for pid in "${LANE_PIDS[@]}"; do wait "$pid" || rc=1; done
say "JOB $JOB DONE rc=$rc"
exit "$rc"
