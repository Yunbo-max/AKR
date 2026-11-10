#!/usr/bin/env bash
# AKR final_v1 wave launcher: one 8-GPU job, one cell per lane (or packed lane
# with a floor pad).  Spec JSON: list of 8 lane objects.
#   {"name": "...", "lanes": [{"case","dataset","model","condition","profile",
#    "assets"?, "batch_map"?, "pad_bytes"?}, ...]}
set -uo pipefail
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR
CORE=/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-core/bin/python
GLM=/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-glm/bin/python
SPEC=${1:?usage: run_wave_cells_20261108.sh <spec.json>}
RUN_ID=${RUN_ID:-final_v1}
SEED=${SEED:-20261107}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
DEFAULT_MAP=results/batch_floor60_runner_map_20261108.json
cd "$ROOT" || exit 1
export HF_HOME=$ROOT/.hf-cache ANIMAL_HF_HOME=$ROOT/.hf-cache
export PYTHONPATH=$ROOT/results/run_specs_20261108/envfix:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# qz exports distributed env for multi-GPU jobs; every lane is single-process.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT
JOB=$("$CORE" - "$SPEC" <<'PY'
import json,sys
spec=json.load(open(sys.argv[1]));print(spec['name'])
PY
)
JDIR="$ROOT/logs/run/$JOB"
mkdir -p "$JDIR"
STATUS_LOG=$ROOT/run_final_v1_20261108.log
say(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$JDIR/driver_lines.log" >> "$STATUS_LOG"; }
say "JOB $JOB START spec=$SPEC"

LANE_PIDS=()
PAD_PIDS=()
launch_lane(){
  local i=$1
  local line
  line=$("$CORE" - "$SPEC" "$i" <<'PY'
import json,sys
spec=json.load(open(sys.argv[1])); lane=spec['lanes'][int(sys.argv[2])]
case=lane['case']; ds=lane['dataset']; model=lane['model']; cond=lane['condition']
profile=lane.get('profile','80gb'); assets=lane.get('assets') or '-'
bmap=lane.get('batch_map') or 'results/batch_floor60_runner_map_20261108.json'
pad=str(lane.get('pad_bytes') or 0)
import pathlib
matches=sorted(pathlib.Path('experiments/rebuttal_cases').glob(f'{case}_*.py'))
assert len(matches)==1, (case,matches)
print('\t'.join([case,ds,model,cond,profile,assets,bmap,pad,matches[0].as_posix()]))
PY
)
  IFS=$'\t' read -r case ds model cond profile assets bmap pad script <<< "$line"
  [[ "$assets" == "-" ]] && assets=""
  local py=$CORE
  [[ "$case" == c12 ]] && py=$GLM
  local tag="${case}_${ds}_${model}_${cond//:/_}"
  local log="$JDIR/lane${i}_${tag}.log"
  local cell="$ROOT/results/akr_train50/$RUN_ID/$case/${ds}__${model}__${cond//:/_}__s${SEED}"
  local status="$cell/STATUS.json"
  (
    export CUDA_VISIBLE_DEVICES=$i
    pad_pid=""
    if [[ "$pad" != 0 ]]; then
      "$CORE" -m scripts.floor60_pad --bytes "$pad" --hold-seconds 86400 > "$JDIR/lane${i}_${tag}_pad.log" 2>&1 &
      pad_pid=$!
    fi
    rc=1
    for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
      if [[ -s "$status" ]]; then
        done_flag=$("$CORE" - "$status" <<'PY'
import json,sys
try: print('1' if json.load(open(sys.argv[1])).get('complete') else '0')
except Exception: print('0')
PY
)
        if [[ "$done_flag" == 1 ]]; then
          echo "LANE $i $tag SKIP attempt=$attempt already complete"
          rc=0; break
        fi
      fi
      echo "LANE $i $tag ATTEMPT $attempt START $(date -u +%Y-%m-%dT%H:%M:%SZ) script=$script"
      cmd=("$py" "$script" run --execute --dataset "$ds" --model "$model" --condition "$cond"
           --run-id "$RUN_ID" --seed "$SEED" --profile "$profile" --batch-map "$bmap")
      [[ -n "$assets" ]] && cmd+=(--assets "$assets")
      "${cmd[@]}" > "$log" 2>&1
      rc=$?
      echo "LANE $i $tag ATTEMPT $attempt EXIT $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      if [[ $rc -eq 0 ]]; then break; fi
      if [[ $attempt -lt 3 ]]; then sleep 20; fi
    done
    [[ -n "$pad_pid" ]] && { kill "$pad_pid" 2>/dev/null || true; wait "$pad_pid" 2>/dev/null || true; }
    exit $rc
  ) &
  LANE_PIDS+=("$!")
  echo "LANE $i $tag LAUNCHED pid=${LANE_PIDS[-1]}"
}

NL=$("$CORE" - "$SPEC" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))['lanes']))
PY
)
for ((i=0; i<NL; i++)); do launch_lane "$i"; done
rc=0
for pid in "${LANE_PIDS[@]}"; do wait "$pid" || rc=1; done
say "JOB $JOB DONE rc=$rc"
exit "$rc"
