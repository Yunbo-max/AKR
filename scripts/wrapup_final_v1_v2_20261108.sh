#!/usr/bin/env bash
# Final_v1 wrap-up: wait for queue drain (or 24h guard), build the grid report,
# commit/push all cell summaries, and log the independent cross-check.
set -uo pipefail
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR
PY=/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-core/bin/python
cd "$ROOT" || exit 1
DEADLINE=$(( $(date +%s) + 259200 ))
while true; do
  DONE=$("$PY" - <<'PY'
import json,sys
from pathlib import Path
s=json.load(open('results/run_specs_20261108/serial_state.json'))
c=s['cells']; c16=s.get('c16',{}).get('status')
open_cells=[t for t,e in c.items() if e.get('status') not in ('OK','FAILED')]
print('1' if (not open_cells and c16 in ('OK','FAILED')) else '0')
PY
)
  if [[ "$DONE" == "1" ]]; then break; fi
  if [[ $(date +%s) -ge $DEADLINE ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WRAPUP deadline reached with open cells; proceeding with current state" >> run_final_v1_20261108.log
    break
  fi
  sleep 300
done
"$PY" scripts/build_final_grid_report_20261108.py > logs/run/final_grid_report.log 2>&1
"$PY" - <<'PY'
import json,subprocess
from pathlib import Path
ROOT=Path('.')
files=[]
for p in (ROOT/'results/akr_train50/final_v1').rglob('*'):
    if p.is_file() and p.name in ('REPORT.json','RESULTS.csv','STATUS.json','CASE.json'):
        files.append(str(p))
(ROOT/'logs/run/final_grid_files.txt').write_text('\n'.join(sorted(files))+'\n')
print('receipt files',len(files))
PY
git add --pathspec-from-file=logs/run/final_grid_files.txt
git add docs/rebuttal/FINAL_GRID_20261110.md scripts/build_final_grid_report_20261108.py scripts/wrapup_final_v1_20261108.sh 2>/dev/null || true
git commit -m "Add final_v1 grid report and all cell summaries" >> logs/run/wrapup_git.log 2>&1 || true
PUSHED=0
for i in 1 2 3 4 5; do
  if git push origin research/rebuttal-train50-cases-20261107 >> logs/run/wrapup_git.log 2>&1; then PUSHED=1; break; fi
  sleep 30
done
COMMIT=$(git rev-parse --short HEAD)
OKCOUNT=$("$PY" - <<'PY'
import json
s=json.load(open('results/run_specs_20261108/serial_state.json'))
print(sum(1 for e in s['cells'].values() if e.get('status')=='OK'))
PY
)
REPORTS=$(git ls-tree -r origin/research/rebuttal-train50-cases-20261107 --name-only 2>/dev/null | grep -c 'results/akr_train50/final_v1/.*/REPORT.json')
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WRAPUP commit=$COMMIT pushed=$PUSHED ok_cells=$OKCOUNT github_reports=$REPORTS" >> run_final_v1_20261108.log
"$PY" - "$COMMIT" "$PUSHED" "$OKCOUNT" "$REPORTS" <<'PY'
import json,sys
from pathlib import Path
json.dump(dict(commit=sys.argv[1],pushed=sys.argv[2],ok_cells=int(sys.argv[3]),github_reports=int(sys.argv[4])),
          open('results/run_specs_20261108/wrapup_summary.json','w'),indent=1)
PY
echo "WRAPUP_DONE commit=$COMMIT pushed=$PUSHED ok=$OKCOUNT github_reports=$REPORTS"
