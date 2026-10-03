"""Safe release entrypoint: inventory/plan by default; explicit bounded execution."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def load_registry(path: Path) -> dict:
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema_version')!='1.0':raise ValueError('Unsupported registry schema')
    if set(value['default_tasks'])-{'S1','F1','F2','F3'}:raise ValueError('Unknown default task')
    for key in ['qwen7b','qwen3b']:
        if not re.fullmatch(r'[0-9a-f]{40}',value['models'][key]['revision']):
            raise ValueError('Core models require an immutable revision')
    return value


def make_command(root: Path, run_id: str, tasks: list[str], profile: str, execute: bool) -> list[str]:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_id):raise ValueError('Invalid run ID')
    if not tasks or len(set(tasks))!=len(tasks) or set(tasks)-{'S1','F1','F2','F3'}:
        raise ValueError('Choose a distinct subset of S1,F1,F2,F3')
    if profile not in {'16gb','24gb','48gb'}:raise ValueError('Unknown memory profile')
    cmd=[sys.executable,str(root/'scripts/run_akr_standalone.py'),'--root',str(root),
         '--config',str(root/'configs/akr_standalone.yaml'),'--run-id',run_id,
         '--profile',profile,'--tasks',','.join(tasks)]
    return cmd+(['--execute'] if execute else [])


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['plan','check','run'],nargs='?',default='plan')
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--run-id',default='akr_final_s1_v1')
    ap.add_argument('--tasks',default='S1')
    ap.add_argument('--profile',choices=['16gb','24gb','48gb'],default='24gb')
    ap.add_argument('--budget-hours',type=float,default=24.)
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args(argv);root=args.root.resolve()
    registry=load_registry(root/'configs/akr_registry.json')
    tasks=args.tasks.split(',')
    command=make_command(root,args.run_id,tasks,args.profile,args.execute)
    if not math.isfinite(args.budget_hours) or not 0<args.budget_hours<=24:
        raise ValueError('Budget must be finite and within (0,24] hours')
    if args.action=='plan':
        print(json.dumps({'tasks':{t:registry['experiments'][t] for t in tasks},
                          'models':registry['models'],'command':command,
                          'raw_audio_required':True,'old_RUN_LOCK_required':False,
                          'gpu_started':False},indent=2));return 0
    env=os.environ.copy();env['PYTHONPATH']=str(root/'src')+os.pathsep+str(root)
    env.setdefault('OMP_NUM_THREADS','1');env.setdefault('TOKENIZERS_PARALLELISM','false')
    # Always run file/hash/partition preflight before any model load.
    preflight=make_command(root,args.run_id,tasks,args.profile,False)
    start=time.monotonic()
    status=subprocess.run(preflight,cwd=root,env=env,check=False).returncode
    if status or args.action=='check' or not args.execute:return status
    remaining=args.budget_hours*3600-(time.monotonic()-start)
    if remaining<=0: return 124
    proc=subprocess.Popen(command,cwd=root,env=env,start_new_session=True)
    try:
        return proc.wait(timeout=remaining)
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        print('Budget/interrupt stop: keep partial journals; no partial aggregate is a completed result.',file=sys.stderr)
        return 124

if __name__=='__main__':raise SystemExit(main())
