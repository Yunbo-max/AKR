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


def validate_default_config(root: Path, registry: dict) -> None:
    """Ensure the displayed release pins are the ones the runner will load."""
    import yaml
    cfg=yaml.safe_load((root/'configs/akr_standalone.yaml').read_text(encoding='utf-8'))
    for tag,key in [('qwen7b','model_7b'),('qwen3b','model_3b')]:
        actual=cfg.get(key,{})
        for field in ['id','revision']:
            if actual.get(field)!=registry['models'][tag][field]:
                raise ValueError(f'{key} {field} differs from release registry; declare a new protocol before changing pins')
    s1=registry['experiments']['S1']
    fixed=cfg.get('fixed_settings',{})
    expected={'rank':s1['fixed_rank'],'alpha':s1['relative_alpha'],'ridge_alpha':s1['ridge_alpha']}
    if fixed!=expected or cfg.get('seeds')!=[s1['seed']] or cfg.get('k_per_class')!=s1['support_per_class']:
        raise ValueError('Standalone numerical settings differ from release registry')
    if cfg.get('labels')!=registry['datasets']['MA-CT']['labels'] or cfg.get('conditions')!=s1['conditions']:
        raise ValueError('Standalone labels/conditions differ from release registry')


def make_command(root: Path, run_id: str, tasks: list[str], profile: str, execute: bool) -> list[str]:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_id):raise ValueError('Invalid run ID')
    if not tasks or len(set(tasks))!=len(tasks) or set(tasks)-{'S1','F1','F2','F3'}:
        raise ValueError('Choose a distinct subset of S1,F1,F2,F3')
    if profile not in {'16gb','24gb','48gb'}:raise ValueError('Unknown memory profile')
    cmd=[sys.executable,str(root/'scripts/run_akr_standalone.py'),'--root',str(root),
         '--config',str(root/'configs/akr_standalone.yaml'),'--run-id',run_id,
         '--profile',profile,'--tasks',','.join(tasks)]
    return cmd+(['--execute'] if execute else [])


def _bounded(command: list[str], root: Path, env: dict, deadline: float) -> int:
    """Apply one shared deadline to preflight and execution; retain checkpoints."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 124
    proc = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
    old_term = signal.getsignal(signal.SIGTERM)
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        return proc.wait(timeout=remaining)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        print('Stopped: checkpoints retained; incomplete tasks have no final score.', file=sys.stderr)
        return 124
    finally:
        signal.signal(signal.SIGTERM, old_term)


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
    validate_default_config(root,registry)
    tasks=args.tasks.split(',')
    command=make_command(root,args.run_id,tasks,args.profile,args.execute)
    if not math.isfinite(args.budget_hours) or not 0<args.budget_hours<=24:
        raise ValueError('Budget must be finite and within (0,24] hours')
    if args.action=='plan':
        print(json.dumps({'tasks':{t:registry['experiments'][t] for t in tasks},
                          'models':registry['models'],'command':command,
                          'raw_audio_required':True,'old_RUN_LOCK_required':False,
                          'gpu_started':False},indent=2));return 0
    if args.action == 'run' and not args.execute:
        print('Use run --execute to authorize model loading; use plan/check otherwise.', file=sys.stderr)
        return 2
    env=os.environ.copy();env['PYTHONPATH']=str(root/'src')+os.pathsep+str(root)
    env.setdefault('OMP_NUM_THREADS','1');env.setdefault('TOKENIZERS_PARALLELISM','false')
    # Preflight shares the time cap, rather than running outside an unbounded wait.
    deadline=time.monotonic()+args.budget_hours*3600
    preflight=make_command(root,args.run_id,tasks,args.profile,False)
    status=_bounded(preflight,root,env,deadline)
    if status or args.action=='check':return status
    return _bounded(command,root,env,deadline)

if __name__=='__main__':raise SystemExit(main())
