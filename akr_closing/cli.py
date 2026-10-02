"""CLI; --execute is required to start any GPU work."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
import json
import re
from pathlib import Path
import signal
import sys
import yaml
from .core import atomic_json,digest,file_hash


def validate_run_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value):
        raise ValueError('run-id must be a nonempty simple name, not a path')
    return value


def load_config(root,path):
    cfg=yaml.safe_load(path.read_text())
    if cfg.get('router_implementation')!='repository':raise ValueError('production CLI requires the existing ConditionalGradientRouter')
    for ds in cfg['datasets']:
        if 'labels_from_config' in ds:
            ds['labels']=yaml.safe_load((root/ds['labels_from_config']).read_text())['dataset']['labels']
    return cfg


@contextmanager
def run_lock(path):
    import fcntl
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('another AKR runner holds this repository lock') from exc
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def environment():
    versions={}
    for p in ['torch','transformers','numpy','scipy','scikit-learn','peft','qwen-omni-utils']:
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]=None
    return {'python':sys.version,'packages':versions}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--config',type=Path)
    ap.add_argument('--run-id',default='akr_closing_v1')
    ap.add_argument('--tasks',default='E1,E3,E2,E4,E5,E7,E8',help='ordered E1..E9, or scaling; E6/E9 optional')
    ap.add_argument('--profile',choices=['16gb','24gb','48gb'],default='24gb')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--resume',action='store_true',help='resumption always uses immutable fingerprints')
    ap.add_argument('--report-only',action='store_true')
    args=ap.parse_args(argv);root=args.root.resolve()
    validate_run_id(args.run_id)
    tasks=args.tasks.split(',')
    if any(t not in {f'E{i}' for i in range(1,10)}|{'scaling'} for t in tasks):raise ValueError('unknown task')
    out=root/'results'/'akr_closing'/args.run_id;out.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        from .report import report_run
        print(json.dumps(report_run(out),indent=2));return 0
    cfg=load_config(root,args.config or root/'configs/akr_closing.yaml')
    from .pipeline import Study,SafeStop,request_stop
    from .backend import OmniBackend
    import akr_closing.pipeline as pipe
    pipe.STOP=False
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    identity={'config':cfg,'profile':args.profile,'code':{p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'environment':environment()}
    identitypath=out/'EXECUTION_IDENTITY.json'
    if identitypath.exists() and json.loads(identitypath.read_text())!=identity:
        raise RuntimeError('code/config/profile/environment changed; choose a new run ID')
    atomic_json(identitypath,identity)
    with run_lock(root/'results'/'akr_closing'/'.runner.lock'):
        # A plan checks all declared data and dependencies without loading model weights.
        study=Study(root,cfg,OmniBackend,profile=args.profile,run_id=args.run_id)
        if any(t not in {'E1','E6','E7'} for t in tasks):study.preflight()
        status={};statuspath=out/'TASK_STATUS.json'
        if statuspath.exists():status=json.loads(statuspath.read_text())
        for task in tasks:
            if pipe.STOP:
                status[task]={'status':'safe_stop','reason':'stop requested before next task'}
                atomic_json(statuspath,status)
                return 130
            status[task]={'status':'running' if args.execute else 'checking'};atomic_json(statuspath,status)
            try:
                if task=='E1':
                    from .legacy import resume_e1
                    result=resume_e1(root,cfg,run_id=args.run_id,execute=args.execute)
                    if args.execute and result.get('status')!='complete':raise RuntimeError('E1 remains incomplete')
                elif task=='E7':
                    from .orders import run_orders
                    result=run_orders(root,cfg,run_id=args.run_id,execute=args.execute)
                    if args.execute and not result.get('complete'):raise RuntimeError('E7 remains incomplete')
                elif task=='E6':
                    from .optional import run_factorized
                    result=run_factorized(root,cfg,run_id=args.run_id,execute=args.execute)
                elif task=='E8':
                    from .lora import run_lora
                    result={'jobs':len(run_lora(root,cfg,OmniBackend,run_id=args.run_id,profile=args.profile,execute=args.execute))}
                elif task=='E9':
                    from .scaling import confirm_3b
                    result=confirm_3b(root,cfg,OmniBackend,run_id=args.run_id,profile=args.profile,execute=args.execute)
                else:
                    if args.execute:study.run(task)
                    result={'phase':task,'run_id':args.run_id,'execute_requested':args.execute}
                    if args.execute and (pipe.STOP or not study.completed_execution):raise SafeStop('phase did not finish')
                    if args.execute and study.blocked:raise RuntimeError('one or more jobs blocked; inspect COVERAGE and BLOCKED artifacts')
                status[task]={'status':'completed' if args.execute else 'ready','result':result}
            except SafeStop:
                status[task]={'status':'safe_stop','resume':'rerun the same command; no partial accuracy is published'}
                atomic_json(statuspath,status);return 130
            except Exception as exc:
                status[task]={'status':'blocked_or_incomplete','error':type(exc).__name__,'reason':str(exc)}
                atomic_json(statuspath,status);raise
            atomic_json(statuspath,status)
        from .report import report_run
        report_run(out)
        print(f'Outputs: {out}; '+('execution finished' if args.execute else 'preflight only; no GPU work'))
    return 0
