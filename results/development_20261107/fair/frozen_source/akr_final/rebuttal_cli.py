"""Plan/check/run/report the rebuttal matrix, with bounded opt-in execution."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import yaml
from akr_closing.cli import run_lock,validate_run_id
from .release import _bounded
from .rebuttal_plan import build_study
from .rebuttal_runner import RebuttalStudy,report_study,expected_methods


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','check','run','report'],nargs='?',default='plan')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--config',type=Path,default=Path('configs/akr_rebuttal.yaml'))
    p.add_argument('--tasks',default='core,scale,dogs',help='core,scale,dogs; optional watkins; all expands all four')
    p.add_argument('--run-id',default='iclr_rebuttal_v1')
    p.add_argument('--profile',choices=['16gb','24gb','48gb','80gb'],default='24gb')
    p.add_argument('--budget-hours',type=float,default=24.)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--enable-conditional',action='store_true',help='explicitly activate GLM/Qwen3 suites')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv);root=args.root.resolve();validate_run_id(args.run_id)
    if not math.isfinite(args.budget_hours) or not 0<args.budget_hours<=24:raise ValueError('wall cap must be in (0,24] hours')
    config=args.config if args.config.is_absolute() else root/args.config
    if args.action=='report':
        result=report_study(root/'results/akr_rebuttal'/args.run_id)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return 0 if result['complete'] else 1
    if args.action=='run' and not args.execute:
        print('run requires --execute; plan/check never load a model',file=sys.stderr);return 2
    cfg=yaml.safe_load(config.read_text())
    from akr_closing.model_factory import MODEL_SPECS
    conditional_ids={m['id'] for m in MODEL_SPECS.values() if m['conditional']}
    conditional_tasks={name for name,suite in cfg['suites'].items()
                       if cfg['models'][suite['model']]['id'] in conditional_ids}
    tasks=[name for name in cfg['suites'] if args.enable_conditional or name not in conditional_tasks] if args.tasks=='all' else args.tasks.split(',')
    if set(tasks)&conditional_tasks and not args.enable_conditional:
        raise ValueError('conditional models require --enable-conditional')
    if args.action=='plan':
        study=build_study(root,config,tasks);jobs=[]
        for job in study['jobs']:
            entry=study['datasets'][job['dataset']]['plan'];e=next(e for e in entry['episodes'] if e['id']==job['episode'])
            jobs.append({**job,'methods':expected_methods(job,study['config']['settings']),
                **{k:len(e[k]) for k in ['fit_ids','calibration_ids','guard_ids','query_ids']}})
        print(json.dumps({'jobs':jobs,'job_count':len(jobs),'models':study['config']['models'],
            'settings':study['config']['settings'],'old_RUN_LOCK_required':False,'gpu_started':False,
            'scientific_status':'new variants implemented; no GPU result claimed'},indent=2));return 0
    from akr_closing.model_factory import model_family,validate_backend_environment
    selected_ids={cfg['models'][cfg['suites'][task]['model']]['id'] for task in tasks}
    if len({model_family(model_id) for model_id in selected_ids})>1:
        raise ValueError('mixed model families require scripts/run_akr_comprehensive.py for separate environments')
    if not args.worker:
        cmd=[sys.executable,str(root/'scripts/run_akr_rebuttal.py'),args.action,'--root',str(root),
             '--config',str(config),'--tasks',','.join(tasks),'--run-id',args.run_id,'--profile',args.profile,
             '--budget-hours',str(args.budget_hours),'--worker']+(['--execute'] if args.execute else [])
        if args.enable_conditional:cmd.append('--enable-conditional')
        env=os.environ.copy();env['PYTHONPATH']=str(root/'src')+os.pathsep+str(root)
        env.setdefault('OMP_NUM_THREADS','1');env.setdefault('TOKENIZERS_PARALLELISM','false')
        return _bounded(cmd,root,env,time.monotonic()+args.budget_hours*3600)
    from . import runner as runtime
    for model_id in selected_ids:validate_backend_environment(model_id)
    runtime.STOP=False
    signal.signal(signal.SIGINT,runtime._stop);signal.signal(signal.SIGTERM,runtime._stop)
    try:
        with run_lock(root/'results/akr_closing/.runner.lock'):
            study=RebuttalStudy(root,config,args.run_id,tasks,args.profile)
            result=study.run(execute=args.action=='run')
        print(json.dumps({k:v for k,v in result.items() if k not in ['rows','jobs']},indent=2));return 0
    except InterruptedError:
        print('Stopped safely; resume the same run ID, code, environment and config.',file=sys.stderr);return 130

if __name__=='__main__':raise SystemExit(main())
