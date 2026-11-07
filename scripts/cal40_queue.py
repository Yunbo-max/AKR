#!/usr/bin/env python3
"""Durable two-GPU queue for the registered full-population comparison."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from akr_closing.core import atomic_json,file_hash
from akr_closing.cli import run_lock,validate_run_id
from scripts.cal40_full_eval import aggregate,validate_sources


def gpu_samples():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                           '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    rows=[]
    for line in result.stdout.splitlines():
        i,used,total,util=[int(v.strip()) for v in line.split(',')]
        rows.append(dict(index=i,used_mib=used,total_mib=total,utilization_percent=util))
    return rows


def can_admit(used,baseline,reservations,request,total,target=90.):
    committed=max(used,baseline+sum(reservations))
    return committed+request<=min(target,total-3.) and used+request<=total-3.


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run-id',required=True)
    ap.add_argument('--execute',action='store_true'); ap.add_argument('--hours',type=float,default=24.)
    args=ap.parse_args(); validate_run_id(args.run_id)
    if not args.execute: raise ValueError('--execute required')
    if not 0<args.hours<=24: raise ValueError('queue must have a positive maximum24-hour window')
    out=ROOT/'results/cal40_full'/args.run_id
    with run_lock(out/'queue.lock'):
        reg=json.loads((out/'REGISTRATION.json').read_text()); validate_sources(reg)
        samples=gpu_samples(); baseline={s['index']:s['used_mib']/1024 for s in samples}
        if not {0,1}<=set(baseline): raise ValueError('two GPUs required')
        active={}; attempted=set(); failures={}; started=time.time(); stop=False
        def stopped(*_):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
        lanes=[(0,'qwen3b',18.),(0,'qwen7b',36.),(1,'qwen3b',18.),(1,'qwen7b',36.)]
        def launch(job,gpu,budget,phase):
            logdir=out/'logs'; logdir.mkdir(parents=True,exist_ok=True)
            handle=(logdir/(job['id']+'.'+phase+'.log')).open('a')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                 'TOKENIZERS_PARALLELISM':'false'}
            cmd=[sys.executable,'-u',str(ROOT/'scripts/cal40_full_eval.py'),'--run-id',args.run_id,
                 '--stage',phase,'--job',job['id']]
            if phase=='job': cmd+=['--execute','--memory-gib',str(budget)]
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            item=dict(job=job,gpu=gpu,budget=budget,phase=phase,pid=process.pid,started=time.time(),process=process,handle=handle)
            print(json.dumps({k:v for k,v in item.items() if k not in ['process','handle']}),flush=True)
            return item
        try:
            while not stop and time.time()-started<args.hours*3600:
                samples=gpu_samples()
                with (out/'GPU_SAMPLES.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(unix_time=time.time(),gpus=samples))+'\n')
                for lane,item in list(active.items()):
                    code=item['process'].poll()
                    if code is None: continue
                    item['handle'].close(); del active[lane]
                    if code==0 and item['phase']=='job':
                        active[lane]=launch(item['job'],item['gpu'],item['budget'],'verify')
                    elif code!=0:
                        failures[item['job']['id']]=dict(phase=item['phase'],returncode=code)
                        atomic_json(out/'QUEUE_FAILURES.json',failures)
                validate_sources(reg)
                summary=aggregate(out,reg)
                finished={s['job']['id'] for s in summary['completed']}
                for gpu,model,budget in lanes:
                    lane=(gpu,model)
                    if lane in active: continue
                    job=next((j for j in reg['jobs'] if j['model']==model and j['id'] not in attempted|finished),None)
                    if job is None: continue
                    s=next(v for v in samples if v['index']==gpu)
                    reserved=[v['budget'] for v in active.values() if v['gpu']==gpu]
                    if not reserved: baseline[gpu]=s['used_mib']/1024
                    if not can_admit(s['used_mib']/1024,baseline[gpu],reserved,budget,s['total_mib']/1024): continue
                    attempted.add(job['id']); active[lane]=launch(job,gpu,budget,'job')
                state=dict(complete=summary['complete'] and not active and not failures,
                    pid=os.getpid(),elapsed_seconds=time.time()-started,max_hours=args.hours,
                    active=[{k:v for k,v in x.items() if k not in ['process','handle']} for x in active.values()],
                    attempted=len(attempted),completed_jobs=summary['completed_jobs'],expected_jobs=len(reg['jobs']),
                    failures=failures,waiting=[j['id'] for j in reg['jobs'] if j['id'] not in attempted|finished])
                atomic_json(out/'QUEUE.json',state)
                if not active and not state['waiting']: break
                time.sleep(5)
        finally:
            # Only terminate processes created by this queue when its window ends.
            for item in active.values():
                if item['process'].poll() is None: os.killpg(item['pid'],signal.SIGTERM)
            for item in active.values():
                try: item['process'].wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(item['pid'],signal.SIGKILL)
                item['handle'].close()
            summary=aggregate(out,reg)
            atomic_json(out/'QUEUE_EXIT.json',dict(complete=summary['complete'] and not failures,
                stopped=stop,window_exhausted=time.time()-started>=args.hours*3600,failures=failures,
                completed_jobs=summary['completed_jobs'],expected_jobs=len(reg['jobs'])))


if __name__=='__main__': main()
