"""Auxiliary 2x2 rank/ridge CE check; NEVER enters main AKR selection."""
from pathlib import Path
import sys
import time
import signal
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.fair_speech_lm import load_registration,FairRunner,support_arrays,gpu_budget
from akr_closing.core import Journal,atomic_json,digest,file_hash
from akr_closing.pipeline import prompt_for
from akr_final.predictability import fit_repository
from akr_final.rebuttal_runner import measured,array_hash

out=ROOT/'results/fair_speech_lm'/sys.argv[1]
def expire(*_): raise TimeoutError('auxiliary parameter diagnostic timed out')
signal.signal(signal.SIGALRM,expire); signal.alarm(1800)
runner=None
try:
    atomic_json(out/'PARAMETER_CE_STATUS.json',dict(complete=False))
    atomic_json(out/'telemetry/parameter_ce.json',gpu_budget(90.))
    runner=FairRunner(out,'parameter_ce',load_registration(out))
    e,pool,x,g,y=support_arrays(runner)
    rows=[]
    for rank,ridge in [(4,10.),(4,3000.),(16,10.),(16,3000.)]:
        losses=[]
        for fold,held in enumerate(e['folds']):
            h=[pool.index(i) for i in held]; t=[j for j in range(len(pool)) if j not in h]
            fields=fit_repository(x[t],g[t],rank=rank,alpha=ridge).predict(x[h])
            ident=dict(run=runner.fingerprint,script=file_hash(Path(__file__)),rank=rank,ridge=ridge,
                alpha=.03,mode='relative_norm',fold=fold,ids=held,fields=array_hash(fields))
            journal=Journal(runner.out/'loss_cache'/digest(ident),ident,held)
            for j,event in enumerate(held):
                if journal.has(event): continue
                backend=runner._model('qwen3b',e)
                value=measured(backend,lambda:backend.support_answer_loss(runner._query(event,e['condition']),
                    runner.lookup[event]['label'],prompt_for(runner.labels),direction=fields[j],alpha=.03,scale_mode='relative_norm'))
                if not np.isfinite(value['answer_loss']): raise ValueError('nonfinite auxiliary CE')
                journal.add(event,value)
            losses.extend(r['answer_loss'] for r in journal.results())
        row=dict(rank=rank,ridge=ridge,alpha=.03,mean_ce=float(np.mean(losses)),losses=losses)
        rows.append(row); print({k:v for k,v in row.items() if k!='losses'},flush=True)
    result=dict(complete=True,calibration_only=True,auxiliary_not_used_for_primary_selection=True,rows=rows,
        script_sha256=file_hash(Path(__file__)),note='2x2 interaction check at fixed alpha .03. Additional diagnostic compute disclosed; no development queries or main-policy changes.')
    atomic_json(out/'PARAMETER_CE.json',result)
    atomic_json(out/'PARAMETER_CE_STATUS.json',dict(complete=True,result_sha256=file_hash(out/'PARAMETER_CE.json')))
finally:
    if runner is not None: runner.close()
    signal.alarm(0)
