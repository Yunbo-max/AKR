#!/usr/bin/env python3
"""Bounded subsequent development round: centered AKR and recognition selection."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import signal
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import scripts.fair_speech_lm as F
from akr_closing.core import Journal,atomic_json,digest,file_hash,metrics,paired_stats
from akr_closing.cli import run_lock,validate_run_id
from akr_closing.pipeline import prompt_for
from akr_closing.repair import RidgeReadout
from akr_final.predictability import fit_repository
from akr_final.rebuttal_runner import measured,array_hash
from akr_final.runner import prediction_summary

PRIOR='fair_dogs_3b_20261107_v2'
KINDS=('shrunk','centered_regression','centered_class')
ALPHAS=(.03,.1,.3)


class CenteredRouter:
    def __init__(self,x,g,y,nclass,kind):
        if kind not in KINDS: raise ValueError('unknown mapping')
        self.kind=kind; self.mean=np.asarray(g).mean(0)
        if kind=='centered_class':
            self.readout=RidgeReadout.fit(x,y,nclass,alpha=10.)
            self.centroids=np.stack([g[y==j].mean(0) for j in range(nclass)])
        else: self.router=fit_repository(x,g,rank=4,alpha=10.)
    def predict(self,x):
        if self.kind=='centered_class': return self.centroids[self.readout.predict(x)]-self.mean
        pred=self.router.predict(x)
        if self.kind=='centered_regression': return pred-self.mean
        return self.mean+.25*(pred-self.mean)


def select_recognition(predictions,truth,labels,order):
    if set(predictions)!=set(order) or len(set(order))!=len(order) or order[0]!='native':
        raise ValueError('complete predeclared candidate order required')
    scores={k:metrics(truth,predictions[k],labels) for k in order}
    def key(k):
        s=scores[k]; return (s['accuracy'],s['macro_f1'],-s['invalid_rate'],-order.index(k))
    best=max(order,key=key)
    return dict(candidate=best,accepted=best!='native',scores=scores,
        criterion='OOF accuracy, then macro-F1, then lower invalid rate, then native/simpler predeclared order')


def candidate_specs(method):
    if method=='akr':
        return [('native',{})]+[(f'{k}_a{a:g}',dict(kind=k,alpha=a)) for a in ALPHAS for k in KINDS]
    if method=='lora':
        return [('native',{})]+[(f'lr_{lr:g}_epoch_{ep}',dict(lr=lr,epoch=ep)) for ep in F.EPOCHS for lr in F.LRS]
    raise ValueError('unknown method')


def prepare(out):
    prior=ROOT/'results/fair_speech_lm'/PRIOR
    old=F.read(prior/'REGISTRATION.json'); report=F.read(prior/'REPORT.json')
    status=F.read(prior/'STATUS.json'); cached=F.read(prior/'CACHED_VALIDATION.json')
    sha=file_hash(prior/'REPORT.json')
    if not all(v['complete'] for v in [report,status,cached]) or status['report_sha256']!=sha or cached['report_sha256']!=sha:
        raise RuntimeError('complete verified prior report required')
    archive=F.read(prior/'FROZEN_SOURCE_MANIFEST.json')['files']
    entry=deepcopy(old['entry']); sources={}
    for name,expected in entry['plan']['source_hashes'].items():
        p=Path(name)
        if p.is_file() and file_hash(p)==expected: sources[name]=expected
        else:
            if p.resolve()!=Path(F.__file__).resolve(): raise RuntimeError('changed source beyond declared instrumentation maintenance')
            if name not in archive or archive[name]['sha256']!=expected: raise RuntimeError('unarchived changed prior source')
            frozen=prior/archive[name]['archive']
            if file_hash(frozen)!=expected: raise RuntimeError('frozen source checksum')
            sources[str(frozen)]=expected
    e,=entry['plan']['episodes']; pool=e['support_ids']; adapters={}
    bound=[prior/'REGISTRATION.json',prior/'REPORT.json',prior/'CACHED_VALIDATION.json',
           prior/'FROZEN_SOURCE_MANIFEST.json',prior/'POLICY_LOCK.json',Path(__file__),Path(F.__file__),
           ROOT/'docs/superpowers/plans/2026-11-07-accuracy-selected-akr.md']
    for fold in range(2):
        identity_path=prior/'workers'/f'fold{fold}'/'IDENTITY.json'
        identity=F.read(identity_path); old_fp=digest(identity); bound.append(identity_path)
        for name,expected_sha in identity['code'].items():
            if file_hash(Path(name))!=expected_sha: raise RuntimeError('backbone/pipeline code changed since imported training')
        if identity['plan_sha256']!=digest(old['entry']['plan']): raise RuntimeError('prior fold plan mismatch')
        held=e['folds'][fold]; train=[i for i in pool if i not in held]
        for lr in F.LRS:
            directory=prior/'training'/f'fold_{fold}'/f'lr_{lr:g}'
            expected=dict(run=old_fp,train=train,held=held,lr=lr,max_epoch=5,seed=20261108+fold,accumulation=4)
            stage=F.verify_stage(directory,expected)
            if stage is None or stage['train_ids']!=train or stage['heldout_ids']!=held:
                raise RuntimeError('foreign prior calibration adapter')
            bound.append(directory/'DONE.json'); bound.extend(Path(p) for p in stage['files'])
            for epoch in F.EPOCHS:
                adapters[f'{fold}:lr_{lr:g}_epoch_{epoch}']=str(directory/f'epoch_{epoch}')
    prior_refit=F.read(prior/'training/refit/DONE.json')['result']
    old_refit_fp=digest(F.read(prior/'workers/lora_refit/IDENTITY.json'))
    choice=report['policies']['lora']
    expected=dict(run=old_refit_fp,train=pool,held=[],lr=choice['lr'],max_epoch=choice['epoch'],seed=20261110,accumulation=4)
    F.verify_stage(prior/'training/refit',expected)
    bound.extend([prior/'training/refit/DONE.json',prior/'workers/lora_refit/IDENTITY.json'])
    bound.extend(Path(p) for p in prior_refit['files'])
    sources.update({str(p):file_hash(p) for p in bound})
    entry['plan']['source_hashes']=sources
    entry['plan']['evaluation_status']='second development iteration after inspecting prior results; official test unused'
    registration=dict(entry=entry,arrays=old['arrays'],adapters=adapters,prior_run=PRIOR,
        prior_lora_choice=choice,prior_full_adapter=prior_refit['final_adapter'],prior_report_sha256=sha,
        akr_candidates=candidate_specs('akr'),lora_candidates=candidate_specs('lora'))
    path=out/'REGISTRATION.json'
    if path.exists() and F.read(path)!=json.loads(json.dumps(registration)):
        raise RuntimeError('changed registration; new run ID required')
    atomic_json(path,registration); return registration


def specs(runner,method,fold):
    e,pool,x,g,y=F.support_arrays(runner); held=e['folds'][fold]
    h=[pool.index(i) for i in held]; t=[j for j in range(len(pool)) if j not in h]
    result=[]
    for name,settings in candidate_specs(method):
        fields=None; adapter=None
        if name!='native':
            if method=='akr': fields=CenteredRouter(x[t],g[t],y[t],len(runner.labels),settings['kind']).predict(x[h])
            else: adapter=runner.registration['adapters'][f'{fold}:{name}']
        result.append((name,settings,fields,adapter))
    return held,result


def prediction_journal(runner,name,ids,settings,fields=None,adapter=None):
    adapter_hashes=None if adapter is None else {n:file_hash(Path(adapter)/n) for n in ['adapter_config.json','adapter_model.safetensors']}
    identity=dict(run=runner.fingerprint,name=name,ids=ids,settings=settings,
        fields=None if fields is None else array_hash(fields),adapter=adapter_hashes,
        prompt=prompt_for(runner.labels),decoding='greedy12',scale_mode='relative_norm')
    return Journal(runner.out/'trials'/name,identity,ids)


def generate(runner,name,ids,settings,fields=None,adapter=None):
    e,=runner.plan['episodes']; journal=prediction_journal(runner,name,ids,settings,fields,adapter)
    backend=None
    try:
        for j,event in enumerate(ids):
            if journal.has(event): continue
            if backend is None:
                backend=runner._model('qwen3b',e)
                if adapter:
                    from peft import PeftModel
                    backend.thinker=PeftModel.from_pretrained(backend.thinker,adapter,is_trainable=False).eval()
                    if any(p.requires_grad for p in backend.thinker.parameters()): raise RuntimeError('trainable inference weights')
            result=measured(backend,lambda:backend.predict(runner._query(event,e['condition']),prompt_for(runner.labels),
                runner.labels,direction=None if fields is None else fields[j],alpha=settings.get('alpha',0.),
                scale_mode='relative_norm',max_new_tokens=12))
            journal.add(event,result)
        return journal.results()
    finally:
        backend=None; runner.close(); gc.collect()


def cv(runner,method,fold):
    held,candidates=specs(runner,method,fold)
    for name,settings,fields,adapter in candidates:
        result=generate(runner,name,held,settings,fields,adapter)
        truth=[runner.lookup[i]['label'] for i in held]
        print(json.dumps(dict(method=method,fold=fold,candidate=name,**metrics(truth,[r['prediction'] for r in result],runner.labels))),flush=True)


def selection(out,registration,publish=False):
    all_records={}; choices={}; e,=registration['entry']['plan']['episodes']
    labels=registration['entry']['plan']['labels']; lookup={r['event_id']:r for r in registration['entry']['plan']['rows']}
    ordered_ids=e['folds'][0]+e['folds'][1]; truth=[lookup[i]['label'] for i in ordered_ids]
    for method in ['akr','lora']:
        collected={name:[] for name,_ in candidate_specs(method)}
        for fold in range(2):
            runner=F.FairRunner(out,f'cv_{method}{fold}',registration)
            try:
                held,candidates=specs(runner,method,fold)
                for name,settings,fields,adapter in candidates:
                    result=prediction_journal(runner,name,held,settings,fields,adapter).results()
                    collected[name].extend(result)
            finally: runner.close()
        all_records[method]=collected
        choice=select_recognition({k:[r['prediction'] for r in v] for k,v in collected.items()},truth,labels,
                                  [k for k,_ in candidate_specs(method)])
        choice['settings']=dict(candidate_specs(method))[choice['candidate']]
        choice['prediction_forward_calls']=sum(r['new_model_calls'] for v in collected.values() for r in v)
        choice['prediction_seconds']=sum(r['wall_seconds'] for v in collected.values() for r in v)
        choices[method]=choice
    for a,b in zip(all_records['akr']['native'],all_records['lora']['native']):
        if any(a[k]!=b[k] for k in ['prediction','raw_prediction','generated_tokens']):
            raise RuntimeError('native CV outputs differ between method pipelines')
    result=dict(registration=digest(registration),choices=choices,native_cv_parity=True,
        calibration_ids=ordered_ids,development_labels_used_for_selection=False)
    if publish: atomic_json(out/'SELECTION.json',result)
    return result


def refit(out,registration):
    selected=selection(out,registration,publish=True); choice=selected['choices']['lora']
    runner=F.FairRunner(out,'refit',registration); e,=runner.plan['episodes']
    try:
        if choice['candidate']=='native': result=dict(kind='native',adapter=None)
        elif choice['settings']=={k:registration['prior_lora_choice'][k] for k in ['lr','epoch']}:
            result=dict(kind='imported_identical_full_pool_adapter',adapter=registration['prior_full_adapter'],
                prior_receipt_sha256=file_hash(ROOT/'results/fair_speech_lm'/PRIOR/'training/refit/DONE.json'))
        else:
            s=choice['settings']
            trained=F.train_stage(runner,out/'training/refit',e['support_ids'],[],s['lr'],s['epoch'],20261110)
            result=dict(kind='new_full_pool_fit',adapter=trained['final_adapter'],receipt_sha256=file_hash(out/'training/refit/DONE.json'))
        result.update(selection_sha256=file_hash(out/'SELECTION.json'),run=runner.fingerprint,
            adapter_files=None if result['adapter'] is None else {n:file_hash(Path(result['adapter'])/n) for n in ['adapter_config.json','adapter_model.safetensors']})
        atomic_json(out/'REFIT.json',result)
    finally: runner.close()


def lock(out,registration):
    import fcntl
    with (out/'.policy.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:
            selected=selection(out,registration)
            if selected!=F.read(out/'SELECTION.json'): raise RuntimeError('selection evidence changed')
            refitted=F.read(out/'REFIT.json'); choice=selected['choices']['lora']
            if refitted['selection_sha256']!=file_hash(out/'SELECTION.json'): raise RuntimeError('refit selection mismatch')
            runner=F.FairRunner(out,'refit',registration); e,=runner.plan['episodes']
            try:
                if refitted['run']!=runner.fingerprint: raise RuntimeError('foreign refit')
                if refitted['kind']=='native':
                    if choice['candidate']!='native' or refitted['adapter'] is not None: raise RuntimeError('invalid native refit')
                elif refitted['kind']=='imported_identical_full_pool_adapter':
                    if choice['settings']!={k:registration['prior_lora_choice'][k] for k in ['lr','epoch']} or refitted['adapter']!=registration['prior_full_adapter']:
                        raise RuntimeError('mismatched imported refit')
                elif refitted['kind']=='new_full_pool_fit':
                    s=choice['settings']; expected=dict(run=runner.fingerprint,train=e['support_ids'],held=[],lr=s['lr'],max_epoch=s['epoch'],seed=20261110,accumulation=4)
                    result=F.verify_stage(out/'training/refit',expected)
                    if result is None or result['final_adapter']!=refitted['adapter']: raise RuntimeError('invalid new refit')
                else: raise RuntimeError('unknown refit kind')
                if refitted['adapter']:
                    for name,sha in refitted['adapter_files'].items():
                        if file_hash(Path(refitted['adapter'])/name)!=sha: raise RuntimeError('adapter changed')
            finally: runner.close()
            result=dict(selection=selected,refit=refitted)
            path=out/'POLICY_LOCK.json'
            if path.exists():
                if F.read(path)!=result: raise RuntimeError('policy lock changed')
            else: atomic_json(path,result)
            return result
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def evaluate(runner,out,method):
    policy=lock(out,runner.registration); choice=policy['selection']['choices'][method]
    e,pool,x,g,y=F.support_arrays(runner); fields=None; adapter=None
    if choice['candidate']!='native':
        if method=='akr':
            q=runner.matrix(e['query_ids'],e,'qwen3b')
            fields=CenteredRouter(x,g,y,len(runner.labels),choice['settings']['kind']).predict(q)
        else: adapter=policy['refit']['adapter']
    return generate(runner,method,e['query_ids'],choice['settings'],fields,adapter)


def report(out,registration):
    policy=lock(out,registration); prior=F.read(ROOT/'results/fair_speech_lm'/PRIOR/'REPORT.json')
    e,=registration['entry']['plan']['episodes']; lookup={r['event_id']:r for r in registration['entry']['plan']['rows']}
    records={k:prior['predictions'][k] for k in ['native','akr_reference']}
    for method in ['akr','lora']:
        runner=F.FairRunner(out,'eval_'+method,registration); runner.cache_only=True
        try: records[method]=evaluate(runner,out,method)
        finally: runner.close()
    truth=[lookup[i]['label'] for i in e['query_ids']]; native=[r['prediction'] for r in records['native']]
    for key in ['native','akr_reference']:
        if [r['event_id'] for r in records[key]]!=e['query_ids'] or [r['target'] for r in records[key]]!=truth:
            raise RuntimeError('imported baseline population mismatch')
    rows=[]
    for method,rs in records.items():
        pred=[r['prediction'] for r in rs]
        row=dict(method=method,**prediction_summary(truth,native,pred,registration['entry']['plan']['labels']))
        if method!='native': row['paired']=paired_stats(truth,native,pred,seed=20261108)
        rows.append(row)
    result=dict(complete=True,rows=rows,policy=policy,previously_inspected_development=True,official_test_used=False,
        label_budget=40,candidate_count_each=9,compute_matched=False,historical_search_matched=False,
        cached_inference_verified=True,predictions=records)
    atomic_json(out/'REPORT.json',result)
    atomic_json(out/'STATUS.json',dict(complete=True,report_sha256=file_hash(out/'REPORT.json')))
    print(json.dumps({'complete':True,'rows':rows}),flush=True)
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--stage',required=True,choices=['prepare','cv_akr0','cv_akr1','cv_lora0','cv_lora1','refit','lock','eval_akr','eval_lora','report'])
    ap.add_argument('--execute',action='store_true'); ap.add_argument('--gpu-target-gib',type=float,default=90.)
    args=ap.parse_args(argv); validate_run_id(args.run_id)
    real=args.stage.startswith(('cv_','eval_')) or args.stage=='refit'
    if real and not args.execute: raise ValueError('--execute required')
    if not 20<=args.gpu_target_gib<=90: raise ValueError('GPU target out of range')
    out=ROOT/'results/accuracy_selected_akr'/args.run_id; runner=None
    def expired(*_): raise TimeoutError('stage exceeded90minute budget')
    signal.signal(signal.SIGALRM,expired); signal.alarm(5400)
    with run_lock(out/(args.stage+'.lock')):
        atomic_json(out/'stages'/(args.stage+'.json'),dict(complete=False))
        try:
            registration=prepare(out) if args.stage=='prepare' else F.load_registration(out)
            if real: atomic_json(out/'telemetry'/(args.stage+'.json'),F.gpu_budget(args.gpu_target_gib))
            if args.stage=='prepare':
                runner=F.FairRunner(out,args.stage,registration)
            elif args.stage.startswith('cv_'):
                runner=F.FairRunner(out,args.stage,registration); cv(runner,args.stage[3:-1],int(args.stage[-1]))
            elif args.stage=='refit': refit(out,registration)
            elif args.stage=='lock': lock(out,registration)
            elif args.stage.startswith('eval_'):
                runner=F.FairRunner(out,args.stage,registration); evaluate(runner,out,args.stage[5:])
            elif args.stage=='report': report(out,registration)
            atomic_json(out/'stages'/(args.stage+'.json'),dict(complete=True))
        except BaseException as exc:
            atomic_json(out/'stages'/(args.stage+'.json'),dict(complete=False,error=str(exc)))
            atomic_json(out/'STATUS.json',dict(complete=False,failed_stage=args.stage,error=str(exc)))
            raise
        finally:
            if runner is not None: runner.close()
            signal.alarm(0)
    return 0


if __name__=='__main__': raise SystemExit(main())
