#!/usr/bin/env python3
"""Fixed forty-label adaptation with complete, independent evaluation populations."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from akr_closing.core import Journal,atomic_json,digest,file_hash,read_manifest,paired_stats
from akr_closing.cli import run_lock,validate_run_id
from akr_closing.pipeline import prompt_for
from akr_final.beans_zero_protocol import canonical_pcm_hash
from akr_final.calibration_pilot import PilotRunner
from akr_final.rebuttal_runner import measured,array_hash
from akr_final.runner import prediction_summary
import scripts.fair_speech_lm as F
from scripts.accuracy_selected_akr import CenteredRouter

SEED=20261107
POLICY=dict(akr=dict(kind='centered_class',alpha=.3,ridge=10.,scale_mode='relative_norm'),
            lora=dict(lr=5e-4,epochs=5,rank=8,alpha=16,dropout=.05,effective_batch=4),
            selection='fixed transferred hyperparameters before any current query prediction',
            prior_development='Dogs inspected development subset, Qwen2.5-Omni-3B',
            label_budget=40,query_tuning=False,historical_search_matched=False)


def recording_groups(rows,key):
    """Source identity and decoded audio equivalence compose transitively."""
    lookup={r['event_id']:r for r in rows}
    if len(lookup)!=len(rows): raise ValueError('duplicate event IDs')
    parents={i:i for i in lookup}
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    seen={}
    for r in sorted(rows,key=lambda r:r['event_id']):
        if not r.get(key) or not r.get('pcm_sha256'): raise ValueError('source group and PCM hash required')
        i=r['event_id']
        for token in [('source',r[key]),('pcm',r['pcm_sha256'])]:
            if token in seen:
                a,b=find(i),find(seen[token]); parents[max(a,b)]=min(a,b)
            seen[token]=i
    members={}
    for i in lookup: members.setdefault(find(i),[]).append(i)
    identities={i:digest(sorted(v)) for i,v in members.items()}
    return {i:identities[find(i)] for i in lookup}


def allocate(rows,labels,key,official,seed=SEED):
    if len(labels)>40 or len(set(labels))!=len(labels): raise ValueError('40 labels cannot cover this class vocabulary')
    if set(r['label'] for r in rows)!=set(labels): raise ValueError('manifest/vocabulary mismatch')
    groups=recording_groups(rows,key)
    query=sorted([r['event_id'] for r in rows if r.get('split')=='test']) if official else []
    if official and not query: raise ValueError('official test is empty')
    blocked={groups[i] for i in query}
    candidates=[r for r in rows if (not official or r.get('split')=='train') and groups[r['event_id']] not in blocked]
    candidates.sort(key=lambda r:digest([seed,r['event_id']]))
    counts={c:0 for c in labels}; support=[]; chosen=set()
    remaining={c:{groups[r['event_id']] for r in rows if r['label']==c} for c in labels}
    for _ in range(40):
        selected=None
        for label in sorted(labels,key=lambda c:(counts[c],digest([seed,c]))):
            for r in candidates:
                g=groups[r['event_id']]
                if r['label']!=label or g in chosen: continue
                if not official and any(not (gs-chosen-{g}) for gs in remaining.values()): continue
                selected=r; break
            if selected is not None: break
        if selected is None: raise ValueError('insufficient distinct groups for 40 calibration clips and independent evaluation')
        support.append(selected['event_id']); chosen.add(groups[selected['event_id']]); counts[selected['label']]+=1
    if any(v==0 for v in counts.values()): raise ValueError('40-clip allocation missing class coverage')
    if not official: query=sorted(r['event_id'] for r in rows if groups[r['event_id']] not in chosen)
    if not query or chosen & {groups[i] for i in query}: raise ValueError('empty or overlapping evaluation')
    used=set(support+query)
    excluded=[dict(event_id=r['event_id'],reason=('calibration_recording_group' if groups[r['event_id']] in chosen else 'unused_official_train_or_validation'))
              for r in sorted(rows,key=lambda r:r['event_id']) if r['event_id'] not in used]
    return dict(support_ids=support,query_ids=query,support_class_counts=counts,excluded=excluded,
                groups=groups,official_test=official,query_group_count=len({groups[i] for i in query}),
                complete_official_test=official and set(query)=={r['event_id'] for r in rows if r.get('split')=='test'})


def check_complete(ids,records):
    if [r.get('event_id') for r in records]!=ids: raise ValueError('predictions do not cover exact registered population')


def model_tag(requested,registered):
    # The old, tested PEFT helper calls _model('qwen3b'); only that call is aliased.
    if registered not in ('qwen3b','qwen7b') or requested not in ('qwen3b',registered):
        raise ValueError('unexpected model tag')
    return registered


def validate_sources(reg):
    for p,sha in reg['source_hashes'].items():
        if not Path(p).is_file() or file_hash(Path(p))!=sha: raise RuntimeError(f'changed registered source: {p}')


def prepare(out):
    cfgpath=ROOT/'configs/akr_comprehensive.yaml'; cfg=yaml.safe_load(cfgpath.read_text())
    source_files=[cfgpath,Path(__file__),Path(F.__file__),ROOT/'scripts/accuracy_selected_akr.py',
                  ROOT/'scripts/cal40_queue.py',ROOT/'docs/superpowers/plans/2026-11-07-cal40-full-evaluation.md']
    source_files+=list((ROOT/'akr_final').glob('*.py'))+list((ROOT/'akr_closing').glob('*.py'))
    source_files += [ROOT/'src/animal_omni/conditional_kv.py',ROOT/'src/animal_omni/metrics.py']
    datasets={}
    for name,ds in cfg['datasets'].items():
        path=ROOT/ds['manifest']; source_files.append(path)
        labels=ds.get('labels')
        if labels is None:
            p=ROOT/ds['labels_from_config']; source_files.append(p)
            labels=yaml.safe_load(p.read_text())['dataset']['labels']
        for p in ds.get('registration_sources',[]): source_files.append(ROOT/p)
        original=read_manifest(path,old_root=cfg['old_root'],root=ROOT)
        for row in original:
            audio=Path(row['audio_path']); row['sha256']=file_hash(audio)
            row['pcm_sha256']=canonical_pcm_hash(audio)
        split=allocate(original,labels,ds['group_key'],ds['partition']=='official_test')
        keep=set(split['support_ids']+split['query_ids'])
        selected=[{**r,'recording_id':split['groups'][r['event_id']]} for r in original if r['event_id'] in keep]
        partition='official_test' if split['official_test'] else 'derived_group_disjoint_re_evaluation'
        datasets[name]=dict(labels=labels,rows=sorted(selected,key=lambda r:r['event_id']),split=split,
            partition=partition,source_manifest=str(path),original_rows=len(original),
            max_new_tokens=64 if name.startswith('bz_') else 12,
            original_zero_shot_benchmark=False if name.startswith('bz_') else None,
            source_group_key=ds['group_key'])
        print(json.dumps(dict(dataset=name,classes=len(labels),calibration=40,query=len(split['query_ids']),
                              query_groups=split['query_group_count'],excluded=len(split['excluded']),partition=partition)),flush=True)
    order=['dogs','watkins','marmaudio']+[n for n in datasets if n.startswith('bz_')]
    if set(order)!=set(datasets): raise ValueError('unhandled dataset configuration')
    jobs=[]
    for name in order:
        for condition in ['lp:1000','full']:
            for model in ['qwen3b','qwen7b']:
                jobs.append(dict(id=f'{name}__{model}__{condition.replace(":","_")}',dataset=name,model=model,condition=condition))
    reg=dict(version=1,seed=SEED,policy=POLICY,config=cfg,datasets=datasets,jobs=jobs,
             source_hashes={str(p):file_hash(p) for p in sorted(set(source_files))},
             original_beans_zero_artifacts_unchanged=True,
             test_history='Official split labels are held out from this run; repository contains historical evaluations, so not claimed globally untouched.')
    path=out/'REGISTRATION.json'
    if path.exists() and F.read(path)!=reg: raise RuntimeError('registration changed; new run ID required')
    atomic_json(path,reg)
    atomic_json(out/'DATASET_COUNTS.json',{n:dict(classes=len(d['labels']),calibration=40,
        query=len(d['split']['query_ids']),query_groups=d['split']['query_group_count'],partition=d['partition'],
        complete_official_test=d['split']['complete_official_test'],excluded=len(d['split']['excluded'])) for n,d in datasets.items()})
    return reg


def entry_for(reg,job):
    ds=reg['datasets'][job['dataset']]; split=ds['split']; cfg=reg['config']; condition=job['condition']
    setting={**cfg['settings'],'max_new_tokens':ds['max_new_tokens']}
    ep=dict(id=f's{SEED}_{condition.replace(":","_")}',seed=SEED,condition=condition,
            support_ids=split['support_ids'],fit_ids=split['support_ids'],calibration_ids=[],guard_ids=[],query_ids=split['query_ids'],
            lock=reg['policy'],lock_sha256=digest(reg['policy']))
    internal=dict(models=[dict(tag=job['model'],**cfg['models'][job['model']])],E9=cfg['models']['qwen3b'],
                  **{k:setting[k] for k in ['layer_fractions','feature_fraction','max_context_tokens','ridge_alpha']})
    plan=dict(mode='standalone',config=internal,labels=ds['labels'],rows=ds['rows'],episodes=[ep],
              dataset=dict(name=job['dataset'],partition=ds['partition'],group_key='recording_id'),
              source_hashes=reg['source_hashes'],selection_uses_outcomes=False,
              query_partition=ds['partition'],evaluation_status=reg['test_history'],fixed_policy=reg['policy'])
    return dict(plan=plan,options=dict(dataset=job['dataset'],settings=setting,registered_model=job['model']))


class FullRunner(PilotRunner):
    def __init__(self,out,reg,job,*,cache_only=False):
        self.actual_model=job['model']; self.job=job
        super().__init__(ROOT,out/'jobs',job['id'],entry_for(reg,job),'24gb')
        self.cache_only=cache_only
        self.preflight(); self.verify_caches()
        # Override the legacy development-only descriptor, not its integrity checks.
        p=F.read(self.out/'PLAN.json'); p['no_test_evaluation']=False
        p['query_partition']=reg['datasets'][job['dataset']]['partition']; atomic_json(self.out/'PLAN.json',p)
    def _model(self,tag,episode=None):
        return super()._model(model_tag(tag,self.actual_model),episode)


def journal_for(runner,method,map_hash,adapter_hashes):
    e,=runner.plan['episodes']
    return Journal(runner.out/'trials'/method,
        dict(run=runner.fingerprint,method=method,model=runner.actual_model,policy=POLICY,
             map_hash=map_hash if method=='akr' else None,adapter=adapter_hashes if method=='lora' else None,
             prompt=prompt_for(runner.labels),max_new_tokens=runner.settings['max_new_tokens']),e['query_ids'])


def predictions(runner,method,router,map_hash,adapter,adapter_hashes):
    e,=runner.plan['episodes']; journal=journal_for(runner,method,map_hash,adapter_hashes)
    backend=None
    try:
        for event in e['query_ids']:
            if journal.has(event): continue
            if backend is None:
                backend=runner._model(runner.actual_model,e)
                if method=='lora':
                    from peft import PeftModel
                    backend.thinker=PeftModel.from_pretrained(backend.thinker,adapter,is_trainable=False).eval()
                    if any(p.requires_grad for p in backend.thinker.parameters()): raise RuntimeError('trainable inference adapter')
            field=None
            if method=='akr': field=router.predict(runner.matrix([event],e,runner.actual_model))[0]
            value=measured(backend,lambda:backend.predict(runner._query(event,e['condition']),prompt_for(runner.labels),runner.labels,
                    direction=field,alpha=.3 if method=='akr' else 0.,scale_mode='relative_norm',
                    max_new_tokens=runner.settings['max_new_tokens']))
            journal.add(event,dict(event_id=event,**value))
            atomic_json(runner.out/'PROGRESS.json',dict(complete=False,phase=method,**{k:v for k,v in journal.status().items() if k!='complete'}))
        result=journal.results(); check_complete(e['query_ids'],result); return result
    finally:
        backend=None; runner.close()


def run_job(out,reg,job,verify_only=False):
    validate_sources(reg)
    with run_lock(out/'locks'/(job['id']+'.lock')):
        runner=FullRunner(out,reg,job,cache_only=verify_only); start=time.time()
        try:
            atomic_json(runner.out/'STATUS.json',dict(complete=False,phase='calibration'))
            e,=runner.plan['episodes']; pool=e['support_ids']; tag=job['model']
            x=runner.matrix(pool,e,tag); g=runner.matrix(pool,e,tag,'gradient')
            y=np.array([runner.labels.index(runner.lookup[i]['label']) for i in pool])
            router=CenteredRouter(x,g,y,len(runner.labels),'centered_class')
            map_hash=digest([array_hash(x),array_hash(g),y.tolist()])
            runner.close()
            trained=F.train_stage(runner,runner.out/'training',pool,[],5e-4,5,SEED+3)
            adapter=trained['final_adapter']
            hashes={n:file_hash(Path(adapter)/n) for n in ['adapter_model.safetensors','adapter_config.json']}
            locked=dict(run=runner.fingerprint,policy=POLICY,map_hash=map_hash,adapter=hashes,
                        calibration_ids=pool,query_ids=e['query_ids'],model=job['model'])
            p=runner.out/'POLICY_LOCK.json'
            if p.exists() and F.read(p)!=locked: raise RuntimeError('locked method identity changed')
            atomic_json(p,locked)
            records={method:predictions(runner,method,router,map_hash,adapter,hashes) for method in ['native','akr','lora']}
            # Score only after all three methods have every registered prediction.
            truth=[runner.lookup[i]['label'] for i in e['query_ids']]
            groups=[runner.lookup[i]['recording_id'] for i in e['query_ids']]
            native=[r['prediction'] for r in records['native']]; scores=[]
            for method,rs in records.items():
                pred=[r['prediction'] for r in rs]
                row=dict(method=method,**prediction_summary(truth,native,pred,runner.labels))
                if method!='native': row['paired']=paired_stats(truth,native,pred,groups=groups,seed=SEED)
                scores.append(row)
            report=dict(complete=True,job=job,rows=scores,policy=locked,label_budget=40,
                calibration_class_counts=dict(Counter(runner.lookup[i]['label'] for i in pool)),
                evaluation_partition=reg['datasets'][job['dataset']]['partition'],query_count=len(truth),
                query_ids=e['query_ids'],query_groups=groups,targets=truth,predictions=records,
                test_history=reg['test_history'],native_fallback=False,training_receipt_sha256=file_hash(runner.out/'training/DONE.json'))
            existing=runner.out/'REPORT.json'
            if existing.exists() and F.read(existing)!=report: raise RuntimeError('completed report changed')
            atomic_json(existing,report)
            atomic_json(runner.out/'STATUS.json',dict(complete=True,report_sha256=file_hash(existing),cache_only_verified=verify_only,seconds_this_invocation=time.time()-start))
            print(json.dumps(dict(job=job['id'],complete=True,rows=scores)),flush=True)
        except BaseException as exc:
            atomic_json(runner.out/'STATUS.json',dict(complete=False,error=str(exc))); raise
        finally: runner.close()


def aggregate(out,reg):
    finished=[]; failed=[]; pending=[]
    for job in reg['jobs']:
        directory=out/'jobs'/job['id']; p=directory/'STATUS.json'
        status=F.read(p) if p.exists() else {}
        if status.get('complete'):
            if status['report_sha256']!=file_hash(directory/'REPORT.json'): raise RuntimeError('report checksum changed')
            report=F.read(directory/'REPORT.json')
            for records in report['predictions'].values(): check_complete(report['query_ids'],records)
            finished.append(dict(job=job,rows=report['rows'],report_sha256=status['report_sha256']))
        elif status.get('error'): failed.append(dict(job=job['id'],error=status['error']))
        else: pending.append(job['id'])
    result=dict(complete=len(finished)==len(reg['jobs']),expected_jobs=len(reg['jobs']),completed_jobs=len(finished),
                completed=finished,failed=failed,pending=pending,partial_results_are_complete=False)
    atomic_json(out/'STATUS.json',result); return result


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run-id',required=True)
    ap.add_argument('--stage',choices=['prepare','job','verify','status'],required=True); ap.add_argument('--job')
    ap.add_argument('--execute',action='store_true'); ap.add_argument('--memory-gib',type=float,default=18.)
    args=ap.parse_args(); validate_run_id(args.run_id); out=ROOT/'results/cal40_full'/args.run_id
    if args.stage=='prepare':
        with run_lock(out/'prepare.lock'): reg=prepare(out)
        aggregate(out,reg); return
    reg=F.read(out/'REGISTRATION.json'); validate_sources(reg)
    if args.stage in ('job','verify'):
        if args.stage=='job' and not args.execute: raise ValueError('--execute required')
        job=next(j for j in reg['jobs'] if j['id']==args.job)
        if args.stage=='job':
            import torch
            total=torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(args.memory_gib*1024**3/total,1.))
        run_job(out,reg,job,verify_only=args.stage=='verify')
    else:
        r=aggregate(out,reg); print(json.dumps({k:r[k] for k in ['complete','expected_jobs','completed_jobs','failed','pending']}))


if __name__=='__main__': main()
