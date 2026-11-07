"""Outcome-blind data/model planning for new rebuttal experiments.

All role allocation happens before model evaluation. A bounded class-coverage
allocation is not a search over recognition performance or test outcomes.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import re
import numpy as np
import yaml
from akr_closing.core import digest,file_hash,read_manifest,make_split,support_ids
from akr_closing.pipeline import prompt_for


def positive_int(x,name):
    if isinstance(x,bool) or not isinstance(x,int) or x<1:raise ValueError(f'{name}: positive integer required')
    return x


def allocate_roles(rows,pool,labels,counts,seed,group_key):
    allowed=set(pool);rr=[r for r in rows if r['event_id'] in allowed]
    if not rr or any(not r.get(group_key) for r in rr):raise ValueError('real support groups required')
    groups=sorted({r[group_key] for r in rr});gi={g:i for i,g in enumerate(groups)};li={c:i for i,c in enumerate(labels)}
    matrix=np.zeros((len(groups),len(labels)),int)
    for r in rr:matrix[gi[r[group_key]],li[r['label']]]+=1
    if np.any(matrix.sum(0)<sum(counts.values())):raise ValueError('insufficient support class coverage')
    rng=np.random.default_rng(seed)
    for attempt in range(200):
        remaining=set(range(len(groups)));assigned={};possible=True
        for role in ['calibration','guard']:
            deficit=np.full(len(labels),counts[role]);selected=[]
            while np.any(deficit>0):
                candidates=sorted(remaining)
                if not candidates:possible=False;break
                gain=np.minimum(matrix[candidates],deficit).sum(1)
                score=gain/(1+matrix[candidates].sum(1)*.05)+rng.uniform(0,.2,len(candidates))
                score[gain==0]=-np.inf
                if not np.isfinite(score).any():possible=False;break
                pick=candidates[int(np.argmax(score))];selected.append(pick);remaining.remove(pick)
                deficit=np.maximum(0,deficit-matrix[pick])
            if not possible:break
            assigned[role]={groups[g] for g in selected}
        if not possible or not remaining or np.any(matrix[sorted(remaining)].sum(0)<counts['fit']):continue
        assigned['fit']={groups[g] for g in remaining};out={}
        for j,role in enumerate(['fit','calibration','guard']):
            candidates=[r['event_id'] for r in rr if r[group_key] in assigned[role]]
            out[role+'_ids']=support_ids(rows,candidates,labels,counts[role],seed=seed+100+j)
        return {**out,'allocation_attempt':attempt,'allocation_criterion':'training-side class coverage only'}
    raise ValueError('cannot obtain disjoint support/calibration/guard group coverage')


def build_study(root,config_path,tasks):
    root=Path(root).resolve();path=Path(config_path).resolve();cfg=yaml.safe_load(path.read_text())
    if not isinstance(cfg,dict) or cfg.get('version')!=1:raise ValueError('rebuttal config version 1 required')
    tasks=list(tasks)
    if not tasks or len(set(tasks))!=len(tasks) or any(t not in cfg['suites'] for t in tasks):raise ValueError('unknown/duplicate task')
    seeds=cfg['seeds']
    if not isinstance(seeds,list) or not seeds or len(set(seeds))!=len(seeds):raise ValueError('distinct seeds required')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in seeds):raise ValueError('nonnegative integer seeds required')
    s=cfg['settings'];positive_int(s['rank'],'rank');positive_int(s['max_context_tokens'],'context')
    positive_int(s.get('max_new_tokens',12),'max_new_tokens')
    for key in ['ridge_alpha','raw_eta','relative_alpha','gain_upper','gain_regularization','guard_tolerance','min_improvement']:
        if isinstance(s[key],bool) or not np.isfinite(s[key]) or s[key]<0:raise ValueError(f'invalid finite {key}')
    if s['ridge_alpha']<=0 or s['gain_upper']<1:raise ValueError('positive ridge, gain upper >=1 required')
    if not s['gain_steps'] or any(not np.isfinite(v) or v<=0 for v in s['gain_steps']):raise ValueError('positive gain steps required')
    if not s['layer_fractions'] or any(not np.isfinite(v) or not 0<=v<=1 for v in [*s['layer_fractions'],s['feature_fraction']]):raise ValueError('invalid layer fractions')
    if any(not np.isfinite(v) or v<0 for v in s.get('control_alphas',[])):raise ValueError('invalid control alpha')
    for tag,expected in [('qwen7b','Qwen/Qwen2.5-Omni-7B'),('qwen3b','Qwen/Qwen2.5-Omni-3B')]:
        m=cfg['models'][tag]
        if m['id']!=expected or not re.fullmatch('[0-9a-f]{40}',m['revision']):raise ValueError('pinned supported model revision required')
    from akr_closing.model_factory import model_family
    for tag, model in cfg['models'].items():
        if not re.fullmatch('[A-Za-z0-9_-]+', tag):raise ValueError('unsafe model tag')
        model_family(model['id'])
        if not re.fullmatch('[0-9a-f]{40}',model['revision'] or ''):raise ValueError('pinned model revision required')
    jobs=[];needed={}
    for task in tasks:
        suite=cfg['suites'][task];name=suite['dataset'];tag=suite['model']
        if name not in cfg['datasets'] or tag not in cfg['models'] or suite['seeds'] not in ['all','first']:raise ValueError('invalid suite')
        if not re.fullmatch('[A-Za-z0-9_-]+',name):raise ValueError('unsafe dataset name')
        selected=seeds if suite['seeds']=='all' else seeds[:1]
        control_policy=suite.get('controls','first' if task=='core' else 'none')
        if control_policy not in ['first','all','none']:raise ValueError('controls must be first/all/none')
        for seed in selected:
            needed.setdefault(name,set()).add(seed)
            for c in cfg['datasets'][name]['conditions']:
                jobs.append({'suite':task,'dataset':name,'model':tag,'seed':seed,'condition':c,
                    'episode':f's{seed}_{c.replace(":","_")}',
                    'controls':control_policy=='all' or (control_policy=='first' and seed==selected[0])})
    job_keys=[(j['dataset'],j['model'],j['episode']) for j in jobs]
    if len(set(job_keys))!=len(job_keys):raise ValueError('duplicate model/dataset/episode jobs')
    datasets={}
    for name,used_seeds in needed.items():
        ds=deepcopy(cfg['datasets'][name]);manifest=Path(ds['manifest'])
        settings={**s, **{key:positive_int(ds[key],key) for key in ['rank','max_new_tokens'] if key in ds}}
        if not manifest.is_absolute():manifest=root/manifest
        rows=sorted(read_manifest(manifest,old_root=cfg.get('old_root'),root=root),key=lambda r:r['event_id'])
        labels=ds.get('labels');hashes={str(path):file_hash(path),str(manifest):file_hash(manifest)}
        for source in ds.get('registration_sources',[]):
            source=root/source
            hashes[str(source)]=file_hash(source)
        if labels is None:
            lp=root/ds['labels_from_config'];labels=yaml.safe_load(lp.read_text())['dataset']['labels'];hashes[str(lp)]=file_hash(lp)
        if not labels or len(set(labels))!=len(labels) or set(labels)!={r['label'] for r in rows}:raise ValueError('label vocabulary mismatch')
        if ds['partition']=='beans_zero_supervised':
            from .beans_zero_protocol import validate_registration
            if not ds.get('supervised_registration'):
                raise ValueError('BEANS-Zero supervised registration file is required')
            hashes.update(validate_registration(root,root/ds['supervised_registration'],name,manifest,labels,rows))
        conditions=ds['conditions']
        if not conditions or len(set(conditions))!=len(conditions) or any(c not in ['full','lp:1000'] for c in conditions):raise ValueError('only Full/LP1 supported')
        counts={role:positive_int(ds[role+'_per_class'],role) for role in ['fit','calibration','guard']}
        if settings['rank']>counts['fit']*len(labels)-1:raise ValueError('rank exceeds centered fit capacity')
        group=ds['group_key'];lookup={r['event_id']:r for r in rows};episodes=[];splits={}
        for seed in sorted(used_seeds):
            roles=None
            if ds['partition']=='recording_group_confirmation':
                split=make_split(rows,labels,seed=seed,group_key=group)
                pool=split['train']+split['selection'];query=split['confirmation']
            elif ds['partition']=='official_test':
                if any(r['split'] not in ['train','valid','validation','test'] for r in rows):raise ValueError('official splits required')
                query=[r['event_id'] for r in rows if r['split']=='test']
                query_groups={lookup[i].get(group) for i in query}
                query_hashes={lookup[i]['sha256'] for i in query if lookup[i].get('sha256')}
                excluded=[r['event_id'] for r in rows if r['split']=='train' and
                          (r.get(group) in query_groups or (r.get('sha256') and r['sha256'] in query_hashes))]
                pool=[r['event_id'] for r in rows if r['split']=='train' and r['event_id'] not in set(excluded)]
                split={'training_pool':pool,'test':query,'excluded_train_ids':excluded,
                       'exclusion_rule':'source group or supplied waveform hash overlaps official test; query population unchanged',
                       'unused_valid':[r['event_id'] for r in rows if r['split'] in ['valid','validation']]}
            elif ds['partition']=='beans_zero_supervised':
                from .beans_zero_protocol import allocate
                if group!='recording_group' or set(counts.values())!={1}:
                    raise ValueError('BEANS-Zero supervised protocol requires one example per role/class and recording_group')
                allocation=allocate(rows,labels,seed)
                roles={key:allocation[key] for key in ['fit_ids','calibration_ids','guard_ids']}
                query=allocation['query_ids'];pool=allocation['support_ids']
                split={**allocation,'evaluation_status':'derived supervised re-evaluation; not original zero-shot benchmark'}
            else:raise ValueError('unknown partition policy')
            if not query or any(not lookup[i].get(group) for i in pool+query):raise ValueError('real group values required')
            if roles is None:roles=allocate_roles(rows,pool,labels,counts,seed,group)
            support=roles['fit_ids']+roles['calibration_ids']+roles['guard_ids']
            role_sets=[{lookup[i][group] for i in ids} for ids in [roles['fit_ids'],roles['calibration_ids'],roles['guard_ids'],query]]
            if any(role_sets[a]&role_sets[b] for a in range(4) for b in range(a+1,4)):raise ValueError('role/query group overlap')
            splits[str(seed)]=split
            for c in conditions:
                fixed={'rank':settings['rank'],'ridge_alpha':s['ridge_alpha'],'alpha':s['relative_alpha']}
                episodes.append({'seed':seed,'condition':c,'id':f's{seed}_{c.replace(":","_")}',
                    'support_ids':support,'query_ids':query,**roles,
                    'lock':{'selection':fixed,'support':support,'prompt':prompt_for(labels)},
                    'meta':{'condition':c,'support_ids':support},'lock_sha256':digest(fixed)})
        internal={'models':[{'tag':tag,**spec} for tag,spec in cfg['models'].items()],'E9':cfg['models']['qwen3b'],
                  **{k:s[k] for k in ['layer_fractions','feature_fraction','max_context_tokens','ridge_alpha']}}
        plan={'mode':'standalone','config':internal,'dataset':{'name':name,**ds},'labels':labels,'rows':rows,
              'episodes':episodes,'splits':splits,'source_hashes':hashes,
              'historical_metadata_recovered':False,'selection_uses_outcomes':False,'query_partition':ds['partition'],
              'evaluation_status':'new rebuttal protocol on previously inspected source data; not an untouched test'}
        if ds['partition']=='beans_zero_supervised':
            plan['evaluation_status']='derived supervised BEANS-Zero closed-label re-evaluation; eligible classes only; not original zero-shot benchmark'
        datasets[name]={'plan':plan,'options':{'dataset':name,'settings':settings,'all_labelled_per_class':sum(counts.values())}}
    return {'config':cfg,'config_sha256':file_hash(path),'tasks':tasks,'jobs':jobs,'datasets':datasets,
            'new_scientific_results':False,'old_RUN_LOCK_required':False}
