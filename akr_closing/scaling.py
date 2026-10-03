"""Optional E9: transfer 7B-selected ranks/scales/support to 3B without new search."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .core import atomic_json,file_hash,read_manifest
from .pipeline import Study,prompt_for


def confirm_3b(root,config,factory,*,run_id,profile,execute=False):
    source=root/'results'/'akr_closing'/run_id
    locks=sorted((source/'qwen7b').rglob('LOCK.json')) if (source/'qwen7b').exists() else []
    if not locks:raise FileNotFoundError('E9 requires completed 7B parameter locks; do not choose settings on 3B')
    cfg=deepcopy(config);cfg['models']=[{'tag':'qwen3b','id':cfg['E9']['id'],'revision':cfg['E9']['revision']}]
    cfg['run_test_after_gate']=False
    target=Study(root,cfg,factory,profile=profile,run_id=run_id+'_3b_locked')
    target.preflight();plan=[]
    for path in locks:
        meta=json.loads((path.parent/'EPISODE.json').read_text());lock=json.loads(path.read_text())
        plan.append({'source_lock':str(path),'source_sha256':file_hash(path),'settings':lock['selection'],'episode':meta})
    atomic_json(target.out/'E9_TRANSFER_PLAN.json',{'locks':plan,'new_hyperparameter_search':False,'execute_requested':execute})
    if not execute:return {'planned_locked_episodes':len(plan),'target_run':target.out.name}
    model=cfg['models'][0];target.backend=factory(model['id'],model['revision'],profile=profile,
         layer_fractions=cfg['layer_fractions'],feature_fraction=cfg['feature_fraction'],max_context_tokens=cfg['max_context_tokens'])
    for entry in plan:
        meta=entry['episode'];context=meta['context'];parts=context.split('/');dsname=parts[1];seed=int(parts[2][1:]);mode=parts[3]
        ds=next(d for d in cfg['datasets'] if d['name']==dsname)
        real_labels=ds['labels'];labels=real_labels if mode=='semantic' else [chr(65+i) for i in range(len(real_labels))]
        mapping=dict(zip(real_labels,labels))
        rows=read_manifest(root/ds['manifest'],old_root=cfg.get('old_root'),root=root)
        lookup={r['event_id']:{**r,'label':mapping[r['label']]} for r in rows}
        split=json.loads((source/'splits'/f'{dsname}_{seed}.json').read_text());condition=meta['condition']
        sid=meta['support_ids'];support=[(target._query(lookup[i],dsname,condition),lookup[i]['label']) for i in sid]
        newcontext=context.replace('qwen7b/','qwen3b/',1);base=Path(newcontext)/f"k{meta['k_per_class']}";prompt=prompt_for(labels)
        fs=np.stack([target._feature(q,prompt,newcontext) for q,_ in support]);sy=np.array([labels.index(l) for _,l in support])
        grads=[target._gradients(q,label,prompt,newcontext) for q,label in support]
        matrices={scope:np.stack([g[scope] for g in grads]) for scope in ['audio','text','full_prefill']}
        queries=[target._query(lookup[i],dsname,condition) for i in split['confirmation']]
        features={q.event_id:target._feature(q,prompt,newcontext) for q in queries}
        provenance={**meta,'context':newcontext,'source_7b_lock_sha256':entry['source_sha256'],'backbone':target.backend.provenance}
        target._repair_evaluate(base,'confirmation_from_7b_lock',queries,lookup,labels,cfg['prompts'],
            fs,sy,matrices,features,newcontext,entry['settings'],provenance,seed,ablations=False)
    target.completed_execution=True;target.report();return {'completed_locked_episodes':len(plan),'target_run':target.out.name}
