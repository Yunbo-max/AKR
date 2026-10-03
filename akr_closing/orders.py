"""E7: every held-out query is evaluated under the same counterbalanced orders."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import csv
import json
import time
import numpy as np
import yaml
from .core import Query,Journal,atomic_json,digest,file_hash,read_manifest,metrics,paired_stats
from .legacy import validate_exact_split


def counterbalanced_orders(rows,labels,*,seed):
    ids=[r['event_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(r['label'] for r in rows)!=set(labels):raise ValueError('unique, class-complete support required')
    buckets={c:[r['event_id'] for r in rows if r['label']==c] for c in labels}
    if len({len(v) for v in buckets.values()})!=1:raise ValueError('balanced support required')
    k=len(next(iter(buckets.values())));orders={'historical_fixed':ids}
    rng=np.random.default_rng(seed)
    for i,c in enumerate(labels):
        classes=labels[i:]+labels[:i]
        orders['first_class_'+c]=[buckets[cl][j] for j in range(k) for cl in classes]
        rev=list(reversed(classes))
        orders['reversed_cycle_'+c]=[buckets[cl][j] for j in range(k) for cl in rev]
        p=list(rng.permutation(ids));head=buckets[c][0];p.remove(head)
        orders['random_first_'+c]=[head]+p
    return orders


def summarize_orders(records,queries,orders,labels,*,seed):
    expected={(q,o) for q in queries for o in orders};bykey={}
    for r in records:
        key=(r['event_id'],r['order'])
        if key not in expected or key in bykey:raise ValueError('foreign/duplicate order record')
        bykey[key]=r
    out={'complete':set(bykey)==expected,'completed':len(bykey),'expected':len(expected)}
    if not out['complete']:return out
    results={};agreements=[]
    for order in orders:
        rs=[bykey[q,order] for q in queries];pred=[r['prediction'] for r in rs];target=[r['target'] for r in rs]
        first=float(np.mean([r['prediction']==r['first_label'] for r in rs]));agreements.append(first)
        results[order]={**metrics(target,pred,labels),'first_agreement':first,
                        'last_agreement':float(np.mean([r['prediction']==r['last_label'] for r in rs])),
                        'prediction_counts':dict(Counter(pred))}
    reference=orders[0];comp={}
    for order in orders[1:]:
        comp[order]=paired_stats([bykey[q,reference]['target'] for q in queries],
            [bykey[q,reference]['prediction'] for q in queries],[bykey[q,order]['prediction'] for q in queries],
            [bykey[q,reference].get('recording_id',q) for q in queries],seed=seed)
    out.update(metrics=results,paired_to_fixed=comp,mean_first_agreement=float(np.mean(agreements)),
       prediction_change_rate=float(np.mean([len({bykey[q,o]['prediction'] for o in orders})>1 for q in queries])),
       interpretation='support order intervention; not an isolated causal effect of the first token alone')
    return out


def run_orders(root,config,*,run_id,execute=False):
    e=config['E1'];cfg=yaml.safe_load((root/e['config']).read_text());labels=cfg.get('labels') or cfg['dataset']['labels']
    prompt=cfg.get('prompts',{}).get('bare') or cfg['evaluation']['prompt']
    rows=read_manifest(root/e['manifest'],old_root=config.get('old_root'),root=root);byid={r['event_id']:r for r in rows}
    split=json.loads((root/e['split']).read_text());sid,qid=validate_exact_split(rows,split,labels,k=8,expected_query=75)
    orders=counterbalanced_orders([byid[i] for i in sid],labels,seed=config['seeds'][0])
    folder=root/'results'/'akr_closing'/run_id/'E7'
    meta={'split_sha256':file_hash(root/e['split']),'prompt':prompt,'orders':orders,'model':e['model_id'],
          'model_revision':e['model_revision'],'query_ids':qid,'audio_hashes':{i:file_hash(Path(byid[i]['audio_path'])) for i in sid+qid},
          'backend_hash':file_hash(root/'src/animal_omni/qwen_runner.py'),
          'code':file_hash(Path(__file__)),'stage':'historical-query repeated diagnostic; not untouched'}
    j=Journal(folder,meta,[digest([q,o]) for o in orders for q in qid])
    if execute and not j.complete:
        from animal_omni.qwen_runner import QwenThinkerRunner
        from animal_omni.metrics import normalize_label
        from .pipeline import check_stop
        from huggingface_hub import snapshot_download
        runner=QwenThinkerRunner(snapshot_download(e['model_id'],revision=e['model_revision']))
        for order,ids in orders.items():
            support=[(byid[i]['audio_path'],byid[i]['label']) for i in ids]
            for eid in qid:
                key=digest([eid,order])
                if j.has(key):continue
                check_stop();q=Query(eid,byid[eid]['audio_path'],byid[eid]['recording_id']);start=time.perf_counter()
                raw=runner.predict_icl(support,q.audio_path,prompt,max_new_tokens=8)
                prediction=normalize_label(raw,labels) or ''
                j.add(key,dict(event_id=eid,order=order,raw_prediction=raw,prediction=prediction,
                      target=byid[eid]['label'],recording_id=q.recording_id,first_label=support[0][1],last_label=support[-1][1],
                      wall_seconds=time.perf_counter()-start,support_ids=ids))
    atomic_json(folder/'COVERAGE.json',j.status())
    if not j.complete:return j.status()
    result=summarize_orders(j.results(),qid,list(orders),labels,seed=config['seeds'][0]);atomic_json(folder/'SUMMARY.json',result)
    return result
