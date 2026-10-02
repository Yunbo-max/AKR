"""E1 exact historical resume: no new split, no changed prompt, no row deletion."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
import numpy as np
import yaml
from .core import atomic_json,digest,file_hash,read_manifest,metrics,paired_stats


def validate_exact_split(rows,split,labels,*,k=8,expected_query=75):
    byid={r['event_id']:r for r in rows}
    if len(byid)!=len(rows):raise ValueError('duplicate manifest events; specify one condition')
    support=split['support_sets'][str(k)];query=split['query_events']
    if len(query)!=expected_query or len(query)!=len(set(query)):raise ValueError('query set is not exact or unique')
    if len(support)!=len(set(support)) or len(support)!=k*len(labels):raise ValueError('support count mismatch')
    if set(support)&set(query):raise ValueError('support/query event overlap')
    if not set(support+query)<=set(byid):raise ValueError('events missing from manifest')
    if Counter(byid[i]['label'] for i in support)!=Counter({c:k for c in labels}):raise ValueError('support is not K per class')
    if all(byid[i].get('recording_id') for i in support+query):
        if {byid[i]['recording_id'] for i in support}&{byid[i]['recording_id'] for i in query}:
            raise ValueError('support/query recording overlap')
    return support,query


def candidate_summary(rows,query_ids,labels,*,k=8,groups=None):
    seen=set(); predictions=[];means=[];targets=[];byid={}
    for row in rows:
        eid=row['event_id']
        if eid not in query_ids or eid in seen:raise ValueError('foreign/duplicate query in E1 output')
        if int(row['support_k_per_class'])!=k or row['readout']!='candidate':raise ValueError('wrong K/readout in legacy CSV')
        if int(row['support_k_total'])!=k*len(labels):raise ValueError('legacy support total differs from exact registered episode')
        if row['target_output'] not in labels:raise ValueError('foreign target label in legacy CSV')
        seen.add(eid)
        scores=json.loads(row['candidate_scores_json'])
        if len(scores)!=len(labels) or {s['candidate'] for s in scores}!=set(labels):
            raise ValueError('missing, duplicate, or foreign candidate scores')
        if any(not np.isfinite(s[x]) for s in scores for x in ['sequence_logprob','mean_token_logprob']):
            raise ValueError('nonfinite candidate likelihood')
        byid[eid]=(row['target_output'],max(scores,key=lambda s:s['sequence_logprob'])['candidate'],
                    max(scores,key=lambda s:s['mean_token_logprob'])['candidate'])
    result={'status':'complete' if seen==set(query_ids) else 'incomplete','completed':len(seen),'expected':len(query_ids),
            'missing_ids':[q for q in query_ids if q not in seen]}
    if result['status']!='complete':return result
    for eid in query_ids:
        target,pred,mean=byid[eid]; targets.append(target);predictions.append(pred);means.append(mean)
    result.update(sequence_sum=metrics(targets,predictions,labels),mean_token=metrics(targets,means,labels),
                  sequence_output_counts=dict(Counter(predictions)),mean_output_counts=dict(Counter(means)),
                  sequence_minus_mean=paired_stats(targets,means,predictions,groups,seed=20260914))
    return result


def resume_e1(root:Path,config:dict,*,run_id:str,execute=False):
    e=config['E1'];folder=root/'results'/'akr_closing'/run_id/'E1';folder.mkdir(parents=True,exist_ok=True)
    cfgpath=root/e['config']; splitpath=root/e['split']; manifestpath=root/e['manifest']
    cfg=yaml.safe_load(cfgpath.read_text());labels=cfg.get('labels') or cfg['dataset']['labels']
    rows=read_manifest(manifestpath,old_root=config.get('old_root'),root=root)
    split=json.loads(splitpath.read_text());support,query=validate_exact_split(rows,split,labels,k=8,expected_query=75)
    byid={r['event_id']:r for r in rows}
    missing=[byid[i]['audio_path'] for i in support+query if not Path(byid[i]['audio_path']).exists()]
    if missing:raise FileNotFoundError(f'E1 missing {len(missing)} WAVs; first {missing[0]}')
    source=root/e['predictions']
    checkpoint=folder/'candidate_predictions.csv'
    if not checkpoint.exists() and source.exists():checkpoint.write_bytes(source.read_bytes())
    existing=list(csv.DictReader(checkpoint.open())) if checkpoint.exists() else []
    candidate_summary(existing,query,labels,k=8)
    # Freeze the source mapping and source code BEFORE resuming inherited rows.
    sourcefiles=[cfgpath,splitpath,root/'scripts/evaluate_equal_support_audio_icl.py',root/'src/animal_omni/qwen_runner.py']
    identity={'source_manifest_sha256':file_hash(manifestpath),'support_ids':support,'query_ids':query,
              'source_files':{str(p.relative_to(root)):file_hash(p) for p in sourcefiles},'model_id':e['model_id'],
              'model_revision':e['model_revision'],
              'audio_sha256':{i:file_hash(Path(byid[i]['audio_path'])) for i in support+query},
              'policy':'same legacy backend and native source audio; no new filtering/resampling in E1',
              'inherited_provenance':'legacy partial rows retained; their pre-existing execution provenance is not reconstructed'}
    lock=folder/'IDENTITY.json'
    if lock.exists() and json.loads(lock.read_text())!=identity:raise RuntimeError('E1 identity changed; do not mix checkpoints')
    if not lock.exists():
        atomic_json(lock,identity)
        atomic_json(folder/'INHERITED_ROWS.json',{r['event_id']:digest(r) for r in existing})
    inherited=json.loads((folder/'INHERITED_ROWS.json').read_text())
    for r in existing:
        if r['event_id'] in inherited and digest(r)!=inherited[r['event_id']]:raise RuntimeError('legacy row was changed')
    # Paths are relocated into an equivalent manifest; identities and source audio are unchanged.
    relocated=folder/'resolved_manifest.csv'
    with relocated.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    cmd=[sys.executable,str(root/'scripts/evaluate_equal_support_audio_icl.py'),
         '--config',str(cfgpath),'--manifest',str(relocated),'--split',str(splitpath),
         '--model-id',e['model_id'],'--support-k-per-class','8','--readout','candidate',
         '--output',str(checkpoint),'--summary',str(folder/'legacy_summary.json'),'--resume']
    atomic_json(folder/'COMMAND.json',{'argv':cmd,'execute_requested':execute})
    if execute and set(r['event_id'] for r in existing)!=set(query):
        from huggingface_hub import snapshot_download
        # The legacy wrapper has no revision argument; resolve the immutable snapshot first.
        local=snapshot_download(e['model_id'],revision=e['model_revision'])
        cmd[cmd.index('--model-id')+1]=str(local)
        atomic_json(folder/'EXECUTED_COMMAND.json',{'argv':cmd,'resolved_model_revision':e['model_revision']})
        from .process import run_checked
        run_checked(cmd,cwd=root,log_path=folder/'run.log')
    after=list(csv.DictReader(checkpoint.open())) if checkpoint.exists() else []
    for eid,h in inherited.items():
        match=[r for r in after if r['event_id']==eid]
        if len(match)!=1 or digest(match[0])!=h:raise RuntimeError('inherited result lost or changed')
    summary=candidate_summary(after,query,labels,k=8,groups=[byid[i]['recording_id'] for i in query])
    atomic_json(folder/'SUMMARY.json',summary)
    return summary
