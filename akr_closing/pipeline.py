"""Executable closing-study pipeline. Does not overwrite any historical output.

all = registered support scaling + selection + confirmation + gated locked
re-evaluation + matched ablations + support-only gradient nulls + full reports.
The optional LoRA launcher uses the same declared decoder q/v targets with an
explicit optimizer-step budget and development-only selection. Failed old tokenwise gates are never reopened by this pipeline.
"""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import numpy as np
import yaml
from .core import Query,Journal,atomic_json,digest,file_hash,read_manifest,make_split,support_ids,metrics,paired_stats,confirmation_gate
from .repair import RidgeReadout,GradientRouter,RepositoryGradientRouter,matched_random,geometry
from .backend import ContextLimit


class SafeStop(RuntimeError): pass
STOP=False

def request_stop(*_):
    global STOP
    STOP=True

def check_stop():
    if STOP: raise SafeStop('Stop requested; completed items are durably checkpointed.')


def write_npz(path,**values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**values); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def prepare_audio(path,out,condition):
    import soundfile as sf
    from scipy.signal import butter,sosfiltfilt,resample_poly
    if out.exists(): return out
    x,sr=sf.read(path,dtype='float64',always_2d=True); x=x.mean(1)
    if len(x)<32 or not np.isfinite(x).all(): raise ValueError(f'invalid/too short waveform: {path}')
    if condition!='full':
        kind,cut=condition.split(':',1)
        if kind=='lp':
            cutoff=float(cut)
            if not 0<cutoff<sr/2: raise ValueError('low-pass cutoff must be below original Nyquist')
            sos=butter(10,cutoff,btype='lowpass',fs=sr,output='sos')
        elif kind=='notch':
            low,high=map(float,cut.split('-'))
            if low==0:
                sos=butter(10,high,btype='highpass',fs=sr,output='sos')
            else: sos=butter(10,[low,high],btype='bandstop',fs=sr,output='sos')
        else: raise ValueError('conditions: full, lp:1000, notch:2000-4000')
        x=sosfiltfilt(sos,x,padlen=min(len(x)-1,3*(2*len(sos)+1)))
    divisor=math.gcd(int(sr),16000); x=resample_poly(x,16000//divisor,int(sr)//divisor)
    out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_name(out.stem+'.tmp.wav')
    sf.write(tmp,x,16000,subtype='FLOAT'); os.replace(tmp,out)
    return out


def prompt_for(labels,variant='canonical'):
    ordered=list(reversed(labels)) if variant=='reverse_order' else labels
    choices=', '.join(ordered)
    if variant=='paraphrase': return f'Identify the vocal category of this recording. Reply with exactly one label from: {choices}.'
    if variant not in {'canonical','reverse_order'}: raise ValueError('unknown prompt')
    return f'Listen to the animal vocalization. Classify it using exactly one of these labels: {choices}. Output the label only.'


class Study:
    def __init__(self,root:Path,config:dict,backend_factory,*,profile='24gb',run_id='closing_v1'):
        self.root=root; self.config=config; self.factory=backend_factory; self.profile=profile
        self.out=root/'results'/'akr_closing'/run_id; self.out.mkdir(parents=True,exist_ok=True)
        code=Path(__file__).parent
        self.provenance={'config':config,'profile':profile,'code':{p.name:file_hash(p) for p in sorted(code.glob('*.py'))},
            'manifests':{d['name']:file_hash(root/d['manifest']) for d in config['datasets']},
            'repository_sources':{str(p.relative_to(root)):file_hash(p) for p in
                [root/'src/animal_omni/conditional_kv.py',root/'src/animal_omni/qwen_runner.py',root/'src/animal_omni/metrics.py'] if p.exists()}}
        manifest=self.out/'RUN.json'
        if manifest.exists() and json.loads(manifest.read_text())!=self.provenance:
            raise RuntimeError('run ID already belongs to different config/code; choose a new --run-id')
        atomic_json(manifest,self.provenance)
        self.backend=None; self.records=[]; self.blocked=[]; self.completed_execution=False

    def _journal(self,path,meta,expected):
        return Journal(self.out/path,{'run':digest(self.provenance),**meta},expected)

    def preflight(self):
        info=[]
        for ds in self.config['datasets']:
            rows=read_manifest(self.root/ds['manifest'],old_root=self.config.get('old_root'),root=self.root)
            missing=[r['audio_path'] for r in rows if not Path(r['audio_path']).is_file()]
            if missing: raise FileNotFoundError(f"{ds['name']}: {len(missing)} WAVs missing, first: {missing[0]}; see docs/AKR_RUNBOOK_ZH.md")
            labels=ds['labels']
            if set(r['label'] for r in rows)!=set(labels): raise ValueError(f"{ds['name']}: label mismatch")
            dsinfo={'dataset':ds['name'],'n':len(rows),'classes':len(labels),'manifest_sha256':file_hash(self.root/ds['manifest']),
                    'split_provenance':'existing datasets, previously inspected; all new evaluations are locked re-evaluations'}
            # Hash duplicates across original splits are a fatal train/test leak.
            byhash={}
            for r in rows:
                sha=r.get('sha256')
                if sha: byhash.setdefault(sha,set()).add(r.get('split',''))
            leaks=[s for s in byhash.values() if 'train' in s and 'test' in s]
            if leaks: raise ValueError(f"{ds['name']}: duplicate waveform hash across official train/test")
            for seed in self.config['seeds']:
                split=make_split(rows,labels,seed=seed,group_key=ds.get('group_key'))
                atomic_json(self.out/'splits'/f"{ds['name']}_{seed}.json",split)
                # Validate requested support without borrowing from query partitions.
                support_ids(rows,split['train'],labels,max(ds['support_k']),seed=seed)
            wavehashes={r['event_id']:file_hash(Path(r['audio_path'])) for r in rows}
            actual_splits={}
            for r in rows:
                actual_splits.setdefault(wavehashes[r['event_id']],set()).add(r.get('split',''))
            if any('train' in names and 'test' in names for names in actual_splits.values()):
                raise ValueError(f"{ds['name']}: duplicate source waveform bytes across official train/test")
            hashpath=self.out/'splits'/f"{ds['name']}_waveform_hashes.json"
            if hashpath.exists() and json.loads(hashpath.read_text())!=wavehashes:raise RuntimeError('source waveform content changed within run ID')
            atomic_json(hashpath,wavehashes)
            info.append(dsinfo)
        atomic_json(self.out/'PREFLIGHT.json',{'datasets':info,'status':'passed'})
        return info

    def _query(self,row,dsname,condition):
        source=Path(row['audio_path']); ident=digest({'source':str(source),'hash':file_hash(source),'condition':condition})
        out=self.out/'audio'/dsname/(ident+'.wav'); prepare_audio(source,out,condition)
        return Query(row['event_id'],str(out),row.get('recording_id') or row['event_id'])

    def _feature(self,q,prompt,context):
        path=self.out/'features'/context/(digest({'q':q.event_id,'path':q.audio_path,'prompt':prompt})+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as data: return data['feature']
        check_stop(); start=time.perf_counter();before=getattr(self.backend,'forward_count',None)
        value=self.backend.feature(q,prompt); write_npz(path,feature=value)
        atomic_json(Path(str(path)+'.cost.json'),{'kind':'query_or_support_feature_forward','wall_seconds':time.perf_counter()-start,
            'model_forward_calls':self.backend.forward_count-before if before is not None else None})
        return value

    def _gradients(self,q,target,prompt,context,variant='real'):
        path=self.out/'support_gradients'/context/(digest({'id':q.event_id,'path':q.audio_path,'target':target,'prompt':prompt,'variant':variant})+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}
        check_stop(); start=time.perf_counter();before=getattr(self.backend,'forward_count',None)
        value=self.backend.support_gradients(q,target,prompt); write_npz(path,**value)
        atomic_json(Path(str(path)+'.cost.json'),{'kind':'support_forward_and_backward','wall_seconds':time.perf_counter()-start,
            'model_forward_calls':self.backend.forward_count-before if before is not None else None,
            'target':target,'target_token_ids':self.backend.target_token_ids(target) if hasattr(self.backend,'target_token_ids') else None})
        return value

    def _score(self,records,rows,labels,stage,job):
        target=[r['label'] for r in rows]; preds=[r['prediction'] for r in records]
        result={'job':job,'stage':stage,'status':'complete',**metrics(target,preds,labels)}
        costs=[r.get('wall_seconds') for r in records if r.get('wall_seconds') is not None]
        forwards=[r.get('model_forward_calls') for r in records if r.get('model_forward_calls') is not None]
        peaks=[r.get('peak_cuda_allocated_bytes') for r in records if r.get('peak_cuda_allocated_bytes') is not None]
        result.update(mean_wall_seconds=float(np.mean(costs)) if costs else None,
            mean_model_forwards=float(np.mean(forwards)) if forwards else None,peak_cuda_bytes=max(peaks) if peaks else None)
        atomic_json(self.out/job/'SCORED_PREDICTIONS.json',[{**rec,'event_id':row['event_id'],'target':row['label']} for row,rec in zip(rows,records)])
        atomic_json(self.out/job/'METRICS.json',result)
        self.records.append(result); return result

    def _predictions(self,path,meta,queries,fn):
        journal=self._journal(path,meta,[q.event_id for q in queries])
        if (journal.root/'BLOCKED.json').exists():
            self.blocked.append({'job':str(path),**json.loads((journal.root/'BLOCKED.json').read_text())})
            return None
        for i,q in enumerate(queries):
            if journal.has(q.event_id): continue
            check_stop()
            try:
                started=time.perf_counter()
                try:
                    import torch
                    on_cuda=torch.cuda.is_available()
                    if on_cuda: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                except ImportError: on_cuda=False
                before_forward=getattr(self.backend,'forward_count',None)
                result=fn(q)
                if before_forward is not None:result['model_forward_calls']=self.backend.forward_count-before_forward
                if on_cuda: torch.cuda.synchronize()
                result['wall_seconds']=time.perf_counter()-started
                result['peak_cuda_allocated_bytes']=int(torch.cuda.max_memory_allocated()) if on_cuda else None
            except ContextLimit as exc:
                block={'status':'blocked_context','reason':str(exc),'completed':i,'expected':len(queries)}
                atomic_json(journal.root/'BLOCKED.json',block); self.blocked.append({'job':str(path),**block}); return None
            journal.add(q.event_id,result)
            if i==0 or (i+1)%25==0: print(f'{path}: {i+1}/{len(queries)}',flush=True)
        return journal.results()

    def _readout(self,features,y,query_features,labels,ridge_alpha):
        readout=RidgeReadout.fit(features,y,len(labels),alpha=ridge_alpha)
        ridge=[labels[i] for i in readout.predict(query_features)]
        mean=np.stack([features[y==c].mean(0) for c in range(len(labels))])
        # Raw embedding cosine prototypes; fitted only from identical support.
        cosine=query_features@mean.T/(np.maximum(np.linalg.norm(query_features,axis=1,keepdims=True),1e-12)*
                                     np.maximum(np.linalg.norm(mean,axis=1)[None,:],1e-12))
        centroid=[labels[i] for i in cosine.argmax(1)]
        return readout,ridge,centroid

    def _directions(self,features,y,gradients,query_features,rank,alpha):
        router_type=RepositoryGradientRouter if self.config.get('router_implementation','repository')=='repository' else GradientRouter
        router=router_type.fit(features,gradients,rank=rank,alpha=alpha)
        readout=RidgeReadout.fit(features,y,int(y.max()+1),alpha=alpha)
        centroids=np.stack([gradients[y==c].mean(0) for c in range(int(y.max()+1))])
        return {'continuous_pooled':router.predict(query_features),
                'class_pooled':readout.probabilities(query_features)@centroids,
                'class_hard':centroids[readout.predict(query_features)],
                'fixed_mean':np.repeat(gradients.mean(0)[None,:],len(query_features),0)},readout

    def run(self,phase='all'):
        self.preflight()
        if phase=='preflight': return
        global STOP
        STOP=False
        self.completed_execution=False; self.blocked=[]
        signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
        try:
            for model in self.config['models']:
                self.backend=self.factory(model['id'],model['revision'],profile=self.profile,
                    layer_fractions=self.config['layer_fractions'],feature_fraction=self.config['feature_fraction'],
                    max_context_tokens=self.config['max_context_tokens'])
                atomic_json(self.out/f"BACKEND_{model['tag']}.json",self.backend.provenance)
                for ds in self.config['datasets']:
                    raw=read_manifest(self.root/ds['manifest'],old_root=self.config.get('old_root'),root=self.root)
                    rawby={r['event_id']:r for r in raw}
                    for seed in self.config['seeds']:
                        split=json.loads((self.out/'splits'/f"{ds['name']}_{seed}.json").read_text())
                        for mode in ds['label_modes']:
                            real_labels=ds['labels']; labels=real_labels if mode=='semantic' else [chr(65+i) for i in range(len(real_labels))]
                            if mode!='semantic' and len(labels)>26: raise ValueError('use semantic labels for >26 classes')
                            mapping=dict(zip(real_labels,labels)); rows=[{**r,'label':mapping[r['label']]} for r in raw]
                            lookup={r['event_id']:r for r in rows}
                            for condition in ds['conditions']:
                                context=f"{model['tag']}/{ds['name']}/s{seed}/{mode}/{condition.replace(':','_')}"
                                for k in ds['support_k']:
                                    self._episode(context,ds,lookup,rows,split,labels,seed,k,condition,phase)
                del self.backend; self.backend=None
                import gc; gc.collect()
                try:
                    import torch; torch.cuda.empty_cache()
                except ImportError: pass
            self.completed_execution=True
        except SafeStop:
            atomic_json(self.out/'STOPPED.json',{'status':'safe_stop','message':'Rerun exactly the same command to resume.'})
            print('Safe stop: completed items retained; no partial metrics published.',flush=True)
        finally:
            self.report()

    def _episode(self,context,ds,lookup,rows,split,labels,seed,k,condition,phase):
        if phase not in {'scaling','all'} and k not in ds['repair_k']: return
        base=Path(context)/f'k{k}';prompt=prompt_for(labels)
        sid=support_ids(rows,split['train'],labels,k,seed=seed)
        support=[(self._query(lookup[i],ds['name'],condition),lookup[i]['label']) for i in sid]
        meta={'context':context,'support_ids':sid,'support_labels':[l for _,l in support],
              'k_per_class':k,'condition':condition,'backbone':self.backend.provenance,
              'split_fingerprint':split['fingerprint'],'evaluation_status':'locked_reevaluation',
              'model_parameters':'frozen; auxiliary supervised regression is fitted on support'}
        atomic_json(self.out/base/'EPISODE.json',meta)
        fs=np.stack([self._feature(q,prompt,context) for q,_ in support])
        sy=np.array([labels.index(l) for _,l in support]);ridge_alpha=self.config['ridge_alpha']
        querysets={s:[self._query(lookup[i],ds['name'],condition) for i in split[s]]
                   for s in ['selection','confirmation']}
        featuremap={q.event_id:self._feature(q,prompt,context) for qs in querysets.values() for q in qs}
        # New scaling/order development does not consume confirmation labels before a lock.
        if phase in {'scaling','all'}:
            qs=querysets['selection'];qr=[lookup[q.event_id] for q in qs]
            fq=np.stack([featuremap[q.event_id] for q in qs])
            readout,ridge,centroid=self._readout(fs,sy,fq,labels,ridge_alpha)
            for name,pred in [('ridge',ridge),('centroid',centroid),('probe_to_text',ridge)]:
                self._score([{'prediction':p,'model_forward_calls':0} for p in pred],qr,labels,
                            'selection',str(base/'scaling'/name))
            for name in ['native','native_candidates','icl','icl_candidates']:
                if name.startswith('icl') and k not in ds['icl_k']:continue
                def evaluate(q,name=name):
                    examples=support if name.startswith('icl') else ()
                    if name.endswith('candidates'):return self.backend.candidates(q,prompt,labels,support=examples)
                    value=self.backend.predict(q,prompt,labels,support=examples)
                    if examples:
                        value['first_support_label']=examples[0][1];value['last_support_label']=examples[-1][1]
                    return value
                result=self._predictions(base/'scaling'/name,meta,qs,evaluate)
                if result is not None:
                    self._score(result,qr,labels,'selection',str(base/'scaling'/name))
                    if name.endswith('candidates'):
                        self._score([{**r,'prediction':r['mean_prediction']} for r in result],qr,labels,
                                    'selection',str(base/'scaling'/(name+'_mean')))
            if phase=='scaling':return
        if k not in ds['repair_k']:return
        gs=[self._gradients(q,label,prompt,context) for q,label in support]
        matrices={scope:np.stack([g[scope] for g in gs]) for scope in ['audio','text','full_prefill']}
        for scope,g in matrices.items():atomic_json(self.out/base/'geometry'/f'{scope}.json',geometry(g,sy))
        if phase=='E3':
            predeclared={'rank':self.config.get('comparison_rank',4),'alpha':self.config.get('comparison_alpha',.01),
                         'ridge_alpha':ridge_alpha,'selection':'predeclared E3 setting, not tuned on queries'}
            self._repair_evaluate(base,'selection_E3',querysets['selection'],lookup,labels,['canonical'],
                 fs,sy,matrices,featuremap,context,predeclared,meta,seed,ablations=False)
            return
        selection=querysets['selection'];sf=np.stack([featuremap[q.event_id] for q in selection])
        sr=[lookup[q.event_id] for q in selection];options=[]
        for penalty in self.config.get('ridge_penalties',[ridge_alpha]):
            for rank in self.config['ranks']:
                directions,_=self._directions(fs,sy,matrices['audio'],sf,rank,penalty)
                for alpha in self.config['relative_alphas']:
                    name=f'continuous_r{rank}_a{alpha:g}_ridge{penalty:g}'
                    table={q.event_id:d for q,d in zip(selection,directions['continuous_pooled'])}
                    result=self._predictions(base/'selection'/name,{**meta,'rank':rank,'alpha':alpha,'ridge_alpha':penalty},selection,
                           lambda q:self.backend.predict(q,prompt,labels,direction=table[q.event_id],alpha=alpha))
                    if result is None:return
                    score=self._score(result,sr,labels,'selection',str(base/'selection'/name))
                    if score['invalid_rate']<=self.config['max_invalid_rate']:
                        options.append({'rank':rank,'alpha':alpha,'ridge_alpha':penalty,
                                        'accuracy':score['accuracy'],'macro_f1':score['macro_f1']})
        if not options:
            atomic_json(self.out/base/'BLOCKED.json',{'status':'no_valid_selection','test_permitted':False});return
        chosen=sorted(options,key=lambda x:(-x['accuracy'],-x['macro_f1'],x['rank'],x['alpha'],x['ridge_alpha']))[0]
        lock={'selection':chosen,'support':sid,'protocol':digest(meta),'feature_layer':self.backend.provenance.get('feature_layer'),
              'layers':self.backend.provenance.get('layers'),'parser':'repository normalize_label',
              'prompt':prompt,'scale_type':'relative_active_state_norm','status':'locked_reevaluation',
              'test_gate':'cluster-bootstrap lower CI > 0 vs native; invalid <= registered max'}
        lockpath=self.out/base/'LOCK.json'
        if lockpath.exists() and json.loads(lockpath.read_text())!=lock:raise RuntimeError('selection lock changed')
        atomic_json(lockpath,lock)
        if phase=='selection':return
        if phase=='E4':
            self._null_gradients(support,sy,labels,prompt,context,base,matrices,fs,querysets['confirmation'],
                                 lookup,featuremap,chosen,meta,seed)
            return
        stage='confirmation' if phase!='E5' else 'confirmation_E5'
        confirm=self._repair_evaluate(base,stage,querysets['confirmation'],lookup,labels,self.config['prompts'],
                 fs,sy,matrices,featuremap,context,chosen,meta,seed,ablations=phase in {'E5','all'})
        stats=confirm['comparison'];gate=confirmation_gate(stats,confirm['primary']['invalid_rate'])
        if phase!='E5':
            atomic_json(self.out/base/'GATE.json',{'passed':gate,'comparison':stats,
                    'lock_fingerprint':digest(lock),'old_tokenwise_gate':'unchanged; never reopened'})
        if phase=='E5' or (phase=='all' and self.config.get('gradient_nulls',False) and k==min(ds['repair_k'])):
            self._null_gradients(support,sy,labels,prompt,context,base,matrices,fs,querysets['confirmation'],
                                 lookup,featuremap,chosen,meta,seed)
        if phase in {'all','E2'} and gate and self.config.get('run_test_after_gate',False):
            test=[self._query(lookup[i],ds['name'],condition) for i in split['test']]
            for q in test:featuremap[q.event_id]=self._feature(q,prompt,context)
            self._repair_evaluate(base,'test_locked_reevaluation',test,lookup,labels,self.config['prompts'],
                    fs,sy,matrices,featuremap,context,chosen,meta,seed,ablations=False)

    def _repair_evaluate(self,base,stage,queries,lookup,labels,prompts,fs,sy,matrices,featuremap,context,chosen,meta,seed,ablations=True):
        fq=np.stack([featuremap[q.event_id] for q in queries]);byscope={}
        for scope,g in matrices.items():
            byscope[scope],readout=self._directions(fs,sy,g,fq,chosen['rank'],chosen.get('ridge_alpha',self.config['ridge_alpha']))
        direct_scores=readout.mapping.predict(fq)
        router_prediction=[labels[j] for j in np.argmax(direct_scores,axis=1)]
        qr=[lookup[q.event_id] for q in queries];target=[r['label'] for r in qr]
        groups=[q.recording_id for q in queries];canonical={}
        from .protocol import probe_to_text
        direct=probe_to_text(router_prediction)
        for rec,scores in zip(direct,direct_scores):rec['class_scores']=scores.tolist()
        for name in ['ridge_direct','probe_to_text']:
            self._score(direct,qr,labels,stage,str(base/stage/name))
            atomic_json(self.out/base/stage/name/'predictions.json',[
                {**r,'event_id':q.event_id} for q,r in zip(queries,direct)])
        for variant in prompts:
            prompt=prompt_for(labels,variant);predicted=byscope['audio']['continuous_pooled']
            configs=[('native',None,0.,'audio','kv','all'),
                ('fixed_mean',byscope['audio']['fixed_mean'],chosen['alpha'],'audio','kv','all'),
                ('continuous_pooled',predicted,chosen['alpha'],'audio','kv','all'),
                ('class_pooled',byscope['audio']['class_pooled'],chosen['alpha'],'audio','kv','all'),
                ('class_hard',byscope['audio']['class_hard'],chosen['alpha'],'audio','kv','all'),
                ('zero_alpha',predicted,0.,'audio','kv','all'),
                ('zero_direction',np.zeros_like(predicted),chosen['alpha'],'audio','kv','all')]
            if ablations and variant=='canonical':
                configs += [('K_only',predicted,chosen['alpha'],'audio','k','all'),
                    ('V_only',predicted,chosen['alpha'],'audio','v','all'),
                    ('early_layers',predicted,chosen['alpha'],'audio','kv','early'),
                    ('middle_layers',predicted,chosen['alpha'],'audio','kv','middle'),
                    ('late_layers',predicted,chosen['alpha'],'audio','kv','late'),
                    ('text_scope',byscope['text']['continuous_pooled'],chosen['alpha'],'text','kv','all'),
                    ('full_prefill',byscope['full_prefill']['continuous_pooled'],chosen['alpha'],'full_prefill','kv','all'),
                    ('random_matched',np.stack([matched_random(f,seed+i) for i,f in enumerate(predicted)]),chosen['alpha'],'audio','kv','all'),
                    ('shuffled_query_field',np.roll(predicted,1,axis=0),chosen['alpha'],'audio','kv','all'),
                    ('negative_direction',-predicted,chosen['alpha'],'audio','kv','all')]
            results={};scored={}
            for name,field,alpha,scope,kind,layer_group in configs:
                table={} if field is None else {q.event_id:d for q,d in zip(queries,field)}
                res=self._predictions(base/stage/variant/name,{**meta,'lock':chosen,'prompt':prompt,
                     'method':name,'alpha':alpha,'scope':scope,'kind':kind,'layer_group':layer_group},queries,
                     lambda q:self.backend.predict(q,prompt,labels,direction=table.get(q.event_id),alpha=alpha,
                             scope=scope,kind=kind,layer_group=layer_group))
                if res is None:raise ContextLimit('incomplete repair job; no scientific summary')
                results[name]=res
                score=self._score(res,qr,labels,stage,str(base/stage/variant/name));scored[name]=score
                if name in {'continuous_pooled','class_pooled','class_hard'}:
                    pr=np.array([r['prediction'] for r in res]);correctrouter=np.array(router_prediction)==np.array(target)
                    strata={'router_accuracy':float(correctrouter.mean()),
                        'agreement_with_router':float(np.mean(pr==np.array(router_prediction))),
                        'router_correct_n':int(correctrouter.sum()),'router_wrong_n':int((~correctrouter).sum()),
                        'accuracy_given_router_correct':float(np.mean(pr[correctrouter]==np.array(target)[correctrouter])) if correctrouter.any() else None,
                        'accuracy_given_router_wrong':float(np.mean(pr[~correctrouter]==np.array(target)[~correctrouter])) if (~correctrouter).any() else None}
                    atomic_json(self.out/base/stage/variant/name/'ROUTER_STRATA.json',strata)
            for zero in ['zero_alpha','zero_direction']:
                if [(r.get('raw_prediction'),r['prediction']) for r in results[zero]] != [(r.get('raw_prediction'),r['prediction']) for r in results['native']]:
                    raise RuntimeError(f'{zero} not a strict native no-op; investigate before reporting')
            if self.config.get('probe_to_lm',True):
                table=dict(zip([q.event_id for q in queries],router_prediction))
                res=self._predictions(base/stage/variant/'probe_to_lm',{**meta,'prompt':prompt,'lock':chosen},queries,
                    lambda q:self.backend.predict(q,prompt+f' An external classifier predicts {table[q.event_id]}. Use this evidence when answering.',labels))
                if res is not None:
                    results['probe_to_lm']=res;self._score(res,qr,labels,stage,str(base/stage/variant/'probe_to_lm'))
                    arr=np.array([r['prediction'] for r in res]);rp=np.array(router_prediction);yt=np.array(target)
                    atomic_json(self.out/base/stage/variant/'probe_to_lm'/'COMPLIANCE.json',{
                        'agreement':float(np.mean(arr==rp)),'classifier_correct_lm_wrong':int(np.sum((rp==yt)&(arr!=yt))),
                        'classifier_wrong_lm_correct':int(np.sum((rp!=yt)&(arr==yt)))})
            comparisons={name:paired_stats(target,[r['prediction'] for r in results['native']],
                           [r['prediction'] for r in res],groups,seed=seed) for name,res in results.items() if name!='native'}
            comparisons['continuous_minus_fixed']=paired_stats(target,[r['prediction'] for r in results['fixed_mean']],
                                [r['prediction'] for r in results['continuous_pooled']],groups,seed=seed)
            comparisons['continuous_minus_direct_ridge']=paired_stats(target,router_prediction,
                                [r['prediction'] for r in results['continuous_pooled']],groups,seed=seed)
            atomic_json(self.out/base/stage/variant/'paired_comparisons.json',comparisons)
            if variant=='canonical':canonical={'comparison':comparisons['continuous_pooled'],'primary':scored['continuous_pooled']}
        return canonical

    def _null_gradients(self,support,sy,labels,prompt,context,base,real,fs,queries,lookup,featuremap,chosen,meta,seed):
        from .protocol import matched_audio,compare_gradient_targets,shuffle_support_labels
        shuffled=shuffle_support_labels(sy,seed=seed)
        names=['silence_same_label','wrong_audio_same_label','same_audio_wrong_label','support_label_shuffle']
        fields={name:[] for name in names};assignments=[]
        for i,(q,label) in enumerate(support):
            other=next(sq for sq,sl in support if sl!=label)
            folder=self.out/'audio'/'target_controls'/digest([q.audio_path,other.audio_path])
            silence=matched_audio(Path(q.audio_path),None,folder/'silence.wav')
            wrong=matched_audio(Path(q.audio_path),Path(other.audio_path),folder/'wrong.wav')
            cases={'silence_same_label':(Query(q.event_id,str(silence),q.recording_id),label),
                'wrong_audio_same_label':(Query(q.event_id,str(wrong),q.recording_id),label),
                'same_audio_wrong_label':(q,labels[(labels.index(label)+1)%len(labels)]),
                'support_label_shuffle':(q,labels[int(shuffled[i])])}
            assignments.append({'support_id':q.event_id,'target':label,'wrong_audio_donor':other.event_id,
                                'shuffled_target':labels[int(shuffled[i])]})
            for name,(audio,target) in cases.items():fields[name].append(self._gradients(audio,target,prompt,context,variant=name)['audio'])
        atomic_json(self.out/base/'geometry'/'CONTROL_ASSIGNMENTS.json',assignments)
        summary={'real':geometry(real['audio'],sy)}
        nulls={name:np.stack(value) for name,value in fields.items()}
        centroids={c:real['audio'][sy==c].mean(0) for c in np.unique(sy)}
        shared=np.stack([centroids[c] for c in sy]);nulls['class_centroid_targets']=shared
        nulls['class_residual_targets']=real['audio']-shared
        for name,g in nulls.items():
            y=shuffled if name=='support_label_shuffle' else (sy+1)%len(labels) if name=='same_audio_wrong_label' else sy
            comparison=compare_gradient_targets(real['audio'],g,sy)
            if name in {'support_label_shuffle','same_audio_wrong_label'}:
                comparison['paired_gradient_cosine']=comparison.pop('median_same_target_cosine')
                comparison['target_relation']='changed/permuted targets'
            else:comparison['target_relation']='same targets'
            summary[name]={**geometry(g,y),**comparison}
        atomic_json(self.out/base/'geometry'/'label_controlled_nulls.json',summary)
        fq=np.stack([featuremap[q.event_id] for q in queries]);qr=[lookup[q.event_id] for q in queries]
        # All target-control regressors see the SAME original support/query features.
        for name,g in {'real_targets':real['audio'],**nulls}.items():
            directions,_=self._directions(fs,sy,g,fq,chosen['rank'],chosen.get('ridge_alpha',self.config['ridge_alpha']))
            table=dict(zip([q.event_id for q in queries],directions['continuous_pooled']))
            result=self._predictions(base/'confirmation_E4'/name,{**meta,'lock':chosen,'target_control':name,
                 'regressor_features':'original acoustic features held fixed'},queries,
                 lambda q:self.backend.predict(q,prompt,labels,direction=table[q.event_id],alpha=chosen['alpha']))
            if result is not None:self._score(result,qr,labels,'confirmation_E4',str(base/'confirmation_E4'/name))
        # Paired effects are derived only after every target-control job is complete.
        basepath=base/'confirmation_E4';comparison={}
        ref=self._journal(basepath/'real_targets',{**meta,'lock':chosen,'target_control':'real_targets',
                'regressor_features':'original acoustic features held fixed'},[q.event_id for q in queries])
        if ref.complete:
            rp=[r['prediction'] for r in ref.results()]
            for name in nulls:
                jp=self._journal(basepath/name,{**meta,'lock':chosen,'target_control':name,
                     'regressor_features':'original acoustic features held fixed'},[q.event_id for q in queries])
                if jp.complete:comparison[name]=paired_stats([r['label'] for r in qr],
                    [r['prediction'] for r in jp.results()],rp,[q.recording_id for q in queries],seed=seed)
            atomic_json(self.out/basepath/'REAL_MINUS_CONTROL.json',comparison)

    def report(self):
        from .report import report_run
        result=report_run(self.out)
        self.records=json.loads((self.out/'COMPLETE_METRICS.json').read_text())
        atomic_json(self.out/'SUMMARY.json',{'completed_summaries':self.records,'blocked':self.blocked,
                   'requested_phase_finished':self.completed_execution,
                   'unblocked_phase_complete':self.completed_execution and not STOP and not self.blocked,
                   'warning':'Completion does not imply a positive result. Historical test data remain re-evaluation.'})
        atomic_json(self.out/'COVERAGE.json',json.loads((self.out/'COVERAGE_ALL.json').read_text()))
        return result
