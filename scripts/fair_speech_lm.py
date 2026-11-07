#!/usr/bin/env python3
"""Matched labelled-pool Speech LM comparison; stages allow two real GPUs."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from akr_closing.core import Journal, atomic_json, digest, file_hash, read_manifest, paired_stats
from akr_closing.cli import run_lock, validate_run_id
from akr_closing.pipeline import prompt_for
from akr_closing.repair import RidgeReadout
from akr_final.beans_zero_protocol import canonical_pcm_hash
from akr_final.calibration_pilot import PilotRunner, choose_alpha
from akr_final.predictability import fit_repository
from akr_final.rebuttal_runner import measured, array_hash
from akr_final.runner import prediction_summary

LRS = (5e-5, 2e-4, 5e-4)
EPOCHS = (1, 3, 5)
KINDS = ('regression', 'shrunk', 'class_pooled')
STRENGTHS = (.01, .03, .1)
SOURCE_RUN = 'pool_dogs_3b_20261107_v2'
METHODS = ('native', 'akr_reference', 'akr_selected', 'lora')


def read(path): return json.loads(Path(path).read_text())


def append_answer(inputs, answer, max_context):
    import torch
    n = inputs['input_ids'].shape[1]
    if answer.ndim != 2 or answer.shape[0] != 1 or not answer.shape[1] or n+answer.shape[1] > max_context:
        raise ValueError('empty answer or context limit exceeded')
    result = dict(inputs)
    result['input_ids'] = torch.cat([inputs['input_ids'], answer], 1)
    result['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(answer)], 1)
    labels = torch.full_like(result['input_ids'], -100); labels[:, n:] = answer
    result['labels'] = labels
    return result


def language_targets(model):
    targets = [n for n,m in model.named_modules()
               if re.fullmatch(r'model\.layers\.\d+\.self_attn\.(q_proj|v_proj)', n)]
    if not targets: raise ValueError('no language decoder q/v projections')
    return sorted(targets)


def attach_lora(model, seed):
    import torch
    from peft import LoraConfig, get_peft_model
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    for p in model.parameters(): p.requires_grad_(False)
    targets = language_targets(model)
    adapted = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    trainable = [(n,p) for n,p in adapted.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n or '.self_attn.' not in n or
                            any(t in n for t in ['audio_tower','visual']) for n,p in trainable):
        raise RuntimeError('unexpected trainable parameters')
    return adapted, dict(target_modules=targets, trainable_parameters=sum(p.numel() for n,p in trainable),
                         audio_trainable_parameters=0, rank=8, alpha=16, dropout=.05, seed=seed)


def register_adapter_counter(backend):
    """PEFT direct forward bypasses the base-module hook; generate still uses it."""
    if getattr(backend,'_adapter_count_handle',None) is not None:
        raise RuntimeError('adapter counter already registered')
    def count(*_): backend.forward_count+=1
    backend._adapter_count_handle=backend.thinker.register_forward_hook(count)


def select_epoch(folds):
    if len(folds) != 2: raise ValueError('two folds required')
    for f in folds:
        if set(f) != {0,1,3,5}: raise ValueError('all epoch candidates required')
        choose_alpha(f)  # Also validates per-fold paired lengths and finite values.
    choice = choose_alpha({e:list(folds[0][e])+list(folds[1][e]) for e in [0,1,3,5]})
    return {'epoch':int(choice['alpha']), **{k:v for k,v in choice.items() if k!='alpha'}}


def select_candidate(losses):
    if 'native' not in losses: raise ValueError('native candidate missing')
    arrays = {k:np.asarray(v,float) for k,v in losses.items()}
    if any(v.ndim!=1 or not len(v) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('finite nonempty losses required')
    if len({len(v) for v in arrays.values()}) != 1: raise ValueError('unpaired candidates')
    means = {k:float(v.mean()) for k,v in arrays.items()}
    best = min(means, key=lambda k:(means[k], k))
    accepted = means['native']-means[best] > 1e-6
    return dict(candidate=best if accepted else 'native', accepted=accepted,
                reason='oof_ce_improved' if accepted else 'no_oof_ce_improvement', mean_ce=means)


def fresh_queries(rows, labels, excluded, seed=20261108):
    groups = {r['source_filename'] for r in excluded}
    hashes = {r['pcm_sha256'] for r in excluded}
    selected = []
    for label in labels:
        choices = sorted((r for r in rows if r['label']==label and r['split'] in ['valid','validation']),
                         key=lambda r:digest([seed,r['event_id']]))
        kept = []
        for r in choices:
            if r['source_filename'] in groups or r['pcm_sha256'] in hashes: continue
            kept.append(r); groups.add(r['source_filename']); hashes.add(r['pcm_sha256'])
            if len(kept)==4: break
        if len(kept)!=4: raise ValueError('insufficient distinct new development examples')
        selected.extend(kept)
    return selected


def verify_stage(directory, identity):
    path = Path(directory)/'DONE.json'
    if not path.exists(): return None
    envelope = read(path); result = envelope.get('result', {})
    if envelope.get('sha256') != digest(result) or result.get('identity') != identity or not result.get('complete'):
        raise RuntimeError('changed/corrupt stage receipt')
    for name, sha in result['files'].items():
        if not Path(name).is_file() or file_hash(Path(name)) != sha: raise RuntimeError('changed adapter/checkpoint')
    return result


def finish_stage(directory, result):
    result = {**result, 'complete':True}
    atomic_json(Path(directory)/'DONE.json', dict(result=result, sha256=digest(result)))
    return result


def prepare(out):
    source = ROOT/'results/calibration_pilot'/SOURCE_RUN
    validation = read(source/'VALIDATION.json'); report = read(source/'dogs/REPORT.json')
    status = read(source/'dogs/STATUS.json'); original = read(source/'REGISTRATION.json')
    if not all(v['complete'] for v in [validation, report, status]): raise RuntimeError('source pilot incomplete')
    if file_hash(source/'dogs/REPORT.json') != validation['report_sha256'] or status['report_sha256'] != validation['report_sha256']:
        raise RuntimeError('source report changed')
    saved = read(source/'dogs/IDENTITY.json')
    for name,sha in {**saved['code'], **original['plan']['source_hashes']}.items():
        if file_hash(Path(name)) != sha: raise RuntimeError(f'source changed: {name}')
    entry = deepcopy(original); e, = entry['plan']['episodes']; pool=set(e['support_ids'])
    rows=read_manifest(ROOT/'data/manifests/beans_dogs_protocol.csv', root=ROOT, old_root='/root/animal-omni-kv')
    # Rudy has only four official validation rows, all used by the pilot.
    # Reuse the EXACT development population and disclose prior inspection.
    # Do not fabricate a fresh split or consume official test during debugging.
    entry['plan']['evaluation_status']='previously inspected 40-example development subset; matched comparison; not official test'
    paths=[source/'VALIDATION.json', source/'dogs/REPORT.json', source/'REGISTRATION.json',
           source/'dogs/IDENTITY.json', Path(__file__),
           ROOT/'docs/superpowers/plans/2026-11-07-fair-speech-lm.md']
    arrays={}
    for meta in (source/'dogs/cache').glob('*.npz.json'):
        item=read(meta); ident=item['identity']
        if ident['event'] not in pool: continue
        if ident['run'] != validation['fingerprint'] or ident['tag']!='qwen3b' or ident['condition']!='lp:1000':
            raise RuntimeError('foreign source array')
        path=Path(str(meta)[:-5])
        if file_hash(path)!=item['sha256']: raise RuntimeError('source array checksum')
        key=ident['event']+'/'+ident['kind']
        if key in arrays: raise RuntimeError('duplicate source array')
        arrays[key]=str(path); paths.extend([meta,path])
    if set(arrays)!={i+'/'+k for i in pool for k in ['feature','gradient']}: raise RuntimeError('missing support array')
    entry['plan']['source_hashes'].update({str(p):file_hash(p) for p in paths})
    registration=dict(entry=entry, arrays=arrays, source_run=SOURCE_RUN,
        reference_policy=report['policies']['cv_refit'], lrs=list(LRS), epochs=list(EPOCHS),
        kinds=list(KINDS), strengths=list(STRENGTHS), prior_query_ids=original['plan']['episodes'][0]['query_ids'])
    dest=out/'REGISTRATION.json'
    if dest.exists() and read(dest)!=registration: raise RuntimeError('registration changed; use a new run ID')
    atomic_json(dest,registration)
    return registration


def load_registration(out):
    registration=read(out/'REGISTRATION.json')
    for name,sha in registration['entry']['plan']['source_hashes'].items():
        if not Path(name).is_file() or file_hash(Path(name))!=sha: raise RuntimeError(f'changed source: {name}')
    return registration


class FairRunner(PilotRunner):
    def __init__(self, out, stage, registration):
        super().__init__(ROOT,out/'workers',stage,registration['entry'],'24gb')
        self.registration=registration
        self.preflight(); self.verify_caches()

    def _array(self,event,condition,tag,kind,episode):
        key=event+'/'+kind
        if key in self.registration['arrays']:
            path=Path(self.registration['arrays'][key])
            if file_hash(path)!=self.plan['source_hashes'][str(path)]: raise RuntimeError('imported array changed')
            with np.load(path,allow_pickle=False) as z: value=z['value'].copy()
            if not np.isfinite(value).all(): raise ValueError('nonfinite imported array')
            return value
        return super()._array(event,condition,tag,kind,episode)


def training_inputs(backend, runner, event):
    e,=runner.plan['episodes']
    if event not in e['support_ids']: raise ValueError('training label outside calibration pool')
    x=backend._prepare(runner._query(event,e['condition']),prompt_for(runner.labels))
    answer=backend.processor.tokenizer(runner.lookup[event]['label'],add_special_tokens=False,
        return_tensors='pt')['input_ids'].to(backend.device)
    return append_answer(x,answer,backend.max_context_tokens)


def train_stage(runner, directory, train_ids, held_ids, lr, max_epoch, seed):
    import torch
    e,=runner.plan['episodes']
    if not set(train_ids+held_ids)<=set(e['support_ids']) or set(train_ids)&set(held_ids):
        raise ValueError('invalid LoRA fold boundaries')
    identity=dict(run=runner.fingerprint, train=train_ids, held=held_ids, lr=lr,
                  max_epoch=max_epoch,seed=seed,accumulation=4)
    saved=verify_stage(directory,identity)
    if saved is not None: return saved
    directory.mkdir(parents=True,exist_ok=True)
    ip=directory/'IDENTITY.json'
    if ip.exists() and read(ip)!=identity: raise RuntimeError('changed training stage')
    atomic_json(ip,identity)
    runner.close(); backend=runner._model('qwen3b',e)
    before=backend.forward_count; started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    losses={}; history=[]; updates=0; exposures=0
    def evaluate(ids):
        if not set(ids)<=set(e['support_ids']): raise ValueError('heldout loss outside pool')
        backend.thinker.eval()
        values=[]
        for event in ids:
            x=training_inputs(backend,runner,event)
            with torch.inference_mode(): loss=backend.thinker(**x,use_cache=False,return_dict=True).loss
            value=float(loss.float().cpu())
            if not np.isfinite(value): raise ValueError('nonfinite heldout CE')
            values.append(value)
        return values
    if held_ids: losses['0']=evaluate(held_ids)
    parity_event=train_ids[0]
    parity_x=training_inputs(backend,runner,parity_event)
    with torch.inference_mode(): initial_loss=backend.thinker(**parity_x,use_cache=False).loss.detach().clone()
    backend.thinker,audit=attach_lora(backend.thinker,seed)
    register_adapter_counter(backend)
    backend.thinker.eval()
    with torch.inference_mode(): zero_loss=backend.thinker(**parity_x,use_cache=False).loss.detach().clone()
    if not torch.equal(initial_loss,zero_loss): raise RuntimeError('zero adapter does not match base')
    del parity_x
    backend.thinker.enable_input_require_grads()
    backend.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    parameters=[p for p in backend.thinker.parameters() if p.requires_grad]
    frozen=[(p,p._version) for p in backend.thinker.parameters() if not p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.01)
    files={}; rng=np.random.default_rng(seed)
    for epoch in range(1,max_epoch+1):
        backend.thinker.train()
        base=backend.thinker.get_base_model()
        for name in ['audio_tower','visual']:
            if hasattr(base,name): getattr(base,name).eval()
        order=[train_ids[j] for j in rng.permutation(len(train_ids))]
        train_losses=[]
        optimizer.zero_grad(set_to_none=True)
        for start in range(0,len(order),4):
            chunk=order[start:start+4]
            for event in chunk:
                loss=backend.thinker(**training_inputs(backend,runner,event),use_cache=False,return_dict=True).loss
                if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
                train_losses.append(float(loss.detach().float().cpu()))
                (loss/len(chunk)).backward(); exposures+=1
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); updates+=1
            if any(not torch.isfinite(p).all() for p in parameters): raise ValueError('nonfinite adapter update')
        if any(p.requires_grad or p._version!=v for p,v in frozen): raise RuntimeError('frozen base changed')
        if epoch in EPOCHS or epoch==max_epoch:
            if held_ids: losses[str(epoch)]=evaluate(held_ids)
            checkpoint=directory/f'epoch_{epoch}'
            backend.thinker.save_pretrained(checkpoint,safe_serialization=True)
            for name in ['adapter_model.safetensors','adapter_config.json']:
                path=checkpoint/name; files[str(path)]=file_hash(path)
        item=dict(epoch=epoch,train_mean_ce=float(np.mean(train_losses)),exposures=exposures,
                  optimizer_steps=updates,heldout_mean_ce=float(np.mean(losses[str(epoch)])) if str(epoch) in losses else None)
        history.append(item); atomic_json(directory/'HISTORY.json',history)
        print(json.dumps({'stage':str(directory.name),**item}),flush=True)
    backend.thinker.eval()
    parity_x=training_inputs(backend,runner,parity_event)
    with backend.thinker.disable_adapter(),torch.inference_mode():
        disabled_loss=backend.thinker(**parity_x,use_cache=False).loss
    if not torch.equal(initial_loss,disabled_loss): raise RuntimeError('disabled adapter changed base behavior')
    result=finish_stage(directory,dict(identity=identity,files=files,losses=losses,history=history,
        audit=audit,train_ids=train_ids,heldout_ids=held_ids,train_exposures=exposures,
        optimizer_steps=updates,forward_calls=backend.forward_count-before,
        forward_count_scope='top-level calls including PEFT wrapper; excludes backward/checkpoint recomputation',
        wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        frozen_base_versions_unchanged=True,zero_adapter_parity=True,disabled_adapter_parity=True,
        final_adapter=str(directory/f'epoch_{max_epoch}'),restart_policy='complete stages reused; interrupted stage restarts from deterministic initialization'))
    del optimizer,parameters,frozen,base,backend,parity_x,loss
    runner.close(); gc.collect(); torch.cuda.empty_cache()
    return result


def fold_stage(runner,out,fold):
    e,=runner.plan['episodes']; held=e['folds'][fold]
    train=[i for i in e['support_ids'] if i not in held]
    for lr in LRS:
        train_stage(runner,out/'training'/f'fold_{fold}'/f'lr_{lr:g}',train,held,lr,5,20261108+fold)


def choose_lora(out,registration,publish=True):
    audit=FairRunner(out,'selection_audit',registration)
    fingerprint=audit.fingerprint; audit.close()
    e,=registration['entry']['plan']['episodes']
    candidates={}; stages=[]
    for lr in LRS:
        folds=[]
        for f in range(2):
            path=out/'training'/f'fold_{f}'/f'lr_{lr:g}'
            held=e['folds'][f]; train=[i for i in e['support_ids'] if i not in held]
            expected=dict(run=fingerprint,train=train,held=held,lr=lr,max_epoch=5,seed=20261108+f,accumulation=4)
            stage=verify_stage(path,expected)
            if stage is None or stage['train_ids']!=train or stage['heldout_ids']!=held:
                raise RuntimeError('missing/misidentified LoRA fold')
            if set(stage['losses'])!={'0','1','3','5'} or any(len(v)!=len(held) or not np.isfinite(v).all() for v in stage['losses'].values()):
                raise RuntimeError('invalid heldout loss vectors')
            folds.append(stage); stages.append(stage)
        if 'native' not in candidates:
            candidates['native']=folds[0]['losses']['0']+folds[1]['losses']['0']
        elif candidates['native'] != folds[0]['losses']['0']+folds[1]['losses']['0']:
            raise RuntimeError('base loss differs across LoRA trials')
        for epoch in EPOCHS:
            candidates[f'lr_{lr:g}_epoch_{epoch}']=folds[0]['losses'][str(epoch)]+folds[1]['losses'][str(epoch)]
    choice=select_candidate(candidates)
    if choice['candidate']=='native': choice.update(lr=None,epoch=0)
    else:
        _, lr, _, epoch=choice['candidate'].split('_'); choice.update(lr=float(lr),epoch=int(epoch))
    choice['cv_compute']={k:sum(s[k] for s in stages) for k in ['train_exposures','optimizer_steps','forward_calls','wall_seconds']}
    if publish: atomic_json(out/'LORA_SELECTION.json',choice)
    return choice


class Correction:
    def __init__(self,x,g,y,nclass,kind,rank=4,ridge=10.):
        self.kind=kind; self.mean=g.mean(0)
        if kind=='class_pooled':
            self.readout=RidgeReadout.fit(x,y,nclass,alpha=ridge)
            self.centroids=np.stack([g[y==j].mean(0) for j in range(nclass)])
        else: self.mapping=fit_repository(x,g,rank=rank,alpha=ridge)
    def predict(self,x):
        if self.kind=='class_pooled': return self.centroids[self.readout.predict(x)]
        field=self.mapping.predict(x)
        return self.mean+.25*(field-self.mean) if self.kind=='shrunk' else field


def support_arrays(runner):
    e,=runner.plan['episodes']; pool=e['support_ids']
    x=runner.matrix(pool,e,'qwen3b'); g=runner.matrix(pool,e,'qwen3b','gradient')
    y=np.array([runner.labels.index(runner.lookup[i]['label']) for i in pool])
    return e,pool,x,g,y


def sensitivity(runner,out):
    e,pool,x,g,y=support_arrays(runner); rows=[]
    for rank in [0,1,2,4,8,16]:
        for ridge in [.1,1.,10.,100.,1000.]:
            error=0.; baseline=0.; cos=[]
            for held in e['folds']:
                h=[pool.index(i) for i in held]; t=[j for j in range(len(pool)) if j not in h]
                model=fit_repository(x[t],g[t],rank=rank,alpha=ridge); pred=model.predict(x[h])
                error+=float(np.square(pred-g[h]).sum()); baseline+=float(np.square(g[t].mean(0)-g[h]).sum())
                cos.extend((np.sum(pred*g[h],axis=1)/(np.linalg.norm(pred,axis=1)*np.linalg.norm(g[h],axis=1)).clip(1e-30)).tolist())
            rows.append(dict(rank=rank,ridge=ridge,relative_sse=error/baseline,mean_cosine=float(np.mean(cos))))
    result=dict(calibration_only=True,proxy_only=True,used_for_primary_selection=False,rows=rows,
        note='Gradient prediction sensitivity is not end-task accuracy sensitivity; no new GPU/model calls.')
    atomic_json(out/'PARAMETER_SENSITIVITY_PROXY.json',result)
    return result


def akr_select(runner,out,verify_only=False):
    e,pool,x,g,y=support_arrays(runner); candidates={'native':[]}
    for kind in KINDS:
        for alpha in STRENGTHS: candidates[f'{kind}_a{alpha:g}']=[]
    for fold,held in enumerate(e['folds']):
        h=[pool.index(i) for i in held]; t=[j for j in range(len(pool)) if j not in h]
        for kind in ('native',)+KINDS:
            model=None if kind=='native' else Correction(x[t],g[t],y[t],len(runner.labels),kind)
            fields=None if model is None else model.predict(x[h])
            for alpha in ((0.,) if kind=='native' else STRENGTHS):
                key='native' if kind=='native' else f'{kind}_a{alpha:g}'
                ident=dict(run=runner.fingerprint,fold=fold,kind=kind,alpha=alpha,ids=held,
                           fields=None if fields is None else array_hash(fields),mode='relative_norm')
                journal=Journal(runner.out/'loss_cache'/digest(ident),ident,held)
                for j,event in enumerate(held):
                    if journal.has(event): continue
                    if verify_only: raise RuntimeError('missing AKR selection evidence')
                    backend=runner._model('qwen3b',e)
                    value=measured(backend,lambda:backend.support_answer_loss(
                        runner._query(event,e['condition']),runner.lookup[event]['label'],prompt_for(runner.labels),
                        direction=None if fields is None else fields[j],alpha=alpha,scale_mode='relative_norm'))
                    if not np.isfinite(value['answer_loss']): raise ValueError('nonfinite AKR heldout CE')
                    journal.add(event,value)
                candidates[key].extend(r['answer_loss'] for r in journal.results())
                if not verify_only:
                    print(json.dumps({'stage':'akr_selection','fold':fold,'candidate':key,
                                      'mean_ce':float(np.mean(candidates[key]))}),flush=True)
    choice=select_candidate(candidates)
    if choice['candidate']=='native': choice.update(kind=None,alpha=0.)
    else:
        kind,alpha=choice['candidate'].rsplit('_a',1); choice.update(kind=kind,alpha=float(alpha))
    costs=[]
    for path in (runner.out/'loss_cache').glob('*/manifest.json'):
        meta=read(path); costs.extend(Journal(path.parent,meta['provenance'],meta['expected']).results())
    choice['cv_compute']=dict(forward_calls=sum(r['new_model_calls'] for r in costs),
        wall_seconds=sum(r['wall_seconds'] for r in costs),loss_records=len(costs),
        imported_support_gradients=len(pool),additional_backwards=0)
    if not verify_only: atomic_json(out/'AKR_SELECTION.json',choice)
    return choice


def _lock_policies(out,registration):
    lora=read(out/'LORA_SELECTION.json'); akr=read(out/'AKR_SELECTION.json')
    if lora!=choose_lora(out,registration,publish=False): raise RuntimeError('LoRA selection evidence mismatch')
    audit=FairRunner(out,'akr_select',registration)
    try:
        if akr!=akr_select(audit,out,verify_only=True): raise RuntimeError('AKR selection evidence mismatch')
        fingerprint=audit.fingerprint
    finally: audit.close()
    e,=registration['entry']['plan']['episodes']
    refit=read(out/'training/refit/DONE.json')
    if refit['sha256']!=digest(refit['result']): raise RuntimeError('refit receipt checksum')
    expected=(dict(run=fingerprint,train=e['support_ids'],held=[],lr=lora['lr'],max_epoch=lora['epoch'],
                   seed=20261110,accumulation=4) if lora['epoch'] else dict(run=fingerprint,native_selected=True))
    verify_stage(out/'training/refit',expected)
    lock=dict(registration=digest(registration),lora=lora,akr=akr,
              refit_sha256=file_hash(out/'training/refit/DONE.json'),
              reference=registration['reference_policy'],query_labels_used_for_selection=False)
    path=out/'POLICY_LOCK.json'
    with run_lock(out/'.policy.lock'):
        if path.exists():
            if read(path)!=lock: raise RuntimeError('locked policies changed')
        else: atomic_json(path,lock)
    return lock


def lock_policies(out,registration):
    # Two GPU inference workers may verify identical evidence concurrently.
    # Serialize audit-file creation as well as the final lock write.
    import fcntl
    with (out/'.policy_verification.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try: return _lock_policies(out,registration)
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def infer(runner,out,which):
    lock=lock_policies(out,runner.registration)
    e,pool,x,g,y=support_arrays(runner)
    backend=None
    methods=['lora'] if which=='lora' else ['native','akr_reference','akr_selected']
    fq=None
    for method in methods:
        fields=None; alpha=0.; mode='relative_norm'; raw_eta=300.
        if method in ['akr_reference','akr_selected']:
            if fq is None: fq=runner.matrix(e['query_ids'],e,'qwen3b')
            if method=='akr_reference':
                fields=fit_repository(x,g,rank=4,alpha=10.).predict(fq)
                alpha=lock['reference']['alpha']; mode='relative_cap'
            elif lock['akr']['candidate']!='native':
                fields=Correction(x,g,y,len(runner.labels),lock['akr']['kind']).predict(fq)
                alpha=lock['akr']['alpha']
        ident=dict(run=runner.fingerprint,method=method,policies=lock,
                   fields=None if fields is None else array_hash(fields),alpha=alpha,mode=mode)
        journal=Journal(runner.out/'trials'/method,ident,e['query_ids'])
        for j,event in enumerate(e['query_ids']):
            if journal.has(event): continue
            if backend is None:
                backend=runner._model('qwen3b',e)
                if which=='lora' and lock['lora']['epoch']:
                    from peft import PeftModel
                    adapter=read(out/'training/refit/DONE.json')['result']['final_adapter']
                    backend.thinker=PeftModel.from_pretrained(backend.thinker,adapter,is_trainable=False).eval()
                    if any(p.requires_grad for p in backend.thinker.parameters()): raise RuntimeError('inference adapter not frozen')
            value=measured(backend,lambda:backend.predict(runner._query(event,e['condition']),
                prompt_for(runner.labels),runner.labels,direction=None if fields is None else fields[j],
                alpha=alpha,scale_mode=mode,raw_eta=raw_eta,max_new_tokens=12))
            journal.add(event,value)
        if not journal.complete: raise RuntimeError('incomplete predictions')
        print(json.dumps({'completed_method':method,'query_count':len(e['query_ids'])}),flush=True)


def report(out,registration):
    e,=registration['entry']['plan']['episodes']; labels=registration['entry']['plan']['labels']
    lookup={r['event_id']:r for r in registration['entry']['plan']['rows']}
    records={}; evidence={}
    lock=lock_policies(out,registration)
    for method in METHODS:
        directory=out/'workers'/('eval_lora' if method=='lora' else 'eval_base')/'trials'/method
        meta=read(directory/'manifest.json')
        if meta['expected']!=e['query_ids'] or meta['provenance']['policies']!=lock: raise RuntimeError('foreign query predictions')
        for path in (directory/'items').glob('*.json'):
            item=read(path)
            if item.get('result_sha256')!=digest(item.get('result')): raise RuntimeError('prediction checksum')
        records[method]=Journal(directory,meta['provenance'],meta['expected']).results()
        evidence[str(directory/'manifest.json')]=file_hash(directory/'manifest.json')
    truth=[lookup[i]['label'] for i in e['query_ids']]; native=[r['prediction'] for r in records['native']]
    rows=[]
    for method in METHODS:
        predictions=[r['prediction'] for r in records[method]]
        row=prediction_summary(truth,native,predictions,labels)
        row.update(method=method,prediction_seconds=sum(r['wall_seconds'] for r in records[method]),
                   generation_forward_calls=sum(r['new_model_calls'] for r in records[method]))
        if method!='native': row['paired']=paired_stats(truth,native,predictions,seed=20261108)
        rows.append(row)
    result=dict(complete=True,rows=rows,policies=lock,calibration_count=len(e['support_ids']),
        development_count=len(e['query_ids']),official_test_used=False,prior_queries_used=True,
        equal_label_budget=True,equal_new_candidate_count=9,compute_matched=False,
        evidence=evidence,protocol=registration['entry']['plan']['evaluation_status'],
        predictions={m:[{'event_id':i,'target':t,**r} for i,t,r in zip(e['query_ids'],truth,rs)] for m,rs in records.items()})
    atomic_json(out/'REPORT.json',result)
    atomic_json(out/'STATUS.json',dict(complete=True,report_sha256=file_hash(out/'REPORT.json')))
    return result


def gpu_budget(target_gib):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('select exactly one GPU per worker')
    free,total=torch.cuda.mem_get_info(); cuda_occupied=total-free
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or ',' in visible: raise RuntimeError('explicit single physical CUDA_VISIBLE_DEVICES required')
    output=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=memory.used,memory.total',
                                    '--format=csv,noheader,nounits'],text=True)
    used_mib,total_mib=[int(v.strip()) for v in output.strip().split(',')]
    nvidia_occupied=used_mib*1024**2; nvidia_free=(total_mib-used_mib)*1024**2
    if not 0<=used_mib<total_mib: raise RuntimeError('invalid NVIDIA memory telemetry')
    occupied=max(cuda_occupied,nvidia_occupied)
    budget=min(18*1024**3,target_gib*1024**3-occupied,free-3*1024**3,nvidia_free-3*1024**3)
    if budget < 10*1024**3: raise RuntimeError('less than 10 GiB available within global target; wait for existing workload')
    torch.cuda.set_per_process_memory_fraction(budget/total,0)
    return dict(target_total_gib=target_gib,existing_occupied_bytes=occupied,worker_cap_bytes=budget,
                cuda_occupied_bytes=cuda_occupied,nvidia_occupied_bytes=nvidia_occupied,
                free_before_bytes=free,device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                note='Allocation target includes unrelated workloads; actual usage is measured, not artificially padded.')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--stage',required=True,choices=['prepare','sensitivity','fold0','fold1','akr_select','lora_refit','lock','eval_base','eval_lora','report'])
    ap.add_argument('--execute',action='store_true'); ap.add_argument('--max-seconds',type=int,default=5400)
    ap.add_argument('--gpu-target-gib',type=float,default=85.)
    args=ap.parse_args(argv); validate_run_id(args.run_id)
    if not 1<=args.max_seconds<=5400 or not 20<=args.gpu_target_gib<=90: raise ValueError('invalid runtime/memory budget')
    real=args.stage in ['fold0','fold1','akr_select','lora_refit','eval_base','eval_lora']
    if real and not args.execute: raise ValueError('--execute required for real model work')
    out=ROOT/'results/fair_speech_lm'/args.run_id; runner=None
    def expired(*_): raise TimeoutError('stage runtime budget exhausted')
    signal.signal(signal.SIGALRM,expired); signal.alarm(args.max_seconds)
    with run_lock(out/(args.stage+'.lock')):
        atomic_json(out/'stages'/(args.stage+'.json'),dict(complete=False,phase='starting'))
        try:
            registration=prepare(out) if args.stage=='prepare' else load_registration(out)
            if real: atomic_json(out/'telemetry'/(args.stage+'.json'),gpu_budget(args.gpu_target_gib))
            if args.stage in ['prepare','sensitivity','fold0','fold1','akr_select','lora_refit','eval_base','eval_lora']:
                runner=FairRunner(out,args.stage,registration)
            if args.stage=='sensitivity': sensitivity(runner,out)
            elif args.stage in ['fold0','fold1']: fold_stage(runner,out,int(args.stage[-1]))
            elif args.stage=='akr_select': akr_select(runner,out)
            elif args.stage=='lora_refit':
                choice=choose_lora(out,registration)
                e,=runner.plan['episodes']
                if choice['epoch']:
                    train_stage(runner,out/'training/refit',e['support_ids'],[],choice['lr'],choice['epoch'],20261110)
                else:
                    finish_stage(out/'training/refit',dict(identity={'run':runner.fingerprint,'native_selected':True},files={},final_adapter=None))
            elif args.stage=='lock': lock_policies(out,registration)
            elif args.stage.startswith('eval_'): infer(runner,out,args.stage[5:])
            elif args.stage=='report':
                result=report(out,registration)
                print(json.dumps({'complete':True,'rows':result['rows']}),flush=True)
            atomic_json(out/'stages'/(args.stage+'.json'),dict(complete=True,phase='complete'))
        except BaseException as exc:
            atomic_json(out/'stages'/(args.stage+'.json'),dict(complete=False,error_type=type(exc).__name__,error=str(exc)))
            atomic_json(out/'STATUS.json',dict(complete=False,failed_stage=args.stage,error=str(exc)))
            raise
        finally:
            if runner is not None: runner.close()
            signal.alarm(0)
    return 0


if __name__=='__main__': raise SystemExit(main())
