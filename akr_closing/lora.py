"""E8 optimizer-budget LoRA baselines, development-only selection and safe resume.

Only explicitly requested E8 launches this module. It reuses Study's locked
support/selection/confirmation partitions. No full-dataset or test fitting occurs.
"""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from .core import atomic_json,digest,file_hash,Query,read_manifest,make_split,support_ids,metrics,Journal
from .pipeline import prepare_audio,prompt_for,check_stop


def validate_lora_budget(*,lr,rank,steps,accumulation,eval_every):
    if not np.isfinite(lr) or lr<=0 or min(rank,steps,accumulation,eval_every)<1:
        raise ValueError('LoRA requires finite positive learning rate and positive integer budgets')


def step_schedule(*,max_steps:int,eval_every:int):
    if max_steps<1 or eval_every<1:raise ValueError('positive optimizer budgets required')
    return sorted(set(range(eval_every,max_steps+1,eval_every))|{max_steps})


def microbatch_schedule(ids,*,steps,accumulation,seed):
    if not ids or steps<1 or accumulation<1:raise ValueError('nonempty support and positive budgets required')
    rng=np.random.default_rng(seed);result=[]
    while len(result)<steps*accumulation:result.extend(rng.permutation(ids).tolist())
    return result[:steps*accumulation]


def select_development_checkpoint(records):
    if not records or any(r.get('stage')!='selection' for r in records):raise ValueError('selection-only records required')
    eligible=[r for r in records if r['invalid_rate']<=.01]
    if not eligible:raise ValueError('no valid development checkpoint; no confirmation selection')
    return sorted(eligible,key=lambda r:(-r['accuracy'],-r['macro_f1'],r['step'],r['lr']))[0]


def _teacher_inputs(backend,q,prompt,target,include_eos):
    t=backend.torch;x=backend._prepare(q,prompt);n=x['input_ids'].shape[1]
    tok=backend.processor.tokenizer
    suffix=tok(target,add_special_tokens=False)['input_ids']
    if not suffix:raise ValueError('empty label tokenization')
    if include_eos:
        if tok.eos_token_id is None:raise ValueError('tokenizer has no EOS')
        suffix=suffix+[tok.eos_token_id]
    ids=t.tensor([suffix],device=x['input_ids'].device,dtype=x['input_ids'].dtype)
    x['input_ids']=t.cat([x['input_ids'],ids],1)
    x['attention_mask']=t.cat([x['attention_mask'],t.ones_like(ids)],1)
    losslabels=t.full_like(x['input_ids'],-100);losslabels[:,n:]=ids;x['labels']=losslabels
    assert bool((losslabels[:,:n]==-100).all()) and bool((losslabels[:,n:]!=-100).all())
    return x,{'prefix_tokens':int(n),'target_token_ids':suffix,'supervised_tokens':len(suffix)}


def _torch_save(path,value):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp);os.replace(tmp,path)


def _rng_save(folder):
    import torch
    npstate=np.random.get_state()
    _torch_save(folder/'rng.pt',{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
    atomic_json(folder/'rng.json',{'python':random.getstate(),'numpy':{'kind':npstate[0],'keys':npstate[1].tolist(),
                 'position':npstate[2],'has_gauss':npstate[3],'cached_gaussian':npstate[4]}})


def _rng_restore(folder):
    import torch
    def tuples(x):return tuple(tuples(y) for y in x) if isinstance(x,list) else x
    r=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
    torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
    j=json.loads((folder/'rng.json').read_text());random.setstate(tuples(j['python']))
    n=j['numpy'];np.random.set_state((n['kind'],np.asarray(n['keys'],dtype='uint32'),n['position'],n['has_gauss'],n['cached_gaussian']))


def _evaluate(backend,queries,targets,prompt,labels,folder,meta):
    j=Journal(folder,meta,[q.event_id for q in queries]);backend.thinker.eval()
    for q in queries:
        if j.has(q.event_id):continue
        check_stop();before=time.perf_counter();result=backend.predict(q,prompt,labels)
        result['wall_seconds']=time.perf_counter()-before;j.add(q.event_id,result)
    result=metrics(targets,[r['prediction'] for r in j.results()],labels)
    atomic_json(folder/'METRICS.json',result)
    return result


def fit_one(backend,train,selection,targets,prompt,labels,folder,*,lr,rank,steps,eval_every,
            accumulation,seed,include_eos=True):
    validate_lora_budget(lr=lr,rank=rank,steps=steps,accumulation=accumulation,eval_every=eval_every)
    import torch
    from peft import LoraConfig,get_peft_model,PeftModel
    meta={'support_ids':[q.event_id for q,y in train],'support_targets':[y for q,y in train],
          'selection_ids':[q.event_id for q in selection],'lr':lr,'rank':rank,'optimizer_steps':steps,
          'eval_every':eval_every,'accumulation':accumulation,'seed':seed,'prompt':prompt,
          'model':backend.provenance,'include_eos':include_eos,'code_sha256':file_hash(Path(__file__)),
          'scope':'decoder q_proj/v_proj only; backbone frozen; checkpoints chosen on development'}
    folder.mkdir(parents=True,exist_ok=True);identity=folder/'IDENTITY.json'
    if identity.exists() and json.loads(identity.read_text())!=meta:raise RuntimeError('LoRA run identity changed')
    atomic_json(identity,meta)
    statepath=folder/'STATE.json';state=json.loads(statepath.read_text()) if statepath.exists() else None
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    backend.thinker.enable_input_require_grads()
    backend.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if state:
        backend.thinker=PeftModel.from_pretrained(backend.thinker,str(folder/state['checkpoint']/'adapter'),is_trainable=True)
    else:
        cfg=LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
               target_modules=r'.*model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$')
        backend.thinker=get_peft_model(backend.thinker,cfg)
    params=[p for p in backend.thinker.parameters() if p.requires_grad]
    if not params:raise RuntimeError('no LoRA parameters selected')
    names=[n for n,p in backend.thinker.named_parameters() if p.requires_grad]
    if any('lora_' not in n for n in names):raise RuntimeError('unexpected trainable backbone parameter')
    atomic_json(folder/'TRAINABLE.json',{'names':names,'parameters':sum(p.numel() for p in params)})
    optimizer=torch.optim.AdamW(params,lr=lr)
    if state:
        saved=folder/state['checkpoint']
        optimizer.load_state_dict(torch.load(saved/'optimizer.pt',map_location='cpu',weights_only=True));_rng_restore(saved)
    byid={q.event_id:(q,y) for q,y in train}
    sequence=microbatch_schedule(list(byid),steps=steps,accumulation=accumulation,seed=seed)
    completed=int(state['step']) if state else 0;evalsteps=step_schedule(max_steps=steps,eval_every=eval_every)
    records=[]
    # A checkpoint can be complete while its interrupted dev evaluation still needs resuming.
    def evaluate_step(step,checkpoint):
        rec=_evaluate(backend,selection,targets,prompt,labels,folder/checkpoint/'selection',
                      {'fit':digest(meta),'step':step,'stage':'selection'})
        rec.update(step=step,lr=lr,stage='selection',checkpoint=str(folder/checkpoint/'adapter'))
        atomic_json(folder/checkpoint/'SELECTION.json',rec);return rec
    if state and completed in evalsteps:
        evaluate_step(completed,state['checkpoint'])
    for step in range(completed+1,steps+1):
        # Stop only between optimizer steps; no uncommitted microbatch gradients are resumed.
        check_stop();backend.thinker.train();optimizer.zero_grad(set_to_none=True)
        start=time.perf_counter();losses=[]
        for eid in sequence[(step-1)*accumulation:step*accumulation]:
            q,target=byid[eid];inputs,audit=_teacher_inputs(backend,q,prompt,target,include_eos)
            if step==1:atomic_json(folder/'TARGET_MASK_EXAMPLE.json',audit)
            loss=backend.thinker(**inputs,use_cache=False,return_dict=True).loss
            if not torch.isfinite(loss):raise RuntimeError('non-finite LoRA loss')
            (loss/accumulation).backward();losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
        checkpoint=f'step_{step:06d}' if step in evalsteps else f'_resume_{step%2}'
        # Two alternating atomic resume slots prevent unbounded checkpoint disk growth.
        # Evaluation checkpoints remain immutable; the pointer advances only after all state is saved.
        saved=folder/checkpoint;saved.mkdir(exist_ok=True);backend.thinker.save_pretrained(saved/'adapter')
        _torch_save(saved/'optimizer.pt',optimizer.state_dict());_rng_save(saved)
        trainrec={'step':step,'mean_train_loss':float(np.mean(losses)),
                  'wall_seconds':time.perf_counter()-start,'optimizer_steps':step,
                  'peak_cuda_bytes':int(torch.cuda.max_memory_allocated())}
        atomic_json(folder/'TRAIN_CURVE'/f'step_{step:06d}.json',trainrec)
        atomic_json(statepath,{'step':step,'checkpoint':checkpoint,'fit_fingerprint':digest(meta)})
        if step in evalsteps:evaluate_step(step,checkpoint)
    for step in evalsteps:
        p=folder/f'step_{step:06d}'/'SELECTION.json'
        if not p.exists():raise RuntimeError('missing complete selection evaluation; do not choose best checkpoint')
        records.append(json.loads(p.read_text()))
    atomic_json(folder/'COMPLETE.json',{'completed':True,'optimizer_steps':steps,'selection_steps':evalsteps})
    return records


def run_lora(root,config,backend_factory,*,run_id,profile,execute=False):
    """Return/run exact E8 jobs. No test labels are used for selection."""
    from .pipeline import Study
    e=config['E8'];study=Study(root,config,backend_factory,run_id=run_id,profile=profile);study.preflight()
    base=study.out/'E8';base.mkdir(exist_ok=True);jobs=[]
    for ds in config['datasets']:
        if ds['name'] not in e['datasets']:continue
        rows=read_manifest(root/ds['manifest'],old_root=config.get('old_root'),root=root);lookup={r['event_id']:r for r in rows}
        for model in config['models']:
            for seed in config['seeds']:
                split=json.loads((study.out/'splits'/f"{ds['name']}_{seed}.json").read_text())
                for k in e['k_per_class']:
                    sid=support_ids(rows,split['train'],ds['labels'],k,seed=seed)
                    for condition in e['conditions']:
                        job={'dataset':ds['name'],'model':model,'seed':seed,'k_per_class':k,'condition':condition,
                             'support_ids':sid,'selection_ids':split['selection'],'confirmation_ids':split['confirmation'],
                             'test_used':False,'lrs':e['learning_rates'],'max_steps':e['max_steps']}
                        jobs.append(job)
                        if not execute:continue
                        jobdir=base/ds['name']/model['tag']/f's{seed}_k{k}'/condition.replace(':','_')
                        labels=ds['labels'];prompt=prompt_for(labels)
                        def query(eid):
                            r=lookup[eid];source=Path(r['audio_path'])
                            dest=study.out/'audio'/ds['name']/(digest({'source':str(source),'hash':file_hash(source),'condition':condition})+'.wav')
                            prepare_audio(source,dest,condition);return Query(eid,str(dest),r.get('recording_id',eid))
                        train=[(query(i),lookup[i]['label']) for i in sid]
                        dev=[query(i) for i in split['selection']];dy=[lookup[i]['label'] for i in split['selection']]
                        records=[]
                        for lr in e['learning_rates']:
                            backend=backend_factory(model['id'],model['revision'],profile=profile,
                                 layer_fractions=config['layer_fractions'],feature_fraction=config['feature_fraction'],
                                 max_context_tokens=config['max_context_tokens'])
                            records.extend(fit_one(backend,train,dev,dy,prompt,labels,jobdir/f'lr_{lr:g}',lr=lr,rank=e['rank'],
                                  steps=e['max_steps'],eval_every=e['eval_every'],accumulation=e['accumulation'],seed=seed,
                                  include_eos=e.get('include_eos',True)))
                            del backend;gc.collect();__import__('torch').cuda.empty_cache()
                        chosen=select_development_checkpoint(records)
                        atomic_json(jobdir/'LOCK.json',chosen)
                        from peft import PeftModel
                        backend=backend_factory(model['id'],model['revision'],profile=profile,
                            layer_fractions=config['layer_fractions'],feature_fraction=config['feature_fraction'],
                            max_context_tokens=config['max_context_tokens'])
                        backend.thinker=PeftModel.from_pretrained(backend.thinker,chosen['checkpoint'],is_trainable=False)
                        backend.thinker.eval()
                        # Reload smoke checks exact saved development outputs before confirmation.
                        q=dev[0];testout=backend.predict(q,prompt,labels)
                        previous=Path(chosen['checkpoint']).parent/'selection'/'items'/(digest(q.event_id)+'.json')
                        expected=json.loads(previous.read_text())['result']
                        if testout['prediction']!=expected['prediction'] or testout['raw_prediction']!=expected['raw_prediction']:
                            raise RuntimeError('adapter reload prediction parity failed; confirmation blocked')
                        conf=[query(i) for i in split['confirmation']];cy=[lookup[i]['label'] for i in split['confirmation']]
                        _evaluate(backend,conf,cy,prompt,labels,jobdir/'confirmation',{'lock':chosen,'stage':'locked_reevaluation'})
                        del backend;gc.collect();__import__('torch').cuda.empty_cache()
    atomic_json(base/'PLAN.json',{'jobs':jobs,'execute_requested':execute,
                  'warning':'E8 settings are selected on development; old one-epoch results are not overwritten'})
    return jobs
