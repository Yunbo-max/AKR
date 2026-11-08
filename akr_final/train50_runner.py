"""Standalone Train-50 case execution with immutable, exact-population receipts.

No historical worker/cache is required. A CPU contract run is not a scientific
experiment. Query targets enter only the final scoring path.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import yaml

from akr_closing.cli import run_lock, validate_run_id, environment
from akr_closing.core import Journal, atomic_json, digest, file_hash, paired_stats
from akr_closing.pipeline import InputDomainInfeasible, input_domain_infeasible, prompt_for, write_npz
from akr_closing.repair import RidgeReadout
from .runner import FinalRunner, prediction_summary, _check_stop
from .rebuttal_runner import measured, array_hash
from .simple_baselines import evidence_prompt
from .train50_cases import CASES, case_plan
from .train50_methods import fit_correction
from .train50_protocol import (prepare_registration, lock_registration,
                               validate_registered_inputs, validate_population)


def read(path):return json.loads(Path(path).read_text())


def asset_identity(value):
    """Bind local asset bytes, including code and model shards, before execution."""
    if isinstance(value,dict):return {k:asset_identity(v) for k,v in sorted(value.items())}
    if isinstance(value,list):return [asset_identity(v) for v in value]
    if isinstance(value,str):
        p=Path(value)
        if p.is_file():return {'path':str(p.resolve()),'sha256':file_hash(p)}
        if p.is_dir():
            suffixes={'.py','.json','.yaml','.yml','.pt','.pth','.bin','.safetensors',
                      '.model','.txt','.tiktoken','.jinja','.vocab','.merges','.spm','.so'}
            files=sorted(f for f in p.rglob('*') if f.is_file() and f.suffix in suffixes and '.git' not in f.parts)
            return {'path':str(p.resolve()),'files':{str(f.relative_to(p)):file_hash(f) for f in files}}
    return value


def optional_preflight(job,assets):
    result={}
    for feature in sorted({s['feature'] for s in job['methods']} - {'omni'}):
        from .train50_features import feature_provenance
        result[feature]=feature_provenance(feature,assets.get(feature,{}))
    if any(s['kind']=='naturelm' for s in job['methods']):
        spec=assets.get('naturelm',{})
        required=('naturelm_code','naturelm_checkpoint','naturelm_llama','naturelm_bert_config','naturelm_config',
                  'naturelm_source_revision','naturelm_checkpoint_revision','naturelm_llama_revision')
        if any(not spec.get(k) for k in required):raise ValueError('NatureLM local assets and pinned revisions required; see NATURELM_SCOPE.md')
        for key in required:
            if key.endswith('_revision'):
                if not re.fullmatch('[0-9a-f]{40}',spec[key]):raise ValueError('NatureLM revisions must be full commit SHAs')
            elif not Path(spec[key]).exists():raise FileNotFoundError(f'missing NatureLM asset: {key}')
        from importlib.metadata import version,PackageNotFoundError
        packages={}
        for name in ('torch','transformers','peft','safetensors','huggingface-hub','resampy','soundfile','pydantic','NatureLM-audio'):
            try:packages[name]=version(name)
            except PackageNotFoundError:packages[name]=None
        result['naturelm']={'runtime_packages':packages,
                            'status':'local assets present; architecture/weights compatibility requires production acceptance run'}
    return result


def entry_for(config,job,registration):
    biggest=registration['splits'][str(max(job['support_budgets']))]
    settings=config['settings'];condition=job['condition']
    episode=dict(id=f's{job["seed"]}_{condition.replace(":","_")}',seed=job['seed'],condition=condition,
                 support_ids=biggest['support_ids'],query_ids=registration['query_ids'])
    cfg=dict(models=[{'tag':job['model'],**config['models'][job['model']]}],E9=config['models']['qwen3b'],
             **{k:settings[k] for k in ('layer_fractions','feature_fraction','max_context_tokens','ridge_alpha')})
    rows=[{**r,'recording_id':biggest['groups'][r['event_id']]} for r in registration['rows']]
    plan=dict(mode='standalone',config=cfg,labels=registration['labels'],rows=rows,episodes=[episode],
              dataset={'name':job['dataset'],'partition':'official_test','group_key':'recording_id'},
              source_hashes=registration['source_hashes'],query_partition='complete_official_test',
              selection_uses_outcomes=False,evaluation_status=registration['test_history'])
    return plan,dict(dataset=job['dataset'],case=job['case'],job=job)


class CaseWorker(FinalRunner):
    def __init__(self,root,out,config,job,registration,profile,assets,backend_factory=None,cache_only=False):
        plan,options=entry_for(config,job,registration)
        super().__init__(root,Path(root)/'results/.train50_no_private_source','worker',options,profile,
                         backend_factory=backend_factory,cache_only=cache_only,
                         standalone_plan=plan,output_root=out)
        self.job,self.registration,self.settings=job,registration,config['settings']
        self.assets=assets;self.extractors={};self.naturelm=None;self.stage_costs=[]
        self.case_out=Path(out)
        self.exclusion_path=self.case_out/'INPUT_DOMAIN_EXCLUSIONS.json'
        self._exclusion_doc=None;self._exclusion_index={}

    def _model(self,tag,episode=None):
        # The existing, tested LoRA helper calls qwen3b. Alias only this explicit
        # two-size family, retaining the actual pinned model in every receipt.
        if tag=='qwen3b' and self.job['model'] in ('qwen3b','qwen7b'):tag=self.job['model']
        loading=self.backend is None or self.model_tag!=tag
        start=time.perf_counter();backend=super()._model(tag,episode)
        if loading:
            path=self.case_out/'MODEL_LOAD_COSTS.json';values=read(path) if path.exists() else []
            values.append(dict(model=tag,wall_seconds=time.perf_counter()-start))
            atomic_json(path,values)
        return backend

    def prepare_exclusions(self):
        """Lock the run-level lp input-domain ledger before any method runs.

        Only events whose lp cutoff is at or above their own Nyquist are
        recorded. Every method consumes this one ledger; ``full`` contributes
        no query entries and is unaffected. The ledger is immutable within a
        run, and registered support budgets are never reduced to hide it.
        """
        wanted={}
        for eid in self.registration['query_ids']:
            wanted.setdefault(self.job['condition'],{}).setdefault(eid,'query')
        for spec in self.job['methods']:
            fit=spec.get('fit_condition',self.job['condition'])
            support_condition=self.job['condition'] if spec['kind'] in ('icl','lora') else fit
            for eid in self.registration['splits'][str(spec['support_budget'])]['support_ids']:
                wanted.setdefault(support_condition,{}).setdefault(eid,'support')
        lp_conditions=sorted({condition for condition in wanted if condition.startswith('lp:')})
        hits={}
        for condition in lp_conditions:
            for row in self.registration['rows']:
                hit=input_domain_infeasible(row['audio_path'],condition)
                if hit is not None:hits[(row['event_id'],condition)]=hit
        entries=[]
        for condition in sorted(wanted):
            for eid in sorted(wanted[condition]):
                hit=hits.get((eid,condition))
                if hit is not None:
                    rate,cutoff=hit
                    entries.append(dict(event_id=eid,sample_rate=float(rate),cutoff=float(cutoff),
                                        condition=condition,role=wanted[condition][eid],
                                        reason='input-domain-infeasible-lowpass-nyquist'))
        dataset_scan=None
        if lp_conditions:
            by_condition={}
            for condition in lp_conditions:
                counts={}
                for row in self.registration['rows']:
                    if (row['event_id'],condition) in hits:
                        split=row.get('split','');counts[split]=counts.get(split,0)+1
                by_condition[condition]=dict(split_counts=counts,total=sum(counts.values()))
            dataset_scan=dict(registered_events=len(self.registration['rows']),by_condition=by_condition,
                              description=('every registered row scanned at its original rate; only the '
                                           'query and selected-support roles are excluded from execution'))
        document=dict(version=1,case=self.job['case'],dataset=self.job['dataset'],model=self.job['model'],
                      reason='input-domain-infeasible-lowpass-nyquist',
                      definition=('lp cutoff is not below the recording original Nyquist; the event '
                                  'cannot enter the lp input domain and is excluded explicitly, never '
                                  'replaced by another waveform or a smaller support budget'),
                      dataset_scan=dataset_scan,entries=entries)
        if self.exclusion_path.exists():
            if read(self.exclusion_path)!=document:
                raise RuntimeError('input-domain exclusion ledger changed within run; use a new run ID')
        else:atomic_json(self.exclusion_path,document)
        self._exclusion_doc=document
        self._exclusion_index={(e['event_id'],e['condition']):e for e in entries}
        return document

    def exclusion_entry(self,event,condition):
        if self._exclusion_doc is None:self.prepare_exclusions()
        return self._exclusion_index.get((event,condition))

    def _query(self,event,condition):
        try:
            q=super()._query(event,condition)
        except InputDomainInfeasible as error:
            entry=self.exclusion_entry(event,condition)
            if entry is None:
                raise RuntimeError(f'event {event} is outside the lp input domain but absent '
                                   'from the exclusion ledger') from error
            raise RuntimeError(f'excluded event {event} unexpectedly reached audio preparation '
                               f'for condition {condition}') from error
        path=Path(q.audio_path);meta=path.with_suffix('.sha256.json');sha=file_hash(path)
        if meta.exists() and read(meta).get('sha256')!=sha:raise RuntimeError('prepared audio cache changed')
        if not meta.exists():atomic_json(meta,{'sha256':sha})
        return q

    def array(self,event,condition,kind='feature',feature='omni'):
        episode=self.plan['episodes'][0]
        if kind=='gradient' and event not in episode['support_ids']:raise ValueError('gradient requested outside registered training support')
        target=self.lookup[event]['label'] if kind=='gradient' else None
        identity=dict(run=self.fingerprint,event=event,condition=condition,kind=kind,feature=feature,
                      waveform=self.wave_hashes[event],target=target,prompt=prompt_for(self.labels))
        path=self.out/'cache'/(digest(identity)+'.npz');meta=Path(str(path)+'.json')
        if path.exists() or meta.exists():
            if not path.is_file() or not meta.is_file():raise RuntimeError('incomplete array cache pair')
            info=read(meta)
            if info.get('identity')!=identity or info.get('sha256')!=file_hash(path):raise RuntimeError('changed array cache')
            with np.load(path,allow_pickle=False) as z:value=z['value'].copy()
        else:
            if self.cache_only:raise FileNotFoundError('required array missing in cache-only mode')
            _check_stop();q=self._query(event,condition)
            if feature=='omni':
                backend=self._model(self.job['model'],episode)
                def op():
                    v=backend.support_gradients(q,target,prompt_for(self.labels))['audio'] if kind=='gradient' else backend.feature(q,prompt_for(self.labels))
                    return {'value':v}
                result=measured(backend,op);value=result.pop('value')
            else:
                if kind!='feature':raise ValueError('external feature extraction cannot produce K/V gradients')
                from .train50_features import FeatureExtractor
                if feature not in self.extractors:self.extractors[feature]=FeatureExtractor(feature,self.assets.get(feature,{}))
                start=time.perf_counter();value=self.extractors[feature].extract(q.audio_path)
                result=dict(wall_seconds=time.perf_counter()-start,new_model_calls=0,
                            external_feature_calls=1,feature_provenance=self.extractors[feature].provenance())
            value=np.asarray(value,np.float32)
            if value.ndim!=1 or not len(value) or not np.isfinite(value).all():raise ValueError('invalid feature/gradient array')
            write_npz(path,value=value)
            atomic_json(meta,dict(identity=identity,sha256=file_hash(path),cost=result))
        if value.ndim!=1 or not len(value) or not np.isfinite(value).all():raise ValueError('invalid saved array')
        return value

    def matrix(self,ids,condition,kind='feature',feature='omni'):
        return np.stack([self.array(i,condition,kind,feature) for i in ids])

    def payload(self,spec):
        support=self.registration['splits'][str(spec['support_budget'])]['support_ids']
        query=self.registration['query_ids'];condition=self.job['condition'];fit=spec.get('fit_condition',condition)
        excluded_set={eid for eid in query if self.exclusion_entry(eid,condition) is not None}
        excluded=[eid for eid in query if eid in excluded_set]
        active=[eid for eid in query if eid not in excluded_set]
        uses_support=spec['kind'] not in ('native','naturelm')
        support_condition=condition if spec['kind'] in ('icl','lora') else fit
        blocked_support=([eid for eid in support
                          if self.exclusion_entry(eid,support_condition) is not None]
                         if uses_support else [])
        if blocked_support:
            raise RuntimeError(f'input-domain-infeasible lp support: {blocked_support}; the registered '
                               'support budget cannot be silently reduced (re-register support from the '
                               'lp-feasible pool or obtain explicit approval for an effective-support '
                               'exclusion)')
        payload={'support_ids':support,'query_ids':query,'fit_condition':fit,'query_condition':condition,
                 'excluded_query_ids':excluded,'scored_query_ids':active}
        kind=spec['kind'];fields=None;external=None;adapter=None
        if kind in ('native','icl','naturelm'):return payload,fields,external,adapter
        if kind=='lora':
            if self.job['model'] not in ('qwen7b','qwen3b'):raise ValueError('matched LoRA supports the pinned Qwen2.5 sizes')
            from scripts.fair_speech_lm import train_stage, verify_stage
            identity=dict(run=self.fingerprint,train=support,held=[],lr=spec['lr'],max_epoch=spec['epochs'],seed=self.job['seed'],accumulation=4)
            directory=self.case_out/'training'/'lora_same50'
            result=verify_stage(directory,identity)
            if result is None:
                if self.cache_only:raise FileNotFoundError('missing LoRA stage receipt')
                result=train_stage(self,directory,support,[],spec['lr'],spec['epochs'],self.job['seed'])
            adapter=result['final_adapter'];payload['adapter_hashes']=result['files'];payload['training']=result
            return payload,fields,external,adapter
        x=self.matrix(support,fit,feature=spec['feature']);q=self.matrix(active,condition,feature=spec['feature'])
        y=np.array([self.labels.index(self.lookup[i]['label']) for i in support])
        if kind in ('ridge','probe_lm','probability_lm'):
            start=time.perf_counter()
            mapping=RidgeReadout.fit(x,y,len(self.labels),alpha=spec['ridge'])
            scores=mapping.mapping.predict(q);external=[self.labels[int(i)] for i in scores.argmax(1)]
            if kind=='probability_lm':
                from .train50_evidence import class_probabilities
                probabilities=class_probabilities(scores,spec['temperature'])
                priors=np.bincount(y,minlength=len(self.labels))/len(y)
                payload.update(class_probabilities=probabilities.tolist(),support_priors=priors.tolist(),
                               text_evidence='full ordered probabilities and support priors; no query truth')
            cost=dict(method=spec['name'],stage='fit_and_route',wall_seconds=time.perf_counter()-start,
                      scores_sha256=array_hash(scores))
            cost_path=self.case_out/'maps'/(spec['name']+'.cost.json')
            if cost_path.exists():
                saved=read(cost_path)
                if saved['scores_sha256']!=cost['scores_sha256']:raise RuntimeError('readout changed within run')
                cost=saved
            else:atomic_json(cost_path,cost)
            self.stage_costs.append(cost)
            payload.update(scores_sha256=array_hash(scores),external=external,
                           feature=spec['feature'],output_source='external_ridge' if kind=='ridge' else 'speech_lm_with_ridge_evidence')
        else:
            g=self.matrix(support,fit,kind='gradient')
            start=time.perf_counter()
            if kind=='zero':fields=np.zeros((len(active),g.shape[1]),dtype=np.float32);meta={'kind':'zero_field'}
            else:
                correction=fit_correction(x,g,y,len(self.labels),spec,seed=self.job['seed'])
                fields=correction.predict(q);meta=correction.metadata
                if kind in ('soft','hard','random','shuffled_prototypes'):
                    probabilities=correction.probabilities(q)
                    payload['routing_probabilities_sha256']=array_hash(probabilities)
                    write_npz(self.case_out/'maps'/(spec['name']+'.npz'),probabilities=probabilities,fields=fields)
                else:write_npz(self.case_out/'maps'/(spec['name']+'.npz'),fields=fields)
            elapsed=time.perf_counter()-start
            payload.update(fields_sha256=array_hash(fields),mapping=meta)
            cost_path=self.case_out/'maps'/(spec['name']+'.cost.json')
            cost=dict(method=spec['name'],stage='fit_and_route',wall_seconds=elapsed,fields_sha256=array_hash(fields))
            if cost_path.exists():
                saved=read(cost_path)
                if saved['fields_sha256']!=cost['fields_sha256']:raise RuntimeError('mapping changed within run')
                cost=saved
            else:atomic_json(cost_path,cost)
            self.stage_costs.append(cost)
        return payload,fields,external,adapter

    def trial(self,spec,case_fingerprint):
        payload,fields,external,adapter=self.payload(spec)
        identity=dict(case_fingerprint=case_fingerprint,spec=spec,payload=payload,
                      registration_sha256=digest(self.registration))
        base=self.case_out/'trials'/spec['name'];ids=self.registration['query_ids']
        journal=Journal(base,identity,ids)
        method_meta={'identity':identity,'sha256':digest(identity)}
        lock_registration(base/'METHOD.json',method_meta)
        episode=self.plan['episodes'][0];condition=self.job['condition']
        active=payload['scored_query_ids'];position={event:i for i,event in enumerate(active)}
        loaded_adapter=False
        for event in ids:
            entry=self.exclusion_entry(event,condition)
            if entry is not None:
                if not journal.has(event):
                    journal.add(event,dict(event_id=event,prediction='excluded-infeasible',
                                           raw_prediction='excluded-infeasible',excluded=True,
                                           exclusion_reason=entry['reason'],exclusion=entry,
                                           wall_seconds=0.,new_model_calls=0))
                continue
            if journal.has(event):continue
            if self.cache_only:raise FileNotFoundError('required prediction missing in cache-only mode')
            _check_stop();kind=spec['kind'];j=position[event];prompt=prompt_for(self.labels)
            if kind=='ridge':
                result=dict(prediction=external[j],raw_prediction=external[j],output_source='external_ridge',
                            wall_seconds=0.,new_model_calls=0)
            elif kind=='naturelm':
                if self.naturelm is None:
                    self.close()
                    from .train50_naturelm import NatureLMBackend
                    self.naturelm=NatureLMBackend(self.assets.get('naturelm',{}))
                result=measured(self.naturelm,lambda:self.naturelm.predict(self._query(event,condition),prompt,self.labels))
            else:
                backend=self._model(self.job['model'],episode)
                if adapter and not loaded_adapter:
                    from peft import PeftModel
                    backend.thinker=PeftModel.from_pretrained(backend.thinker,adapter,is_trainable=False).eval()
                    if any(p.requires_grad for p in backend.thinker.parameters()):raise RuntimeError('trainable parameter during inference')
                    loaded_adapter=True
                support=()
                if kind=='icl':
                    support_ids=list(payload['support_ids'])
                    rng=np.random.default_rng(self.job['seed']+spec['order_seed'])
                    support_ids=[support_ids[k] for k in rng.permutation(len(support_ids))]
                    support=[(self._query(i,condition),self.lookup[i]['label']) for i in support_ids]
                if kind=='probe_lm':prompt=evidence_prompt(prompt,external[j],self.labels)
                if kind=='probability_lm':
                    from .train50_evidence import probability_evidence_prompt
                    prompt=probability_evidence_prompt(prompt,self.labels,payload['class_probabilities'][j],payload['support_priors'])
                field=None if fields is None else fields[j]
                result=measured(backend,lambda:backend.predict(self._query(event,condition),prompt,self.labels,
                    support=support,direction=field,alpha=spec['alpha'] if field is not None else 0.,
                    scope=spec['scope'],kind=spec['kv_kind'],layer_group=spec['layer_group'],
                    scale_mode=spec['scale_mode'],raw_eta=spec['raw_eta'],max_new_tokens=self.settings['max_new_tokens']))
                if external is not None:result['external_prediction']=external[j]
            if result.get('prediction') not in self.labels and result.get('prediction')!='':raise ValueError('backend returned an unnormalized prediction')
            if result.get('event_id',event)!=event:raise ValueError('backend returned a foreign event identity')
            journal.add(event,{**result,'event_id':event})
        if adapter:self.close()
        records=journal.results();validate_population(ids,records)
        return records


def _summary_cost(out):
    arrays=[read(p) for p in (out/'worker/cache').glob('*.npz.json')]
    return dict(feature_records=sum(x['identity']['kind']=='feature' for x in arrays),
                support_backward_records=sum(x['identity']['kind']=='gradient' for x in arrays),
                shared_array_wall_seconds=sum(x.get('cost',{}).get('wall_seconds',0.) for x in arrays),
                shared_model_forward_calls=sum(x.get('cost',{}).get('new_model_calls',0) for x in arrays),
                fitting=read(out/'FIT_COSTS.json') if (out/'FIT_COSTS.json').exists() else [],
                model_loads=read(out/'MODEL_LOAD_COSTS.json') if (out/'MODEL_LOAD_COSTS.json').exists() else [],
                note='Shared support/query extraction is charged once per case. Add first-pass query feature cost to generation for end-to-end inference; model loading and disk cache I/O reported separately.')


def method_cost(out,spec,job,registration,metadata):
    arrays=[read(p) for p in (out/'worker/cache').glob('*.npz.json')]
    support=set(registration['splits'][str(spec['support_budget'])]['support_ids'])
    query=set(registration['query_ids']);fit=spec.get('fit_condition',job['condition'])
    readouts={'ridge','probe_lm','probability_lm'};corrections={'original','hard','soft','fixed','random','shuffled_pair','shuffled_prototypes','zero'}
    feature_needed=spec['kind'] in readouts|corrections
    def select(ids,condition,kind):
        return [r for r in arrays if r['identity']['event'] in ids and r['identity']['condition']==condition
                and r['identity']['kind']==kind and r['identity']['feature']==spec['feature']]
    qf=select(query,job['condition'],'feature') if feature_needed else []
    sf=select(support,fit,'feature') if feature_needed else []
    sg=select(support,fit,'gradient') if spec['kind'] in corrections else []
    def seconds(records):return sum(r.get('cost',{}).get('wall_seconds',0.) for r in records)
    training=metadata.get('payload',{}).get('training',{})
    cost_path=out/'maps'/(spec['name']+'.cost.json')
    fitting=read(cost_path)['wall_seconds'] if cost_path.exists() else 0.
    return dict(query_feature_records=len(qf),query_feature_wall_seconds=seconds(qf),
                support_feature_records=len(sf),support_feature_wall_seconds=seconds(sf),
                support_backward_records=len(sg),support_backward_wall_seconds=seconds(sg),
                fit_and_route_wall_seconds=fitting,lora_training_wall_seconds=training.get('wall_seconds',0.),
                lora_train_exposures=training.get('train_exposures',0),
                lora_optimizer_steps=training.get('optimizer_steps',0),
                model_load_excluded=True,cache_io_excluded=True,
                timing_note='Component costs from first completed computations; standalone inference includes query features plus generation. Fit/route is combined and may include routing over all queries.')


def report_case(out,*,write=True):
    """Rebuild from registered IDs and validated per-item journals, never REPORT."""
    out=Path(out)
    if write:atomic_json(out/'STATUS.json',dict(complete=False,status='verifying'))
    lock=read(out/'CASE.json');registration=read(out/'REGISTRATION.json')
    if digest(registration)!=lock['registration_sha256']:raise RuntimeError('registration identity changed')
    case_fingerprint=digest(lock);job=lock['job'];ids=registration['query_ids'];records={};incomplete=[];metadata={}
    lookup={r['event_id']:r for r in registration['rows']}
    ledger_path=out/'INPUT_DOMAIN_EXCLUSIONS.json'
    ledger=read(ledger_path) if ledger_path.is_file() else dict(version=1,entries=[])
    index={(e['event_id'],e['condition']):e for e in ledger.get('entries',[])}
    excluded=[eid for eid in ids if (eid,job['condition']) in index]
    excluded_set=set(excluded);scored=[eid for eid in ids if eid not in excluded_set]
    if not scored:
        raise RuntimeError('every registered query is outside the lp input domain; no scientific summary')
    recomputed={eid for eid in ids
                if input_domain_infeasible(lookup[eid]['audio_path'],job['condition']) is not None}
    if recomputed!=set(excluded):
        raise RuntimeError('input-domain exclusion ledger disagrees with the registered query audio')
    if any(index[(eid,job['condition'])]['reason']!='input-domain-infeasible-lowpass-nyquist'
           for eid in excluded):
        raise RuntimeError('input-domain exclusion ledger has an unknown reason')
    for entry in ledger.get('entries',[]):
        row=lookup.get(entry.get('event_id'))
        hit=None if row is None else input_domain_infeasible(row['audio_path'],entry['condition'])
        if (hit!=(entry.get('sample_rate'),entry.get('cutoff')) or
                entry.get('reason')!='input-domain-infeasible-lowpass-nyquist'):
            raise RuntimeError('input-domain exclusion ledger entry disagrees with registered audio')
    scan=ledger.get('dataset_scan')
    if scan:
        for condition,recorded in scan.get('by_condition',{}).items():
            counts={}
            for row in registration['rows']:
                if input_domain_infeasible(row['audio_path'],condition) is not None:
                    split=row.get('split','');counts[split]=counts.get(split,0)+1
            if dict(counts)!=recorded.get('split_counts') or sum(counts.values())!=recorded.get('total'):
                raise RuntimeError('input-domain dataset scan disagrees with registered audio')
    for spec in job['methods']:
        base=out/'trials'/spec['name'];meta_path=base/'METHOD.json'
        if not meta_path.is_file():incomplete.append(spec['name']);continue
        meta=read(meta_path);identity=meta.get('identity',{})
        if (digest(identity)!=meta.get('sha256') or identity.get('case_fingerprint')!=case_fingerprint or
            identity.get('spec')!=spec or identity.get('registration_sha256')!=digest(registration)):
            raise RuntimeError('foreign or corrupt method receipt')
        payload=identity.get('payload',{})
        if (payload.get('query_ids')!=ids or payload.get('support_ids')!=registration['splits'][str(spec['support_budget'])]['support_ids'] or
            payload.get('query_condition')!=job['condition'] or payload.get('fit_condition')!=spec.get('fit_condition',job['condition'])):
            raise RuntimeError('method population/protocol differs from registration')
        if payload.get('excluded_query_ids')!=excluded or payload.get('scored_query_ids')!=scored:
            raise RuntimeError('method input-domain exclusions differ from the run ledger')
        if not (base/'manifest.json').is_file():incomplete.append(spec['name']);continue
        journal=Journal(base,identity,ids)
        if not journal.complete:incomplete.append(spec['name']);continue
        values=journal.results();validate_population(ids,values)
        for event,value in zip(ids,values):
            entry=index.get((event,job['condition']))
            if entry is not None:
                if (value.get('prediction')!='excluded-infeasible' or
                        value.get('raw_prediction')!='excluded-infeasible' or
                        value.get('exclusion_reason')!=entry['reason'] or value.get('exclusion')!=entry):
                    raise RuntimeError('journal does not honor the input-domain exclusion ledger')
            elif value.get('prediction')=='excluded-infeasible' or value.get('excluded'):
                raise RuntimeError('journal excludes an event outside the input-domain ledger')
            elif value.get('prediction') not in registration['labels']+['']:
                raise RuntimeError('invalid normalized prediction in journal')
        records[spec['name']]=values
        metadata[spec['name']]=identity
    scores={};truth_all=[lookup[i]['label'] for i in ids];truth=[lookup[i]['label'] for i in scored]
    def scored_values(values):
        return [v for i,v in zip(ids,values) if i not in excluded_set]
    if 'native' in records:
        native=[r['prediction'] for r in scored_values(records['native'])]
        groups=[registration['splits']['50']['groups'][i] for i in scored]
        for name,values in records.items():
            active_values=scored_values(values)
            predicted=[r['prediction'] for r in active_values]
            summary=prediction_summary(truth,native,predicted,registration['labels'])
            nb=lock['config']['settings'].get('bootstrap_samples',2000)
            if nb>0:summary['paired']=paired_stats(truth,native,predicted,groups,seed=job['seed'],n_boot=nb)
            summary.update(generation_wall_seconds=sum(v.get('wall_seconds',0.) for v in active_values),
                           generation_model_calls=sum(v.get('new_model_calls',0) for v in active_values),
                           generation_model_call_units=sorted({v.get('model_call_unit','instrumented backend forward calls') for v in active_values}),
                           excluded_queries=len(excluded),excluded_query_ids=excluded,scored_queries=len(scored))
            spec=next(s for s in job['methods'] if s['name']==name)
            summary['cost_components']=method_cost(out,spec,job,registration,metadata[name])
            summary['query_feature_plus_generation_seconds']=summary['generation_wall_seconds']+summary['cost_components']['query_feature_wall_seconds']
            effective=[x for v in active_values for x in v.get('effective_relative_norms',{}).values()]
            summary['mean_effective_relative_norm']=float(np.mean(effective)) if effective else None
            planned=[x for v in active_values for x in v.get('applied_relative_norms',{}).values()]
            summary['mean_planned_relative_norm']=float(np.mean(planned)) if planned else None
            unchanged=[b['unchanged_fraction'] for v in active_values for b in v.get('applied_norms',{}).values() if 'unchanged_fraction' in b]
            summary['mean_unchanged_fraction']=float(np.mean(unchanged)) if unchanged else None
            if name in ('probe_to_lm','probe_distribution_to_lm'):
                pairs=[(v,t) for i,v,t in zip(ids,values,truth_all) if i not in excluded_set]
                summary.update(probe_agreement=sum(v['prediction']==v['external_prediction'] for v,_ in pairs),
                               correct_probe_spoiled=sum(v['external_prediction']==t and v['prediction']!=t for v,t in pairs),
                               incorrect_probe_rescued=sum(v['external_prediction']!=t and v['prediction']==t for v,t in pairs))
            scores[name]=summary
        def signatures(values):
            return [(r['prediction'],r['raw_prediction'],r.get('generated_tokens')) for r in scored_values(values)]
        for name in ('zero_alpha','zero_field'):
            if name in records and signatures(records[name])!=signatures(records['native']):
                raise RuntimeError(f'{name} failed exact native no-op check')
        if ('ridge_direct' in records and 'probe_to_text' in records and
                [r['prediction'] for r in scored_values(records['ridge_direct'])]!=
                [r['prediction'] for r in scored_values(records['probe_to_text'])]):
            raise RuntimeError('ridge text rendering changed class predictions')
    complete=bool(job['methods']) and not incomplete and len(scores)==len(job['methods'])
    report=dict(complete=complete,case=job['case'],job=job,case_fingerprint=case_fingerprint,
                registration_sha256=digest(registration),expected_queries=len(ids),query_ids=ids,
                scored_queries=len(scored),excluded_queries=len(excluded),excluded_query_ids=excluded,
                input_domain_exclusions=dict(reason='input-domain-infeasible-lowpass-nyquist',
                                             query_condition=job['condition'],query_count=len(excluded),
                                             query_ids=excluded,scored_queries=len(scored),
                                             registered_low_sr_events=((scan or {}).get('by_condition',{})
                                                                       .get(job['condition'],{}).get('total')),
                                             dataset_scan=scan,ledger_sha256=digest(ledger),
                                             entries=ledger.get('entries',[])),
                methods=scores,incomplete_methods=incomplete,complete_official_test=registration['complete_official_test'],
                costs=_summary_cost(out),new_scientific_results=complete and not lock.get('injected_backend',False),
                interpretation='Actual results if produced by real backends; contract-test injected backends are software validation only.')
    if write:
        atomic_json(out/'REPORT.json',report)
        with (out/'RESULTS.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['case','dataset','model','condition','seed','method','n','accuracy','macro_f1','invalid_rate','repaired','harmed','gain_pp'])
            writer.writeheader()
            for name,s in scores.items():writer.writerow({**{k:job[k] for k in ('case','dataset','model','condition','seed')},'method':name,**{k:s[k] for k in ('n','accuracy','macro_f1','invalid_rate','repaired','harmed','gain_pp')}})
        if not complete:atomic_json(out/'STATUS.json',dict(complete=False,status='incomplete',incomplete_methods=incomplete))
    return report


def run_case(root,out,config,job,*,execute=False,profile='24gb',assets=None,backend_factory=None,
             cache_only=False,config_path=None,audio_root=None):
    root=Path(root).resolve();out=Path(out).resolve();assets=assets or {};worker=None
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out/'.run.lock'):
        atomic_json(out/'STATUS.json',dict(complete=False,status='preflight',case=job['case']))
        started=time.perf_counter()
        try:
            reg_path=out/'REGISTRATION.json'
            if reg_path.exists():
                registration=read(reg_path);validate_registered_inputs(root,registration)
            else:
                registration=prepare_registration(root,config,job['dataset'],job['seed'],job['support_budgets'],config_path=config_path,audio_root=audio_root)
                lock_registration(reg_path,registration)
            extras=optional_preflight(job,assets)
            identity=dict(job=job,config=config,profile=profile,assets=asset_identity(assets),extra_provenance=extras,
                          registration_sha256=digest(registration),environment=environment(),
                          requested_audio_root=None if audio_root is None else str(Path(audio_root).resolve()),
                          injected_backend=backend_factory is not None)
            lock_registration(out/'CASE.json',identity)
            if job['case']=='c00' or not execute:
                status=dict(complete=False,status='preflight_passed',case=job['case'],
                            protocol_complete=True,scientific_results=False,expected_queries=len(registration['query_ids']),
                            support_counts={n:s['support_class_counts'] for n,s in registration['splits'].items()})
                atomic_json(out/'STATUS.json',status);return status
            worker=CaseWorker(root,out,config,job,registration,profile,assets,backend_factory,cache_only)
            worker.preflight();worker.prepare_exclusions()
            # FinalRunner's legacy descriptor means no model was run at preflight.
            # Here the locked query population is explicitly the official test.
            plan=read(worker.out/'PLAN.json');plan['no_test_evaluation']=False;plan['query_partition']='complete_official_test'
            atomic_json(worker.out/'PLAN.json',plan)
            for spec in job['methods']:
                atomic_json(out/'STATUS.json',dict(complete=False,status='running',method=spec['name'],case=job['case']))
                worker.trial(spec,digest(identity))
            atomic_json(out/'FIT_COSTS.json',worker.stage_costs)
            result=report_case(out)
            if not result['complete']:raise RuntimeError('required methods/query population incomplete')
            # A separate cache-only re-read checks coverage before the success bit.
            atomic_json(out/'CACHED_VALIDATION.json',dict(complete=True,report_sha256=file_hash(out/'REPORT.json'),
                         case_fingerprint=digest(identity),registration_sha256=digest(registration)))
            atomic_json(out/'STATUS.json',dict(complete=True,status='complete',case=job['case'],
                         report_sha256=file_hash(out/'REPORT.json'),invocation_seconds=time.perf_counter()-started,
                         scientific_results=backend_factory is None))
            return result
        except BaseException as error:
            atomic_json(out/'STATUS.json',dict(complete=False,status='stopped' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'error',
                         case=job['case'],error=type(error).__name__,reason=str(error)))
            raise
        finally:
            if worker is not None:worker.close()


def main(case=None,argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','check','run','report'],nargs='?',default='plan')
    if case is None:p.add_argument('--case',choices=sorted(CASES),required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--config',type=Path)
    p.add_argument('--dataset',default='dogs')
    p.add_argument('--model')
    p.add_argument('--condition',choices=['full','lp:1000'],default='full')
    p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--run-id',default='final_v1')
    p.add_argument('--output-root',type=Path)
    p.add_argument('--audio-root',type=Path)
    p.add_argument('--assets',type=Path,help='JSON with feature/naturelm asset mappings')
    p.add_argument('--profile',choices=['16gb','24gb','48gb','80gb'],default='24gb')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--cache-only',action='store_true')
    args=p.parse_args(argv);case=case or args.case;validate_run_id(args.run_id)
    job=case_plan(case,args.dataset,args.model,args.condition,args.seed)
    if args.action=='plan':print(json.dumps(job,indent=2));return 0
    root=args.root.resolve();config_path=args.config or root/'configs/akr_train50.yaml'
    cell=f'{job["dataset"]}__{job["model"]}__{job["condition"].replace(":","_")}__s{job["seed"]}'
    out=(args.output_root or root/'results/akr_train50')/args.run_id/case/cell
    try:
        config=yaml.safe_load(config_path.read_text());assets=read(args.assets) if args.assets else {}
    except BaseException as error:
        with run_lock(out/'.run.lock'):
            atomic_json(out/'STATUS.json',dict(complete=False,status='error',case=case,
                         error=type(error).__name__,reason=str(error)))
        raise
    if args.action=='report':
        with run_lock(out/'.run.lock'):
            atomic_json(out/'STATUS.json',dict(complete=False,status='verifying'))
            try:
                validate_registered_inputs(root,read(out/'REGISTRATION.json'))
                result=report_case(out)
                atomic_json(out/'STATUS.json',dict(complete=result['complete'],status='complete' if result['complete'] else 'incomplete',
                             report_sha256=file_hash(out/'REPORT.json')))
            except BaseException as error:
                atomic_json(out/'STATUS.json',dict(complete=False,status='error',reason=str(error)));raise
    else:
        if args.action=='run' and not args.execute and not args.cache_only:p.error('run requires --execute or --cache-only')
        result=run_case(root,out,config,job,execute=args.action=='run',profile=args.profile,assets=assets,
                        cache_only=args.cache_only,config_path=config_path,audio_root=args.audio_root)
    print(json.dumps({'output':str(out),**{k:v for k,v in result.items() if k not in ('job','query_ids')}},indent=2))
    return 0 if result.get('complete') or result.get('protocol_complete') or args.action=='check' else 2


if __name__=='__main__':raise SystemExit(main())
