"""F1/F2/F3: narrow, source-locked closing experiments; no backbone fitting.

Legacy mode reads source locks; standalone mode requires no old metadata.
S1 adds a same-support comparison with simpler readout-to-text alternatives.
Output goes to results/akr_final/<run_id>, never into the source run.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
from copy import deepcopy
import numpy as np
import yaml
from akr_closing.core import Query, Journal, atomic_json, digest, file_hash, read_manifest, metrics
from akr_closing.pipeline import prepare_audio, prompt_for, write_npz
from akr_closing.repair import RidgeReadout
from .predictability import grouped_predictability, fit_repository, derangement

STOP = False

def _stop(*_):
    global STOP
    STOP = True


def _check_stop():
    if STOP:
        raise InterruptedError('Safe stop requested; rerun the same command to resume.')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hashes(paths):
    return {str(p):file_hash(p) for p in paths if p.is_file()}


def select_episodes(root: Path, source: Path, options: dict) -> dict:
    run = _json(source/'RUN.json'); cfg = run['config']
    ds = next(d for d in cfg['datasets'] if d['name'] == options['dataset'])
    if options['label_mode'] != 'semantic':
        raise ValueError('This bounded addendum uses semantic labels only.')
    rows = read_manifest(root/ds['manifest'], old_root=cfg.get('old_root'), root=root)
    lookup = {r['event_id']:r for r in rows}
    labels = ds.get('labels')
    if labels is None:
        labels = yaml.safe_load((root/ds['labels_from_config']).read_text())['dataset']['labels']
    model = next(m for m in cfg['models'] if m['tag'] == 'qwen7b')
    seeds = list(cfg['seeds'])[:options['max_seeds']]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Nonempty distinct source seeds required.')
    conditions = options['conditions']
    if conditions != ['full','lp:1000']:
        raise ValueError('This final scope is exactly full and lp:1000.')
    episodes, used_paths = [], [source/'RUN.json', root/ds['manifest']]
    for seed in seeds:
        splitpath = source/'splits'/f"{ds['name']}_{seed}.json"
        split = _json(splitpath); used_paths.append(splitpath)
        support_reference = None
        for condition in conditions:
            context = f"qwen7b/{ds['name']}/s{seed}/semantic/{condition.replace(':','_')}"
            base = source/context/f"k{options['k_per_class']}"
            metapath, lockpath = base/'EPISODE.json', base/'LOCK.json'
            meta, lock = _json(metapath), _json(lockpath); used_paths += [metapath,lockpath]
            ids = meta['support_ids']; qids = split['confirmation']
            if not ids or not qids or len(set(ids)) != len(ids) or len(set(qids)) != len(qids):
                raise ValueError('Support/query IDs must be nonempty and unique.')
            if set(ids)&set(qids) or not set(ids)<=set(split['train']):
                raise ValueError('Support/query leakage or support outside original train split.')
            if any(i not in lookup for i in ids+qids):
                raise ValueError('An episode ID is absent from its manifest.')
            expected_labels = [lookup[i]['label'] for i in ids]
            if meta['support_labels'] != expected_labels or any(expected_labels.count(l) != options['k_per_class'] for l in labels):
                raise ValueError('Support labels/coverage differ from registered episode.')
            if lock.get('support',ids) != ids:
                raise ValueError('LOCK and EPISODE support differ.')
            if support_reference is not None and ids != support_reference:
                raise ValueError('Bandwidth comparison requires identical support IDs/order.')
            support_reference = ids
            if ds.get('group_key'):
                key = ds['group_key']
                if any(not lookup[i].get(key) for i in ids+qids):
                    raise ValueError('Real recording groups required.')
                if {lookup[i][key] for i in ids}&{lookup[i][key] for i in qids}:
                    raise ValueError('Support/query recording overlap.')
            if meta['condition'] != condition or meta['context'] != context or meta['k_per_class'] != options['k_per_class']:
                raise ValueError('Episode identity mismatch.')
            if meta.get('split_fingerprint') != split.get('fingerprint'):
                raise ValueError('Episode and source split fingerprints differ.')
            if lock.get('scale_type') != 'relative_active_state_norm':
                raise ValueError('Raw eta locks cannot be silently used as relative alpha.')
            if lock.get('prompt') != prompt_for(labels):
                raise ValueError('Noncanonical source prompt; do not silently replace it.')
            settings = lock['selection']
            if (not np.isfinite(settings['alpha']) or settings['alpha'] < 0 or int(settings['rank']) < 1
                    or settings.get('ridge_alpha',cfg['ridge_alpha']) <= 0):
                raise ValueError('Invalid locked settings.')
            b = meta['backbone']
            if b['model_id'] != model['id'] or b['revision'] != model['revision']:
                raise ValueError('Backbone identity does not match source config.')
            episodes.append({'seed':seed, 'condition':condition, 'context':context, 'source_base':str(base),
                             'support_ids':ids, 'query_ids':qids, 'lock':lock, 'meta':meta,
                             'lock_sha256':file_hash(lockpath), 'id':f"s{seed}_{condition.replace(':','_')}"})
    return {'config':cfg, 'dataset':ds, 'labels':labels, 'rows':rows, 'episodes':episodes,
            'source_hashes':_hashes(used_paths), 'selection_uses_outcomes':False,
            'evaluation_status':'previously inspected confirmation, locked re-evaluation; no new test'}


def prediction_summary(target, native, prediction, labels):
    base = np.asarray(native) == np.asarray(target)
    pred = np.asarray(prediction) == np.asarray(target)
    return {**metrics(target,prediction,labels), 'gain_pp':float((pred.astype(float)-base).mean()*100),
            'repaired':int(np.sum(pred&~base)), 'harmed':int(np.sum(base&~pred)),
            'both_correct':int(np.sum(base&pred)), 'both_wrong':int(np.sum(~base&~pred)),
            'scientific_success_required':False}


class FinalRunner:
    def __init__(self,root,source,run_id,options,profile='24gb',*,backend_factory=None,router_fit=None,cache_only=False,standalone_plan=None):
        import re
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_id):
            raise ValueError('run_id must be a simple name.')
        self.root,self.source = Path(root).resolve(),Path(source).resolve()
        self.out = self.root/'results/akr_final'/run_id
        if self.out == self.source or self.source in self.out.parents:
            raise ValueError('Output must not be inside source run.')
        self.options,self.profile,self.cache_only = deepcopy(options),profile,cache_only
        self.standalone = standalone_plan is not None
        if self.standalone and standalone_plan.get('mode') != 'standalone':
            raise ValueError('Explicit standalone plan required; old locks are not auto-fabricated.')
        self.plan=deepcopy(standalone_plan) if self.standalone else select_episodes(self.root,self.source,options)
        self.cfg=self.plan['config'];self.labels=self.plan['labels']
        self.lookup={r['event_id']:r for r in self.plan['rows']}
        self.backend_factory=backend_factory
        self.router_fit=router_fit or fit_repository
        self.backend=None;self.model_tag=None
        self.models={'qwen7b':next(m for m in self.cfg['models'] if m['tag']=='qwen7b'),
                     'qwen3b':{'tag':'qwen3b','id':self.cfg['E9']['id'],'revision':self.cfg['E9']['revision']}}
        self.spec={k:self.cfg[k] for k in ['layer_fractions','feature_fraction','max_context_tokens']}
        self.imports=[]
        files=list(Path(__file__).parent.glob('*.py'))
        files += [self.root/'akr_closing'/n for n in ['backend.py','core.py','repair.py','pipeline.py']]
        files += [self.root/'src/animal_omni/conditional_kv.py',self.root/'src/animal_omni/metrics.py']
        self.identity={'source':str(self.source),'source_hashes':self.plan['source_hashes'],
                       'options':options,'profile':profile,'code':_hashes(files),'models':self.models,
                       'environment':self._environment(), 'mode':'standalone' if self.standalone else 'source_locked',
                       'plan_sha256':digest(self.plan) if self.standalone else None}
        ip=self.out/'IDENTITY.json'
        if ip.exists() and _json(ip)!=self.identity:
            raise RuntimeError('Source/code/config/environment changed: choose a new run ID.')
        atomic_json(ip,self.identity)
        self.fingerprint=digest(self.identity)
        self.wave_hashes={}

    @staticmethod
    def _environment():
        from importlib.metadata import version,PackageNotFoundError
        result={'python':sys.version.split()[0]}
        for package in ['numpy','scipy','scikit-learn','torch','transformers','qwen-omni-utils']:
            try:result[package]=version(package)
            except PackageNotFoundError:result[package]=None
        return result

    def close(self):
        if self.backend is not None:
            self.backend=None;self.model_tag=None;gc.collect()
            if 'torch' in sys.modules:
                torch=sys.modules['torch']
                if torch.cuda.is_available():torch.cuda.empty_cache()

    def _model(self,tag,episode=None):
        if self.backend is None or self.model_tag!=tag:
            self.close()
            if self.cache_only:raise FileNotFoundError('Required saved state/prediction unavailable in --cache-only mode.')
            _check_stop()
            factory=self.backend_factory
            if factory is None:
                from akr_closing.backend import OmniBackend
                factory=OmniBackend
            model=self.models[tag]
            self.backend=factory(model['id'],model['revision'],profile=self.profile,**self.spec)
            self.model_tag=tag
            atomic_json(self.out/f'BACKEND_{tag}.json',self.backend.provenance)
        if tag=='qwen7b' and episode is not None and not self.standalone:
            expected=episode['meta']['backbone']
            for key in ['model_id','revision','layers','feature_layer','dtype','batch_size']:
                if self.backend.provenance.get(key)!=expected.get(key):
                    raise ValueError(f'Loaded backend {key} differs from source episode.')
        return self.backend

    def preflight(self):
        ids=sorted({i for e in self.plan['episodes'] for i in e['support_ids']+e['query_ids']})
        missing=[self.lookup[i]['audio_path'] for i in ids if not Path(self.lookup[i]['audio_path']).is_file()]
        if missing:raise FileNotFoundError(f'{len(missing)} waveform files missing; first: {missing[0]}')
        hashes={i:file_hash(Path(self.lookup[i]['audio_path'])) for i in ids}
        if self.standalone:
            for e in self.plan['episodes']:
                if {hashes[i] for i in e['support_ids']} & {hashes[i] for i in e['query_ids']}:
                    raise ValueError('Duplicate waveform bytes across standalone support/query partitions.')
            atomic_json(self.out/'STANDALONE_PROTOCOL.json', self.plan)
        previous=self.out/'WAVEFORM_HASHES.json'
        if previous.exists() and _json(previous)!=hashes:
            raise RuntimeError('Waveform bytes changed within this run.')
        atomic_json(previous,hashes);self.wave_hashes=hashes
        saved=self.source/'splits'/f"{self.options['dataset']}_waveform_hashes.json"
        if saved.exists():
            recorded=_json(saved)
            if any(recorded.get(i)!=hashes[i] for i in ids):
                raise ValueError('Waveform bytes no longer match source experiment.')
        self.source_cache_compatible=self._source_code_matches(self.source)
        atomic_json(self.out/'PLAN.json',{'episodes':self.plan['episodes'],'source':str(self.source),
                    'scope':self.options,'selection_uses_outcomes':False,'query_partition':self.plan.get('query_partition','existing confirmation'),
                    'source_cache_compatible':self.source_cache_compatible,'no_test_evaluation':True})

    def _source_code_matches(self,source):
        if self.standalone:return False
        rp=source/'RUN.json'
        if not rp.exists():return False
        old=_json(rp); current=old.get('config',{})
        for key in ['layer_fractions','feature_fraction','max_context_tokens']:
            if current.get(key)!=self.cfg.get(key):return False
        code=old.get('code',{})
        names=['backend.py','pipeline.py','repair.py','core.py']
        if not all((self.root/'akr_closing'/n).is_file() and code.get(n)==file_hash(self.root/'akr_closing'/n) for n in names):
            return False
        sources=old.get('repository_sources',{})
        for name in ['src/animal_omni/conditional_kv.py','src/animal_omni/metrics.py']:
            if not (self.root/name).is_file() or sources.get(name)!=file_hash(self.root/name):return False
        return True

    def _query(self,event,condition):
        row=self.lookup[event];p=Path(row['audio_path']);sha=self.wave_hashes[event]
        name=digest({'source':str(p),'hash':sha,'condition':condition})+'.wav'
        path=self.out/'audio'/self.options['dataset']/name
        prepare_audio(p,path,condition)
        return Query(event,str(path),row.get('recording_id') or event)

    def _old_qpath(self,event,condition,source):
        p=Path(self.lookup[event]['audio_path'])
        name=digest({'source':str(p),'hash':self.wave_hashes[event],'condition':condition})+'.wav'
        return source/'audio'/self.options['dataset']/name

    def _array(self,event,condition,tag,kind,episode):
        target=self.lookup[event]['label'] if kind=='gradient' else None
        if kind=='gradient' and event not in episode['support_ids']:
            raise ValueError('Only registered support examples may generate gradients.')
        prompt=prompt_for(self.labels)
        ident={'run':self.fingerprint,'event':event,'condition':condition,'tag':tag,'kind':kind,
               'waveform':self.wave_hashes[event],'prompt':prompt,'target':target}
        path=self.out/'cache'/(digest(ident)+'.npz');info=Path(str(path)+'.json')
        if path.exists() and info.exists():
            if _json(info)['sha256']!=file_hash(path):raise RuntimeError('Saved-array checksum mismatch.')
            with np.load(path,allow_pickle=False) as z:return z['value']
        oldsource=self.source if tag=='qwen7b' else self.source.with_name(self.source.name+'_3b_locked')
        oldpath=None
        if self._source_code_matches(oldsource):
            oldcfg=_json(oldsource/'RUN.json')['config']
            if any(m['id']==self.models[tag]['id'] and m['revision']==self.models[tag]['revision'] for m in oldcfg['models']):
                oldq=self._old_qpath(event,condition,oldsource)
                context=episode['context'].replace('qwen7b/',tag+'/',1)
                if kind=='feature':
                    key=digest({'q':event,'path':str(oldq),'prompt':prompt})
                    oldpath=oldsource/'features'/context/(key+'.npz'); field='feature'
                else:
                    key=digest({'id':event,'path':str(oldq),'target':target,'prompt':prompt,'variant':'real'})
                    oldpath=oldsource/'support_gradients'/context/(key+'.npz'); field='audio'
        start=time.perf_counter()
        if oldpath is not None and oldpath.is_file():
            with np.load(oldpath,allow_pickle=False) as z:value=np.asarray(z[field],dtype=np.float32)
            origin={'kind':'imported_saved_array','source':str(oldpath),'source_sha256':file_hash(oldpath),'new_model_calls':0}
        else:
            if self.cache_only:raise FileNotFoundError(f'Missing {tag} {kind}: {event}, {condition}')
            backend=self._model(tag,episode);q=self._query(event,condition)
            if kind=='gradient': value=backend.support_gradients(q,target,prompt)['audio']
            else:value=backend.feature(q,prompt)
            value=np.asarray(value,dtype=np.float32)
            origin={'kind':'new_support_backward' if kind=='gradient' else 'new_feature_forward',
                    'wall_seconds':time.perf_counter()-start}
        if value.ndim!=1 or not value.size or not np.isfinite(value).all():raise ValueError('Invalid cached feature/gradient.')
        write_npz(path,value=value);atomic_json(info,{'identity':ident,'sha256':file_hash(path),'origin':origin})
        return value

    def _support(self,e,tag):
        fs=np.stack([self._array(i,e['condition'],tag,'feature',e) for i in e['support_ids']])
        gs=np.stack([self._array(i,e['condition'],tag,'gradient',e) for i in e['support_ids']])
        return fs,gs

    def _queries(self,e,tag):
        return np.stack([self._array(i,e['condition'],tag,'feature',e) for i in e['query_ids']])

    def _existing_e9(self,e,method):
        return self._existing_predictions(e,'qwen3b',method)

    def _existing_predictions(self,e,tag,method):
        """Import a complete matching 7B confirmation or 3B E9 journal."""
        source=self.source if tag=='qwen7b' else self.source.with_name(self.source.name+'_3b_locked')
        if not self._source_code_matches(source):return None
        run=_json(source/'RUN.json')
        if run.get('profile')!=self.profile:return None
        if not any(m['id']==self.models[tag]['id'] and m['revision']==self.models[tag]['revision'] for m in run['config']['models']):return None
        stage='confirmation' if tag=='qwen7b' else 'confirmation_from_7b_lock'
        directory=source/e['context'].replace('qwen7b/',tag+'/',1)/f"k{self.options['k_per_class']}"/stage/'canonical'/method
        mp=directory/'manifest.json'
        if not mp.exists():return None
        manifest=_json(mp);p=manifest.get('provenance',{})
        if (manifest.get('expected')!=e['query_ids'] or p.get('support_ids')!=e['support_ids']
                or (tag=='qwen3b' and p.get('source_7b_lock_sha256')!=e['lock_sha256']) or p.get('prompt')!=prompt_for(self.labels)
                or p.get('lock')!=e['lock']['selection'] or p.get('scope')!='audio'
                or p.get('kind')!='kv' or p.get('layer_group')!='all'):
            return None
        if digest(p)!=manifest.get('fingerprint'):raise RuntimeError('E9 manifest checksum mismatch.')
        rows=[]
        for event in e['query_ids']:
            fp=directory/'items'/(digest(event)+'.json')
            if not fp.exists():return None
            item=_json(fp)
            if item.get('key')!=event or digest(item['result'])!=item.get('result_sha256'):
                raise RuntimeError('E9 prediction checksum mismatch.')
            rows.append(item['result'])
        return rows,{'source':str(directory),'manifest_sha256':file_hash(mp)}

    def _predict(self,path,e,tag,method,settings,fields,*,e9=False,extra=None):
        ids=e['query_ids'];ident={'run':self.fingerprint,'episode':e['id'],'tag':tag,'method':method,
                              'settings':settings,'extra':extra or {},'scope':'audio','ids':ids}
        if fields is not None:
            ident['field_sha256']=__import__('hashlib').sha256(np.ascontiguousarray(fields).tobytes()).hexdigest()
        journal=Journal(self.out/path,ident,ids)
        if journal.complete:return journal.results()
        same_map=(settings==e['lock']['selection'] and
                  (not extra or extra.get('source_condition')==e['condition']))
        eligible=method=='native' or same_map
        reused=self._existing_predictions(e,tag,method) if eligible else None
        if reused:
            records,origin=reused
            for event,record in zip(ids,records):
                if not journal.has(event):journal.add(event,{**record,'reuse_origin':origin,'new_model_calls':0})
            return journal.results()
        for j,event in enumerate(ids):
            if journal.has(event):continue
            _check_stop();q=self._query(event,e['condition']);t=time.perf_counter()
            field=None if fields is None else fields[j]
            alpha=0. if fields is None or method=='zero_alpha' else float(settings['alpha'])
            key={'run':self.fingerprint,'tag':tag,'event':event,'condition':e['condition'],
                 'alpha':alpha,'has_direction':field is not None,
                 'field':None if field is None else __import__('hashlib').sha256(np.ascontiguousarray(field).tobytes()).hexdigest()}
            cached=self.out/'prediction_cache'/(digest(key)+'.json')
            if cached.exists():
                item=_json(cached)
                if item['result_sha256']!=digest(item['result']):raise RuntimeError('Prediction-cache checksum mismatch.')
                journal.add(event,{**item['result'],'new_model_calls':0,'reuse_origin':str(cached)})
                continue
            backend=self._model(tag,e);before=getattr(backend,'forward_count',0)
            value=backend.predict(q,prompt_for(self.labels),self.labels,direction=field,alpha=alpha,scope='audio',kind='kv',layer_group='all')
            atomic_json(cached,{'identity':key,'result':value,'result_sha256':digest(value)})
            journal.add(event,{**value,'wall_seconds':time.perf_counter()-t,'new_model_calls':getattr(backend,'forward_count',0)-before})
        return journal.results()

    def _score_trial(self,path,e,predictions,native):
        y=[self.lookup[i]['label'] for i in e['query_ids']]
        pred=[r['prediction'] for r in predictions];base=[r['prediction'] for r in native]
        value=prediction_summary(y,base,pred,self.labels)
        value.update(episode=e['id'],query_ids=e['query_ids'],protocol=self.plan['evaluation_status'])
        atomic_json(self.out/path/'METRICS.json',value)
        atomic_json(self.out/path/'SCORED_PREDICTIONS.json',[{'event_id':i,'target':t,**r} for i,t,r in zip(e['query_ids'],y,predictions)])
        return value

    @staticmethod
    def _no_op(native,zero):
        a=[(r['prediction'],r.get('raw_prediction')) for r in native]
        b=[(r['prediction'],r.get('raw_prediction')) for r in zero]
        if a!=b:raise RuntimeError('Zero-alpha intervention was not a strict no-op.')

    def _settings_provenance(self,e,legacy_key='source_7b_lock_sha256'):
        if self.standalone:
            return {'fixed_settings_sha256':e['lock_sha256'],
                    'settings_origin':'standalone fixed YAML; no historical RUN/LOCK read'}
        return {legacy_key:e['lock_sha256'],
                'settings_origin':'existing 7B selection lock'}

    def run_f1(self):
        for e in self.plan['episodes']:
            path=self.out/'F1'/e['id']/'RESULT.json'
            if path.exists():
                saved=_json(path)
                if not saved.get('complete') or saved.get('support_ids')!=e['support_ids']:
                    raise RuntimeError('Invalid F1 completion record.')
                continue
            fs,gs=self._support(e,'qwen7b')
            groups=[self.lookup[i].get('recording_id') or i for i in e['support_ids']]
            labels=[self.lookup[i]['label'] for i in e['support_ids']]
            res=grouped_predictability(fs,gs,groups,labels,self.options['ranks'],
                  e['lock']['selection'].get('ridge_alpha',self.cfg['ridge_alpha']),
                  self.options['folds'],e['seed'],router_fit=self.router_fit)
            res.update(support_ids=e['support_ids'],episode=e['id'],**self._settings_provenance(e,'source_lock_sha256'))
            atomic_json(path,res)

    def run_f2(self):
        for seed in sorted(set(e['seed'] for e in self.plan['episodes'])):
            episodes={e['condition']:e for e in self.plan['episodes'] if e['seed']==seed}
            anchor=episodes[self.options['anchor_condition']]
            settings=anchor['lock']['selection'];penalty=settings.get('ridge_alpha',self.cfg['ridge_alpha'])
            fitted={};supports={}
            for source,e in episodes.items():
                x,g=self._support(e,'qwen7b');supports[source]=(x,g)
                fitted[source]=self.router_fit(x,g,rank=settings['rank'],alpha=penalty)
            for target,e in episodes.items():
                fq=self._queries(e,'qwen7b');base=Path('F2')/e['id']
                native=self._predict(base/'native',e,'qwen7b','native',settings,None)
                self._score_trial(base/'native',e,native,native)
                for source in self.options['conditions']:
                    model=fitted[source];x,g=supports[source]
                    pred=model.predict(fq);fixed=np.repeat(g.mean(0)[None,:],len(fq),axis=0)
                    control=pred[derangement(len(pred),seed)]
                    stage=base/('from_'+source.replace(':','_'))
                    for method,fields in [('continuous_pooled',pred),('fixed_mean',fixed),('shuffled_query_field',control)]:
                        rows=self._predict(stage/method,e,'qwen7b',method,settings,fields,
                                           extra={'source_condition':source,**self._settings_provenance(anchor,'anchor_lock_sha256')})
                        self._score_trial(stage/method,e,rows,native)
                    labels=np.array([self.labels.index(self.lookup[i]['label']) for i in e['support_ids']])
                    readout=RidgeReadout.fit(x,labels,len(self.labels),alpha=penalty)
                    rows=[{'prediction':self.labels[i],'raw_prediction':self.labels[i],'new_model_calls':0} for i in readout.predict(fq)]
                    self._score_trial(stage/'ridge_direct',e,rows,native)
                    if source==target:
                        zero=self._predict(stage/'zero_alpha',e,'qwen7b','zero_alpha',settings,pred)
                        self._no_op(native,zero)
                atomic_json(self.out/base/'COMPLETE.json',{'complete':True,'all_source_conditions':self.options['conditions'],
                            'no_hyperparameter_search':True,
                            **self._settings_provenance(anchor,'anchor_lock')})

    def run_f3(self):
        for tag in ['qwen7b','qwen3b']:
            for e in self.plan['episodes']:
                base=Path('F3')/tag/e['id'];settings=e['lock']['selection']
                x,g=self._support(e,tag);fq=self._queries(e,tag)
                model=self.router_fit(x,g,rank=settings['rank'],alpha=settings.get('ridge_alpha',self.cfg['ridge_alpha']))
                pred=model.predict(fq);fixed=np.repeat(g.mean(0)[None,:],len(fq),axis=0)
                native=self._predict(base/'native',e,tag,'native',settings,None,e9=(tag=='qwen3b'))
                self._score_trial(base/'native',e,native,native)
                zero=self._predict(base/'zero_alpha',e,tag,'zero_alpha',settings,pred,e9=(tag=='qwen3b'))
                self._no_op(native,zero)
                for name,field in [('fixed_mean',fixed),('continuous_pooled',pred)]:
                    rows=self._predict(base/name,e,tag,name,settings,field,e9=(tag=='qwen3b'))
                    self._score_trial(base/name,e,rows,native)
                y=np.array([self.labels.index(self.lookup[i]['label']) for i in e['support_ids']])
                readout=RidgeReadout.fit(x,y,len(self.labels),alpha=settings.get('ridge_alpha',self.cfg['ridge_alpha']))
                rows=[{'prediction':self.labels[i],'raw_prediction':self.labels[i],'new_model_calls':0} for i in readout.predict(fq)]
                self._score_trial(base/'ridge_direct',e,rows,native)
                atomic_json(self.out/base/'COMPLETE.json',{'complete':True,**self._settings_provenance(e),
                            'independent_model_specific_fit':True,'scope':'same-family cross-size replication, not cross-family'})
            self.close()

    def run_s1(self):
        from .simple_baselines import run_simple_baselines
        return run_simple_baselines(self)

    def run(self,tasks,execute=False):
        if not tasks or any(t not in {'F1','F2','F3','S1'} for t in tasks) or len(set(tasks))!=len(tasks):
            raise ValueError('Use a unique subset of F1,F2,F3,S1.')
        self.preflight()
        statuspath=self.out/'TASK_STATUS.json';status=_json(statuspath) if statuspath.exists() else {}
        failed=False
        for task in tasks:
            if not execute:
                print(f'{task}: planned ({len(self.plan["episodes"])} source episodes); no GPU work',flush=True)
                continue
            status[task]={'status':'running'};atomic_json(statuspath,status)
            atomic_json(self.out/'FINAL_STATUS.json',{'complete':False,'required_tasks':tasks,'tasks':status})
            try:
                _check_stop();getattr(self,'run_'+task.lower())()
                status[task]={'status':'complete','positive_result_required':False}
            except InterruptedError:
                status[task]={'status':'stopped','resume':'same command'};atomic_json(statuspath,status);self.close();raise
            except Exception as exc:
                status[task]={'status':'blocked','error':type(exc).__name__,'reason':str(exc)}
                failed=True;self.close()
            atomic_json(statuspath,status)
        final={'complete':execute and not failed and all(status.get(t,{}).get('status')=='complete' for t in tasks),
               'execute_requested':execute,'required_tasks':tasks,'tasks':status,
               'not_a_scientific_success_gate':True}
        atomic_json(self.out/'FINAL_STATUS.json',final)
        self.report()
        if failed:raise RuntimeError('At least one requested task is blocked; inspect TASK_STATUS.json. No missing score was filled.')
        return final

    def report(self):
        rows=[]
        for p in sorted(list(self.out.glob('F*/**/METRICS.json'))+list(self.out.glob('S1/**/METRICS.json'))):
            r=_json(p);rows.append({'path':str(p.parent.relative_to(self.out)),**{k:r[k] for k in
                ['n','accuracy','macro_f1','invalid_rate','gain_pp','repaired','harmed']}})
        if rows:
            with (self.out/'RESULTS.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        atomic_json(self.out/'COVERAGE.json',{'complete_trial_tables':len(rows),'task_status':_json(self.out/'TASK_STATUS.json') if (self.out/'TASK_STATUS.json').exists() else {},
                     'warning':'A trial table is not completion of all requested tasks. See FINAL_STATUS.json.'})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--source-run',default='akr_closing_all_v4')
    p.add_argument('--run-id',default='akr_final_three_v1')
    p.add_argument('--config',type=Path)
    p.add_argument('--tasks',default='F1,F2,F3')
    p.add_argument('--profile',choices=['16gb','24gb','48gb'],default='24gb')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--cache-only',action='store_true',help='Never load a model; missing arrays/predictions block the task.')
    args=p.parse_args(argv);root=args.root.resolve()
    options=yaml.safe_load((args.config or root/'configs/akr_final_three.yaml').read_text())
    source=Path(args.source_run)
    if not source.is_absolute():source=root/'results/akr_closing'/source
    signal.signal(signal.SIGINT,_stop);signal.signal(signal.SIGTERM,_stop)
    runner=FinalRunner(root,source,args.run_id,options,args.profile,cache_only=args.cache_only)
    from akr_closing.cli import run_lock
    try:
        with run_lock(root/'results/akr_closing/.runner.lock'):
            runner.run(args.tasks.split(','),args.execute)
    except InterruptedError:
        return 130
    finally:runner.close()
    print(f'Output: {runner.out}');return 0
