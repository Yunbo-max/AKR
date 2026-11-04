"""Rebuttal evaluation reusing the original source-frozen AKR cache pipeline.

Real-model runs are opt-in. Synthetic tests check software contracts only.
Historical result directories and original S1/F1/F2/F3 settings are untouched.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from akr_closing.core import Journal,atomic_json,digest,file_hash
from akr_closing.cli import validate_run_id
from akr_closing.pipeline import prompt_for
from akr_closing.repair import RidgeReadout,matched_random
from .runner import FinalRunner,_check_stop,prediction_summary
from .predictability import derangement
from .simple_baselines import evidence_prompt
from .rebuttal_math import GainMap,fit_gains
from .rebuttal_plan import build_study


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def read_json(p):return json.loads(Path(p).read_text())


def measured(backend,operation):
    """Wall time includes synchronization, excludes checkpoint loading/features."""
    torch=getattr(backend,'torch',None)
    devices=list(range(torch.cuda.device_count())) if torch is not None and torch.cuda.is_available() else []
    for d in devices:torch.cuda.synchronize(d);torch.cuda.reset_peak_memory_stats(d)
    before=getattr(backend,'forward_count',0);start=time.perf_counter()
    value=operation()
    for d in devices:torch.cuda.synchronize(d)
    return {**value,'wall_seconds':time.perf_counter()-start,
            'new_model_calls':getattr(backend,'forward_count',0)-before,
            'peak_allocated_bytes':{str(d):torch.cuda.max_memory_allocated(d) for d in devices},
            'timing_scope':'this model call; excludes model load, first-pass features and support fitting'}


def expected_methods(job,settings):
    names=['native','zero','zero_field','fixed_relative','akr_relative','akr_cap','scalar_cap','gain_cap',
           'akr_all','ridge_fit','ridge_all','probe_to_text','probe_to_lm']
    if job['controls']:
        names += [f'control_{label}_a{a:g}' for a in settings.get('control_alphas',[settings['relative_alpha']])
                  for label in ['fixed','predicted','random','shuffled_pair']]
        names += ['k_only_cap','v_only_cap']
    return names


class DatasetRunner(FinalRunner):
    def __init__(self,root,out,name,entry,profile,backend_factory=None):
        super().__init__(root,Path(root)/'results/.no_historical_run_required',name,entry['options'],profile,
                         standalone_plan=entry['plan'],backend_factory=backend_factory,output_root=out)
        self.settings=entry['options']['settings']

    def preflight(self):
        super().preflight()
        for e in self.plan['episodes']:
            roles=[e[k] for k in ['fit_ids','calibration_ids','guard_ids','query_ids']]
            sets=[{self.wave_hashes[i] for i in ids} for ids in roles]
            if any(sets[a]&sets[b] for a in range(4) for b in range(a+1,4)):
                raise ValueError('Duplicate waveform bytes across fit/calibration/guard/query roles')

    def matrix(self,ids,e,tag,kind='feature'):
        values=[]
        for event in ids:
            _check_stop()
            values.append(self._array(event,e['condition'],tag,kind,e))
        return np.stack(values)

    def _calibration(self,e,tag,mapping,scalar):
        base=self.out/'policies'/tag/e['id']/('scalar' if scalar else 'coordinates')
        data={part:self.matrix(e[part+'_ids'],e,tag) for part in ['calibration','guard']}
        gm=GainMap(mapping);s=self.settings
        # Include numerical map predictions in the identity, not only a name.
        ident={'run':self.fingerprint,'episode':e['id'],'model':tag,'scalar':scalar,
               'size':gm.size,'parts':{p:e[p+'_ids'] for p in data},
               'map_sha256':array_hash(np.concatenate([gm.predict(data[p],np.ones(gm.size)) for p in data]))}
        saved=base/'POLICY.json'
        if saved.exists():
            item=read_json(saved)
            if item['identity']!=ident or digest(item['result'])!=item['result_sha256']:
                raise RuntimeError('Changed/corrupt gain policy')
            return item['result']
        def evaluate(gains,part):
            if part not in ['calibration','guard']:raise ValueError('Only registered calibration/guard losses allowed')
            ids=e[part+'_ids']
            if set(ids)&set(e['query_ids']):raise ValueError('query label leakage in calibration')
            fields=None if gains is None else gm.predict(data[part],gains)
            ci={'run':self.fingerprint,'episode':e['id'],'tag':tag,'partition':part,'ids':ids,
                'fields':None if fields is None else array_hash(fields),
                'alpha':s['relative_alpha'],'raw_eta':s['raw_eta'],'mode':'relative_cap'}
            journal=Journal(self.out/'loss_cache'/digest(ci),ci,ids)
            for j,event in enumerate(ids):
                if journal.has(event):continue
                _check_stop();backend=self._model(tag,e);q=self._query(event,e['condition'])
                # This is the only labelled endpoint; ids are registered above.
                target=self.lookup[event]['label'];field=None if fields is None else fields[j]
                value=measured(backend,lambda:backend.support_answer_loss(q,target,prompt_for(self.labels),
                    direction=field,alpha=s['relative_alpha'],scale_mode='relative_cap',raw_eta=s['raw_eta']))
                journal.add(event,value)
            return np.array([row['answer_loss'] for row in journal.results()])
        result=fit_gains(gm.size,evaluate,steps=s['gain_steps'],regularization=s['gain_regularization'],
                        upper=s['gain_upper'],min_improvement=s['min_improvement'],
                        guard_tolerance=s['guard_tolerance'],scalar=scalar)
        result.update(fit_ids=e['fit_ids'],calibration_ids=e['calibration_ids'],guard_ids=e['guard_ids'],
                      applied_on_actual_backend=True,backbone_updates=0)
        atomic_json(saved,{'identity':ident,'result':result,'result_sha256':digest(result)})
        return result

    def trial(self,e,tag,name,fields,native,*,mode='relative_norm',alpha=None,kind='kv',
              prompts=None,external=None,labelled_count=0,policy=None):
        s=self.settings;alpha=s['relative_alpha'] if alpha is None else alpha
        base=self.out/'trials'/tag/e['id']/name;ids=e['query_ids']
        identity={'run':self.fingerprint,'episode':e['id'],'model':tag,'method':name,
            'mode':mode,'alpha':alpha,'raw_eta':s['raw_eta'],'kind':kind,
            'fields':None if fields is None else array_hash(fields),'prompts':prompts,
            'external':external,'labelled_count':labelled_count,'policy':policy}
        journal=Journal(base,identity,ids)
        for j,event in enumerate(ids):
            if journal.has(event):continue
            _check_stop()
            if external is not None and prompts is None:
                label=external[j];value={'prediction':label,'raw_prediction':label,
                    'new_model_calls':0,'wall_seconds':0.,'output_source':'external_ridge'}
            else:
                backend=self._model(tag,e);q=self._query(event,e['condition'])
                prompt=prompt_for(self.labels) if prompts is None else prompts[j]
                field=None if fields is None else fields[j]
                value=measured(backend,lambda:backend.predict(q,prompt,self.labels,
                    direction=field,alpha=alpha if field is not None else 0.,scope='audio',kind=kind,
                    layer_group='all',scale_mode=mode,raw_eta=s['raw_eta'],
                    max_new_tokens=s.get('max_new_tokens',12)))
                if external is not None:value['external_prediction']=external[j]
            journal.add(event,value)
        records=journal.results()
        if native is None:native=records
        # Labels enter only AFTER this method has complete query predictions.
        truth=[self.lookup[i]['label'] for i in ids]
        score=prediction_summary(truth,[r['prediction'] for r in native],[r['prediction'] for r in records],self.labels)
        planned=[v for r in records for v in r.get('applied_relative_norms',{}).values()]
        effective=[v for r in records for v in r.get('effective_relative_norms',{}).values()]
        unchanged=[b['unchanged_fraction'] for r in records for b in r.get('applied_norms',{}).values() if 'unchanged_fraction' in b]
        score.update(method=name,model=tag,episode=e['id'],seed=e['seed'],condition=e['condition'],
            dataset=self.options['dataset'],labelled_count=labelled_count,
            planned_ratio_mean=float(np.mean(planned)) if planned else None,
            effective_ratio_mean=float(np.mean(effective)) if effective else None,
            unchanged_fraction_mean=float(np.mean(unchanged)) if unchanged else None,
            prediction_seconds=sum(r.get('wall_seconds',0) for r in records),
            generation_forward_calls=sum(r.get('new_model_calls',0) for r in records),
            policy_accepted=None if policy is None else policy['accepted'],
            policy_reason=None if policy is None else policy['reason'],
            cost_note='prediction calls only; see shared feature/gradient and calibration costs separately',
            query_ids=ids,protocol=self.plan['evaluation_status'])
        if prompts is not None and external is not None:
            score.update(probe_agreement=sum(r['prediction']==p for r,p in zip(records,external)),
                correct_probe_spoiled=sum(p==t and r['prediction']!=t for r,p,t in zip(records,external,truth)),
                incorrect_probe_rescued=sum(p!=t and r['prediction']==t for r,p,t in zip(records,external,truth)))
        atomic_json(base/'METRICS.json',score)
        atomic_json(base/'SCORED_PREDICTIONS.json',[{'event_id':i,'target':t,**r} for i,t,r in zip(ids,truth,records)])
        return records

    def run_job(self,job):
        e=next(e for e in self.plan['episodes'] if e['id']==job['episode']);tag=job['model'];s=self.settings
        x=self.matrix(e['fit_ids'],e,tag);g=self.matrix(e['fit_ids'],e,tag,'gradient')
        mapping=self.router_fit(x,g,rank=s['rank'],alpha=s['ridge_alpha']);gm=GainMap(mapping)
        # Lock both policies before generating/scoring any query answer.
        scalar=self._calibration(e,tag,mapping,True);coords=self._calibration(e,tag,mapping,False)
        fq=self.matrix(e['query_ids'],e,tag);field=mapping.predict(fq)
        fixed=np.repeat(g.mean(0)[None,:],len(fq),axis=0);fit_n=len(e['fit_ids']);all_n=len(e['support_ids'])
        methods=[]
        def run(name,arr,native,**kwargs):
            records=self.trial(e,tag,name,arr,native,**kwargs);methods.append(name);return records
        native=run('native',None,None)
        zero=run('zero',field,native,alpha=0.)
        self._no_op(native,zero)
        zero_field=run('zero_field',np.zeros_like(field),native)
        self._no_op(native,zero_field)
        run('fixed_relative',fixed,native,labelled_count=fit_n)
        run('akr_relative',field,native,labelled_count=fit_n)
        run('akr_cap',field,native,mode='relative_cap',labelled_count=fit_n)
        for name,policy in [('scalar_cap',scalar),('gain_cap',coords)]:
            run(name,gm.predict(fq,policy['gains']),native,mode='relative_cap',labelled_count=all_n,
                policy={'accepted':policy['accepted'],'reason':policy['reason'],'gains':policy['gains']})
        # Simple alternatives receive the entire annotated budget (incl. cal/guard).
        xa=self.matrix(e['support_ids'],e,tag);ga=self.matrix(e['support_ids'],e,tag,'gradient')
        allmap=self.router_fit(xa,ga,rank=s['rank'],alpha=s['ridge_alpha'])
        run('akr_all',allmap.predict(fq),native,labelled_count=all_n)
        predicted={}
        for name,features,ids in [('ridge_fit',x,e['fit_ids']),('ridge_all',xa,e['support_ids'])]:
            labels=np.array([self.labels.index(self.lookup[i]['label']) for i in ids])
            readout=RidgeReadout.fit(features,labels,len(self.labels),alpha=s['ridge_alpha'])
            predicted[name]=[self.labels[k] for k in readout.predict(fq)]
            run(name,None,native,external=predicted[name],labelled_count=len(ids))
        p=predicted['ridge_all'];text=run('probe_to_text',None,native,external=p,labelled_count=all_n)
        if [r['prediction'] for r in text]!=p:raise RuntimeError('probe-to-text changed class identity')
        prompts=[evidence_prompt(prompt_for(self.labels),label,self.labels) for label in p]
        run('probe_to_lm',None,native,external=p,prompts=prompts,labelled_count=all_n)
        if job['controls']:
            random=np.stack([matched_random(f,e['seed']+j) for j,f in enumerate(field)])
            shuffle=self.router_fit(x[derangement(len(x),e['seed'])],g,rank=s['rank'],alpha=s['ridge_alpha']).predict(fq)
            for a in s.get('control_alphas',[s['relative_alpha']]):
                for label,arr in [('fixed',fixed),('predicted',field),('random',random),('shuffled_pair',shuffle)]:
                    run(f'control_{label}_a{a:g}',arr,native,alpha=a,labelled_count=fit_n)
            for k in ['k','v']:
                run(k+'_only_cap',field,native,mode='relative_cap',kind=k,labelled_count=fit_n)
        if methods!=expected_methods(job,s):raise RuntimeError('Missing planned method in completed job')
        receipt={'complete':True,'job':job,'methods':methods,'fit_ids':e['fit_ids'],
                 'calibration_ids':e['calibration_ids'],'guard_ids':e['guard_ids'],'query_ids':e['query_ids'],
                 'query_labels_for_inference':False,'nominal_rank':s['rank'],'effective_rank':gm.size-1,
                 'positive_result_required':False,'protocol':self.plan['evaluation_status']}
        atomic_json(self.out/'jobs'/tag/(e['id']+'.json'),receipt)
        return receipt


class RebuttalStudy:
    def __init__(self,root,config,run_id,tasks,profile='24gb',*,backend_factory=None):
        self.root=Path(root).resolve();self.out=self.root/'results/akr_rebuttal'/validate_run_id(run_id)
        self.study=build_study(self.root,config,tasks);self.profile=profile;self.factory=backend_factory
        code={str(p.relative_to(self.root)):file_hash(p) for directory in ['akr_final','akr_closing','src/animal_omni']
              for p in (self.root/directory).glob('*.py')}
        self.identity={'study':digest(self.study),'profile':profile,'code':code,'environment':FinalRunner._environment()}
        dest=self.out/'IDENTITY.json'
        if dest.exists() and read_json(dest)!=self.identity:raise RuntimeError('Study/code/config/environment changed: choose a new run ID')
        atomic_json(dest,self.identity);atomic_json(self.out/'STUDY.json',self.study)

    def dataset_runner(self,name):
        return DatasetRunner(self.root,self.out,name,self.study['datasets'][name],self.profile,self.factory)

    def run(self,execute=False):
        if execute:atomic_json(self.out/'FINAL_STATUS.json',{'complete':False,'status':'preflight'})
        # Preflight every selected dataset before allocating any model.
        runners={}
        for name in self.study['datasets']:
            runner=self.dataset_runner(name);runners[name]=runner
            try:runner.preflight()
            finally:runner.close()
        if not execute:
            result={'complete':False,'preflight_complete':True,'gpu_started':False,'jobs':self.study['jobs']}
            atomic_json(self.out/'PREFLIGHT.json',result);return result
        status={}
        try:
            for job in sorted(self.study['jobs'],key=lambda j:({'core':0,'scale':1,'dogs':2,'watkins':3}[j['suite']],j['seed'],j['condition'])):
                key=f"{job['suite']}/{job['model']}/{job['dataset']}/{job['episode']}"
                # Retain at most one loaded model across dataset runners.
                for r in runners.values():
                    if r.options['dataset']!=job['dataset'] or r.model_tag not in (None,job['model']):r.close()
                runner=runners[job['dataset']]
                _check_stop();status[key]='running';atomic_json(self.out/'JOB_STATUS.json',status)
                try:runner.run_job(job);status[key]='complete'
                except BaseException as ex:
                    status[key]='stopped' if isinstance(ex,(InterruptedError,KeyboardInterrupt)) else 'blocked'
                    atomic_json(self.out/'JOB_STATUS.json',status)
                    atomic_json(self.out/'ERROR.json',{'job':job,'type':type(ex).__name__,'message':str(ex)})
                    atomic_json(self.out/'FINAL_STATUS.json',{'complete':False,'status':status[key],'job':job})
                    raise
                atomic_json(self.out/'JOB_STATUS.json',status)
        finally:
            for r in runners.values():r.close()
        report=self.report();atomic_json(self.out/'FINAL_STATUS.json',{k:v for k,v in report.items() if k!='rows'})
        return report

    def report(self):
        return report_study(self.out)


def report_study(out):
    """Verify journal completeness/checksums; do not trust stale METRICS alone."""
    out=Path(out);study=read_json(out/'STUDY.json');rows=[];incomplete=[];completed=0
    for job in study['jobs']:
        directory=out/job['dataset'];receipt=directory/'jobs'/job['model']/(job['episode']+'.json')
        if not receipt.exists():incomplete.append({'job':job,'reason':'no complete receipt'});continue
        rec=read_json(receipt);job_ok=(bool(rec.get('complete')) and rec['job']==job
            and rec.get('methods')==expected_methods(job,study['config']['settings']));job_rows=[]
        plan=study['datasets'][job['dataset']]['plan']
        episode=next(e for e in plan['episodes'] if e['id']==job['episode'])
        if rec['query_ids']!=episode['query_ids']:raise RuntimeError('Receipt/query identity mismatch')
        native_dir=directory/'trials'/job['model']/job['episode']/'native'
        if not (native_dir/'manifest.json').exists():
            incomplete.append({'job':job,'reason':'native journal absent'});continue
        nm=read_json(native_dir/'manifest.json');nj=Journal(native_dir,nm['provenance'],nm['expected'])
        if not nj.complete:
            incomplete.append({'job':job,'reason':'native journal incomplete'});continue
        if nm['expected']!=episode['query_ids']:raise RuntimeError('Native/query identity mismatch')
        native_prediction=[r['prediction'] for r in nj.results()]
        lookup={r['event_id']:r for r in plan['rows']}
        truth=[lookup[i]['label'] for i in episode['query_ids']]
        for name in rec['methods']:
            base=directory/'trials'/job['model']/job['episode']/name;mp=base/'manifest.json'
            if not mp.exists():job_ok=False;continue
            meta=read_json(mp);journal=Journal(base,meta['provenance'],meta['expected'])
            if not journal.complete or not (base/'METRICS.json').exists():job_ok=False;continue
            # Validate complete expected identity, not just a count.
            if meta['expected']!=rec['query_ids']:raise RuntimeError('Report/query identity mismatch')
            score=read_json(base/'METRICS.json')
            recomputed=prediction_summary(truth,native_prediction,[r['prediction'] for r in journal.results()],plan['labels'])
            for metric,value in recomputed.items():
                if metric not in score or not np.isclose(score[metric],value,rtol=1e-9,atol=1e-12):
                    raise RuntimeError(f'Stale/corrupt metric {metric} in {base}')
            if any(score.get(k)!=v for k,v in [('method',name),('model',job['model']),('episode',job['episode']),('dataset',job['dataset'])]):
                raise RuntimeError('Metric identity mismatch')
            job_rows.append({k:v for k,v in score.items() if k not in ['query_ids','protocol','cost_note']})
        if job_ok:completed+=1;rows.extend(job_rows)
        else:incomplete.append({'job':job,'reason':'missing or incomplete method journal'})
    result={'complete':completed==len(study['jobs']),'completed_jobs':completed,'expected_jobs':len(study['jobs']),
            'incomplete_trials':incomplete,'rows':rows,'completion_requires_positive_gain':False,
            'note':'Repeated support/group draws may overlap; no pooled independent-trial claim.'}
    if rows:
        fields=sorted({k for r in rows for k in r})
        with (out/'RESULTS.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    elif (out/'RESULTS.csv').exists():(out/'RESULTS.csv').unlink()
    atomic_json(out/'COVERAGE.json',{k:v for k,v in result.items() if k!='rows'})
    # Summarize only cohorts whose full expected set of support draws exists.
    from collections import defaultdict
    expected=defaultdict(int);grouped=defaultdict(list)
    for job in study['jobs']:
        for method in expected_methods(job,study['config']['settings']):
            expected[(job['dataset'],job['model'],job['condition'],method)]+=1
    for r in rows:grouped[(r['dataset'],r['model'],r['condition'],r['method'])].append(r)
    summaries=[]
    for key,group in sorted(grouped.items()):
        if len(group)!=expected[key]:continue
        row=dict(zip(['dataset','model','condition','method'],key));row['support_draws']=len(group)
        for metric in ['accuracy','macro_f1','invalid_rate','gain_pp','repaired','harmed']:
            values=np.array([r[metric] for r in group],dtype=float)
            row[metric+'_mean']=float(values.mean())
            row[metric+'_sample_sd']=float(values.std(ddof=1)) if len(values)>1 else None
        summaries.append(row)
    if summaries:
        with (out/'SUMMARY.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
    elif (out/'SUMMARY.csv').exists():(out/'SUMMARY.csv').unlink()
    text=['# AKR rebuttal execution report', '',
          f"Complete jobs: {completed}/{len(study['jobs'])}. Missing jobs are not zero scores.",
          'New protocol; support draws may share source recordings. No significance gate is used.', '',
          '| Dataset | Model | Condition | Method | Draws | Accuracy (%) | Macro-F1 (%) |',
          '|---|---|---|---|---:|---:|---:|']
    for r in summaries:
        text.append('| '+ ' | '.join([r['dataset'],r['model'],r['condition'],r['method'],str(r['support_draws']),
                    f"{100*r['accuracy_mean']:.2f}",f"{100*r['macro_f1_mean']:.2f}"])+ ' |')
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    costs={'shared_feature_arrays':0,'shared_gradient_arrays':0,'stored_feature_gradient_seconds':0.,
           'support_answer_loss_calls':0,'stored_support_answer_loss_seconds':0.,
           'note':'Shared cache costs counted once. Prediction-only costs are in RESULTS.csv; cache reuse is not zero intrinsic latency.'}
    for name in study['datasets']:
        for fp in (out/name/'cache').glob('*.npz.json'):
            item=read_json(fp);kind=item.get('identity',{}).get('kind')
            if kind in ['feature','gradient']:costs['shared_'+kind+'_arrays']+=1
            costs['stored_feature_gradient_seconds']+=item.get('origin',{}).get('wall_seconds',0.)
        for fp in (out/name/'loss_cache').glob('*/items/*.json'):
            item=read_json(fp);costs['support_answer_loss_calls']+=1
            costs['stored_support_answer_loss_seconds']+=item['result'].get('wall_seconds',0.)
    atomic_json(out/'SUPPORT_COSTS.json',costs)
    return result
