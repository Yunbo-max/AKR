from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from akr_final.predictability import grouped_predictability, error_decomposition, group_folds, derangement
from akr_final.runner import select_episodes, prediction_summary, FinalRunner
from akr_closing.core import atomic_json, digest, file_hash
from akr_closing.repair import GradientRouter


def test_group_folds_are_disjoint_and_exhaustive():
    groups=np.repeat(np.arange(6),3).astype(str)
    folds=group_folds(groups,5,23)
    assert sorted(np.concatenate([b for a,b in folds]).tolist())==list(range(18))
    for a,b in folds:
        assert not set(groups[a]) & set(groups[b])
        assert set(a)|set(b)==set(range(18))

@pytest.mark.parametrize('n',[0,1])
def test_derangement_rejects_too_few_queries(n):
    with pytest.raises(ValueError): derangement(n,5)

@pytest.mark.parametrize('n',[2,3,40])
def test_derangement_is_permutation_without_self(n):
    p=derangement(n,8)
    assert sorted(p.tolist())==list(range(n))
    assert np.all(p!=np.arange(n))
    np.testing.assert_array_equal(p,derangement(n,8))


def test_exact_error_decomposition():
    rng=np.random.default_rng(3)
    basis=np.linalg.qr(rng.normal(size=(9,3)))[0].T
    truth=rng.normal(size=(4,9)); mean=rng.normal(size=9); coef=rng.normal(size=(4,3))
    pred=mean+coef@basis
    z=error_decomposition(truth,pred,mean,basis)
    np.testing.assert_allclose(z['total_sse'],z['projection_sse']+z['coefficient_sse'],rtol=1e-6)


def test_predictability_refits_in_each_fold():
    rng=np.random.default_rng(8); x=rng.normal(size=(36,4)); g=x@rng.normal(size=(4,12))
    groups=np.repeat(np.arange(12),3).astype(str); labels=np.tile(np.arange(3),12)
    calls=[]
    from akr_closing.repair import GradientRouter
    def fit(a,b,rank,alpha):
        calls.append((a.copy(),b.copy()));return GradientRouter.fit(a,b,rank=rank,alpha=alpha)
    out=grouped_predictability(x,g,groups,labels,[1,4],.001,3,7,router_fit=fit)
    assert out['complete'] and len(out['rows'])==2
    assert all(len(a)<len(x) for a,b in calls)
    assert out['rows'][1]['predicted_nmse']<out['rows'][0]['predicted_nmse']
    assert out['rows'][1]['predicted_nmse']<out['rows'][1]['shuffled_nmse']
    assert out['query_gradients_computed']==0
    for fold in out['folds']:
        assert not set(fold['train_groups']) & set(fold['heldout_groups'])


def test_predictability_nonfinite_rejected():
    x=np.ones((8,3));x[0,0]=np.nan
    with pytest.raises(ValueError):
        grouped_predictability(x,np.ones((8,5)),np.repeat(['a','b'],4),np.zeros(8),[1],1,2,0)


def test_single_group_not_treated_as_independent_events():
    with pytest.raises(ValueError,match='group'):
        group_folds(['r']*10,3,1)


def test_summary_counts_repairs_harms_and_no_p_value_gate():
    out=prediction_summary(['A','A','B','B'],['A','B','B','A'],['A','A','','B'],['A','B'])
    assert out['repaired']==2 and out['harmed']==1
    assert out['gain_pp']==25.0 and out['invalid_rate']==.25
    assert out['scientific_success_required'] is False


def build_source(tmp, seeds=(11,12)):
    import csv, soundfile as sf
    cfg={'datasets':[{'name':'marmaudio','manifest':'data/marm.csv','labels':['A','B'], 'group_key':'recording_id'}],
         'seeds':list(seeds),'models':[{'tag':'qwen7b','id':'Qwen/Qwen2.5-Omni-7B','revision':'rev7'}],
         'E9':{'id':'Qwen/Qwen2.5-Omni-3B','revision':'rev3'},
         'layer_fractions':[.25,.5],'feature_fraction':.5,'max_context_tokens':4096,'ridge_alpha':1.0}
    root=tmp/'repo';root.mkdir();(root/'data').mkdir()
    rows=[]
    for i in range(12):
        p=root/'data'/f'{i}.wav';sf.write(p,np.linspace(-.1,.1,1600),16000,subtype='FLOAT')
        rows.append({'event_id':str(i),'audio_path':str(p),'label':'A' if i%2==0 else 'B','recording_id':f'r{i//2}'})
    with (root/'data/marm.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    source=root/'results/akr_closing/old';source.mkdir(parents=True)
    atomic_json(source/'RUN.json',{'config':cfg,'profile':'24gb','code':{}})
    atomic_json(source/'splits/marmaudio_waveform_hashes.json',{r['event_id']:file_hash(Path(r['audio_path'])) for r in rows})
    for seed in seeds:
        split={'train':[str(i) for i in range(8)],'selection':['8','9'],'confirmation':['10','11'],'test':[], 'fingerprint':'split'+str(seed)}
        atomic_json(source/f'splits/marmaudio_{seed}.json',split)
        for cond in ['full','lp:1000']:
            context=f'qwen7b/marmaudio/s{seed}/semantic/{cond.replace(":","_")}'
            base=source/context/'k4'
            meta={'context':context,'support_ids':split['train'],'support_labels':['A','B']*4,'k_per_class':4,'condition':cond,
                  'backbone':{'model_id':cfg['models'][0]['id'],'revision':'rev7','layers':[1,2],'feature_layer':2,'dtype':'bfloat16','batch_size':1},
                  'split_fingerprint':split['fingerprint']}
            atomic_json(base/'EPISODE.json',meta)
            atomic_json(base/'LOCK.json',{'selection':{'rank':2,'alpha':.01,'ridge_alpha':1},'support':split['train'],
                        'prompt':'Listen to the animal vocalization. Classify it using exactly one of these labels: A, B. Output the label only.',
                        'layers':[1,2],'feature_layer':2,'scale_type':'relative_active_state_norm'})
            atomic_json(base/'GATE.json',{'passed':False})
    opts={'dataset':'marmaudio','label_mode':'semantic','k_per_class':4,'max_seeds':3,'conditions':['full','lp:1000'],
          'anchor_condition':'lp:1000','ranks':[1,2],'folds':2,'bootstrap_samples':0}
    return root,source,opts


def test_select_all_declared_seeds_even_if_gates_failed(tmp_path):
    root,source,opts=build_source(tmp_path)
    plan=select_episodes(root,source,opts)
    assert len(plan['episodes'])==4
    assert set(e['seed'] for e in plan['episodes'])=={11,12}
    assert plan['selection_uses_outcomes'] is False


def test_missing_lock_is_blocked_not_dropped(tmp_path):
    root,source,opts=build_source(tmp_path)
    next(source.glob('qwen7b/**/LOCK.json')).unlink()
    with pytest.raises(FileNotFoundError):select_episodes(root,source,opts)


def test_different_support_across_bandwidth_rejected(tmp_path):
    root,source,opts=build_source(tmp_path)
    p=source/'qwen7b/marmaudio/s11/semantic/full/k4/EPISODE.json'
    z=json.loads(p.read_text());z['support_ids'][0]='8';atomic_json(p,z)
    with pytest.raises(ValueError):select_episodes(root,source,opts)


def test_recording_overlap_rejected(tmp_path):
    root,source,opts=build_source(tmp_path)
    import csv
    p=root/'data/marm.csv';rows=list(csv.DictReader(p.open()));rows[-1]['recording_id']='r0'
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='recording'): select_episodes(root,source,opts)


class FakeBackend:
    loads=0;grad_calls=[];predict_calls=0
    def __init__(self,model_id,revision,**kw):
        type(self).loads+=1;self.forward_count=0
        self.provenance={'model_id':model_id,'revision':revision,'layers':[1,2],'feature_layer':2,'dtype':'bfloat16','batch_size':1}
    def feature(self,q,prompt):
        self.forward_count+=1;i=int(q.event_id)
        return np.array([(-1.)**i,i/10,1,0],np.float32)
    def support_gradients(self,q,label,prompt):
        self.grad_calls.append(q.event_id);self.forward_count+=1
        h=self.feature(q,prompt)
        return {'audio':np.r_[h,h*2].astype(np.float32)}
    def predict(self,q,prompt,labels,*,direction=None,alpha=0,**kw):
        assert not hasattr(q,'label');type(self).predict_calls+=1;self.forward_count+=1
        prediction=labels[0] if direction is None or alpha==0 else labels[int(direction[0]<0)]
        return {'prediction':prediction,'raw_prediction':prediction,'applied_relative_norms':{} if alpha==0 else {'block':alpha},
                'applied_norms':{},'generated_tokens':1,'input_tokens':20}


def test_full_synthetic_workflow_and_resume_no_new_calls(tmp_path):
    root,source,opts=build_source(tmp_path,seeds=(11,))
    FakeBackend.grad_calls=[];FakeBackend.loads=0;FakeBackend.predict_calls=0
    r=FinalRunner(root,source,'final_v1',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)
    r.run(['F1','F2','F3'],execute=True)
    assert set(FakeBackend.grad_calls)<=set(str(i) for i in range(8))
    assert (r.out/'F1').exists() and (r.out/'F2').exists() and (r.out/'F3').exists()
    first=FakeBackend.predict_calls;loads=FakeBackend.loads
    r.close()
    r2=FinalRunner(root,source,'final_v1',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)
    r2.run(['F1','F2','F3'],execute=True)
    assert FakeBackend.predict_calls==first and FakeBackend.loads==loads
    status=json.loads((r2.out/'FINAL_STATUS.json').read_text())
    assert status['complete'] is True
    assert status['required_tasks']==['F1','F2','F3']


def test_preflight_never_loads_model(tmp_path):
    root,source,opts=build_source(tmp_path)
    FakeBackend.loads=0
    r=FinalRunner(root,source,'preflight',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)
    r.run(['F1','F2','F3'],execute=False)
    assert FakeBackend.loads==0


def test_changed_configuration_requires_new_run(tmp_path):
    root,source,opts=build_source(tmp_path)
    FinalRunner(root,source,'freeze',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)
    with pytest.raises(RuntimeError): FinalRunner(root,source,'freeze',{**opts,'folds':3},'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)


def test_production_router_matches_reference_when_package_available():
    pytest.importorskip('animal_omni.conditional_kv')
    from akr_final.predictability import fit_repository
    rng=np.random.default_rng(90);x=rng.normal(size=(12,4)).astype('float32');g=rng.normal(size=(12,8)).astype('float32')
    a=fit_repository(x,g,3,2);b=GradientRouter.fit(x,g,rank=3,alpha=2)
    np.testing.assert_allclose(a.predict(x),b.predict(x),atol=2e-5,rtol=2e-5)


def test_f2_uses_anchor_lock_not_target_selected_scale(tmp_path):
    root,source,opts=build_source(tmp_path,seeds=(11,))
    p=source/'qwen7b/marmaudio/s11/semantic/full/k4/LOCK.json';lock=json.loads(p.read_text())
    lock['selection']['alpha']=.3;atomic_json(p,lock)
    r=FinalRunner(root,source,'anchor',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)
    r.run(['F2'],execute=True)
    manifests=list((r.out/'F2').rglob('manifest.json'))
    assert len(manifests)>4
    for p in manifests:
        assert json.loads(p.read_text())['provenance']['settings']['alpha']==.01


def test_missing_cache_never_silently_loads_model(tmp_path):
    root,source,opts=build_source(tmp_path,seeds=(11,))
    FakeBackend.loads=0
    r=FinalRunner(root,source,'cached',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit,cache_only=True)
    with pytest.raises(RuntimeError,match='blocked'):r.run(['F1'],execute=True)
    assert FakeBackend.loads==0
    assert json.loads((r.out/'FINAL_STATUS.json').read_text())['complete'] is False


def test_changed_waveform_rejected_on_resume(tmp_path):
    root,source,opts=build_source(tmp_path,seeds=(11,))
    r=FinalRunner(root,source,'audio_changed',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)
    r.run(['F1'],execute=False)
    p=root/'data/0.wav';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(RuntimeError,match='Waveform'):r.preflight()


def test_cache_corruption_is_not_used(tmp_path):
    root,source,opts=build_source(tmp_path,seeds=(11,))
    r=FinalRunner(root,source,'corrupt',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)
    r.run(['F1'],execute=True)
    p=next((r.out/'cache').glob('*.npz'));p.write_bytes(b'corrupted')
    e=r.plan['episodes'][0]
    # Request every support state: one of them must detect its changed bytes.
    with pytest.raises(RuntimeError,match='checksum'):
        for e in r.plan['episodes']:r._support(e,'qwen7b')


def test_query_gradient_request_is_rejected(tmp_path):
    root,source,opts=build_source(tmp_path,seeds=(11,))
    r=FinalRunner(root,source,'labels',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)
    r.preflight();e=r.plan['episodes'][0]
    with pytest.raises(ValueError,match='support'):r._array(e['query_ids'][0],e['condition'],'qwen7b','gradient',e)


def test_bad_noop_blocks_instead_of_reporting_success(tmp_path):
    class BadBackend(FakeBackend):
        def predict(self,*args,direction=None,alpha=0,**kwargs):
            value=super().predict(*args,direction=direction,alpha=alpha,**kwargs)
            if direction is not None and alpha==0:value['raw_prediction']='different text'
            return value
    root,source,opts=build_source(tmp_path,seeds=(11,))
    r=FinalRunner(root,source,'badzero',opts,'24gb',backend_factory=BadBackend,router_fit=GradientRouter.fit)
    with pytest.raises(RuntimeError,match='blocked'):r.run(['F2'],execute=True)
    assert json.loads((r.out/'TASK_STATUS.json').read_text())['F2']['status']=='blocked'


def test_existing_e9_requires_matching_lock_and_complete_journal(tmp_path,monkeypatch):
    from akr_closing.core import Journal
    from akr_closing.pipeline import prompt_for
    root,source,opts=build_source(tmp_path,seeds=(11,))
    r=FinalRunner(root,source,'e9reuse',opts,'24gb',backend_factory=FakeBackend,router_fit=GradientRouter.fit)
    e=r.plan['episodes'][0];e9=source.with_name('old_3b_locked')
    cfg=dict(r.cfg);cfg['models']=[r.models['qwen3b']]
    atomic_json(e9/'RUN.json',{'config':cfg,'profile':'24gb'})
    monkeypatch.setattr(r,'_source_code_matches',lambda source:True)
    base=e9/e['context'].replace('qwen7b/','qwen3b/',1)/'k4/confirmation_from_7b_lock/canonical/native'
    meta={'support_ids':e['support_ids'],'source_7b_lock_sha256':e['lock_sha256'],'prompt':prompt_for(r.labels),
          'lock':e['lock']['selection'],'scope':'audio','kind':'kv','layer_group':'all'}
    j=Journal(base,meta,e['query_ids'])
    j.add(e['query_ids'][0],{'prediction':'A','raw_prediction':'A'})
    assert r._existing_e9(e,'native') is None
    j.add(e['query_ids'][1],{'prediction':'B','raw_prediction':'B'})
    assert len(r._existing_e9(e,'native')[0])==2
    changed=dict(e);changed['lock_sha256']='unmatched'
    assert r._existing_e9(changed,'native') is None
