"""CPU contracts for the additive final suite; no model downloads or GPU work."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest


def module(name):
    spec = importlib.util.find_spec('akr_closing.' + name)
    assert spec is not None, 'final-suite implementation has not been added'
    return __import__('akr_closing.' + name, fromlist=['*'])


def test_core_module_exists():
    assert importlib.util.find_spec('akr_closing.core') is not None


def test_query_has_no_target_field():
    c = module('core')
    q = c.Query('x', '/audio.wav', 'recording1')
    assert not hasattr(q, 'label') and not hasattr(q, 'target')


def test_nested_balanced_support_excludes_query_groups():
    c = module('core')
    rows = [dict(event_id=f'{cl}_{g}_{j}', audio_path='x', label=cl,
                 recording_id=f'g{g}', split='')
            for g in range(15) for cl in ['a','b'] for j in range(2)]
    split = c.make_split(rows, ['a','b'], seed=9, group_key='recording_id')
    train = c.support_ids(rows, split['train'], ['a','b'], 2, seed=9)
    small = c.support_ids(rows, split['train'], ['a','b'], 1, seed=9)
    assert set(small) <= set(train)
    assert len(train) == 4
    lookup = {r['event_id']:r for r in rows}
    sets = [{lookup[i]['recording_id'] for i in split[s]}
            for s in ['train','selection','confirmation','test']]
    assert all(not sets[i] & sets[j] for i in range(4) for j in range(i+1,4))


def test_official_test_is_not_relabeled_untouched():
    c = module('core')
    rows = [dict(event_id=f'{sp}_{cl}_{j}', audio_path='x', label=cl,
                 recording_id=f'{sp}_{cl}_{j}', split=sp)
            for sp in ['train','valid','test'] for cl in ['a','b'] for j in range(10)]
    split = c.make_split(rows,['a','b'], seed=1, group_key=None)
    assert split['evaluation_status'] == 'locked_reevaluation'
    assert set(split['test']) == {r['event_id'] for r in rows if r['split']=='test'}


def test_resume_rejects_changed_fingerprint_and_partial_metrics(tmp_path):
    c = module('core')
    j = c.Journal(tmp_path/'job', {'config':1}, ['a','b'])
    j.add('a', {'prediction':'a'})
    assert not j.complete
    with pytest.raises(RuntimeError): j.results()
    with pytest.raises(RuntimeError): c.Journal(tmp_path/'job', {'config':2}, ['a','b'])
    j2 = c.Journal(tmp_path/'job', {'config':1}, ['a','b'])
    assert j2.has('a')
    j2.add('b', {'prediction':'b'})
    assert len(j2.results()) == 2


def test_duplicate_or_foreign_prediction_rejected(tmp_path):
    c = module('core'); j = c.Journal(tmp_path/'j', {}, ['x'])
    with pytest.raises(ValueError): j.add('other',{})
    j.add('x',{'prediction':'a'})
    with pytest.raises(ValueError): j.add('x',{'prediction':'b'})


def test_ridge_and_router_use_only_support():
    r = module('repair')
    x=np.array([[-2,0],[-1,0],[1,0],[2,0]],float); y=np.array([0,0,1,1])
    readout=r.RidgeReadout.fit(x,y,2,alpha=.1)
    assert readout.predict(np.array([[1.5,0]]))[0] == 1
    g=np.c_[x[:,0],-x[:,0],x[:,0]*0]
    router=r.GradientRouter.fit(x,g,rank=1,alpha=.1)
    pred=router.predict(np.array([[1.5,0]]))
    assert pred.shape == (1,3) and pred[0,0]>0 and pred[0,1]<0


def test_rank_is_bounded_by_support_and_mean_baseline():
    r=module('repair')
    router=r.GradientRouter.fit(np.array([[1.,2.]]),np.array([[3.,4.,5.]]),rank=8,alpha=1.)
    np.testing.assert_allclose(router.predict(np.array([[99.,-99.]])),[[3.,4.,5.]])
    assert router.rank == 0


def test_random_fields_have_matched_norm():
    r=module('repair'); x=np.arange(12,dtype=float).reshape(3,4)
    z=r.matched_random(x,seed=2)
    assert np.isclose(np.linalg.norm(z),np.linalg.norm(x))
    assert not np.allclose(z,x)


def test_centroid_geometry_uses_held_out_class_centroid():
    r=module('repair'); g=np.array([[1.,0.],[3.,0.],[0.,1.],[0.,3.]])
    result=r.geometry(g,np.array([0,0,1,1]))
    assert result['loo_label_accuracy'] == 1.
    assert result['n_support'] == 4
    assert result['loo_relative_squared_error'] > result['in_sample_relative_squared_error']
    assert 'not independent evidence of acoustic grounding' in result['interpretation']


def test_candidate_scoring_causal_positions():
    b=module('backend'); import torch
    logits=torch.zeros(1,5,7); ids=torch.tensor([[4,5]])
    logits[0,2,4]=9; logits[0,3,5]=9
    result=b.candidate_logprob(logits,ids,prompt_length=3)
    assert result['sequence_logprob'] > -.01


def test_hook_masks_audio_only_and_cleans_up():
    b=module('backend'); import torch
    linear=torch.nn.Linear(3,3,bias=False)
    linear.weight.data.copy_(torch.eye(3))
    x=torch.ones(1,4,3); mask=torch.tensor([[False,True,True,False]])
    with b.ProjectionIntervention({(0,'k'):linear},{(0,'k'):torch.ones(3)},mask,alpha=.1) as hook:
        z=linear(x)
        assert torch.equal(z[:,[0,3]],x[:,[0,3]])
        assert not torch.equal(z[:,1:3],x[:,1:3])
        ratio=torch.norm((z-x)[mask])/torch.norm(x[mask])
        assert abs(float(ratio.detach())-.1)<1e-6
        # Do not modify autoregressive decode positions.
        torch.testing.assert_close(linear(torch.ones(1,1,3)),torch.ones(1,1,3))
    torch.testing.assert_close(linear(x),x)
    assert not linear._forward_hooks


def test_zero_alpha_is_exact_identity():
    b=module('backend'); import torch
    linear=torch.nn.Linear(2,2,bias=False); x=torch.randn(1,3,2); base=linear(x)
    with b.ProjectionIntervention({(0,'v'):linear},{(0,'v'):torch.ones(2)},torch.ones(1,3,dtype=torch.bool),alpha=0.):
        torch.testing.assert_close(linear(x),base,rtol=0,atol=0)


def test_clustered_stats_identical_predictions():
    c=module('core')
    stats=c.paired_stats(['a','b','a'],['a','a','a'],['a','a','a'],['g1','g1','g2'],seed=1,n_boot=200)
    assert stats['gain_pp'] == 0 and stats['ci95_pp']==[0.,0.]
    assert stats['mcnemar_p']==1.


def test_gate_requires_positive_ci_and_low_invalid():
    c=module('core')
    assert c.confirmation_gate({'ci95_pp':[1,4],'gain_pp':2}, .0)
    assert not c.confirmation_gate({'ci95_pp':[-1,4],'gain_pp':2}, .0)
    assert not c.confirmation_gate({'ci95_pp':[1,4],'gain_pp':2}, .03)


def test_full_pipeline_on_synthetic_backend_never_leaks_query_labels(tmp_path):
    """Integration test only: synthetic classifications are not research results."""
    import csv
    import soundfile as sf
    p=module('pipeline')
    c=module('core')
    data=tmp_path/'data'; data.mkdir()
    rows=[]
    for split in ['train','valid','test']:
        for cl in ['a','b']:
            for i in range(20):
                event=f'{split}_{cl}_{i}'; path=data/(event+'.wav')
                sf.write(path,np.full(1600,(len(rows)+1)/200.),16000,subtype='FLOAT')
                rows.append(dict(event_id=event,audio_path=str(path),label=cl,split=split,recording_id=event))
    manifest=data/'manifest.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    class ToyBackend:
        def __init__(self,*args,**kw): self.provenance={'backend':'synthetic test fixture'}
        def feature(self,q,prompt):
            assert not hasattr(q,'label')
            return np.array([1.,0.]) if '_a_' in q.event_id else np.array([0.,1.])
        def support_gradients(self,q,target,prompt):
            assert q.event_id.startswith('train_')
            g=np.array([1.,-1.]) if target=='a' else np.array([-1.,1.])
            return {k:g for k in ['audio','text','full_prefill']}
        def predict(self,q,prompt,labels,**kw):
            assert not hasattr(q,'label')
            field=kw.get('direction'); alpha=kw.get('alpha',0)
            pred='a' if field is None or alpha==0 or not np.any(field) or field[0]>field[1] else 'b'
            return {'prediction':pred,'raw_prediction':pred}
        def candidates(self,q,prompt,labels,**kw): return {'prediction':'a','mean_prediction':'a','scores':[]}
    cfg={'router_implementation':'reference','run_test_after_gate':True,'datasets':[{'name':'toy','manifest':'data/manifest.csv','labels':['a','b'],'group_key':None,
         'label_modes':['semantic'],'conditions':['full'],'support_k':[2],'icl_k':[2],'repair_k':[2]}],
         'seeds':[3],'models':[{'tag':'toy','id':'toy','revision':'test'}],
         'layer_fractions':[.5],'feature_fraction':.5,'max_context_tokens':100,
         'ridge_alpha':1.,'ranks':[1],'relative_alphas':[.01],
         'max_invalid_rate':.01,'prompts':['canonical','paraphrase'],'gradient_nulls':False,
         'probe_to_lm':True,'lora_optimizer_steps':[32]}
    s=p.Study(tmp_path,cfg,ToyBackend,run_id='unit')
    s.run('all')
    summary=json.loads((s.out/'SUMMARY.json').read_text())
    assert summary['requested_phase_finished']
    assert any(r['stage']=='test_locked_reevaluation' for r in summary['completed_summaries'])
    before=len(list(s.out.rglob('items/*.json')))
    s2=p.Study(tmp_path,cfg,ToyBackend,run_id='unit'); s2.run('all')
    assert len(list(s.out.rglob('items/*.json')))==before


def test_audio_lowpass_resampling_preserves_duration(tmp_path):
    p=module('pipeline'); import soundfile as sf
    sr=48000; t=np.arange(sr//10)/sr
    x=np.sin(2*np.pi*400*t)+np.sin(2*np.pi*5000*t)
    source=tmp_path/'src.wav'; sf.write(source,x,sr,subtype='FLOAT')
    full=p.prepare_audio(source,tmp_path/'full.wav','full')
    low=p.prepare_audio(source,tmp_path/'low.wav','lp:1000')
    a,sa=sf.read(full); b,sb=sf.read(low)
    assert sa==sb==16000 and len(a)==len(b)==1600
    assert np.mean(b**2)<np.mean(a**2)


def test_prompt_reverse_does_not_change_vocabulary():
    p=module('pipeline')
    a=p.prompt_for(['A','B','C'],'canonical');b=p.prompt_for(['A','B','C'],'reverse_order')
    assert 'A, B, C' in a and 'C, B, A' in b


def test_lowpass_rejects_impossible_cutoff(tmp_path):
    p=module('pipeline');import soundfile as sf
    source=tmp_path/'x.wav'; sf.write(source,np.ones(1600),16000)
    with pytest.raises(ValueError):p.prepare_audio(source,tmp_path/'bad.wav','lp:20000')


def test_support_cannot_borrow_missing_class_from_query():
    c=module('core')
    rows=[{'event_id':'a','label':'a'},{'event_id':'b','label':'b'}]
    with pytest.raises(ValueError):c.support_ids(rows,['a'],['a','b'],1,seed=1)
