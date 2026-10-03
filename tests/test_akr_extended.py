"""CPU integration of the final staged suite. Synthetic results are test fixtures."""
from pathlib import Path
import csv
import json
import numpy as np
import pytest
from akr_closing.core import Journal,atomic_json,digest,read_manifest
from akr_closing.pipeline import Study


class ToyBackend:
    """Deterministic fixture; never a scientific model or reported benchmark."""
    def __init__(self,*args,**kw):
        self.provenance={'backend':'CPU unit fixture','feature_layer':2,'layers':[1,2,3]}
        self.forward_count=0
    def feature(self,q,prompt):
        assert not hasattr(q,'target') and not hasattr(q,'label')
        self.forward_count+=1
        return np.array([1.,0.]) if '_a_' in q.event_id else np.array([0.,1.])
    def support_gradients(self,q,target,prompt):
        assert q.event_id.startswith('train_')
        self.forward_count+=1
        direction=np.array([1.,-1.]) if target=='a' else np.array([-1.,1.])
        return {s:direction for s in ['audio','text','full_prefill']}
    def predict(self,q,prompt,labels,**kwargs):
        assert not hasattr(q,'target') and not hasattr(q,'label')
        self.forward_count+=1
        v=kwargs.get('direction');alpha=kwargs.get('alpha',0)
        pred='a' if v is None or alpha==0 or not np.any(v) or v[0]>v[1] else 'b'
        return {'prediction':pred,'raw_prediction':pred,'generated_tokens':1,'applied_relative_norms':{}}
    def candidates(self,q,prompt,labels,**kwargs):
        return {'prediction':'a','mean_prediction':'a','scores':[]}


@pytest.fixture
def study_data(tmp_path):
    import soundfile as sf
    rows=[];data=tmp_path/'data';data.mkdir()
    for split in ['train','valid','test']:
        for cl in ['a','b']:
            for i in range(10):
                eid=f'{split}_{cl}_{i}';p=data/f'{eid}.wav'
                sf.write(p,np.ones(1600)*(len(rows)+1)/100.,16000,subtype='FLOAT')
                rows.append(dict(event_id=eid,audio_path=str(p),label=cl,split=split,recording_id=eid))
    with (data/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    cfg={'router_implementation':'reference','datasets':[{'name':'toy','manifest':'data/manifest.csv',
         'labels':['a','b'],'group_key':None,'label_modes':['semantic'],'conditions':['full'],
         'support_k':[2],'icl_k':[2],'repair_k':[2]}],
         'seeds':[4],'models':[{'tag':'toy','id':'toy','revision':'fixture'}],
         'layer_fractions':[.25,.5,.75],'feature_fraction':.5,'max_context_tokens':100,
         'ridge_alpha':1.,'ranks':[1],'relative_alphas':[.01],'max_invalid_rate':.01,
         'prompts':['canonical'],'gradient_nulls':True,'probe_to_lm':True,'run_test_after_gate':False}
    return tmp_path,cfg


def test_E3_is_selection_only_and_reports_all_probe_routes(study_data):
    root,cfg=study_data;s=Study(root,cfg,ToyBackend,run_id='E3');s.run('E3')
    paths=[str(p) for p in s.out.rglob('METRICS.json')]
    assert any('ridge_direct' in p for p in paths) and any('probe_to_text' in p for p in paths)
    assert any('probe_to_lm' in p for p in paths) and any('continuous_pooled' in p for p in paths)
    assert not any('confirmation' in p or 'test_locked' in p for p in paths)
    assert list(s.out.rglob('ROUTER_STRATA.json'))


def test_E4_evaluates_null_target_predictors_on_complete_queries(study_data):
    root,cfg=study_data;s=Study(root,cfg,ToyBackend,run_id='E4');s.run('E4')
    files=list(s.out.rglob('REAL_MINUS_CONTROL.json'));assert len(files)==1
    comparisons=json.loads(files[0].read_text())
    assert {'silence_same_label','wrong_audio_same_label','support_label_shuffle','class_residual_targets'} <= set(comparisons)
    stats=json.loads(next(s.out.rglob('label_controlled_nulls.json')).read_text())
    assert stats['real']['n_support']==4
    assert 'residual_cosine_median' in stats['wrong_audio_same_label']
    assert not list(s.out.rglob('test_locked_reevaluation'))


def test_report_excludes_partial_job_even_if_stale_metrics_exist(tmp_path):
    from akr_closing.report import report_run
    j=Journal(tmp_path/'partial',{},['a','b']);j.add('a',{'prediction':'A'})
    metric={'n':2,'accuracy':.5,'macro_f1':.5,'invalid_rate':0}
    atomic_json(tmp_path/'partial'/'METRICS.json',metric)
    atomic_json(tmp_path/'complete'/'METRICS.json',metric)
    result=report_run(tmp_path)
    assert result['complete_result_cells']==1 and result['incomplete_jobs']==1
    assert not result['all_requested_tasks_complete']
    text=(tmp_path/'complete_metrics.tex').read_text()
    assert 'partial' not in text and '50.00 & 50.00 & 0.00 \\\\' in text


def test_reference_router_and_ridge_are_not_class_dictionary():
    from akr_closing.repair import GradientRouter
    x=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    # Continuous targets within a class may differ: no hard class lookup is possible here.
    g=np.c_[x[:,0]+2*x[:,1],3*x[:,0]-x[:,1]]
    model=GradientRouter.fit(x,g,rank=2,alpha=.01)
    preds=model.predict(np.array([[.25,.25],[.75,.75]]))
    assert not np.allclose(preds[0],preds[1])


def test_repository_router_has_no_silent_fallback(monkeypatch):
    from akr_closing.repair import RepositoryGradientRouter
    import builtins
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name=='animal_omni.conditional_kv':raise ImportError('not installed')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ImportError):RepositoryGradientRouter.fit(np.eye(2),np.eye(2),rank=1,alpha=1.)


def test_nonofficial_missing_group_is_not_fabricated_from_event_id(tmp_path):
    p=tmp_path/'m.csv';p.write_text('event_id,audio_path,label\na,a.wav,A\nb,b.wav,B\n')
    rows=read_manifest(p)
    from akr_closing.core import make_split
    with pytest.raises(ValueError,match='recording-group'):make_split(rows,['A','B'],seed=0)


def test_lora_teacher_mask_is_label_only():
    from akr_closing.lora import _teacher_inputs
    import torch
    class Tok:
        eos_token_id=9
        def __call__(self,*a,**kw):return {'input_ids':[7,8]}
    class Backend:
        processor=type('P',(),{'tokenizer':Tok()})()
        def _prepare(self,*a):return {'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
    Backend.torch=torch
    data,audit=_teacher_inputs(Backend(),None,'prompt','label',True)
    assert data['labels'].tolist()==[[-100,-100,-100,7,8,9]]
    assert audit['target_token_ids']==[7,8,9]


def test_hook_never_modifies_prompt_when_mask_is_audio():
    import torch
    from akr_closing.backend import ProjectionIntervention
    layer=torch.nn.Linear(2,2,bias=False);x=torch.randn(1,5,2);baseline=layer(x)
    mask=torch.tensor([[False,True,False,True,False]])
    with ProjectionIntervention({(1,'k'):layer},{(1,'k'):np.ones(2)},mask,.03) as h:
        got=layer(x)
        torch.testing.assert_close(got[~mask],baseline[~mask],atol=0,rtol=0)
        assert h.norms["(1, 'k')"]['active_tokens']==2
        assert h.norms["(1, 'k')"]['delta_frobenius']>0


def test_journal_checks_result_checksum(tmp_path):
    j=Journal(tmp_path,{},['x']);j.add('x',{'prediction':'A'})
    p=next((tmp_path/'items').glob('*.json'));v=json.loads(p.read_text());v['result']['prediction']='B';p.write_text(json.dumps(v))
    with pytest.raises(RuntimeError,match='checksum'):j.results()


def test_support_masks_never_include_answer_tokens():
    import torch
    from akr_closing.backend import OmniBackend
    b=object.__new__(OmniBackend);b.torch=torch;b.audio_token_id=99
    x={'input_ids':torch.tensor([[2,99,99,3,7,8]])}
    assert b._mask(x,'audio',4).tolist()==[[False,True,True,False,False,False]]
    assert b._mask(x,'text',4).tolist()==[[True,False,False,True,False,False]]
    assert b._mask(x,'full_prefill',4).tolist()==[[True,True,True,True,False,False]]


def test_zero_lora_steps_or_single_class_shuffle_are_rejected():
    from akr_closing.lora import microbatch_schedule
    from akr_closing.protocol import shuffle_support_labels
    with pytest.raises(ValueError):microbatch_schedule([],steps=1,accumulation=1,seed=0)
    with pytest.raises(ValueError):shuffle_support_labels(np.array(['A']),seed=0)


def test_subprocess_failures_are_not_reported_as_complete(tmp_path):
    from akr_closing.process import run_checked
    import sys
    with pytest.raises(RuntimeError,match='exited 3'):
        run_checked([sys.executable,'-c','raise SystemExit(3)'],cwd=tmp_path,log_path=tmp_path/'log',stop_requested=lambda:False)


def test_owned_subprocess_stop_preserves_finished_file(tmp_path):
    from akr_closing.process import run_checked
    from akr_closing.pipeline import SafeStop
    import sys
    code="from pathlib import Path; import time; Path('finished').write_text('done'); time.sleep(10)"
    with pytest.raises(SafeStop):
        run_checked([sys.executable,'-c',code],cwd=tmp_path,log_path=tmp_path/'log',stop_requested=lambda:(tmp_path/'finished').exists())
    assert (tmp_path/'finished').read_text()=='done'


def test_fingerprint_is_independent_of_dictionary_order():
    from akr_closing.core import digest
    assert digest({'rank':4,'alpha':.01})==digest({'alpha':.01,'rank':4})


def test_nan_alpha_is_rejected_before_hook_registration():
    from akr_closing.backend import ProjectionIntervention
    with pytest.raises(ValueError):ProjectionIntervention({}, {},None,float('nan'))


def test_report_only_works_without_models_or_manifest_config(tmp_path):
    from akr_closing.cli import main
    assert main(['--root',str(tmp_path),'--run-id','empty','--report-only'])==0
    result=json.loads((tmp_path/'results/akr_closing/empty/FINAL_STATUS.json').read_text())
    assert result['complete_result_cells']==0 and not result['all_requested_tasks_complete']


def test_study_report_handles_complete_lora_and_ignores_partial(study_data):
    root,cfg=study_data;s=Study(root,cfg,ToyBackend,run_id='report')
    metric={'n':2,'accuracy':.5,'macro_f1':.5,'invalid_rate':0}
    atomic_json(s.out/'E8/completed/METRICS.json',metric)
    j=Journal(s.out/'E8/partial',{},['x','y']);j.add('x',{'prediction':'a'})
    atomic_json(j.root/'METRICS.json',metric)
    s.report()
    summary=json.loads((s.out/'SUMMARY.json').read_text())
    assert len(summary['completed_summaries'])==1
    assert summary['completed_summaries'][0]['job']=='E8/completed'


def test_execution_run_id_cannot_escape_results_directory():
    from akr_closing.cli import validate_run_id
    for bad in ['../escape','/tmp/escape','','a/b','a b']:
        with pytest.raises(ValueError):validate_run_id(bad)
    assert validate_run_id('akr_closing_v1')=='akr_closing_v1'


def test_nonfinite_lora_hyperparameters_rejected():
    from akr_closing.lora import validate_lora_budget
    with pytest.raises(ValueError):validate_lora_budget(lr=float('nan'),rank=8,steps=128,accumulation=8,eval_every=32)
    with pytest.raises(ValueError):validate_lora_budget(lr=2e-4,rank=0,steps=128,accumulation=8,eval_every=32)


def test_preflight_detects_duplicate_source_bytes_without_declared_hash(study_data):
    root,cfg=study_data
    rows=list(csv.DictReader((root/'data/manifest.csv').open()))
    train=next(r for r in rows if r['split']=='train')
    test=next(r for r in rows if r['split']=='test')
    Path(test['audio_path']).write_bytes(Path(train['audio_path']).read_bytes())
    s=Study(root,cfg,ToyBackend,run_id='dup')
    with pytest.raises(ValueError,match='waveform.*across'):
        s.preflight()


def test_revision_is_declared_for_legacy_resume():
    import yaml
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/akr_closing.yaml').read_text())
    assert len(cfg['E1']['model_revision'])==40
