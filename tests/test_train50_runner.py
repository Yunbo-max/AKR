from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import yaml

from akr_final.train50_cases import case_plan
from akr_final.train50_runner import run_case, report_case, asset_identity, main

ROOT=Path(__file__).resolve().parents[1]


class ToyBackend:
    loads=0
    predictions=0
    gradients=[]
    def __init__(self,model_id,revision,**kwargs):
        type(self).loads+=1;self.forward_count=0
        self.provenance=dict(model_id=model_id,revision=revision,fixture=True)
    def feature(self,q,prompt):
        self.forward_count+=1
        x,sr=sf.read(q.audio_path)
        return np.array([x.mean(),x.std(),x.max(),x.min(),*x[:4]],np.float32)
    def support_gradients(self,q,target,prompt):
        type(self).gradients.append(q.event_id)
        x=self.feature(q,prompt)
        return {'audio':np.r_[x,-x]*(1 if target=='a' else -1)}
    def predict(self,q,prompt,labels,*,direction=None,alpha=0,support=(),**kw):
        type(self).predictions+=1;self.forward_count+=1
        prediction=labels[0] if direction is None or alpha==0 or not np.any(direction) else labels[1]
        return dict(prediction=prediction,raw_prediction=prediction,generated_tokens=1,input_tokens=10,
                    applied_relative_norms={},effective_relative_norms={},applied_norms={})


def fixture(tmp_path,case='c01'):
    cfg=yaml.safe_load((ROOT/'configs/akr_train50.yaml').read_text())
    cfg['settings']['bootstrap_samples']=20
    rows=[]
    for c in ['a','b']:
        for i in range(30):
            event=f'{c}{i}';p=tmp_path/(event+'.wav')
            sf.write(p,np.random.default_rng(i+(0 if c=='a' else 100)).normal(0,.05,256),16000,subtype='FLOAT')
            rows.append(dict(event_id=event,audio_path=str(p),label=c,split='train',source_filename=event))
        for i in range(2):
            event=f'q{c}{i}';p=tmp_path/(event+'.wav')
            sf.write(p,np.random.default_rng(i+(200 if c=='a' else 300)).normal(0,.05,256),16000,subtype='FLOAT')
            rows.append(dict(event_id=event,audio_path=str(p),label=c,split='test',source_filename=event))
    manifest=tmp_path/'data.csv'
    with manifest.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cfg['datasets']['dogs'].update(manifest=str(manifest),expected_test_count=4)
    job=case_plan(case,'dogs','qwen3b','full',17)
    ToyBackend.loads=0;ToyBackend.predictions=0;ToyBackend.gradients=[]
    return cfg,job,tmp_path/'run'


def test_end_to_end_complete_support_only_gradients_and_cache_only_resume(tmp_path):
    cfg,job,out=fixture(tmp_path)
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)
    assert result['complete'] and result['expected_queries']==4
    reg=json.loads((out/'REGISTRATION.json').read_text())
    assert set(ToyBackend.gradients)==set(reg['splits']['50']['support_ids'])
    assert not set(ToyBackend.gradients)&set(reg['query_ids'])
    assert set(result['methods'])=={'native','akr_regression_original','akr_hard_centered','akr_soft_cap'}
    assert result['methods']['akr_soft_cap']['cost_components']['query_feature_records']==4
    assert result['costs']['support_backward_records']==50
    before=(ToyBackend.loads,ToyBackend.predictions)
    again=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend,cache_only=True)
    assert again['complete'] and (ToyBackend.loads,ToyBackend.predictions)==before


def test_preflight_failure_clears_old_success(tmp_path):
    cfg,job,out=fixture(tmp_path,'c00')
    run_case(ROOT,out,cfg,job,execute=False,backend_factory=ToyBackend)
    (out/'STATUS.json').write_text('{"complete":true}')
    (tmp_path/'a0.wav').unlink()
    with pytest.raises((FileNotFoundError,RuntimeError)):
        run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)
    status=json.loads((out/'STATUS.json').read_text())
    assert not status['complete'] and status['status']=='error'
    assert ToyBackend.loads==0


def test_aggregate_rejects_missing_query_and_tampered_self_report(tmp_path):
    cfg,job,out=fixture(tmp_path)
    run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)
    item=next((out/'trials/native/items').glob('*.json'));item.unlink()
    report=json.loads((out/'REPORT.json').read_text());report['expected_queries']=3
    (out/'REPORT.json').write_text(json.dumps(report))
    incomplete=report_case(out)
    assert not incomplete['complete'] and 'native' in incomplete['incomplete_methods']
    assert json.loads((out/'STATUS.json').read_text())['complete'] is False


def test_changed_case_and_config_cannot_resume_old_results(tmp_path):
    cfg,job,out=fixture(tmp_path)
    run_case(ROOT,out,cfg,job,execute=False,backend_factory=ToyBackend)
    modified=deepcopy(job);modified['methods'][-1]['temperature']=4.
    with pytest.raises(RuntimeError,match='changed'):
        run_case(ROOT,out,cfg,modified,execute=True,backend_factory=ToyBackend)
    assert not json.loads((out/'STATUS.json').read_text())['complete']


def test_noop_and_ridge_rendering_are_checked_on_saved_records(tmp_path):
    cfg,job,out=fixture(tmp_path,'c14')
    assert run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)['complete']
    cfg2,job2,out2=fixture(tmp_path,'c02')
    out2=out2.with_name('second_run')
    r=run_case(ROOT,out2,cfg2,job2,execute=True,backend_factory=ToyBackend)
    assert r['complete']
    assert r['methods']['ridge_direct']['accuracy']==r['methods']['probe_to_text']['accuracy']
    assert r['methods']['ridge_direct']['cost_components']['fit_and_route_wall_seconds']>0


def test_specialist_event_identity_is_preserved_without_duplicate_keyword(tmp_path,monkeypatch):
    from akr_final import train50_naturelm
    class Specialist:
        def __init__(self,assets):pass
        def predict(self,q,prompt,labels):
            return dict(event_id=q.event_id,prediction=labels[0],raw_prediction=labels[0])
    monkeypatch.setattr(train50_naturelm,'NatureLMBackend',Specialist)
    cfg,job,out=fixture(tmp_path,'c15')
    assets={'naturelm':{k:str(tmp_path) for k in ('naturelm_code','naturelm_checkpoint','naturelm_llama','naturelm_bert_config','naturelm_config')}}
    assets['naturelm'].update({k:'a'*40 for k in ('naturelm_source_revision','naturelm_checkpoint_revision','naturelm_llama_revision')})
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend,assets=assets)
    assert result['complete'] and result['methods']['naturelm_native']['n']==4


def test_synthetic_backend_results_are_never_labeled_scientific(tmp_path):
    cfg,job,out=fixture(tmp_path)
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)
    assert result['new_scientific_results'] is False


def test_missing_beats_assets_fail_check_before_loading_generalist(tmp_path):
    cfg,job,out=fixture(tmp_path,'c03')
    with pytest.raises((ValueError,RuntimeError),match='BEATs|beats|opensmile'):
        run_case(ROOT,out,cfg,job,execute=False,backend_factory=ToyBackend)
    assert ToyBackend.loads==0
    assert json.loads((out/'STATUS.json').read_text())['status']=='error'


def test_sentencepiece_asset_change_invalidates_identity(tmp_path):
    p=tmp_path/'tokenizer.model';p.write_bytes(b'first tokenizer')
    first=asset_identity(str(tmp_path));p.write_bytes(b'changed tokenizer')
    assert asset_identity(str(tmp_path))!=first


def test_cli_missing_config_clears_previous_completion(tmp_path):
    out=tmp_path/'cli_run'/'c00'/'dogs__qwen7b__full__s20261107'
    out.mkdir(parents=True)
    (out/'STATUS.json').write_text('{"complete":true}')
    with pytest.raises(FileNotFoundError):
        main('c00',['check','--run-id','cli_run','--output-root',str(tmp_path),
                    '--config',str(tmp_path/'missing_config.yaml')])
    assert json.loads((out/'STATUS.json').read_text())['complete'] is False


def test_distribution_text_baseline_preserves_the_soft_router_information(tmp_path):
    cfg,job,out=fixture(tmp_path,'c02')
    assert 'probe_distribution_to_lm' in {s['name'] for s in job['methods']}
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)
    payload=json.loads((out/'trials/probe_distribution_to_lm/METHOD.json').read_text())['identity']['payload']
    with np.load(out/'maps/akr_soft_cap.npz') as fields:
        np.testing.assert_allclose(payload['class_probabilities'],fields['probabilities'],rtol=1e-12,atol=1e-12)
    counts=json.loads((out/'REGISTRATION.json').read_text())['splits']['50']['support_class_counts']
    np.testing.assert_allclose(payload['support_priors'],[counts[c]/50 for c in ('a','b')])
    assert result['methods']['probe_distribution_to_lm']['n']==4
    assert 'correct_probe_spoiled' in result['methods']['probe_distribution_to_lm']


def low_sr_rewrite(path,rate=600,samples=600):
    """Replace one registered waveform with a band that cannot host lp:1000."""
    seed=int(hashlib.md5(str(path).encode()).hexdigest()[:8],16)
    sf.write(path,np.random.default_rng(seed).normal(0,.05,samples).astype('float32'),
             rate,subtype='FLOAT')


def test_lp_native_bandwidth_is_annotated_and_scored_as_is(tmp_path):
    base=tmp_path/'lp';base.mkdir()
    cfg,job,out=fixture(base,'c01')
    low_sr_rewrite(base/'qa0.wav')
    job=case_plan('c01','dogs','qwen3b','lp:1000',17)
    ToyBackend.loads=0;ToyBackend.predictions=0;ToyBackend.gradients=[]
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)
    assert result['complete'] and result['expected_queries']==4
    assert result['scored_queries']==4 and result['native_bw_queries']==1
    assert 'excluded_queries' not in result
    entry=dict(event_id='qa0',kind='native_bw_within_passband',sample_rate=600.,cutoff=1000.,
               condition='lp:1000',role='query',used_as_is=True,
               reason='native-bandwidth-within-lowpass-passband',
               annotation='used as-is; native bandwidth <= cutoff')
    ledger=json.loads((out/'INPUT_DOMAIN_EXCLUSIONS.json').read_text())
    assert ledger['version']==2 and ledger['entries']==[entry]
    annotations=result['native_bandwidth_annotations']
    assert annotations['kind']=='native_bw_within_passband'
    assert annotations['query_count']==1 and annotations['query_ids']==['qa0']
    assert annotations['registered_low_sr_events']==1
    assert ledger['dataset_scan']['by_condition']['lp:1000']==dict(split_counts={'test':1},total=1)
    for name in result['methods']:
        summary=result['methods'][name]
        assert summary['n']==4 and summary['scored_queries']==4
        assert summary['native_bw_queries']==1 and summary['native_bw_query_ids']==['qa0']
        items=[json.loads(p.read_text())['result']
               for p in (out/'trials'/name/'items').glob('*.json')]
        annotated=[r for r in items if r['event_id']=='qa0']
        assert len(annotated)==1 and annotated[0]['prediction'] in ('a','b')
        assert annotated[0]['native_bandwidth']==entry
        assert annotated[0]['new_model_calls']>0
        assert all('native_bandwidth' not in r for r in items if r['event_id']!='qa0')
    # The native-bandwidth query is prepared as-is and scored by every method.
    assert ToyBackend.predictions==4*len(job['methods'])
    before=ToyBackend.predictions
    again=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend,cache_only=True)
    assert again['complete'] and ToyBackend.predictions==before


def test_lp_native_bandwidth_support_participates_in_fit(tmp_path):
    base=tmp_path/'support';base.mkdir()
    cfg,job,out=fixture(base,'c01')
    for path in sorted(base.glob('[ab]*.wav')):  # every official-train candidate
        low_sr_rewrite(path)
    job=case_plan('c01','dogs','qwen3b','lp:1000',17)
    ToyBackend.loads=0;ToyBackend.predictions=0;ToyBackend.gradients=[]
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)
    assert result['complete'] and result['native_bw_queries']==0
    reg=json.loads((out/'REGISTRATION.json').read_text())
    support_ids=reg['splits']['50']['support_ids']
    ledger=json.loads((out/'INPUT_DOMAIN_EXCLUSIONS.json').read_text())
    support=[e for e in ledger['entries'] if e['role']=='support']
    assert {e['event_id'] for e in support}==set(support_ids)
    assert all(e['sample_rate']==600. and e['condition']=='lp:1000' for e in support)
    assert all(e['kind']=='native_bw_within_passband' and e['used_as_is'] for e in support)
    assert all(e['reason']=='native-bandwidth-within-lowpass-passband' for e in support)
    # All 50 registered support events, including the low-rate ones, build the fit.
    assert set(ToyBackend.gradients)==set(support_ids)
    for name in result['methods']:
        spec=next(s for s in job['methods'] if s['name']==name)
        payload=json.loads((out/'trials'/name/'METHOD.json').read_text())['identity']['payload']
        expected=support_ids if spec['kind'] not in ('native','naturelm') else []
        assert payload['native_bw_support_ids']==expected


def test_full_condition_has_no_native_bandwidth_annotations(tmp_path):
    base=tmp_path/'full';base.mkdir()
    cfg,job,out=fixture(base,'c01')
    low_sr_rewrite(base/'qa0.wav')
    job=case_plan('c01','dogs','qwen3b','full',17)
    ToyBackend.loads=0;ToyBackend.predictions=0;ToyBackend.gradients=[]
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)
    assert result['complete'] and result['native_bw_queries']==0
    assert result['scored_queries']==4 and result['expected_queries']==4
    assert result['native_bandwidth_annotations']['entries']==[]
    assert all(s['n']==4 and s['native_bw_queries']==0 for s in result['methods'].values())
    ledger=json.loads((out/'INPUT_DOMAIN_EXCLUSIONS.json').read_text())
    assert ledger['entries']==[]


def test_lp_report_counts_13_native_bandwidth_queries_and_2_support(tmp_path):
    """Mirror the real Watkins shape: 13 low-rate test queries, 2 low-rate support events."""
    cfg=yaml.safe_load((ROOT/'configs/akr_train50.yaml').read_text())
    cfg['settings']['bootstrap_samples']=20
    rows=[]
    for c in ['a','b']:
        for i in range(30):
            event=f'{c}{i}';p=tmp_path/(event+'.wav')
            sf.write(p,np.random.default_rng(i+(0 if c=='a' else 100)).normal(0,.05,256),16000,subtype='FLOAT')
            rows.append(dict(event_id=event,audio_path=str(p),label=c,split='train',source_filename=event))
    for i in range(10):
        for c in ['a','b']:
            event=f'q{c}{i}';p=tmp_path/(event+'.wav')
            sf.write(p,np.random.default_rng(i+(200 if c=='a' else 300)).normal(0,.05,256),16000,subtype='FLOAT')
            rows.append(dict(event_id=event,audio_path=str(p),label=c,split='test',source_filename=event))
    native_bw_ids=[f'q{c}{i}' for i in range(10) for c in ['a','b']][:13]
    for event in native_bw_ids:low_sr_rewrite(tmp_path/(event+'.wav'))
    manifest=tmp_path/'data.csv'
    with manifest.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cfg['datasets']['dogs'].update(manifest=str(manifest),expected_test_count=20)
    job=case_plan('c01','dogs','qwen3b','lp:1000',17)
    # Choose two events that the deterministic registration actually uses as support.
    from akr_final.train50_protocol import prepare_registration
    reg=prepare_registration(ROOT,cfg,'dogs',17,[50])
    support_ids=reg['splits']['50']['support_ids'];low_support=support_ids[:2]
    for eid in low_support:low_sr_rewrite(tmp_path/(eid+'.wav'))
    out=tmp_path/'run'
    ToyBackend.loads=0;ToyBackend.predictions=0;ToyBackend.gradients=[]
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=ToyBackend)
    assert result['complete'] and result['expected_queries']==20
    assert result['scored_queries']==20 and result['native_bw_queries']==13
    assert sorted(result['native_bw_query_ids'])==sorted(native_bw_ids)
    annotations=result['native_bandwidth_annotations']
    assert annotations['query_count']==13
    assert annotations['registered_low_sr_events']==15
    assert annotations['dataset_scan']['by_condition']['lp:1000']==dict(
        split_counts={'test':13,'train':2},total=15)
    roles={}
    for entry in annotations['entries']:roles.setdefault(entry['role'],[]).append(entry['event_id'])
    assert sorted(roles['query'])==sorted(native_bw_ids)
    assert sorted(roles['support'])==sorted(low_support)
    for summary in result['methods'].values():
        assert summary['n']==20 and summary['scored_queries']==20
        assert summary['native_bw_queries']==13


class BatchToyBackend(ToyBackend):
    batch_predict_calls=0
    batch_feature_calls=0
    batch_rows=0
    def feature_many(self,items):
        type(self).batch_feature_calls+=1
        return [self.feature(item['q'],item['prompt']) for item in items]
    def predict_many(self,items,labels):
        type(self).batch_predict_calls+=1
        rows=[]
        for item in items:
            type(self).batch_rows+=1;self.forward_count+=1
            direction=item.get('direction')
            prediction=labels[0] if direction is None or item.get('alpha',0)==0 or not np.any(direction) else labels[1]
            rows.append(dict(prediction=prediction,raw_prediction=prediction,generated_tokens=1,input_tokens=10,
                             applied_relative_norms={},effective_relative_norms={},applied_norms={}))
        return rows


class FlakyBatchToyBackend(BatchToyBackend):
    fail_at=None
    def predict_many(self,items,labels):
        type(self).batch_predict_calls+=1
        if type(self).fail_at is not None and type(self).batch_predict_calls>=type(self).fail_at:
            raise RuntimeError('injected batch interruption')
        rows=[]
        for item in items:
            type(self).batch_rows+=1;self.forward_count+=1
            direction=item.get('direction')
            prediction=labels[0] if direction is None or item.get('alpha',0)==0 or not np.any(direction) else labels[1]
            rows.append(dict(prediction=prediction,raw_prediction=prediction,generated_tokens=1,input_tokens=10,
                             applied_relative_norms={},effective_relative_norms={},applied_norms={}))
        return rows


def test_batched_predicts_keep_one_journal_record_per_event(tmp_path):
    cfg,job,out=fixture(tmp_path)
    BatchToyBackend.batch_predict_calls=0;BatchToyBackend.batch_feature_calls=0;BatchToyBackend.batch_rows=0
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=BatchToyBackend,
                    batch_map={'datasets':{'dogs':2}})
    assert result['complete']
    # c01 has native + three gradient methods, four query ids each.
    assert BatchToyBackend.batch_predict_calls==4*2
    assert BatchToyBackend.batch_rows==4*4
    assert BatchToyBackend.batch_feature_calls>0
    for spec in job['methods']:
        items=list((out/'trials'/spec['name']/'items').glob('*.json'))
        assert len(items)==4
        for path in items:
            value=json.loads(path.read_text())['result']
            assert value['event_id']==json.loads(path.read_text())['key']
            if spec['kind']!='ridge':
                assert value['batch_size']==2
    before=(BatchToyBackend.batch_predict_calls,BatchToyBackend.batch_feature_calls,BatchToyBackend.batch_rows)
    again=run_case(ROOT,out,cfg,job,execute=True,backend_factory=BatchToyBackend,
                   batch_map={'datasets':{'dogs':2}},cache_only=True)
    assert again['complete']
    assert (BatchToyBackend.batch_predict_calls,BatchToyBackend.batch_feature_calls,BatchToyBackend.batch_rows)==before


def test_batched_resume_does_not_duplicate_journal_events(tmp_path):
    cfg,job,out=fixture(tmp_path)
    FlakyBatchToyBackend.batch_predict_calls=0;FlakyBatchToyBackend.batch_rows=0
    FlakyBatchToyBackend.fail_at=2
    with pytest.raises(RuntimeError):
        run_case(ROOT,out,cfg,job,execute=True,backend_factory=FlakyBatchToyBackend,
                 batch_map={'datasets':{'dogs':2}})
    native_items=list((out/'trials'/'native'/'items').glob('*.json'))
    assert len(native_items)==2  # first batch committed, second interrupted
    BatchToyBackend.batch_predict_calls=0;BatchToyBackend.batch_rows=0
    BatchToyBackend.batch_feature_calls=0
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=BatchToyBackend,
                    batch_map={'datasets':{'dogs':2}})
    assert result['complete']
    # native finishes the two remaining rows; three gradient methods run four each.
    assert BatchToyBackend.batch_rows==2+3*4
    assert BatchToyBackend.batch_predict_calls==1+3*2
    for spec in job['methods']:
        assert len(list((out/'trials'/spec['name']/'items').glob('*.json')))==4


def test_batched_feature_extraction_for_ridge_paths(tmp_path):
    cfg,job,out=fixture(tmp_path,case='c02')
    BatchToyBackend.batch_predict_calls=0;BatchToyBackend.batch_feature_calls=0;BatchToyBackend.batch_rows=0
    result=run_case(ROOT,out,cfg,job,execute=True,backend_factory=BatchToyBackend,
                    batch_map={'datasets':{'dogs':3}})
    assert result['complete']
    assert BatchToyBackend.batch_feature_calls>0
    for spec in job['methods']:
        assert len(list((out/'trials'/spec['name']/'items').glob('*.json')))==4
