"""Regression contracts for the user-specified E1--E9 protocol (CPU only)."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def mod(name):
    return importlib.import_module('akr_closing.' + name)


def fixture_rows():
    return [dict(event_id=f'{c}{i}', label=c, audio_path='a.wav', recording_id=f'{c}{i}')
            for c in 'ABC' for i in range(3)]


def test_counterbalanced_orders_keep_exact_same_support_set():
    m = mod('orders'); rows = fixture_rows()
    orders = m.counterbalanced_orders(rows, list('ABC'), seed=12)
    expected = {r['event_id'] for r in rows}
    for ids in orders.values():
        assert len(ids) == len(set(ids)) == len(expected)
        assert set(ids) == expected
    byid = {r['event_id']: r for r in rows}
    heads = [byid[orders['first_class_' + c][0]]['label'] for c in 'ABC']
    assert heads == list('ABC')
    assert orders == m.counterbalanced_orders(rows, list('ABC'), seed=12)


def test_order_builder_rejects_missing_class_and_duplicate():
    m=mod('orders'); rows=fixture_rows()
    with pytest.raises(ValueError): m.counterbalanced_orders(rows,list('ABCD'),seed=0)
    with pytest.raises(ValueError): m.counterbalanced_orders(rows+[rows[0]],list('ABC'),seed=0)


def test_legacy_candidate_requires_unique_complete_query_ids():
    m=mod('legacy')
    row = dict(event_id='q1',support_k_per_class='8',support_k_total='16',readout='candidate',target_output='A',
               candidate_scores_json=json.dumps([dict(candidate='A',sequence_logprob=-4.,mean_token_logprob=-2.),
                   dict(candidate='B',sequence_logprob=-3.,mean_token_logprob=-3.)]))
    assert m.candidate_summary([row], ['q1','q2'], ['A','B'], k=8)['status']=='incomplete'
    out=m.candidate_summary([row], ['q1'], ['A','B'], k=8)
    assert out['sequence_sum']['accuracy']==0 and out['mean_token']['accuracy']==1
    with pytest.raises(ValueError): m.candidate_summary([row,row],['q1'],['A','B'],k=8)
    with pytest.raises(ValueError): m.candidate_summary([dict(row,event_id='other')],['q1'],['A','B'],k=8)


def test_invalid_candidate_scores_are_not_scored_as_zero():
    m=mod('legacy'); row=dict(event_id='q',support_k_per_class='8',support_k_total='16',readout='candidate',
      target_output='A',candidate_scores_json='[]')
    with pytest.raises(ValueError): m.candidate_summary([row],['q'],['A','B'],k=8)


def test_exact_legacy_split_rejects_support_query_recording_overlap():
    m=mod('legacy')
    rows=[dict(event_id=f's{i}',label=str(i),recording_id=f'r{i}') for i in range(6)]
    rows += [dict(event_id='q',label='0',recording_id='r0')]
    split={'support_sets':{'1':[f's{i}' for i in range(6)]},'query_events':['q']}
    with pytest.raises(ValueError): m.validate_exact_split(rows,split,[str(i) for i in range(6)],k=1,expected_query=1)


def test_null_audio_matches_duration_and_samples_without_time_stretch(tmp_path):
    m=mod('protocol'); import soundfile as sf
    original=tmp_path/'a.wav'; donor=tmp_path/'b.wav'
    sf.write(original,np.ones(1600)*.1,16000,subtype='FLOAT')
    sf.write(donor,np.arange(400)/400.,16000,subtype='FLOAT')
    silence=m.matched_audio(original,None,tmp_path/'sil.wav')
    wrong=m.matched_audio(original,donor,tmp_path/'wrong.wav')
    x,sr=sf.read(wrong); z,_=sf.read(silence)
    assert sr==16000 and len(x)==len(z)==1600
    assert np.count_nonzero(z)==0 and np.allclose(x[400:],0)
    np.testing.assert_allclose(x[:400],np.arange(400)/400.,atol=1e-7)


def test_label_shuffle_preserves_counts_but_not_association():
    m=mod('protocol'); y=np.array([0,0,1,1,2,2])
    yp=m.shuffle_support_labels(y,seed=13)
    assert sorted(yp)==sorted(y) and np.any(yp!=y)
    np.testing.assert_array_equal(yp,m.shuffle_support_labels(y,seed=13))
    with pytest.raises(ValueError): m.shuffle_support_labels(np.array([0,0]),seed=0)


def test_gradient_comparison_is_label_controlled_not_label_accuracy_only():
    m=mod('protocol'); y=np.array([0,0,1,1]); a=np.array([[1,0],[2,0],[0,1],[0,2.]])
    out=m.compare_gradient_targets(a,a.copy(),y)
    assert out['median_same_target_cosine']==pytest.approx(1)
    assert out['relative_error']==0
    assert out['residual_cosine_median']==pytest.approx(1)
    assert 'not_acoustic_semantics' in out['interpretation']


def test_complete_only_order_summary_detects_first_class_following():
    m=mod('orders')
    records=[dict(event_id=q,order=order,prediction=label,first_label=label,last_label='C',target='A')
             for q in ['q1','q2'] for order,label in [('first_A','A'),('first_B','B')]]
    result=m.summarize_orders(records,['q1','q2'],['first_A','first_B'],list('ABC'),seed=0)
    assert result['complete'] and result['prediction_change_rate']==1.
    assert result['mean_first_agreement']==1.
    partial=m.summarize_orders(records[:-1],['q1','q2'],['first_A','first_B'],list('ABC'),seed=0)
    assert not partial['complete'] and 'metrics' not in partial


def test_literal_probe_to_text_does_not_invoke_lm():
    m=mod('protocol'); values=['A','C','B']
    result=m.probe_to_text(values)
    assert [r['prediction'] for r in result]==values
    assert all(r['model_forward_calls']==0 for r in result)


def test_lora_schedule_counts_optimizer_steps_not_epochs():
    m=mod('lora'); schedule=m.step_schedule(max_steps=128,eval_every=32)
    assert schedule==[32,64,96,128]
    assert m.step_schedule(max_steps=35,eval_every=16)==[16,32,35]
    with pytest.raises(ValueError): m.step_schedule(max_steps=0,eval_every=1)


def test_microbatch_schedule_has_full_accumulation_groups():
    m=mod('lora'); ids=['a','b','c']; schedule=m.microbatch_schedule(ids,steps=3,accumulation=4,seed=1)
    assert len(schedule)==12 and set(schedule)==set(ids)
    assert schedule==m.microbatch_schedule(ids,steps=3,accumulation=4,seed=1)


def test_selection_uses_development_only_and_rejects_test():
    m=mod('lora')
    records=[dict(step=16,lr=.01,stage='selection',accuracy=.5,macro_f1=.4,invalid_rate=0),
             dict(step=32,lr=.01,stage='selection',accuracy=.6,macro_f1=.5,invalid_rate=0)]
    assert m.select_development_checkpoint(records)['step']==32
    with pytest.raises(ValueError): m.select_development_checkpoint([dict(records[0],stage='test')])


def test_zero_direction_at_nonzero_alpha_is_bitwise_noop():
    m=mod('backend'); import torch
    layer=torch.nn.Linear(4,4,bias=False);x=torch.randn(1,5,4);y=layer(x)
    with m.ProjectionIntervention({(0,'k'):layer},{(0,'k'):np.zeros(4)},torch.ones(1,5,dtype=torch.bool),alpha=.1):
        assert torch.equal(layer(x),y)


def test_nonfinite_intervention_rejected():
    m=mod('backend'); import torch
    layer=torch.nn.Linear(2,2,bias=False)
    with pytest.raises(ValueError):
        with m.ProjectionIntervention({(0,'k'):layer},{(0,'k'):np.array([np.nan,0])},torch.ones(1,3,dtype=torch.bool),alpha=.1):
            layer(torch.ones(1,3,2))


def test_journal_rejects_corrupted_or_misidentified_item(tmp_path):
    m=mod('core'); j=m.Journal(tmp_path/'j',{},['a']);j.add('a',{'prediction':'A'})
    p=next((tmp_path/'j'/'items').glob('*.json')); item=json.loads(p.read_text());item['key']='b';p.write_text(json.dumps(item))
    with pytest.raises(RuntimeError): j.results()


def test_factorized_test_requires_real_passed_gate(tmp_path):
    m=mod('protocol');path=tmp_path/'gate.json'
    with pytest.raises(FileNotFoundError): m.require_external_gate(path)
    path.write_text(json.dumps({'gate':{'passed':False}}))
    with pytest.raises(RuntimeError):m.require_external_gate(path)
    path.write_text(json.dumps({'gate':{'passed':True},'registered_protocol':'old'}))
    assert m.require_external_gate(path)['gate']['passed'] is True


def test_e1_support_count_is_not_silently_subsampled():
    m=mod('legacy'); rows=[dict(event_id=f's{c}{i}',label=c,recording_id=f's{c}{i}') for c in 'AB' for i in range(2)]
    rows += [dict(event_id='q',label='A',recording_id='query')]
    split={'support_sets':{'2':['sA0','sB0','sA1']},'query_events':['q']}
    with pytest.raises(ValueError):m.validate_exact_split(rows,split,list('AB'),k=2,expected_query=1)
