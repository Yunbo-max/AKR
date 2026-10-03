"""No-old-RUN/LOCK regression tests; synthetic CPU execution is not a GPU result."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from akr_closing.repair import GradientRouter
from akr_final.runner import FinalRunner
from akr_final.standalone import build_plan
from test_akr_final_three import FakeBackend


def fixture_config(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'data').mkdir()
    rows = []
    for i in range(80):
        path = root / 'data' / f'{i}.wav'
        wave = np.sin(np.arange(1600) * (i + 1) / 300.) * .1
        sf.write(path, wave, 16000, subtype='FLOAT')
        rows.append(dict(event_id=str(i), audio_path=str(path),
                         label='A' if i % 2 == 0 else 'B', recording_id=f'g{i//2}'))
    with (root / 'data/marm.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    cfg = dict(dataset='marmaudio', manifest='data/marm.csv', labels=['A','B'],
               group_key='recording_id', seeds=[11], k_per_class=4,
               conditions=['full','lp:1000'], ranks=[1,2], folds=2,
               fixed_settings=dict(rank=2, alpha=.01, ridge_alpha=1.),
               model_7b=dict(id='Qwen/Qwen2.5-Omni-7B', revision='rev7'),
               model_3b=dict(id='Qwen/Qwen2.5-Omni-3B', revision='rev3'),
               layer_fractions=[.25,.5], feature_fraction=.5, max_context_tokens=4096)
    path=root/'standalone.yaml';path.write_text(yaml.safe_dump(cfg))
    return root,path,cfg


def make_runner(root, path, name='fresh', **kwargs):
    plan, opts = build_plan(root, path)
    return FinalRunner(root, root/'NO_OLD_RUN', name, opts,
        backend_factory=FakeBackend, router_fit=GradientRouter.fit,
        standalone_plan=plan, **kwargs)


def test_no_old_metadata_required_and_plan_repeatable(tmp_path):
    root,path,cfg=fixture_config(tmp_path)
    a,opts=build_plan(root,path); b,_=build_plan(root,path)
    assert a==b and a['mode']=='standalone'
    assert a['selection_uses_outcomes'] is False
    assert len(a['episodes'])==2
    assert a['episodes'][0]['support_ids']==a['episodes'][1]['support_ids']
    lookup={r['event_id']:r for r in a['rows']}
    for e in a['episodes']:
        assert len(e['support_ids'])==8 and len(e['query_ids'])>1
        assert not {lookup[i]['recording_id'] for i in e['support_ids']} & {lookup[i]['recording_id'] for i in e['query_ids']}
    assert not list(root.rglob('RUN.json')) and not list(root.rglob('LOCK.json'))


def test_all_tasks_from_empty_results_and_cache_only_resume(tmp_path):
    root,path,_=fixture_config(tmp_path)
    FakeBackend.loads=0;FakeBackend.grad_calls=[];FakeBackend.predict_calls=0
    r=make_runner(root,path)
    assert r.run(['F1','F2','F3'],execute=True)['complete']
    allowed={i for e in r.plan['episodes'] for i in e['support_ids']}
    assert set(FakeBackend.grad_calls)<=allowed
    calls=(FakeBackend.loads,FakeBackend.predict_calls,len(FakeBackend.grad_calls))
    r.close()
    again=make_runner(root,path,cache_only=True)
    assert again.run(['F1','F2','F3'],execute=True)['complete']
    assert calls==(FakeBackend.loads,FakeBackend.predict_calls,len(FakeBackend.grad_calls))
    assert not list(root.rglob('RUN.json')) and not list(root.rglob('LOCK.json'))
    assert (r.out/'STANDALONE_PROTOCOL.json').is_file()
    result=json.loads(next((r.out/'F3').rglob('METRICS.json')).read_text())
    assert 'standalone' in result['protocol']
    assert 'source_7b_lock_sha256' not in json.loads(next((r.out/'F3').rglob('COMPLETE.json')).read_text())


def test_preflight_only_never_loads_model(tmp_path):
    root,path,_=fixture_config(tmp_path);FakeBackend.loads=0
    r=make_runner(root,path)
    assert not r.run(['F1','F3'],execute=False)['complete']
    assert FakeBackend.loads==0


def test_changed_settings_cannot_reuse_run_id(tmp_path):
    root,path,cfg=fixture_config(tmp_path);make_runner(root,path)
    cfg['fixed_settings']['alpha']=.03;path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError,match='changed'):make_runner(root,path)


def test_no_cache_means_recompute_or_explicit_cache_only_failure(tmp_path):
    root,path,_=fixture_config(tmp_path);FakeBackend.loads=0
    r=make_runner(root,path,cache_only=True)
    with pytest.raises(RuntimeError,match='blocked'):r.run(['F1'],execute=True)
    assert FakeBackend.loads==0
    r=make_runner(root,path)
    assert r.run(['F1'],execute=True)['complete']


@pytest.mark.parametrize('change', ['duplicate_seed','bad_alpha','bad_model','short_support','bad_group','bad_conditions'])
def test_invalid_config_is_not_silently_repaired(tmp_path,change):
    root,path,cfg=fixture_config(tmp_path)
    if change=='duplicate_seed':cfg['seeds']=[11,11]
    elif change=='bad_alpha':cfg['fixed_settings']['alpha']=-1
    elif change=='bad_model':cfg['model_3b']['id']='GLM/not_supported'
    elif change=='short_support':cfg['k_per_class']=100
    elif change=='bad_group':cfg['group_key']=None
    elif change=='bad_conditions':cfg['conditions']=['full']
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises((ValueError,KeyError)):build_plan(root,path)


def test_query_gradients_still_rejected(tmp_path):
    root,path,_=fixture_config(tmp_path);r=make_runner(root,path);r.preflight();e=r.plan['episodes'][0]
    with pytest.raises(ValueError,match='support'):
        r._array(e['query_ids'][0],e['condition'],'qwen7b','gradient',e)


def test_missing_audio_reports_real_input_not_old_json(tmp_path):
    root,path,_=fixture_config(tmp_path);r=make_runner(root,path)
    Path(r.lookup[r.plan['episodes'][0]['support_ids'][0]]['audio_path']).unlink()
    with pytest.raises(FileNotFoundError,match='waveform'):r.run(['F1'],execute=False)


def test_cross_partition_duplicate_audio_rejected(tmp_path):
    root,path,_=fixture_config(tmp_path);r=make_runner(root,path);e=r.plan['episodes'][0]
    Path(r.lookup[e['query_ids'][0]]['audio_path']).write_bytes(Path(r.lookup[e['support_ids'][0]]['audio_path']).read_bytes())
    with pytest.raises(ValueError,match='waveform'):r.preflight()
