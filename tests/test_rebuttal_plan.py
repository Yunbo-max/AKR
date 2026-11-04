import csv
import yaml
import pytest
from test_akr_standalone import fixture_config


def config_fixture(tmp_path):
    from pathlib import Path
    root,_,_=fixture_config(tmp_path)
    cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/akr_rebuttal.yaml').read_text())
    cfg['datasets']={'marmaudio':dict(manifest='data/marm.csv',labels=['A','B'],group_key='recording_id',
       partition='recording_group_confirmation',fit_per_class=4,calibration_per_class=2,guard_per_class=2,
       conditions=['full','lp:1000'])}
    cfg['seeds']=[11,12,13];cfg['settings']['rank']=2;cfg['settings']['ridge_alpha']=1.
    cfg['settings']['gain_steps']=[.5]
    cfg['suites']={k:v for k,v in cfg['suites'].items() if k in ['core','scale']}
    path=root/'rebuttal.yaml';path.write_text(yaml.safe_dump(cfg))
    return root,path,cfg


def test_repeatable_plan_and_role_group_separation(tmp_path):
    from akr_final.rebuttal_plan import build_study
    root,path,cfg=config_fixture(tmp_path);a=build_study(root,path,['core','scale'])
    assert a==build_study(root,path,['core','scale']) and len(a['jobs'])==8
    plan=a['datasets']['marmaudio']['plan'];lookup={r['event_id']:r for r in plan['rows']}
    for e in plan['episodes']:
        roles=[{lookup[i]['recording_id'] for i in e[k]} for k in ['fit_ids','calibration_ids','guard_ids','query_ids']]
        assert all(not roles[i]&roles[j] for i in range(4) for j in range(i+1,4))
        assert len(e['support_ids'])==16 and len(e['fit_ids'])==8


def test_official_test_unchanged(tmp_path):
    from akr_final.rebuttal_plan import build_study
    root,path,cfg=config_fixture(tmp_path);manifest=root/'data/marm.csv'
    with manifest.open() as f:rows=list(csv.DictReader(f))
    for j,r in enumerate(rows):r['split']='train' if j<64 else 'test'
    with manifest.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    cfg['datasets']['marmaudio']['partition']='official_test';path.write_text(yaml.safe_dump(cfg))
    a=build_study(root,path,['core'])
    for e in a['datasets']['marmaudio']['plan']['episodes']:
        assert set(e['query_ids'])=={r['event_id'] for r in rows if r['split']=='test'}


@pytest.mark.parametrize('change',['coverage','revision','eta','seed','condition','task'])
def test_invalid_config_before_model_load(tmp_path,change):
    from akr_final.rebuttal_plan import build_study
    root,path,cfg=config_fixture(tmp_path);tasks=['core']
    if change=='coverage':cfg['datasets']['marmaudio']['fit_per_class']=999
    if change=='revision':cfg['models']['qwen7b']['revision']='main'
    if change=='eta':cfg['settings']['raw_eta']=-1
    if change=='seed':cfg['seeds']=[11,11]
    if change=='condition':cfg['datasets']['marmaudio']['conditions']=['bad']
    if change=='task':tasks=['bad']
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):build_study(root,path,tasks)


def test_official_cross_split_source_removed_only_from_training_pool(tmp_path):
    from akr_final.rebuttal_plan import build_study
    root,path,cfg=config_fixture(tmp_path);manifest=root/'data/marm.csv'
    with manifest.open() as f:rows=list(csv.DictReader(f))
    for j,r in enumerate(rows):r['split']='train' if j<64 else 'test'
    rows[0]['recording_id']=rows[-1]['recording_id']
    with manifest.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    cfg['datasets']['marmaudio']['partition']='official_test';path.write_text(yaml.safe_dump(cfg))
    a=build_study(root,path,['core']);plan=a['datasets']['marmaudio']['plan']
    assert rows[0]['event_id'] in plan['splits']['11']['excluded_train_ids']
    assert all(rows[0]['event_id'] not in e['support_ids'] for e in plan['episodes'])
    assert all(len(e['query_ids'])==16 for e in plan['episodes'])
