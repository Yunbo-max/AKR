import yaml
import pytest
from test_rebuttal_plan import config_fixture


def test_additional_family_and_configured_controls_are_preserved(tmp_path):
    from akr_final.rebuttal_plan import build_study
    from akr_final.rebuttal_runner import RebuttalStudy
    root, path, cfg = config_fixture(tmp_path)
    cfg['models']['glm4voice9b'] = {
        'id': 'zai-org/glm-4-voice-9b',
        'revision': '352e5a64063448d5e46394753fb65bf1dbab975b',
    }
    cfg['suites']['glm_marm'] = dict(dataset='marmaudio', model='glm4voice9b',
                                    seeds='first', controls='first')
    path.write_text(yaml.safe_dump(cfg))
    study = build_study(root, path, ['glm_marm'])
    assert len(study['jobs']) == 2
    assert all(j['controls'] for j in study['jobs'])
    runner = RebuttalStudy(root, path, 'optional', ['glm_marm']).dataset_runner('marmaudio')
    assert runner.models['glm4voice9b']['id'] == 'zai-org/glm-4-voice-9b'


def test_primary_selection_covers_both_sizes_on_every_core_dataset():
    from pathlib import Path
    from akr_final.comprehensive import selected_tasks
    cfg = yaml.safe_load((Path(__file__).parents[1] / 'configs/akr_comprehensive.yaml').read_text())
    tasks = selected_tasks(cfg, ['qwen7b', 'qwen3b'])
    assert {(cfg['suites'][t]['dataset'], cfg['suites'][t]['model']) for t in tasks} == {
        (d, m) for d in cfg['datasets'] for m in ['qwen7b', 'qwen3b']}
    assert {'marmaudio', 'dogs', 'watkins'} <= set(cfg['datasets'])
    assert all(cfg['suites'][t]['controls'] == 'first' for t in tasks)


def test_conditional_models_require_explicit_activation():
    from akr_final.comprehensive import selected_tasks
    cfg = {'suites': {'g': {'model': 'glm4voice9b'}, 'q': {'model': 'qwen7b'}}}
    with pytest.raises(ValueError, match='conditional'):
        selected_tasks(cfg, ['glm4voice9b'])
    assert selected_tasks(cfg, ['glm4voice9b'], enable_conditional=True) == ['g']
    with pytest.raises(ValueError, match='unknown'):
        selected_tasks(cfg, ['imaginary'])


def test_factory_resolves_family_without_instantiating_models():
    from akr_closing.model_factory import model_family
    assert model_family('Qwen/Qwen2.5-Omni-7B') == 'qwen25'
    assert model_family('Qwen/Qwen3-Omni-30B-A3B-Instruct') == 'qwen3'
    assert model_family('zai-org/glm-4-voice-9b') == 'glm'
    with pytest.raises(ValueError, match='unsupported'):
        model_family('unknown/model')


def test_custom_suite_executes_all_methods_and_reports_complete(tmp_path):
    from akr_final.rebuttal_runner import RebuttalStudy
    from test_rebuttal_runner import RebuttalFake
    root, path, cfg = config_fixture(tmp_path)
    cfg['seeds'] = cfg['seeds'][:1]
    cfg['settings']['control_alphas'] = [.01]
    cfg['models']['glm4voice9b'] = {'id': 'zai-org/glm-4-voice-9b',
                                  'revision': '352e5a64063448d5e46394753fb65bf1dbab975b'}
    cfg['suites']['custom_glm'] = dict(dataset='marmaudio', model='glm4voice9b',
                                      seeds='first', controls='first')
    path.write_text(yaml.safe_dump(cfg))
    study = RebuttalStudy(root, path, 'custom', ['custom_glm'], backend_factory=RebuttalFake)
    result = study.run(execute=True)
    assert result['complete'] and result['completed_jobs'] == 2
    assert {r['model'] for r in result['rows']} == {'glm4voice9b'}
    assert 'k_only_cap' in {r['method'] for r in result['rows']}


def test_duplicate_suite_alias_cannot_overwrite_same_episode(tmp_path):
    from akr_final.rebuttal_plan import build_study
    root, path, cfg = config_fixture(tmp_path)
    cfg['suites']['alias'] = dict(cfg['suites']['core'])
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='duplicate model/dataset/episode'):
        build_study(root, path, ['core', 'alias'])


def test_missing_baseline_is_nonzero_in_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from akr_final.comprehensive import _worker
    import akr_final.comprehensive_baselines as baselines
    monkeypatch.setattr(baselines, 'run_baseline_job', lambda *a, **k: {'ready': False, 'complete': False})
    args = SimpleNamespace(worker_kind='baseline', worker_id='b', action='check', profile='80gb')
    assert _worker(args, tmp_path, tmp_path/'config.yaml', {}, {'baseline_jobs': [{'id': 'b'}]}, {}, tmp_path/'out') != 0


def test_failed_preflight_prevents_all_run_workers(tmp_path, monkeypatch):
    import sys
    import akr_final.comprehensive as cli
    config = tmp_path/'config.yaml'
    config.write_text('{}')
    plan = {'models': {'qwen7b': {}}, 'sections': ['baselines'],
            'baseline_jobs': [{'id': 'b', 'kind': 'aves_ridge'}]}
    monkeypatch.setattr(cli, 'make_plan', lambda *a, **k: ({}, plan))
    calls = []
    def bounded(command, *args):
        calls.append(command)
        return 2 if command[command.index('--worker-kind')+1].startswith('baseline') else 0
    monkeypatch.setattr(cli, '_bounded', bounded)
    assert cli.main(['run', '--execute', '--root', str(tmp_path), '--config', str(config),
                     '--sections', 'baselines', '--models', 'qwen7b',
                     '--model-python', 'qwen7b='+sys.executable]) == 2
    assert calls and all(c[2] == 'check' for c in calls)


def test_worker_interpreters_and_single_shared_deadline(tmp_path, monkeypatch):
    import akr_final.comprehensive as cli
    config = tmp_path/'config.yaml'
    config.write_text('{}')
    primary, optional = tmp_path/'core-python', tmp_path/'optional-python'
    primary.touch(); optional.touch()
    plan = {'models': {'qwen7b': {}, 'qwen3omni': {}}, 'sections': ['diagnostic'],
            'baseline_jobs': [{'id': 'bz', 'kind': 'beans_zero', 'model': 'qwen3omni'}]}
    monkeypatch.setattr(cli, 'make_plan', lambda *a, **k: ({}, plan))
    calls = []
    monkeypatch.setattr(cli, '_bounded', lambda command, root, env, deadline: calls.append((command, deadline)) or 0)
    assert cli.main(['check', '--root', str(tmp_path), '--config', str(config),
                     '--sections', 'diagnostic', '--models', 'qwen7b,qwen3omni', '--enable-conditional',
                     '--model-python', 'qwen7b='+str(primary), '--model-python', 'qwen3omni='+str(optional)]) == 0
    assert [c[0][0] for c in calls] == [str(primary), str(optional), str(optional)]
    assert len({c[1] for c in calls}) == 1


def test_aggregate_identity_rejects_changed_registration(tmp_path, monkeypatch):
    import sys
    import akr_final.comprehensive as cli
    root, path, cfg = config_fixture(tmp_path)
    monkeypatch.setattr(cli, '_bounded', lambda *a: 0)
    arguments = ['check', '--root', str(root), '--config', str(path), '--sections', 'akr',
                 '--models', 'qwen7b', '--model-python', 'qwen7b='+sys.executable]
    assert cli.main(arguments) == 0
    manifest = root/'data/marm.csv'
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(RuntimeError, match='changed'):
        cli.main(arguments)


def test_timeout_stops_later_workers(tmp_path, monkeypatch):
    import sys
    import akr_final.comprehensive as cli
    config = tmp_path/'config.yaml'; config.write_text('{}')
    plan = {'models': {'qwen7b': {}}, 'sections': ['akr'], 'baseline_jobs': []}
    monkeypatch.setattr(cli, 'make_plan', lambda *a, **k: ({}, plan))
    calls = []
    monkeypatch.setattr(cli, '_bounded', lambda command, *a: calls.append(command) or 124)
    assert cli.main(['run', '--execute', '--root', str(tmp_path), '--config', str(config),
                     '--models', 'qwen7b', '--model-python', 'qwen7b='+sys.executable]) == 124
    assert len(calls) == 1 and calls[0][2] == 'check'


def test_akr_only_plan_does_not_depend_on_beans_zero(tmp_path):
    from akr_final.comprehensive import make_plan
    root, path, cfg = config_fixture(tmp_path)
    diagnostic = root/'data/manifests/beans_zero_targets_fullscan_cap10.csv'
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.write_text('malformed unrelated diagnostic')
    study, plan = make_plan(root, path, ['qwen7b'], ['akr'])
    assert study['jobs'] and not plan['baseline_jobs']


def test_dataset_rank_and_token_budget_reach_runner(tmp_path):
    from akr_final.rebuttal_plan import build_study
    root, path, cfg = config_fixture(tmp_path)
    cfg['datasets']['marmaudio'].update(rank=1, max_new_tokens=64)
    path.write_text(yaml.safe_dump(cfg))
    entry = build_study(root, path, ['core'])['datasets']['marmaudio']
    assert entry['options']['settings']['rank'] == 1
    assert entry['options']['settings']['max_new_tokens'] == 64
    assert entry['plan']['episodes'][0]['lock']['selection']['rank'] == 1
    assert cfg['settings']['rank'] == 2


def test_beans_supervised_partition_uses_disjoint_roles_and_all_methods(tmp_path):
    import csv
    from akr_final.beans_zero_protocol import prepare_registration
    from akr_final.rebuttal_runner import RebuttalStudy
    from test_rebuttal_runner import RebuttalFake
    root, path, cfg = config_fixture(tmp_path)
    manifest = root/'data/marm.csv'
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row.update(dataset_name='calls', source_dataset='synthetic',
                   file_name=row['event_id']+'.wav')
    with manifest.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    registration = prepare_registration(root, root/'derived', source_manifest=manifest,
                                        expected_source_count=None)
    component = registration['components']['calls']
    name = component['dataset_key']
    dataset = cfg['datasets'].pop('marmaudio')
    dataset.update(partition='beans_zero_supervised', manifest=component['manifest'],
        group_key='recording_group', fit_per_class=1, calibration_per_class=1,
        guard_per_class=1, rank=1, max_new_tokens=64, conditions=['full'],
        supervised_registration='derived/registration.json')
    cfg['datasets'][name] = dataset
    cfg['seeds'] = [11]
    cfg['suites'] = {'bz': dict(dataset=name, model='qwen7b', seeds='all', controls='first')}
    path.write_text(yaml.safe_dump(cfg))
    runner = RebuttalStudy(root, path, 'bz_smoke', ['bz'], backend_factory=RebuttalFake)
    entry = runner.study['datasets'][name]
    episode = entry['plan']['episodes'][0]
    assert len(episode['support_ids']) == 6
    assert len(episode['query_ids']) == len(rows) - 6
    assert not set(episode['support_ids']) & set(episode['query_ids'])
    assert 'supervised' in entry['plan']['evaluation_status']
    result = runner.run(execute=True)
    assert result['complete'] and len(result['rows']) == 27
    import soundfile as sf
    signal, rate = sf.read(rows[0]['audio_path'])
    sf.write(rows[0]['audio_path'], signal*.9, rate, subtype='FLOAT')
    changed = RebuttalStudy(root, path, 'changed_bz', ['bz'], backend_factory=RebuttalFake)
    with pytest.raises(ValueError, match='waveform changed'):
        changed.run(execute=False)
