"""Baseline data isolation, identity locking, and honest completion contracts."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def fixture_study(tmp_path):
    rows = []
    for event, label, value in [('s0', 'a', -.5), ('s1', 'b', .5),
                                 ('q0', 'a', -.4), ('q1', 'b', .4)]:
        path = tmp_path / (event + '.wav')
        sf.write(path, np.full(320, value), 16000, subtype='FLOAT')
        rows.append(dict(event_id=event, label=label, audio_path=str(path), recording_id=event))
    episode = dict(id='s7_full', seed=7, condition='full', support_ids=['s0', 's1'],
                   query_ids=['q0', 'q1'])
    plan = dict(rows=rows, labels=['a', 'b'], episodes=[episode], source_hashes={})
    study = dict(config=dict(settings=dict(ridge_alpha=10.), old_root='/old'),
                 config_sha256='fixture', datasets={'dogs': {'plan': plan}},
                 jobs=[dict(dataset='dogs', model='qwen7b', episode='s7_full', condition='full')])
    checkpoint = tmp_path / 'aves.onnx'
    checkpoint.write_bytes(b'fixture checkpoint')
    return study, {'qwen7b': {'id': 'Qwen/Qwen2.5-Omni-7B', 'revision': 'a' * 40},
                   'qwen3b': {'id': 'Qwen/Qwen2.5-Omni-3B', 'revision': 'b' * 40}}, {'aves': str(checkpoint)}


class AudioFeatures:
    def feature(self, query, prompt):
        assert not hasattr(query, 'label')
        signal, _ = sf.read(query.audio_path)
        return np.array([signal.mean(), 1.])


def test_supervised_beans_gets_matched_aves_without_unrelated_lora(tmp_path):
    from akr_final.comprehensive_baselines import baseline_jobs, run_baseline_job
    from akr_closing.core import file_hash
    study, models, assets = fixture_study(tmp_path)
    entry = study['datasets'].pop('dogs')
    entry['plan']['dataset'] = {'partition': 'beans_zero_supervised'}
    for row in entry['plan']['rows']:
        row['sha256'] = file_hash(Path(row['audio_path']))
    study['datasets']['bz_calls'] = entry
    study['jobs'][0]['dataset'] = 'bz_calls'
    jobs = baseline_jobs(tmp_path, study, models, assets, include_diagnostic=False)
    assert len(jobs) == 1 and jobs[0]['kind'] == 'aves_ridge'
    result = run_baseline_job(tmp_path, tmp_path/'run', jobs[0], study, assets,
                              execute=True, backend_factory=lambda *a, **k: AudioFeatures())
    assert result['complete'] and result['metrics']['accuracy'] == 1.
    entry['plan']['rows'][0]['sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='waveform changed'):
        run_baseline_job(tmp_path, tmp_path/'changed', jobs[0], study, assets)


def test_aves_uses_registered_support_labels_only_and_resumes(tmp_path):
    from akr_final.comprehensive_baselines import baseline_jobs, run_baseline_job
    study, models, assets = fixture_study(tmp_path)
    job = next(j for j in baseline_jobs(tmp_path, study, models, assets) if j['kind'] == 'aves_ridge')
    assert job['episode'] == 's7_full'
    factory = lambda *args, **kwargs: AudioFeatures()
    first = run_baseline_job(tmp_path, tmp_path/'run', job, study, assets, execute=True, backend_factory=factory)
    assert first['complete'] and first['metrics']['accuracy'] == 1.
    assert first['support_ids'] == ['s0', 's1']
    assert first['query_ids'] == ['q0', 'q1']
    # Query labels are swapped: predictions must stay the same, accuracy must change.
    study['datasets']['dogs']['plan']['rows'][2]['label'] = 'b'
    study['datasets']['dogs']['plan']['rows'][3]['label'] = 'a'
    second = run_baseline_job(tmp_path, tmp_path/'different', job, study, assets,
                              execute=True, backend_factory=factory)
    assert second['metrics']['accuracy'] == 0.
    with pytest.raises(RuntimeError, match='changed'):
        run_baseline_job(tmp_path, tmp_path/'run', job, study, assets, execute=True, backend_factory=factory)
    study['datasets']['dogs']['plan']['rows'][2]['label'] = 'a'
    study['datasets']['dogs']['plan']['rows'][3]['label'] = 'b'
    resumed = run_baseline_job(tmp_path, tmp_path/'run', job, study, assets, execute=True,
                               backend_factory=lambda *a, **k: pytest.fail('complete job reloaded model'))
    assert resumed['complete']
    from akr_final.comprehensive_baselines import report_baseline_job
    reported = report_baseline_job(tmp_path, tmp_path/'run', job, study, assets)
    assert reported['metrics']['accuracy'] == 1.
    item = next((Path(first['output_dir'])/'predictions'/'items').glob('*.json'))
    item.unlink()
    status = run_baseline_job(tmp_path, tmp_path/'run', job, study, assets)
    assert not status['complete'] and status['completed'] == 1


def test_preflight_does_not_load_models_and_smoke_cannot_complete(tmp_path):
    from akr_final.comprehensive_baselines import baseline_jobs, run_baseline_job
    study, models, assets = fixture_study(tmp_path)
    job = next(j for j in baseline_jobs(tmp_path, study, models, assets) if j['kind'] == 'aves_ridge')
    check = run_baseline_job(tmp_path, tmp_path/'run', job, study, assets,
                             backend_factory=lambda *a, **k: pytest.fail('preflight loaded model'))
    assert check['ready'] and not check['complete']
    smoke = run_baseline_job(tmp_path, tmp_path/'run', job, study, assets, execute=True,
                             backend_factory=lambda *a, **k: AudioFeatures(), limit=100)
    assert smoke['smoke'] and not smoke['complete'] and smoke['processed_complete']
    assert 'smoke' in Path(smoke['output_dir']).parts
    Path(assets['aves']).write_bytes(b'changed checkpoint')
    with pytest.raises(RuntimeError, match='changed'):
        run_baseline_job(tmp_path, tmp_path/'run', job, study, assets, limit=100)


def test_overlapping_support_query_is_rejected(tmp_path):
    from akr_final.comprehensive_baselines import baseline_jobs, run_baseline_job
    study, models, assets = fixture_study(tmp_path)
    job = next(j for j in baseline_jobs(tmp_path, study, models, assets) if j['kind'] == 'aves_ridge')
    study['datasets']['dogs']['plan']['episodes'][0]['query_ids'] = ['s0']
    with pytest.raises(ValueError, match='overlap'):
        run_baseline_job(tmp_path, tmp_path/'run', job, study, assets)


def make_beans_zero(tmp_path, study, count=2950):
    path = tmp_path/'data/manifests/beans_zero_targets_fullscan_cap10.csv'
    path.parent.mkdir(parents=True)
    audio = study['datasets']['dogs']['plan']['rows'][0]['audio_path']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['event_id', 'audio_path', 'label', 'instruction_text', 'dataset_name'])
        writer.writeheader()
        writer.writerows(dict(event_id=f'bz{i}', audio_path=audio, label='Species A',
                              instruction_text='Name the species.', dataset_name='component') for i in range(count))
    return path


def test_beans_zero_exact_registered_ids_instruction_and_smoke(tmp_path):
    from akr_final.comprehensive_baselines import baseline_jobs, run_baseline_job
    study, models, assets = fixture_study(tmp_path)
    path = make_beans_zero(tmp_path, study)
    jobs = [j for j in baseline_jobs(tmp_path, study, models, assets) if j['kind'] == 'beans_zero']
    assert {j['model'] for j in jobs} == {'qwen7b', 'qwen3b'}
    class InstructionBackend:
        def predict(self, query, prompt, labels, **kwargs):
            assert query.event_id == 'bz0' and prompt == 'Name the species.'
            assert not labels and not hasattr(query, 'label')
            return {'raw_prediction': 'Species A!', 'prediction': ''}
    result = run_baseline_job(tmp_path, tmp_path/'run', jobs[0], study, assets, execute=True,
                              backend_factory=lambda *a, **k: InstructionBackend(), limit=1)
    assert result['expected'] == 2950 and result['completed'] == 1 and not result['complete']
    assert result['metrics']['exact_match'] == 1.
    text = path.read_text().replace('bz2949,', 'foreign,')
    path.write_text(text)
    with pytest.raises((ValueError, RuntimeError), match='changed|registered'):
        run_baseline_job(tmp_path, tmp_path/'run', jobs[0], study, assets, limit=1)


def test_beans_zero_rejects_wrong_cardinality(tmp_path):
    from akr_final.comprehensive_baselines import baseline_jobs, run_baseline_job
    study, models, assets = fixture_study(tmp_path)
    make_beans_zero(tmp_path, study, count=3)
    with pytest.raises(ValueError, match='2950'):
        jobs = baseline_jobs(tmp_path, study, models, assets)
        job = next(j for j in jobs if j['kind'] == 'beans_zero')
        run_baseline_job(tmp_path, tmp_path/'run', job, study, assets)


def test_specialist_keeps_original_96khz_and_applies_registered_lowpass(tmp_path):
    from akr_final.comprehensive_baselines import baseline_jobs, run_baseline_job
    study, models, assets = fixture_study(tmp_path)
    study['datasets']['marmaudio'] = study['datasets'].pop('dogs')
    study['jobs'][0]['dataset'] = 'marmaudio'
    plan = study['datasets']['marmaudio']['plan']
    plan['episodes'][0].update(id='s7_lp_1000', condition='lp:1000')
    for row in plan['rows']:
        t = np.arange(9600) / 96000
        sf.write(row['audio_path'], np.sin(2*np.pi*4000*t), 96000, subtype='FLOAT')
    code = tmp_path/'specialist/marmaudio'
    code.mkdir(parents=True)
    for name in ['classifier.py', 'filterbank.py']:
        (code/name).write_text('# fixture')
    checkpoint, label_path = tmp_path/'specialist.pt', tmp_path/'labels.tsv'
    checkpoint.write_bytes(b'fixture')
    label_path.write_text('label\na\nb\n')
    assets.update(marmaudio_code=str(code.parent), marmaudio_checkpoint=str(checkpoint), marmaudio_labels=str(label_path))
    job = next(j for j in baseline_jobs(tmp_path, study, models, assets) if j['kind'] == 'marmaudio_specialist')
    class Specialist:
        def predict(self, q, prompt, labels, **kwargs):
            signal, rate = sf.read(q.audio_path)
            assert rate == 96000 and len(signal) == 9600
            assert np.std(signal[1000:-1000]) < .01
            assert not hasattr(q, 'label')
            return {'prediction': 'a' if q.event_id == 'q0' else 'b'}
    result = run_baseline_job(tmp_path, tmp_path/'run', job, study, assets, execute=True,
                              backend_factory=lambda *a, **k: Specialist())
    assert result['complete'] and result['metrics']['accuracy'] == 1.
    assert 'unmatched' in result['supervision'] and result['support_ids'] == []
    sf.write(plan['rows'][2]['audio_path'], np.ones(320), 16000)
    with pytest.raises(ValueError, match='96 kHz'):
        run_baseline_job(tmp_path, tmp_path/'run', job, study, assets)


def test_published_lora_attaches_direct_thinker_and_declares_unmatched(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from akr_final.comprehensive_baselines import baseline_jobs, run_baseline_job
    study, models, assets = fixture_study(tmp_path)
    adapter = tmp_path/'adapter'
    adapter.mkdir()
    (adapter/'adapter_config.json').write_text(json.dumps(dict(peft_type='LORA', base_model_name_or_path='Qwen2.5-Omni-7B/thinker')))
    (adapter/'adapter_model.safetensors').write_bytes(b'fixture')
    (adapter/'task_config.yaml').write_text('model: {id: Qwen/Qwen2.5-Omni-7B}\ndataset: {labels: [a, b]}\nevaluation: {prompt: Original adapter prompt, max_new_tokens: 16}\n')
    assets['lora_dogs'] = str(adapter)
    job = next(j for j in baseline_jobs(tmp_path, study, models, assets) if j['kind'] == 'published_lora')
    original = object()
    adapted = SimpleNamespace(eval=lambda: None)
    def attach(thinker, path, *, is_trainable):
        assert thinker is original and Path(path) == adapter and not is_trainable
        return adapted
    monkeypatch.setitem(__import__('sys').modules, 'peft', SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=attach)))
    class Backend:
        thinker = original
        def predict(self, q, prompt, labels, **kwargs):
            assert prompt == 'Original adapter prompt'
            assert kwargs['max_new_tokens'] == 16
            return {'prediction': ('a' if q.event_id == 'q0' else 'b') if self.thinker is adapted else ''}
    result = run_baseline_job(tmp_path, tmp_path/'run', job, study, assets, execute=True,
                              backend_factory=lambda *a, **k: Backend())
    assert result['complete'] and result['metrics']['accuracy'] == 1.
    assert 'unmatched' in result['supervision'] and result['query_ids'] == ['q0', 'q1']


def test_resume_rejects_changed_environment_but_reporting_accepts_saved_environment(tmp_path, monkeypatch):
    import akr_final.comprehensive_baselines as module
    study, models, assets = fixture_study(tmp_path)
    job = next(j for j in module.baseline_jobs(tmp_path, study, models, assets) if j['kind'] == 'aves_ridge')
    module.run_baseline_job(tmp_path, tmp_path/'run', job, study, assets, execute=True,
                            backend_factory=lambda *a, **k: AudioFeatures())
    monkeypatch.setattr(module, '_environment', lambda: {'test': 'changed environment'})
    with pytest.raises(RuntimeError, match='changed'):
        module.run_baseline_job(tmp_path, tmp_path/'run', job, study, assets, execute=True)
    assert module.report_baseline_job(tmp_path, tmp_path/'run', job, study, assets)['complete']
