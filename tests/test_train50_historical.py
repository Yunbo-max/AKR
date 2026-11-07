import json
from pathlib import Path

import pytest
import yaml

from akr_closing.core import atomic_json, file_hash
from akr_closing.repair import GradientRouter
from experiments.rebuttal_cases.c16_historical_readouts import execute
from test_akr_final_three import FakeBackend, build_source


def fixture(tmp_path):
    root, source, options = build_source(tmp_path, seeds=(11,))
    config = root / 'historical.yaml'
    config.write_text(yaml.safe_dump(options))
    return root, source, config


def test_historical_full_suite_preserves_source_and_exact_ids(tmp_path):
    root, source, config = fixture(tmp_path)
    before = {str(p): file_hash(p) for p in source.rglob('*') if p.is_file()}
    FakeBackend.loads = 0
    report = execute('run', root, source, config, 'historical', backend_factory=FakeBackend,
                     router_fit=GradientRouter.fit)
    assert report['complete'] is True
    assert report['historical_raw_eta_parity'] is False
    assert report['new_protocol_readout_extension'] is True
    assert report['episodes'][0]['query_ids'] == ['10', '11']
    assert report['episodes'][0]['support_ids'] == [str(i) for i in range(8)]
    assert before == {str(p): file_hash(p) for p in source.rglob('*') if p.is_file()}
    out = root / 'results/akr_historical_readouts/historical'
    assert json.loads((out / 'STATUS.json').read_text())['complete']
    complete = json.loads(next((out / 'S1').rglob('COMPLETE.json')).read_text())
    assert complete['settings_origin'] == 'registered source LOCK; relative_active_state_norm'
    execute('report', root, source, config, 'historical')
    scored = next((out / 'S1').glob('qwen7b/*/ridge_direct/SCORED_PREDICTIONS.json'))
    rows = json.loads(scored.read_text()); rows.reverse(); atomic_json(scored, rows)
    with pytest.raises(RuntimeError):
        execute('report', root, source, config, 'historical')
    assert json.loads((out / 'STATUS.json').read_text())['complete'] is False


def test_check_no_gpu_and_missing_source_resets_stale_status(tmp_path):
    root, source, config = fixture(tmp_path)
    FakeBackend.loads = 0
    report = execute('check', root, source, config, 'checked', backend_factory=FakeBackend)
    assert report['complete'] is False and report['preflight_passed'] is True
    assert FakeBackend.loads == 0
    status = root / 'results/akr_historical_readouts/checked/STATUS.json'
    atomic_json(status, {'complete': True})
    (source / 'RUN.json').unlink()
    with pytest.raises(ValueError, match='raw protocol not recovered'):
        execute('check', root, source, config, 'checked')
    assert json.loads(status.read_text())['complete'] is False


def test_unknown_raw_scale_explicitly_fails(tmp_path):
    root, source, config = fixture(tmp_path)
    path = next(source.rglob('LOCK.json'))
    lock = json.loads(path.read_text()); lock['scale_type'] = 'raw_eta'; atomic_json(path, lock)
    with pytest.raises(ValueError, match='raw protocol not recovered'):
        execute('check', root, source, config, 'unsupported')


def test_incomplete_method_is_never_reported_complete(tmp_path):
    root, source, config = fixture(tmp_path)
    execute('run', root, source, config, 'partial', backend_factory=FakeBackend, router_fit=GradientRouter.fit)
    out = root / 'results/akr_historical_readouts/partial'
    next((out / 'S1').glob('qwen7b/*/probe_to_text/METRICS.json')).unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        execute('report', root, source, config, 'partial')
    assert not json.loads((out / 'STATUS.json').read_text())['complete']


def test_preflight_and_backend_failure_leave_incomplete(tmp_path):
    root, source, config = fixture(tmp_path)
    out = root / 'results/akr_historical_readouts/bad_backend'
    atomic_json(out / 'STATUS.json', {'complete': True})
    class ChangedBackend(FakeBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.provenance['dtype'] = 'float32'
    with pytest.raises(ValueError, match='dtype differs'):
        execute('run', root, source, config, 'bad_backend', backend_factory=ChangedBackend,
                router_fit=GradientRouter.fit)
    assert not json.loads((out / 'STATUS.json').read_text())['complete']
    (root / 'data/0.wav').unlink()
    with pytest.raises(FileNotFoundError, match='waveform'):
        execute('check', root, source, config, 'bad_backend', backend_factory=FakeBackend)
    assert not json.loads((out / 'STATUS.json').read_text())['complete']


def test_plan_needs_explicit_source_but_never_constructs_model(tmp_path):
    report = execute('plan', tmp_path, Path('missing-source'), Path('missing.yaml'), 'plan')
    assert report['loads_gpu'] is False and report['needs_registered_source'] is True
    assert report['fresh_split_fallback'] is False
    assert not (tmp_path / 'results').exists()
