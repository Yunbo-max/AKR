"""Offline validation: rebuilt registration is accepted by the C16 consumer.

No model weights are loaded.  `c16_historical_readouts.execute('check')` only
constructs FinalRunner, validates RUN/EPISODE/LOCK population and hashes audio.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from akr_closing.core import atomic_json
from akr_closing.pipeline import prompt_for
from experiments.rebuttal_cases.c16_historical_readouts import execute
from historical_rerun.register_rebuild import rebuild


def _fixture(tmp_path):
    labels = ['A', 'B']
    events = [f'e{i}' for i in range(12)]
    groups = {f'e{i}': f'g{i // 2}' for i in range(12)}
    audio = tmp_path / 'audio'
    audio.mkdir()
    rows = []
    for event in events:
        path = audio / f'{event}.wav'
        path.write_bytes(b'RIFF' + event.encode())
        rows.append({'event_id': event, 'audio_path': str(path),
                     'label': 'A' if int(event[1:]) % 2 == 0 else 'B',
                     'recording_id': groups[event]})
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('event_id,audio_path,label,recording_id\n' + ''.join(
        f'{r["event_id"]},{r["audio_path"]},{r["label"]},{r["recording_id"]}\n'
        for r in rows))
    train = events[0:6]
    selection = events[6:8]
    confirmation = events[8:10]
    test = events[10:12]
    support = ['e0', 'e1', 'e4', 'e5', 'e6', 'e7']
    fit_ids, calibration_ids, guard_ids = ['e0', 'e1', 'e4'], ['e5', 'e6'], ['e7']
    assert support == fit_ids + calibration_ids + guard_ids
    revision = 'a' * 40
    config = {
        'datasets': [{'name': 'marmaudio', 'manifest': str(manifest),
                      'labels': labels, 'group_key': 'recording_id'}],
        'seeds': [42],
        'models': [{'tag': 'qwen7b', 'id': 'Qwen/Qwen2.5-Omni-7B', 'revision': revision}],
        'E9': {'id': 'Qwen/Qwen2.5-Omni-3B', 'revision': 'b' * 40},
        'layer_fractions': [0.5], 'feature_fraction': 0.5,
        'max_context_tokens': 4096, 'ridge_alpha': 10.0,
    }
    episodes = []
    for condition in ['full', 'lp:1000']:
        episodes.append({
            'id': f's42_{condition.replace(":", "_")}', 'seed': 42, 'condition': condition,
            'support_ids': support, 'query_ids': confirmation,
            'fit_ids': fit_ids, 'calibration_ids': calibration_ids, 'guard_ids': guard_ids,
            'lock': {'selection': {'rank': 2, 'alpha': 0.01, 'ridge_alpha': 10.0},
                     'support': support, 'prompt': prompt_for(labels)},
            'lock_sha256': 'historical-lock-hash',
            'meta': {'condition': condition, 'support_ids': support},
        })
    protocol = {
        'mode': 'standalone', 'labels': labels,
        'dataset': {'name': 'marmaudio', 'group_key': 'recording_id'},
        'config': config, 'episodes': episodes,
        'splits': {'42': {'train': train, 'selection': selection,
                          'confirmation': confirmation, 'test': test,
                          'seed': 42, 'fingerprint': 'historical-fingerprint'}},
        'rows': rows,
        'historical_metadata_recovered': False,
    }
    protocol_path = tmp_path / 'STANDALONE_PROTOCOL.json'
    atomic_json(protocol_path, protocol)
    backend_path = tmp_path / 'BACKEND_qwen7b.json'
    atomic_json(backend_path, {'model_id': 'Qwen/Qwen2.5-Omni-7B', 'revision': revision,
                               'layers': [1], 'feature_layer': 1,
                               'dtype': 'bfloat16', 'batch_size': 1})
    return tmp_path, protocol_path, manifest, backend_path, protocol


def test_rebuilt_registration_is_accepted_by_c16_check(tmp_path):
    root, protocol, manifest, backend, _ = _fixture(tmp_path)
    source = tmp_path / 'source'
    receipt = rebuild(root, protocol, manifest, backend, source)
    assert receipt['complete'] and receipt['episodes'][0]['support'] == 6
    assert receipt['historical_raw_eta_300_parity'] is False
    assert receipt['original_run_episode_lock_recovered'] is False
    options = json.loads((source / 'C16_OPTIONS.json').read_text())
    assert options['k_per_class'] == 3 and options['conditions'] == ['full', 'lp:1000']
    report = execute('check', root, source, source / 'C16_OPTIONS.json', 'checked')
    assert report['preflight_passed'] is True
    assert report['loads_gpu'] is False
    assert report['n_episodes'] == 2
    assert report['historical_raw_eta_parity'] is False
    # The widened train pool must contain selection-drawn support IDs unchanged.
    from akr_final.runner import select_episodes
    plan = select_episodes(root, source, yaml.safe_load((source / 'C16_OPTIONS.json').read_text()))
    assert plan['episodes'][0]['support_ids'] == ['e0', 'e1', 'e4', 'e5', 'e6', 'e7']
    assert plan['episodes'][0]['query_ids'] == ['e8', 'e9']
    assert set(plan['episodes'][0]['support_ids']) <= set(
        json.loads((source / 'splits/marmaudio_42.json').read_text())['train'])


def test_c16_check_rejects_a_tampered_lock(tmp_path):
    root, protocol, manifest, backend, _ = _fixture(tmp_path)
    source = tmp_path / 'source'
    rebuild(root, protocol, manifest, backend, source)
    lock_path = source / 'qwen7b/marmaudio/s42/semantic/full/k3/LOCK.json'
    lock = json.loads(lock_path.read_text())
    lock['prompt'] = 'Some other prompt'
    atomic_json(lock_path, lock)
    with pytest.raises(ValueError, match='raw protocol not recovered'):
        execute('check', root, source, source / 'C16_OPTIONS.json', 'checked')


def test_rebuild_refuses_support_outside_historical_pool(tmp_path):
    root, protocol, manifest, backend, _ = _fixture(tmp_path)
    data = json.loads(protocol.read_text())
    support = ['e0', 'e1', 'e4', 'e5', 'e6', 'e11']  # e11 is in test, class balance kept
    for episode in data['episodes']:
        episode['support_ids'] = support
        for role in ['fit_ids', 'calibration_ids', 'guard_ids']:
            episode.pop(role)
        episode['lock']['support'] = support
    atomic_json(protocol, data)
    with pytest.raises(ValueError, match='support outside historical train\\+selection'):
        rebuild(root, protocol, manifest, backend, tmp_path / 'source')
