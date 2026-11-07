"""Derived supervised BEANS registration uses no model predictions or outcomes."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from akr_closing.core import file_hash, read_manifest
from akr_final.beans_zero_protocol import allocate, canonical_pcm_hash, prepare_registration


def fixture_source(tmp_path):
    rows = []
    def add(event, label, *, component='calls', source='archive', filename=None, signal=None, subtype='PCM_16'):
        path = tmp_path / 'audio' / (event + '.wav')
        path.parent.mkdir(exist_ok=True)
        if signal is None:
            signal = np.full(160, (len(rows) + 1) / 100, dtype=np.float32)
        sf.write(path, signal, 16000, subtype=subtype)
        rows.append(dict(event_id=event, audio_path=str(path.relative_to(tmp_path)), label=label,
                         dataset_name=component, source_dataset=source,
                         file_name=filename if filename is not None else event + '.flac',
                         instruction_text='Which call?', split='test'))
        return path
    def save():
        path = tmp_path / 'source.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        return path
    return rows, add, save


def test_canonical_pcm_hash_ignores_container_and_channel_layout(tmp_path):
    signal = np.arange(128, dtype=np.float32) / 32768
    a, b = tmp_path / 'a.wav', tmp_path / 'b.wav'
    sf.write(a, signal, 16000, subtype='PCM_16')
    sf.write(b, np.column_stack([signal, signal]), 16000, subtype='FLOAT')
    assert file_hash(a) != file_hash(b)
    assert canonical_pcm_hash(a) == canonical_pcm_hash(b)
    sf.write(b, signal, 8000, subtype='FLOAT')
    assert canonical_pcm_hash(a) != canonical_pcm_hash(b)


def test_registration_transitive_duplicates_conflicts_rare_and_blank_sources(tmp_path):
    source, add, save = fixture_source(tmp_path)
    for label in ['A', 'B']:
        for i in range(5):
            add(f'{label}{i}', label)
    # Same source recording connects A0 -> copy1, identical decoded PCM joins
    # copy1 -> copy2 despite a different source filename and WAV encoding.
    signal = np.arange(160, dtype=np.float32) / 32768
    add('copy1', 'A', filename='A0.flac', signal=signal)
    add('copy2', 'A', signal=signal, subtype='FLOAT')
    conflict_audio = add('conflict1', 'X', filename='conflict.flac')
    add('conflict2', 'Y', filename='conflict.flac')
    add('conflict3', 'Z', signal=sf.read(conflict_audio, dtype='float32')[0], subtype='FLOAT')
    for i in range(3):
        add(f'rare{i}', 'rare')
    add('rare_duplicate', 'rare', filename='rare0.flac')
    add('empty', 'A', filename='  ')
    manifest = save()
    original = file_hash(manifest)
    out = tmp_path / 'derived'
    registration = prepare_registration(tmp_path, out, source_manifest=manifest, expected_source_count=None)
    component = registration['components']['calls']
    rows = read_manifest(tmp_path / component['manifest'], root=tmp_path)
    assert component['labels'] == ['A', 'B'] and len(rows) == 10
    assert len({r['recording_group'] for r in rows}) == len(rows)
    a0 = next(r for r in rows if r['event_id'] == 'A0')
    group = next(g for g in component['groups'] if g['recording_group'] == a0['recording_group'])
    assert group['member_ids'] == ['A0', 'copy1', 'copy2']
    reasons = {r['event_id']: r['reason'] for r in component['exclusions']}
    assert reasons['copy1'] == reasons['copy2'] == 'duplicate_recording_group'
    assert reasons['conflict1'] == reasons['conflict2'] == reasons['conflict3'] == 'conflicting_group_labels'
    assert reasons['rare0'] == 'fewer_than_four_groups'
    assert reasons['empty'] == 'missing_source_filename'
    assert all(r['sha256'] == file_hash(Path(r['audio_path'])) for r in rows)
    assert file_hash(manifest) == original == registration['source_manifest_sha256']
    assert prepare_registration(tmp_path, out, source_manifest=manifest, expected_source_count=None) == registration


def test_allocation_is_order_independent_seeded_and_has_no_group_or_pcm_leakage(tmp_path):
    _, add, save = fixture_source(tmp_path)
    for label in ['A', 'B']:
        for i in range(8):
            add(f'{label}{i}', label)
    reg = prepare_registration(tmp_path, tmp_path / 'derived', source_manifest=save(), expected_source_count=None)
    rows = read_manifest(tmp_path / reg['components']['calls']['manifest'], root=tmp_path)
    result = allocate(rows, ['A', 'B'], 17)
    assert result == allocate(rows[::-1], ['B', 'A'], 17)
    assert result != allocate(rows, ['A', 'B'], 18)
    lookup = {r['event_id']: r for r in rows}
    roles = [result[k] for k in ['fit_ids', 'calibration_ids', 'guard_ids', 'query_ids']]
    assert [len(ids) for ids in roles] == [2, 2, 2, 10]
    assert set.union(*(set(ids) for ids in roles)) == set(lookup)
    for key in ['recording_group', 'sha256', 'pcm_sha256']:
        sets = [{lookup[i][key] for i in ids} for ids in roles]
        assert all(not sets[a] & sets[b] for a in range(4) for b in range(a + 1, 4))
    assert result['support_ids'] == sum(roles[:3], [])


def test_source_groups_are_component_and_archive_scoped(tmp_path):
    _, add, save = fixture_source(tmp_path)
    for component in ['first', 'second']:
        for label in ['A', 'B']:
            for i in range(4):
                add(f'{component}{label}{i}', label, component=component,
                    source=label, filename=f'shared{i}.wav')
    reg = prepare_registration(tmp_path, tmp_path / 'derived', source_manifest=save(), expected_source_count=None)
    assert all(c['eligible_rows'] == 8 for c in reg['components'].values())


def test_single_class_component_excluded_and_changed_source_cannot_overwrite(tmp_path):
    _, add, save = fixture_source(tmp_path)
    for i in range(4):
        add(f'A{i}', 'A')
    manifest = save(); out = tmp_path / 'derived'
    reg = prepare_registration(tmp_path, out, source_manifest=manifest, expected_source_count=None)
    assert reg['components']['calls']['manifest'] is None
    assert {r['reason'] for r in reg['components']['calls']['exclusions']} == {'fewer_than_two_classes'}
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(RuntimeError, match='changed'):
        prepare_registration(tmp_path, out, source_manifest=manifest, expected_source_count=None)


def test_allocation_rejects_noncanonical_or_changed_registration():
    rows = [dict(event_id=str(i), label=label, recording_group=f'g{i}',
                 pcm_sha256=f'p{i}', sha256=f'h{i}')
            for i, label in enumerate(['A'] * 4 + ['B'] * 4)]
    with pytest.raises(ValueError, match='vocabulary'):
        allocate(rows, ['A'], 1)
    rows[1]['pcm_sha256'] = rows[0]['pcm_sha256']
    with pytest.raises(ValueError, match='duplicate'):
        allocate(rows, ['A', 'B'], 1)


@pytest.fixture
def registered_component(tmp_path):
    _, add, save = fixture_source(tmp_path)
    for label in ['A', 'B']:
        for i in range(4):
            add(f'{label}{i}', label)
    reg = prepare_registration(tmp_path, tmp_path / 'derived', source_manifest=save(), expected_source_count=None)
    component = reg['components']['calls']
    path = tmp_path / 'derived' / 'registration.json'
    manifest = tmp_path / component['manifest']
    rows = read_manifest(manifest, root=tmp_path)
    return tmp_path, path, manifest, reg, component, rows


def test_validate_registration_returns_verified_source_hashes(registered_component):
    from akr_final.beans_zero_protocol import validate_registration
    root, path, manifest, reg, component, rows = registered_component
    result = validate_registration(root, path, component['dataset_key'], manifest, component['labels'], rows)
    assert result == {str(p.resolve()): file_hash(p)
                      for p in [path, manifest, root / reg['source_manifest']]}


@pytest.mark.parametrize('change', [
    'manifest_bytes', 'original_source', 'protocol', 'version', 'boolean_version',
    'ineligible', 'component_path', 'count', 'group_count', 'labels', 'rows',
    'duplicate_component', 'unknown_component', 'missing_group',
])
def test_validate_registration_rejects_unregistered_changes(registered_component, change):
    from akr_final.beans_zero_protocol import validate_registration
    root, path, manifest, reg, component, rows = registered_component
    key, labels = component['dataset_key'], component['labels']
    if change == 'manifest_bytes':
        manifest.write_text(manifest.read_text() + '\n')
    elif change == 'original_source':
        source = root / reg['source_manifest']
        source.write_text(source.read_text() + '\n')
    elif change == 'protocol': reg['protocol'] = 'unregistered'
    elif change == 'version': reg['version'] = 2
    elif change == 'boolean_version': reg['version'] = True
    elif change == 'ineligible': component['eligible'] = False
    elif change == 'component_path': component['manifest'] = 'wrong.csv'
    elif change == 'count': component['eligible_rows'] += 1
    elif change == 'group_count': component['eligible_groups'] -= 1
    elif change == 'labels': labels = list(reversed(labels))
    elif change == 'rows': rows[0]['recording_group'] = 'fabricated-group'
    elif change == 'duplicate_component': reg['components']['duplicate'] = dict(component)
    elif change == 'unknown_component': key = 'unknown'
    elif change == 'missing_group': component['groups'] = []
    path.write_text(json.dumps(reg))
    with pytest.raises((ValueError, RuntimeError)):
        validate_registration(root, path, key, manifest, labels, rows)
