"""Derived supervised BEANS protocol, distinct from the original zero-shot task.

Each component is its own closed-label classification task. Registration uses
only source identifiers, decoded audio and labels for class coverage/conflict
checks; no model outcomes enter eligibility or role allocation. Repeated draws
share the source population and must not be presented as independent datasets.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct

import numpy as np
import soundfile as sf

from akr_closing.core import atomic_json, digest, file_hash, read_manifest

SOURCE_MANIFEST = 'data/manifests/beans_zero_targets_fullscan_cap10.csv'
DEFAULT_OUTPUT = 'data/manifests/beans_zero_akr_v1'
PROTOCOL = 'beans_zero_derived_supervised_v1'


def canonical_pcm_hash(path):
    """Hash the model-observable mono float32 samples, sample rate and length.

    WAV container bytes, metadata and lossless encodings do not define acoustic
    identity. Channels are averaged as in the model audio path; signed zero is
    canonicalized. No lossy quantization or approximate matching is performed.
    """
    samples, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f'empty or nonfinite decoded waveform: {path}')
    mono = np.asarray(samples.mean(axis=1), dtype='<f4')
    mono[mono == 0] = 0
    value = hashlib.sha256(b'AKR-mono-float32-le-v1\0')
    value.update(struct.pack('<QQ', int(rate), len(mono)))
    value.update(mono.tobytes())
    return value.hexdigest()


def _portable(path, root):
    path = Path(path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _component_registration(name, rows):
    # Source and acoustic equivalence compose transitively, including a repeated
    # recording whose different segments connect otherwise distinct PCM groups.
    rows = sorted(rows, key=lambda r: r['event_id'])
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    sources, waveforms = {}, {}
    for index, row in enumerate(rows):
        filename, source = row.get('file_name', '').strip(), row.get('source_dataset', '').strip()
        if filename and source:
            key = (source, filename)
            if key in sources:
                union(index, sources[key])
            sources[key] = index
        pcm = row['pcm_sha256']
        if pcm in waveforms:
            union(index, waveforms[pcm])
        waveforms[pcm] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    excluded, candidates, descriptions = [], [], []
    def exclude(row, reason, **extra):
        excluded.append(dict(event_id=row['event_id'], label=row['label'], reason=reason, **extra))
    for members in groups.values():
        ids = sorted(row['event_id'] for row in members)
        labels = sorted({row['label'] for row in members})
        group = digest({'component': name, 'member_ids': ids,
                        'pcm_sha256': sorted({r['pcm_sha256'] for r in members})})
        descriptions.append(dict(recording_group=group, member_ids=ids, labels=labels))
        if len(labels) != 1:
            for row in members:
                exclude(row, 'conflicting_group_labels', recording_group=group)
            continue
        source_known = []
        for row in members:
            if not row.get('file_name', '').strip():
                exclude(row, 'missing_source_filename', recording_group=group)
            elif not row.get('source_dataset', '').strip():
                exclude(row, 'missing_source_dataset', recording_group=group)
            else:
                source_known.append(row)
        if not source_known:
            continue
        representative = min(source_known, key=lambda r: r['event_id'])
        candidates.append({**representative, 'recording_group': group, 'recording_id': group,
                           'group_member_ids': json.dumps(ids, separators=(',', ':')),
                           'source_split': representative.get('split', ''),
                           'split': 'derived_supervised_pool', 'protocol': PROTOCOL})
        for row in source_known:
            if row is not representative:
                exclude(row, 'duplicate_recording_group', recording_group=group,
                        representative_id=representative['event_id'])
    class_counts = {label: sum(r['label'] == label for r in candidates)
                    for label in sorted({r['label'] for r in candidates})}
    labels = [label for label, count in class_counts.items() if count >= 4]
    retained = []
    for row in candidates:
        if row['label'] not in labels:
            exclude(row, 'fewer_than_four_groups', distinct_groups=class_counts[row['label']])
        elif len(labels) < 2:
            exclude(row, 'fewer_than_two_classes')
        else:
            retained.append(row)
    if len(labels) < 2:
        labels = []
    reasons = {reason: sum(r['reason'] == reason for r in excluded)
               for reason in sorted({r['reason'] for r in excluded})}
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', name).strip('_') or 'component'
    key = 'bz_' + slug + '_' + digest(name)[:8]
    return sorted(retained, key=lambda r: r['event_id']), dict(
        name=name, dataset_key=key, labels=labels, source_rows=len(rows), eligible_rows=len(retained),
        eligible_groups=len(retained), source_groups=len(groups), class_group_counts=class_counts,
        exclusions=sorted(excluded, key=lambda r: r['event_id']), exclusion_counts=reasons,
        groups=sorted(descriptions, key=lambda g: g['recording_group']),
        eligible=bool(retained), manifest=None)


def prepare_registration(root, outdir=None, *, source_manifest=None, expected_source_count=2950):
    """Write immutable per-component canonical-group manifests and registration.json.

    ``sha256`` retains the repository's FILE-byte meaning. ``pcm_sha256`` is the
    separately named decoded acoustic identity. Relative paths resolve at root.
    Repeating an unchanged registration is idempotent; changes require a new
    output directory, so existing experimental registrations cannot be replaced.
    """
    root = Path(root).resolve()
    outdir = Path(DEFAULT_OUTPUT if outdir is None else outdir)
    outdir = (root / outdir).resolve() if not outdir.is_absolute() else outdir.resolve()
    source = Path(SOURCE_MANIFEST if source_manifest is None else source_manifest)
    source = (root / source).resolve() if not source.is_absolute() else source.resolve()
    source_hash = file_hash(source)
    rows = read_manifest(source, old_root='/root/animal-omni-kv', root=root)
    if expected_source_count is not None and len(rows) != expected_source_count:
        raise ValueError(f'expected {expected_source_count} registered BEANS rows, found {len(rows)}')
    components = defaultdict(list)
    for row in rows:
        name = row.get('dataset_name', '').strip()
        if not name or not row['label'].strip() or not row['event_id'].strip():
            raise ValueError('component, event ID and label must be nonempty')
        path = Path(row['audio_path'])
        row = {**row, 'sha256': file_hash(path), 'pcm_sha256': canonical_pcm_hash(path),
               'audio_path': _portable(path, root)}
        components[name].append(row)
    if file_hash(source) != source_hash:
        raise RuntimeError('source registration changed during preparation')
    registration = dict(version=1, protocol=PROTOCOL,
        description='Derived supervised per-component closed-label classification; not original BEANS-Zero evaluation.',
        source_manifest=_portable(source, root), source_manifest_sha256=source_hash,
        source_rows=len(rows), source_waveforms_sha256=digest(
            {r['event_id']: r['sha256'] for values in components.values() for r in values}),
        grouping='component-scoped transitive union of (source_dataset,file_name) and canonical mono PCM hash',
        representative='lexicographically smallest source-identified event ID in each nonconflicting group',
        eligibility='at least four distinct recording groups per class; at least two retained classes per component',
        allocation='one distinct group per class for fit, calibration and guard; remaining groups query',
        rank=1, fit_per_class=1, calibration_per_class=1, guard_per_class=1,
        selection_uses_model_outcomes=False, query_population='remaining eligible canonical source groups',
        components={})
    pending = {}
    for name in sorted(components):
        selected, metadata = _component_registration(name, components[name])
        if selected:
            path = outdir / (metadata['dataset_key'] + '.csv')
            if path == source:
                raise ValueError('output manifest must not overwrite source')
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=sorted({k for r in selected for k in r}), lineterminator='\n')
            writer.writeheader(); writer.writerows(selected)
            pending[path] = stream.getvalue().encode('utf-8')
            metadata.update(manifest=_portable(path, root), manifest_sha256=hashlib.sha256(pending[path]).hexdigest())
        registration['components'][name] = metadata
    registration['eligible_components'] = sum(c['eligible'] for c in registration['components'].values())
    registration['eligible_rows'] = sum(c['eligible_rows'] for c in registration['components'].values())
    registration['excluded_rows'] = len(rows) - registration['eligible_rows']
    destination = outdir / 'registration.json'
    if destination.exists() and json.loads(destination.read_text(encoding='utf-8')) != registration:
        raise RuntimeError('registration inputs changed; use a new output directory')
    for path, content in pending.items():
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError(f'registration manifest changed: {path}; use a new output directory')
    outdir.mkdir(parents=True, exist_ok=True)
    for path, content in pending.items():
        if not path.exists():
            temporary = path.with_suffix('.csv.tmp')
            temporary.write_bytes(content)
            os.replace(temporary, path)
    atomic_json(destination, registration)
    return registration


def validate_registration(root, registration_path, dataset_key, manifest_path, labels, rows):
    """Verify a planned component against its immutable supervised registration.

    Returns absolute path -> verified file SHA256 for the registration, original
    source CSV and derived CSV. This checks registration identity before a new
    run, not merely that a later resume sees the same edited files. Execution
    must additionally compare actual WAV file hashes with each row's sha256.
    """
    root = Path(root).resolve()
    def resolve(value):
        value = Path(value)
        return (value if value.is_absolute() else root / value).resolve()
    registration_path, manifest_path = resolve(registration_path), resolve(manifest_path)
    registration_hash = file_hash(registration_path)
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    if (type(registration.get('version')) is not int or registration['version'] != 1
            or registration.get('protocol') != PROTOCOL):
        raise ValueError('unsupported supervised BEANS registration protocol/version')
    components = registration.get('components')
    if not isinstance(components, dict):
        raise ValueError('supervised registration requires component entries')
    matches = [component for component in components.values()
               if component.get('dataset_key') == dataset_key]
    if len(matches) != 1 or matches[0].get('eligible') is not True:
        raise ValueError('dataset must match exactly one eligible registered component')
    component = matches[0]
    if not component.get('manifest') or resolve(component['manifest']) != manifest_path:
        raise ValueError('component manifest path differs from supervised registration')
    manifest_hash = file_hash(manifest_path)
    if manifest_hash != component.get('manifest_sha256'):
        raise RuntimeError('component manifest bytes changed since supervised registration')
    if list(labels) != component.get('labels') or set(labels) != {row['label'] for row in rows}:
        raise ValueError('component label vocabulary/order differs from supervised registration')
    for key in ['eligible_rows', 'eligible_groups']:
        if type(component.get(key)) is not int or component[key] != len(rows):
            raise ValueError('component row/group count differs from supervised registration')
    expected_rows = read_manifest(manifest_path, old_root='/root/animal-omni-kv', root=root)
    if sorted(rows, key=lambda r: r['event_id']) != sorted(expected_rows, key=lambda r: r['event_id']):
        raise ValueError('planned rows differ from the verified component manifest')
    descriptions = component.get('groups', [])
    groups = {group['recording_group']: group for group in descriptions}
    if len(groups) != len(descriptions):
        raise ValueError('duplicate groups in supervised registration')
    for row in rows:
        group = groups.get(row.get('recording_group'))
        if (not group or row['event_id'] not in group.get('member_ids', [])
                or group.get('labels') != [row['label']]
                or row.get('dataset_name') != component.get('name')
                or row.get('protocol') != PROTOCOL):
            raise ValueError('row/group membership differs from supervised registration')
    # Also enforce one representative per group/hash and four groups per class.
    allocate(rows, labels, 0)
    if not registration.get('source_manifest'):
        raise ValueError('supervised registration has no original source manifest')
    source = resolve(registration['source_manifest'])
    source_hash = file_hash(source)
    if source_hash != registration.get('source_manifest_sha256'):
        raise RuntimeError('original BEANS source manifest changed since supervised registration')
    if file_hash(registration_path) != registration_hash:
        raise RuntimeError('supervised registration changed during validation')
    return {str(registration_path): registration_hash, str(source): source_hash,
            str(manifest_path): manifest_hash}


def allocate(rows, labels, seed):
    """Allocate registered canonical groups without reading predictions or losses."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if len(labels) < 2 or len(set(labels)) != len(labels) or set(labels) != {r['label'] for r in rows}:
        raise ValueError('closed label vocabulary must match at least two eligible classes')
    for key in ['event_id', 'recording_group', 'pcm_sha256', 'sha256']:
        values = [r.get(key, '') for r in rows]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f'noncanonical registration: empty or duplicate {key}')
    result = dict(fit_ids=[], calibration_ids=[], guard_ids=[], query_ids=[])
    for label in sorted(labels):
        candidates = sorted([r for r in rows if r['label'] == label],
                            key=lambda r: (digest([PROTOCOL, seed, label, r['recording_group']]), r['event_id']))
        if len(candidates) < 4:
            raise ValueError(f'{label}: at least four distinct groups required')
        for index, role in enumerate(['fit_ids', 'calibration_ids', 'guard_ids']):
            result[role].append(candidates[index]['event_id'])
        result['query_ids'].extend(r['event_id'] for r in candidates[3:])
    result['support_ids'] = result['fit_ids'] + result['calibration_ids'] + result['guard_ids']
    result.update(allocation_criterion='seeded canonical-group allocation; no model outcomes',
                  protocol=PROTOCOL, seed=seed)
    return result
