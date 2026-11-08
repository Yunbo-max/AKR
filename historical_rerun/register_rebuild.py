#!/usr/bin/env python3
"""Rebuild C16-compatible RUN/EPISODE/LOCK registration from STANDALONE_PROTOCOL.

The original raw-eta=300 RUN/EPISODE/LOCK files were never recovered.  This tool
does NOT pretend to recover them: it constructs NEW registration artifacts that
mirror the exact support/query/lock IDs embedded in a STANDALONE_PROTOCOL.json
produced by `akr_final/standalone.py` or by the `rebuttal_runner` standalone
plan route.  The output is a new identity for a previously inspected split.

Why the embedding is usable
---------------------------
`STANDALONE_PROTOCOL.json` embeds the split membership, the ordered support IDs
(fit + calibration + guard), the query IDs and the lock settings.  On the
recovered MarmAudio core protocol we verified that:

  * `akr_closing.core.make_split` reproduces all four split buckets and the
    fingerprint exactly from the registered manifest (sha256 4b236b7b...);
  * `akr_final.rebuttal_plan.allocate_roles` reproduces fit/calibration/guard
    exactly;
  * all 318 support/query waveform hashes match the audio bytes on disk.

One deliberate deviation is required by the consumer schema: FinalRunner's
`select_episodes` demands `support_ids <= split['train']`, while the historical
role allocation drew support from `train + selection`.  The rebuilt split file
therefore sets `train = train | selection` and recomputes the split fingerprint;
`confirmation` (the query population) is unchanged.  This is recorded in the
split note, in every EPISODE.json and in REBUILD.json.

Consumer: `experiments/rebuttal_cases/c16_historical_readouts.py check/run`.
No model weights are loaded here.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from akr_closing.core import atomic_json, digest, file_hash, read_manifest
from akr_closing.pipeline import prompt_for

BACKBONE_KEYS = ['model_id', 'revision', 'layers', 'feature_layer', 'dtype', 'batch_size']
CODE_FILES = ['backend.py', 'pipeline.py', 'repair.py', 'core.py']
REPOSITORY_FILES = ['src/animal_omni/conditional_kv.py', 'src/animal_omni/metrics.py']
NOTE = ('new registration constructed from STANDALONE_PROTOCOL; the original '
        'historical RUN/EPISODE/LOCK artifacts were not recovered; not raw-eta=300 '
        'parity and not an untouched benchmark')


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _code_identity(root: Path) -> tuple[dict, dict]:
    root = Path(root).resolve()
    code = {name: file_hash(root / 'akr_closing' / name) for name in CODE_FILES
            if (root / 'akr_closing' / name).is_file()}
    repository = {name: file_hash(root / name) for name in REPOSITORY_FILES
                  if (root / name).is_file()}
    return code, repository


def _support_per_class(ids, lookup, labels):
    counts = {label: sum(1 for i in ids if lookup[i]['label'] == label) for label in labels}
    _require(all(counts.values()), f'support class coverage missing: {counts}')
    values = set(counts.values())
    _require(len(values) == 1, f'support per class is not uniform: {counts}')
    return values.pop()


def rebuild(root, protocol_path, manifest_path, backend_path, out, *,
            waveform_hashes=None, profile='80gb', force=False):
    """Write RUN.json, splits/, EPISODE.json, LOCK.json under `out`; return receipt."""
    root = Path(root).resolve()
    protocol_path = Path(protocol_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    backend_path = Path(backend_path).resolve()
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f'{out} is not empty; pass --force to overwrite')

    protocol = _read_json(protocol_path)
    _require(protocol.get('mode') == 'standalone', 'STANDALONE_PROTOCOL mode must be standalone')
    labels = protocol.get('labels')
    _require(isinstance(labels, list) and labels and len(set(labels)) == len(labels)
             and all(isinstance(x, str) and x for x in labels), 'distinct nonempty labels required')
    dataset_cfg = protocol.get('dataset')
    _require(isinstance(dataset_cfg, dict), 'protocol dataset mapping required')
    dataset = dataset_cfg.get('name')
    group_key = dataset_cfg.get('group_key') or 'recording_id'
    _require(isinstance(dataset, str) and dataset, 'dataset name required')

    rows = read_manifest(manifest_path)
    lookup = {row['event_id']: row for row in rows}
    _require(len(rows) == len(lookup), 'duplicate manifest event IDs')

    # If the protocol embeds its row table, demand exact agreement with the manifest.
    embedded = protocol.get('rows')
    if embedded:
        _require({r['event_id'] for r in embedded} == set(lookup),
                 'protocol row IDs differ from manifest population')
        for row in embedded:
            got = lookup[row['event_id']]
            _require(row.get('label') == got.get('label'),
                     f"label mismatch for {row['event_id']}")
            _require(row.get('recording_id') == got.get('recording_id'),
                     f"recording_id mismatch for {row['event_id']}")
            if row.get('audio_path'):
                _require(Path(row['audio_path']).name == Path(got['audio_path']).name,
                         f"audio filename mismatch for {row['event_id']}")

    config = deepcopy(protocol.get('config'))
    _require(isinstance(config, dict), 'protocol config mapping required')
    # rebuttal_plan builds a standalone "internal" config without datasets/seeds;
    # FinalRunner.select_episodes (used by C16) requires both.  Reconstruct them
    # from the embedded dataset entry and episode seeds.
    if not isinstance(config.get('datasets'), list) or not config['datasets']:
        config['datasets'] = [deepcopy(dataset_cfg)]
    dataset_entry = next((d for d in config['datasets'] if d.get('name') == dataset), None)
    _require(dataset_entry is not None, f'config has no dataset {dataset}')
    models = {m['tag']: m for m in config.get('models', [])}
    _require('qwen7b' in models, 'protocol config must declare qwen7b')

    backend = _read_json(backend_path)
    missing = [key for key in BACKBONE_KEYS if key not in backend]
    _require(not missing, f'backend provenance missing {missing}')
    backbone = {key: backend[key] for key in BACKBONE_KEYS}
    _require(backbone['model_id'] == models['qwen7b']['id'], 'backend model_id differs from config')
    _require(backbone['revision'] == models['qwen7b']['revision'],
             'backend revision differs from config')
    _require(isinstance(backbone['layers'], list) and backbone['layers'], 'backend layers required')
    _require(isinstance(backbone['feature_layer'], int), 'backend feature_layer required')

    episodes = protocol.get('episodes')
    _require(isinstance(episodes, list) and episodes, 'protocol episodes required')
    seeds = sorted({int(e['seed']) for e in episodes})
    conditions = sorted({e['condition'] for e in episodes})
    _require(conditions == ['full', 'lp:1000'],
             f'C16 consumer requires exactly [full, lp:1000], got {conditions}')
    _require(len({(int(e['seed']), e['condition']) for e in episodes}) == len(episodes),
             'duplicate seed/condition episode')
    _require(len({e['id'] for e in episodes}) == len(episodes), 'duplicate episode id')
    config['seeds'] = seeds
    if 'ridge_alpha' not in config:
        config['ridge_alpha'] = float(episodes[0]['lock']['selection']['ridge_alpha'])
    for key in ['layer_fractions', 'feature_fraction', 'max_context_tokens']:
        _require(key in config, f'protocol config missing {key} required by FinalRunner')

    k_values = set()
    for episode in episodes:
        seed = int(episode['seed'])
        split = protocol.get('splits', {}).get(str(seed))
        _require(split is not None, f'protocol has no split for seed {seed}')
        support = list(episode['support_ids'])
        query = list(episode['query_ids'])
        _require(support and len(set(support)) == len(support), f'{episode["id"]}: bad support IDs')
        _require(query and len(set(query)) == len(query), f'{episode["id"]}: bad query IDs')
        _require(not set(support) & set(query), f'{episode["id"]}: support/query overlap')
        _require(set(support) <= set(lookup) and set(query) <= set(lookup),
                 f'{episode["id"]}: IDs absent from manifest')
        k_values.add(_support_per_class(support, lookup, labels))
        for role in ['fit_ids', 'calibration_ids', 'guard_ids']:
            if role in episode:
                _require(set(episode[role]) <= set(support),
                         f'{episode["id"]}: {role} outside support')
                _require(not set(episode[role]) & set(query),
                         f'{episode["id"]}: {role}/query overlap')
        if all(role in episode for role in ['fit_ids', 'calibration_ids', 'guard_ids']):
            _require(episode['support_ids'] ==
                     list(episode['fit_ids']) + list(episode['calibration_ids'])
                     + list(episode['guard_ids']),
                     f'{episode["id"]}: support is not fit+calibration+guard')
        _require(set(support) <= set(split['train']) | set(split['selection']),
                 f'{episode["id"]}: support outside historical train+selection')
        _require(set(query) == set(split['confirmation']),
                 f'{episode["id"]}: query population differs from confirmation split')
        if group_key:
            support_groups = {lookup[i][group_key] for i in support}
            query_groups = {lookup[i][group_key] for i in query}
            _require(support_groups and query_groups and not support_groups & query_groups,
                     f'{episode["id"]}: support/query recording-group leakage')
        lock_in = episode.get('lock') or {}
        _require(list(lock_in.get('support', support)) == support,
                 f'{episode["id"]}: LOCK/EPISODE support differ')
        if 'prompt' in lock_in:
            _require(lock_in['prompt'] == prompt_for(labels),
                     f'{episode["id"]}: noncanonical historical prompt; refusing to replace it')
        selection = lock_in.get('selection') or {}
        _require(isinstance(selection.get('rank'), int) and selection['rank'] >= 1,
                 f'{episode["id"]}: invalid locked rank')
        _require(float(selection.get('alpha', -1)) >= 0, f'{episode["id"]}: invalid locked alpha')
        _require(float(selection.get('ridge_alpha', config.get('ridge_alpha', 0))) > 0,
                 f'{episode["id"]}: invalid locked ridge_alpha')
    _require(len(k_values) == 1, f'inconsistent support-per-class across episodes: {k_values}')
    k_per_class = k_values.pop()

    out.mkdir(parents=True, exist_ok=True)
    code, repository = _code_identity(root)
    cfg = deepcopy(config)
    cfg['seeds'] = seeds
    ds_entry = next(d for d in cfg['datasets'] if d.get('name') == dataset)
    ds_entry['manifest'] = str(manifest_path)
    ds_entry['labels'] = list(labels)
    ds_entry.setdefault('group_key', group_key)
    run = {
        'config': cfg,
        'profile': profile,
        'code': code,
        'repository_sources': repository,
        'rebuild_provenance': {
            'generator': 'historical_rerun/register_rebuild.py',
            'protocol': str(protocol_path),
            'protocol_sha256': file_hash(protocol_path),
            'manifest': str(manifest_path),
            'manifest_sha256': file_hash(manifest_path),
            'backend_provenance': str(backend_path),
            'backend_provenance_sha256': file_hash(backend_path),
            'note': NOTE,
        },
    }
    atomic_json(out / 'RUN.json', run)

    split_fingerprints = {}
    for seed in seeds:
        historical = protocol['splits'][str(seed)]
        train = sorted(set(historical['train']) | set(historical['selection']))
        split = {
            'train': train,
            'selection': sorted(historical['selection']),
            'confirmation': sorted(historical['confirmation']),
            'test': sorted(historical['test']),
            'seed': seed,
            'evaluation_status': 'reconstructed locked re-evaluation; previously inspected data',
            'note': ('train is historical train UNION selection because the embedded support '
                     'allocation drew from both pools; confirmation/query population unchanged; '
                     'this is NOT a new untouched benchmark'),
            'reconstructed_from_protocol': str(protocol_path),
        }
        _require(not (set(split['train']) & set(split['confirmation'])),
                 f'seed {seed}: train/confirmation overlap')
        split['fingerprint'] = digest(split)
        split_fingerprints[seed] = split['fingerprint']
        atomic_json(out / 'splits' / f'{dataset}_{seed}.json', split)

    for episode in episodes:
        seed = int(episode['seed'])
        condition = episode['condition']
        context = f"qwen7b/{dataset}/s{seed}/semantic/{condition.replace(':', '_')}"
        base = out / context / f'k{k_per_class}'
        support = list(episode['support_ids'])
        selection = episode['lock']['selection']
        meta = {
            'context': context,
            'condition': condition,
            'support_ids': support,
            'support_labels': [lookup[i]['label'] for i in support],
            'k_per_class': k_per_class,
            'split_fingerprint': split_fingerprints[seed],
            'backbone': backbone,
            'source_protocol_episode': episode['id'],
            'source_protocol_lock_sha256': episode.get('lock_sha256'),
            'reconstruction_note': NOTE,
        }
        lock = {
            'selection': {
                'rank': int(selection['rank']),
                'alpha': float(selection['alpha']),
                'ridge_alpha': float(selection.get('ridge_alpha', cfg['ridge_alpha'])),
            },
            'support': support,
            'prompt': prompt_for(labels),
            'layers': backbone['layers'],
            'feature_layer': backbone['feature_layer'],
            'scale_type': 'relative_active_state_norm',
            'source_protocol_lock_sha256': episode.get('lock_sha256'),
            'reconstruction_note': NOTE,
        }
        atomic_json(base / 'EPISODE.json', meta)
        atomic_json(base / 'LOCK.json', lock)

    ids = sorted({i for e in episodes for i in list(e['support_ids']) + list(e['query_ids'])})
    current_hashes = {i: file_hash(Path(lookup[i]['audio_path'])) for i in ids}
    atomic_json(out / 'splits' / f'{dataset}_waveform_hashes.json', current_hashes)
    if waveform_hashes is not None:
        recorded = _read_json(Path(waveform_hashes))
        differing = [i for i in ids
                     if i not in recorded or recorded[i] != current_hashes[i]]
        _require(not differing,
                 f'waveform bytes differ from recorded hashes for {len(differing)} IDs; '
                 f'first: {differing[:3]}')
        waveform_verified = True
    else:
        waveform_verified = False

    settings = {
        'rank': int(episodes[0]['lock']['selection']['rank']),
        'relative_alpha': float(episodes[0]['lock']['selection']['alpha']),
        'ridge_alpha': float(episodes[0]['lock']['selection'].get(
            'ridge_alpha', cfg['ridge_alpha'])),
        'layer_fractions': cfg.get('layer_fractions'),
        'feature_fraction': cfg.get('feature_fraction'),
        'max_context_tokens': cfg.get('max_context_tokens'),
    }
    c16_options = {
        'dataset': dataset,
        'label_mode': 'semantic',
        'k_per_class': k_per_class,
        'max_seeds': len(seeds),
        'conditions': ['full', 'lp:1000'],
        'anchor_condition': 'lp:1000',
        'ranks': [1, 2, 4, 8],
        'folds': 5,
        'bootstrap_samples': 0,
        'all_labelled_per_class': k_per_class,
        'settings': settings,
    }
    atomic_json(out / 'C16_OPTIONS.json', c16_options)

    receipt = {
        'complete': True,
        'note': NOTE,
        'output': str(out),
        'protocol': str(protocol_path),
        'protocol_sha256': file_hash(protocol_path),
        'manifest': str(manifest_path),
        'manifest_sha256': file_hash(manifest_path),
        'waveform_hashes_verified': waveform_verified,
        'dataset': dataset,
        'labels': labels,
        'seeds': seeds,
        'conditions': ['full', 'lp:1000'],
        'k_per_class': k_per_class,
        'support_per_class': k_per_class,
        'episodes': [{'id': e['id'], 'support': len(e['support_ids']),
                      'query': len(e['query_ids'])} for e in episodes],
        'registered_ids': len(ids),
        'split_fingerprints': {str(seed): split_fingerprints[seed] for seed in seeds},
        'backbone': backbone,
        'profile': profile,
        'scale_type': 'relative_active_state_norm',
        'historical_raw_eta_300_parity': False,
        'original_run_episode_lock_recovered': False,
    }
    atomic_json(out / 'REBUILD.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--backend-json', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--waveform-hashes', type=Path,
                        help='optional event_id->sha256 JSON from the same run for byte checks')
    parser.add_argument('--profile', default='80gb')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    receipt = rebuild(args.root, args.protocol, args.manifest, args.backend_json, args.out,
                      waveform_hashes=args.waveform_hashes, profile=args.profile,
                      force=args.force)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
