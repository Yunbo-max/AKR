#!/usr/bin/env python3
"""Source-locked historical readout supplement; never creates a fresh split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import yaml
from akr_closing.core import Journal, atomic_json, digest, file_hash
from akr_final.runner import FinalRunner, prediction_summary
from akr_final.simple_baselines import run_simple_baselines

METHODS = ['native', 'fixed_mean', 'continuous_pooled', 'ridge_direct', 'probe_to_text', 'probe_to_lm']
LIMITATION = ('Historical raw protocol not recovered: explicit registered RUN/LOCK/EPISODE '
              'artifacts with relative_active_state_norm are required. Raw eta=300 legacy '
              'parity is unsupported; no support count or split will be fabricated.')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _files(out):
    paths = list((out / 'S1').rglob('*')) + list(out.glob('BACKEND_*.json'))
    return {str(p.relative_to(out)): file_hash(p) for p in paths if p.is_file()}


def _verify(runner):
    episodes = []
    base = runner.out / 'S1/qwen7b'
    if {p.name for p in base.iterdir() if p.is_dir()} != {e['id'] for e in runner.plan['episodes']}:
        raise RuntimeError('Historical episode population differs from registered source')
    for episode in runner.plan['episodes']:
        path = base / episode['id']
        complete = _read(path / 'COMPLETE.json')
        if (complete.get('complete') is not True or complete.get('methods') != METHODS
                or complete.get('support_ids') != episode['support_ids']
                or complete.get('query_ids') != episode['query_ids']
                or complete.get('source_fingerprint') != runner.fingerprint):
            raise RuntimeError('Historical episode completion/identity mismatch')
        if {p.name for p in path.iterdir() if p.is_dir()} != set(METHODS + ['zero_alpha']):
            raise RuntimeError('Historical method population differs from S1')
        rows_by_method = {}
        for method in METHODS:
            directory = path / method
            rows = _read(directory / 'SCORED_PREDICTIONS.json')
            if ([r.get('event_id') for r in rows] != episode['query_ids']
                    or [r.get('target') for r in rows] != [runner.lookup[i]['label'] for i in episode['query_ids']]
                    or any(not isinstance(r.get('prediction'), str) for r in rows)):
                raise RuntimeError('Historical prediction IDs/order/targets mismatch')
            rows_by_method[method] = rows
            if method in ['ridge_direct', 'probe_to_text']:
                direct = _read(directory / 'PREDICTIONS.json')
                stripped = [{k: v for k, v in r.items() if k not in ['event_id', 'target']} for r in rows]
                if direct != stripped:
                    raise RuntimeError('Historical direct predictions disagree with scored rows')
            else:
                manifest = _read(directory / 'manifest.json')
                provenance = manifest['provenance']
                if (manifest.get('expected') != episode['query_ids']
                        or provenance.get('run') != runner.fingerprint
                        or provenance.get('episode') != episode['id']
                        or provenance.get('method') != method or provenance.get('tag') != 'qwen7b'
                        or digest(provenance) != manifest.get('fingerprint')):
                    raise RuntimeError('Historical prediction journal provenance mismatch')
                journal = Journal(directory, provenance, episode['query_ids'])
                values = journal.results()
                if values != [{k: v for k, v in r.items() if k not in ['event_id', 'target']} for r in rows]:
                    raise RuntimeError('Historical journal/scored predictions disagree')
                expected_items = {digest(i) + '.json' for i in episode['query_ids']}
                if {p.name for p in (directory / 'items').glob('*.json')} != expected_items:
                    raise RuntimeError('Historical journal has foreign query items')
        truth = [runner.lookup[i]['label'] for i in episode['query_ids']]
        native = [r['prediction'] for r in rows_by_method['native']]
        summaries = {}
        for method, rows in rows_by_method.items():
            saved = _read(path / method / 'METRICS.json')
            expected = prediction_summary(truth, native, [r['prediction'] for r in rows], runner.labels)
            if (saved.get('episode') != episode['id'] or saved.get('query_ids') != episode['query_ids']
                    or any(saved.get(k) != v for k, v in expected.items())):
                raise RuntimeError('Historical metrics disagree with complete predictions')
            summaries[method] = expected
        episodes.append({'episode': episode['id'], 'support_ids': episode['support_ids'],
                         'query_ids': episode['query_ids'], 'metrics': summaries,
                         'source_backbone': episode['meta']['backbone'],
                         'source_lock_sha256': episode['lock_sha256']})
    return episodes


def execute(action, root, source, config, run_id, *, profile='24gb', cache_only=False,
            backend_factory=None, router_fit=None):
    """Run/check/report one explicit source registration, with isolated output."""
    if action not in ['plan', 'check', 'run', 'report']:
        raise ValueError('action must be plan/check/run/report')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run_id must be a simple name')
    root = Path(root).resolve()
    source, config = Path(source), Path(config)
    source = (root / source).resolve() if not source.is_absolute() else source.resolve()
    config = (root / config).resolve() if not config.is_absolute() else config.resolve()
    out = root / 'results/akr_historical_readouts' / run_id
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Historical output and original source must be separate directories')
    if action == 'plan':
        return {'complete': False, 'source_run': str(source), 'config': str(config), 'output': str(out),
                'needs_registered_source': True, 'loads_gpu': False, 'methods': METHODS,
                'limitation': LIMITATION, 'fresh_split_fallback': False}
    # Reset before all fallible config/source reads, runner construction and preflight.
    atomic_json(out / 'STATUS.json', {'complete': False, 'action': action, 'status': 'starting'})
    runner = None
    try:
        options = yaml.safe_load(config.read_text(encoding='utf-8'))
        if not isinstance(options, dict):
            raise ValueError('Historical options YAML must be a mapping')
        try:
            runner = FinalRunner(root, source, run_id, options, profile,
                                 output_root=root / 'results/akr_historical_readouts',
                                 backend_factory=backend_factory, router_fit=router_fit,
                                 cache_only=cache_only)
        except (KeyError, StopIteration, FileNotFoundError, ValueError) as error:
            raise ValueError(LIMITATION + f' Source validation: {error}') from error
        runner.preflight()
        if action == 'check':
            report = {'complete': False, 'preflight_passed': True, 'loads_gpu': False,
                      'source_run': str(source), 'n_episodes': len(runner.plan['episodes']),
                      'historical_raw_eta_parity': False, 'limitation': LIMITATION}
            atomic_json(out / 'STATUS.json', report)
            return report
        if action == 'run':
            run_simple_baselines(runner)
            # Correct the S1 helper's generic standalone wording for this source route.
            for episode in runner.plan['episodes']:
                path = out / 'S1/qwen7b' / episode['id'] / 'COMPLETE.json'
                complete = _read(path)
                complete.update(settings_origin='registered source LOCK; relative_active_state_norm',
                                source_fingerprint=runner.fingerprint,
                                historical_raw_eta_parity=False,
                                new_protocol_not_historical_confirmation=True)
                atomic_json(path, complete)
        else:
            receipt = _read(out / 'ARTIFACTS.json')
            if receipt.get('source_fingerprint') != runner.fingerprint or receipt.get('files') != _files(out):
                raise RuntimeError('Historical artifact checksum/population mismatch')
        episodes = _verify(runner)
        report = {'complete': True, 'source_run': str(source), 'source_fingerprint': runner.fingerprint,
                  'methods': METHODS, 'episodes': episodes, 'new_protocol_readout_extension': True,
                  'historical_raw_eta_parity': False, 'scale_type': 'relative_active_state_norm',
                  'source_cache_compatible': runner.source_cache_compatible,
                  'backend_provenance': {p.stem: _read(p) for p in out.glob('BACKEND_*.json')},
                  'limitation': LIMITATION}
        atomic_json(out / 'ARTIFACTS.json', {'source_fingerprint': runner.fingerprint, 'files': _files(out)})
        atomic_json(out / 'REPORT.json', report)
        atomic_json(out / 'STATUS.json', {'complete': True, 'action': action, 'status': 'verified'})
        return report
    except Exception as error:
        atomic_json(out / 'STATUS.json', {'complete': False, 'action': action,
                                         'status': 'blocked', 'error': str(error)})
        raise
    finally:
        if runner is not None:
            runner.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'check', 'run', 'report'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile', choices=['16gb', '24gb', '48gb'], default='24gb')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args(argv)
    result = execute(args.action, args.root, args.source_run, args.config, args.run_id,
                     profile=args.profile, cache_only=args.cache_only)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
