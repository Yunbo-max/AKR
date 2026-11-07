#!/usr/bin/env python3
"""Real Qwen2.5-Omni-3B integration smoke; never a research result.

Uses eight existing Dogs recordings (two classes, three train plus one test
per class), with distinct real source groups and waveform hashes. The normal
RebuttalStudy allocator assigns fit/calibration/guard without model outcomes.
All 13 primary methods and 14 standard first-seed controls run on two queries.
No audio is synthesized, no optional models run, and existing studies are not
modified. Run in WSL with CUDA_VISIBLE_DEVICES selecting an available GPU.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import yaml
import numpy as np
import soundfile as sf

from akr_closing.core import Journal, atomic_json, file_hash, read_manifest
from akr_closing.cli import validate_run_id
from akr_final.rebuttal_runner import RebuttalStudy, expected_methods, report_study


def prepare_config(root, directory):
    """Pick a deterministic, outcome-blind subset of real registered audio."""
    original = root / 'configs/akr_comprehensive.yaml'
    config = yaml.safe_load(original.read_text())
    source = root / 'data/manifests/beans_dogs_protocol.csv'
    population = read_manifest(source, old_root=config.get('old_root'), root=root)
    selected, groups, hashes = [], set(), set()
    for label in ['Mac', 'Rudy']:
        for split, count in [('test', 1), ('train', 3)]:
            choices = sorted((r for r in population if r['label'] == label and r['split'] == split),
                             key=lambda r: r['event_id'])
            kept = 0
            for row in choices:
                if row['source_filename'] in groups:
                    continue
                actual = file_hash(Path(row['audio_path']))
                if actual in hashes:
                    continue
                audio, rate = sf.read(row['audio_path'], dtype='float32', always_2d=True)
                if not audio.size or not np.isfinite(audio).all() or rate <= 0:
                    raise RuntimeError(f'invalid real waveform: {row["event_id"]}')
                # apply_beans_duration_protocol.py retains the embedded source
                # audio hash while rewriting a mono/padded FLOAT WAV. Register
                # both identities explicitly for this derived smoke manifest.
                row = {**row, 'original_embedded_sha256': row['sha256'], 'sha256': actual}
                selected.append(row)
                groups.add(row['source_filename'])
                hashes.add(actual)
                kept += 1
                if kept == count:
                    break
            if kept != count:
                raise RuntimeError(f'insufficient distinct {split} recordings for {label}')
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'SMOKE_MANIFEST.csv'
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in selected for k in r}))
        writer.writeheader()
        writer.writerows(selected)
    config.update(
        smoke_validation_only=True,
        scientific_result=False,
        smoke_note='Eight real Dogs recordings; integration coverage only. Do not report accuracy as research evidence.',
        seeds=[20261106],
        datasets={'smoke_dogs': dict(manifest=str(manifest), labels=['Mac', 'Rudy'],
                                    group_key='source_filename', partition='official_test',
                                    fit_per_class=1, calibration_per_class=1, guard_per_class=1,
                                    conditions=['full'])},
        suites={'smoke3b': dict(dataset='smoke_dogs', model='qwen3b', seeds='first', controls='first')},
    )
    config['settings'].update(rank=1, gain_steps=[.5], control_alphas=[.003, .01, .03],
                              max_new_tokens=12)
    config_path = directory / 'SMOKE_CONFIG.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    atomic_json(directory / 'INPUT_PROVENANCE.json', {
        'smoke_validation_only': True, 'scientific_result': False,
        'source_manifest': str(source), 'source_manifest_sha256': file_hash(source),
        'source_config': str(original), 'source_config_sha256': file_hash(original),
        'selected_ids': [r['event_id'] for r in selected],
        'source_groups': [r['source_filename'] for r in selected],
        'waveform_sha256': {r['event_id']: r['sha256'] for r in selected},
        'original_embedded_sha256': {r['event_id']: r['original_embedded_sha256'] for r in selected},
        'hash_note': 'Original manifest sha256 hashes embedded parquet bytes, before duration protocol rewrites WAV.',
        'audio_cropped_or_synthesized': False,
    })
    return config_path


def inspect_results(study):
    report = study.report()
    job, = study.study['jobs']
    methods = expected_methods(job, study.study['config']['settings'])
    if not report['complete'] or len(methods) != 27 or len(report['rows']) != 27:
        raise AssertionError('smoke must complete one job and exactly 27 method reports')
    base = study.out / job['dataset'] / 'trials' / job['model'] / job['episode']
    records = {}
    for method in methods:
        directory = base / method
        meta = json.loads((directory / 'manifest.json').read_text())
        journal = Journal(directory, meta['provenance'], meta['expected'])
        if len(meta['expected']) != 2 or not journal.complete:
            raise AssertionError(f'incomplete two-query journal: {method}')
        records[method] = journal.results()
    for noop in ['zero', 'zero_field']:
        for native, actual in zip(records['native'], records[noop]):
            for key in ['prediction', 'raw_prediction', 'generated_tokens']:
                if native[key] != actual[key]:
                    raise AssertionError(f'{noop} differs from native on {key}')
    nonzero = ['fixed_relative', 'akr_relative', 'akr_cap', 'akr_all', 'k_only_cap', 'v_only_cap']
    if any(not row.get('applied_norms') for method in nonzero for row in records[method]):
        raise AssertionError('real K/V injection was not recorded for an expected method')
    return {'complete': True, 'method_count': len(methods), 'methods': methods,
            'query_count': 2, 'method_query_records': sum(len(v) for v in records.values()),
            'alpha_zero_matches_native': True, 'zero_field_matches_native': True,
            'interventions_recorded': True, 'report_recomputed_and_verified': True,
            'run_directory': str(study.out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='smoke_methods_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--execute', action='store_true', help='load actual 3B weights and execute all methods')
    parser.add_argument('--inspect-existing', action='store_true',
                        help='verify existing journals without constructing/loading any model')
    parser.add_argument('--minimum-free-gib', type=float, default=24.)
    args = parser.parse_args(argv)
    validate_run_id(args.run_id)
    if not args.run_id.startswith('smoke_'):
        raise ValueError('smoke run IDs must start with smoke_')
    directory = ROOT / 'results/readiness/akr_methods' / args.run_id
    if args.inspect_existing:
        if args.execute:
            raise ValueError('inspection cannot execute a model')
        run = ROOT / 'results/akr_rebuttal' / args.run_id
        original = json.loads((directory / 'VALIDATION.json').read_text())
        registered = json.loads((run / 'STUDY.json').read_text())
        saved_identity = json.loads((run / 'IDENTITY.json').read_text())
        changed = [name for name, sha in saved_identity['code'].items()
                   if file_hash(ROOT / name) != sha]
        study = SimpleNamespace(out=run, study=registered, report=lambda: report_study(run))
        checked = inspect_results(study)
        checked.update(smoke_validation_only=True, scientific_result=False,
                       original_validation=str(directory / 'VALIDATION.json'),
                       original_validation_sha256=file_hash(directory / 'VALIDATION.json'),
                       original_elapsed_seconds=original['elapsed_seconds'],
                       resume_without_model_load=original.get('resume_without_model_load', False),
                       changed_code_files_since_execution=changed,
                       original_error=original.get('error'),
                       resume_status=('rejected_concurrent_source_changes' if changed and
                           original.get('error') == 'Source/code/config/environment changed: choose a new run ID.'
                           else 'see_original_validation'),
                       inspection_script_sha256=file_hash(Path(__file__)),
                       model_loaded_for_inspection=False)
        atomic_json(directory / 'METHOD_VALIDATION.json', checked)
        print(json.dumps(checked), flush=True)
        return 0
    config_path = prepare_config(ROOT, directory)
    started = time.time()
    evidence = {'smoke_validation_only': True, 'scientific_result': False,
                'complete': False, 'run_id': args.run_id, 'script_sha256': file_hash(Path(__file__)),
                'model': 'Qwen/Qwen2.5-Omni-3B', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'config_path': str(config_path), 'execute': args.execute}
    atomic_json(directory / 'VALIDATION.json', evidence)
    try:
        if args.execute:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('select exactly one available GPU using CUDA_VISIBLE_DEVICES')
            free, total = torch.cuda.mem_get_info()
            evidence.update(free_gpu_bytes_before=free, total_gpu_bytes=total)
            if free < args.minimum_free_gib * 1024**3:
                raise RuntimeError('insufficient free GPU memory; do not displace other workloads')
        study = RebuttalStudy(ROOT, config_path, args.run_id, ['smoke3b'], profile='24gb')
        job, = study.study['jobs']
        episode, = study.study['datasets']['smoke_dogs']['plan']['episodes']
        evidence['roles'] = {key: episode[key] for key in ['fit_ids', 'calibration_ids', 'guard_ids', 'query_ids']}
        evidence['expected_methods'] = expected_methods(job, study.study['config']['settings'])
        print(json.dumps({'phase': 'registered', **evidence}), flush=True)
        outcome = study.run(execute=args.execute)
        if args.execute:
            evidence.update(inspect_results(study))
            # A complete run must resume entirely from its real journals/cache.
            def no_reload(*_args, **_kwargs):
                raise AssertionError('completed smoke resume attempted to reload a model')
            study.factory = no_reload
            changed = [name for name, sha in study.identity['code'].items()
                       if file_hash(ROOT / name) != sha]
            if changed:
                evidence.update(resume_without_model_load=False,
                                resume_status='not_attempted_concurrent_source_changes',
                                changed_code_files_since_execution=changed)
            else:
                resumed = study.run(execute=True)
                if not resumed['complete']:
                    raise AssertionError('completed smoke did not resume')
                evidence['resume_without_model_load'] = True
        else:
            evidence['preflight_complete'] = outcome['preflight_complete']
    except BaseException as exc:
        evidence.update(complete=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        evidence['elapsed_seconds'] = time.time() - started
        atomic_json(directory / 'VALIDATION.json', evidence)
        print(json.dumps(evidence), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
