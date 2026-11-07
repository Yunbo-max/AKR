"""Explicit, bounded orchestration of AKR and its applicable baselines."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import yaml

from akr_closing.core import atomic_json, digest, file_hash
from akr_closing.cli import run_lock, validate_run_id
from akr_closing.model_factory import MODEL_SPECS
from .rebuttal_plan import build_study
from .rebuttal_runner import RebuttalStudy, expected_methods, report_study
from .release import _bounded


def selected_tasks(config, models, *, enable_conditional=False):
    if not models or len(set(models)) != len(models):
        raise ValueError('choose distinct model tags')
    for model in models:
        if model not in MODEL_SPECS:
            raise ValueError(f'unknown model: {model}')
        if MODEL_SPECS[model]['conditional'] and not enable_conditional:
            raise ValueError(f'{model} is conditional; explicitly pass --enable-conditional when needed')
    tasks = [name for name, suite in config['suites'].items() if suite['model'] in models]
    if {config['suites'][t]['model'] for t in tasks} != set(models):
        raise ValueError('configuration has no suite for a selected model')
    return tasks


def make_plan(root, config_path, models, sections, enable_conditional=False, assets=None):
    from .comprehensive_baselines import baseline_jobs
    config = yaml.safe_load(config_path.read_text())
    tasks = selected_tasks(config, models, enable_conditional=enable_conditional)
    if not sections or set(sections) - {'akr', 'baselines', 'diagnostic'}:
        raise ValueError('sections must be akr,baselines,diagnostic or all')
    study = build_study(root, config_path, tasks)
    specs = {tag: config['models'][tag] for tag in models}
    baselines = (baseline_jobs(root, study, specs, assets or {}, include_diagnostic='diagnostic' in sections)
                 if set(sections) & {'baselines', 'diagnostic'} else [])
    baselines = [job for job in baselines if
                 ('diagnostic' if job['kind'] == 'beans_zero' else 'baselines') in sections]
    summary = {
        'config_path': str(config_path.resolve()),
        'models': specs, 'conditional_models_enabled': enable_conditional,
        'sections': sections, 'tasks': tasks if 'akr' in sections else [],
        'akr_jobs': study['jobs'] if 'akr' in sections else [],
        'akr_method_units': sum(len(expected_methods(j, config['settings'])) for j in study['jobs'])
                            if 'akr' in sections else 0,
        'baseline_jobs': baselines,
        'scope': {
            'akr': 'Full/LP1 on MarmAudio, Dogs, Watkins and registered supervised BEANS-Zero components; matched support/query identities across models',
            'aves': 'frozen AVES features with Ridge fitted only on the registered support',
            'pretrained': 'published LoRA and MarmAudio specialist have external training supervision',
            'beans_zero': 'original registered 2950-item instruction diagnostic plus explicitly derived supervised component comparisons with disjoint fit/calibration/guard/query recordings',
        },
        'full_experiments_started': False,
    }
    return study, summary


def snapshot_check(spec):
    """Check local index-referenced weights; never download or load a model."""
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(spec['id'], revision=spec['revision'], local_files_only=True))
    indices = list(path.glob('*.index.json'))
    weights = set()
    for index in indices:
        item = json.loads(index.read_text())
        weights.update(item.get('weight_map', {}).values())
    if not weights:
        weights = {p.name for pattern in ['*.safetensors', 'pytorch_model*.bin'] for p in path.glob(pattern)}
    if not weights:
        raise FileNotFoundError(f'no checkpoint weights in {path}')
    missing = [name for name in weights if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f'incomplete checkpoint {spec["id"]}: {missing}')
    return {'snapshot': str(path), 'weight_files': len(weights), 'revision': spec['revision']}


def _worker(args, root, config_path, study, plan, assets, out):
    from .comprehensive_baselines import run_baseline_job
    if args.worker_kind == 'baseline-check':
        selected = [j for j in plan['baseline_jobs'] if (j.get('model') or 'shared') == args.worker_id]
        for job in selected:
            result = run_baseline_job(root, out / 'baselines', job, study, {**assets, 'profile': args.profile})
            print(json.dumps({k: v for k, v in result.items() if k in
                             ['job', 'ready', 'missing', 'expected', 'complete', 'output_dir']}), flush=True)
            if not result.get('ready', False): return 2
        return 0
    if args.worker_kind == 'dependencies':
        from importlib.metadata import version
        from akr_closing.model_factory import backend_class, validate_backend_environment
        spec = study['config']['models'][args.worker_id]
        cls = backend_class(spec['id'])
        result = {**snapshot_check(spec), 'backend': cls.__name__,
                  'python': sys.version.split()[0],
                  'packages': validate_backend_environment(spec['id'])}
        if args.worker_id == 'glm4voice9b':
            result['audio_tokenizer'] = snapshot_check({
                'id': 'THUDM/glm-4-voice-tokenizer',
                'revision': 'a5f2404e63c84e92f5238908e1706316324ebafa'})
        atomic_json(out / 'dependencies' / f'{args.worker_id}.json', result)
    elif args.worker_kind == 'akr':
        tasks = selected_tasks(study['config'], [args.worker_id], enable_conditional=args.enable_conditional)
        runner = RebuttalStudy(root, config_path, args.run_id + '-' + args.worker_id, tasks, args.profile)
        result = runner.run(execute=args.action == 'run')
    else:
        job = next(j for j in plan['baseline_jobs'] if j['id'] == args.worker_id)
        result = run_baseline_job(root, out / 'baselines', job, study,
                                  {**assets, 'profile': args.profile}, execute=args.action == 'run')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ['rows', 'jobs', 'support_ids', 'query_ids', 'execution_environment']}, default=str), flush=True)
    if args.worker_kind == 'baseline' and not result.get('ready', True):
        return 2
    if args.action == 'run' and args.worker_kind != 'dependencies' and not result.get('complete', False):
        return 1
    return 0


def report(root, out, plan, study, assets, run_id):
    from .comprehensive_baselines import report_baseline_job
    jobs = []
    if 'akr' in plan['sections']:
        for tag in plan['models']:
            directory = root / 'results/akr_rebuttal' / (run_id + '-' + tag)
            if (directory / 'STUDY.json').is_file():
                expected = build_study(root, Path(plan['config_path']),
                    selected_tasks(study['config'], [tag], enable_conditional=plan['conditional_models_enabled']))
                if digest(json.loads((directory / 'STUDY.json').read_text())) != digest(expected):
                    raise RuntimeError('saved AKR study differs from the current registration')
                for name, dataset in expected['datasets'].items():
                    hashes_path = directory / name / 'WAVEFORM_HASHES.json'
                    if not hashes_path.is_file():
                        continue
                    hashes = json.loads(hashes_path.read_text())
                    lookup = {r['event_id']: r for r in dataset['plan']['rows']}
                    if any(file_hash(Path(lookup[i]['audio_path'])) != sha for i, sha in hashes.items()):
                        raise RuntimeError('AKR waveform changed since execution')
                result = report_study(directory)
                jobs.append({'id': 'akr-' + tag, 'complete': result['complete'],
                             'completed_jobs': result['completed_jobs'], 'expected_jobs': result['expected_jobs']})
            else:
                jobs.append({'id': 'akr-' + tag, 'complete': False, 'reason': 'not started'})
    for job in plan['baseline_jobs']:
        result = report_baseline_job(root, out / 'baselines', job, study, assets)
        jobs.append({'id': job['id'], **result})
    result = {'complete': bool(jobs) and all(j.get('complete', False) for j in jobs),
              'completed_groups': sum(j.get('complete', False) for j in jobs),
              'expected_groups': len(jobs), 'jobs': jobs}
    atomic_json(out / 'COVERAGE.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'check', 'run', 'report'], default='plan', nargs='?')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, default=Path('configs/akr_comprehensive.yaml'))
    parser.add_argument('--run-id', default='comprehensive_v1')
    parser.add_argument('--models', default='qwen7b,qwen3b')
    parser.add_argument('--sections', default='all')
    parser.add_argument('--profile', choices=['16gb', '24gb', '48gb', '80gb'], default='80gb')
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--model-python', action='append', default=[], metavar='TAG=PATH')
    parser.add_argument('--enable-conditional', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--budget-hours', type=float, default=24.)
    parser.add_argument('--worker-kind', choices=['dependencies', 'akr', 'baseline', 'baseline-check'], help=argparse.SUPPRESS)
    parser.add_argument('--worker-id', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.action == 'run' and not args.execute:
        parser.error('run requires --execute; plan/check do not load model weights')
    if not math.isfinite(args.budget_hours) or not 0 < args.budget_hours <= 24:
        parser.error('budget-hours must be in (0,24]')
    root = args.root.resolve()
    validate_run_id(args.run_id)
    config_path = args.config if args.config.is_absolute() else root / args.config
    models = args.models.split(',')
    sections = ['akr', 'baselines', 'diagnostic'] if args.sections == 'all' else args.sections.split(',')
    assets_path = None if args.assets is None else (args.assets if args.assets.is_absolute() else root / args.assets)
    assets = {} if assets_path is None else yaml.safe_load(assets_path.read_text())
    study, plan = make_plan(root, config_path, models, sections, args.enable_conditional, assets)
    out = root / 'results/akr_comprehensive' / args.run_id
    if args.action == 'plan':
        print(json.dumps(plan, indent=2, default=str)); return 0
    if args.worker_kind:
        return _worker(args, root, config_path, study, plan, assets, out)
    interpreters = {tag: MODEL_SPECS[tag]['python'] for tag in models}
    for value in args.model_python:
        tag, path = value.split('=', 1)
        if tag not in models: raise ValueError('interpreter override for unselected model')
        interpreters[tag] = path
    code = {str(p.relative_to(root)): file_hash(p) for directory in ['akr_final', 'akr_closing', 'src/animal_omni']
            for p in (root / directory).glob('*.py')}
    identity = {'plan': digest(plan), 'study': digest(study), 'config': file_hash(config_path), 'code': code,
                'assets': assets, 'interpreters': interpreters, 'profile': args.profile}
    saved = out / 'IDENTITY.json'
    if saved.exists() and json.loads(saved.read_text()) != identity:
        raise RuntimeError('code/config/selection changed; choose a new run ID')
    if args.action == 'report':
        result = report(root, out, plan, study, {**assets, 'profile': args.profile}, args.run_id)
        print(json.dumps(result, indent=2)); return 0 if result['complete'] else 1
    atomic_json(saved, identity)
    atomic_json(out / 'PLAN.json', plan)
    check_jobs = [('dependencies', tag, tag) for tag in models]
    run_jobs = []
    if 'akr' in sections:
        run_jobs += [('akr', tag, tag) for tag in models]
        check_jobs += run_jobs
    groups = list(dict.fromkeys(j.get('model') or 'shared' for j in plan['baseline_jobs']))
    check_jobs += [('baseline-check', group, None if group == 'shared' else group) for group in groups]
    run_jobs += [('baseline', j['id'], j.get('model')) for j in plan['baseline_jobs']]
    deadline = time.monotonic() + args.budget_hours * 3600
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root) + os.pathsep + str(root / 'src')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    status = {}
    with run_lock(root / 'results/akr_closing/.runner.lock'):
        for action in (['check', 'run'] if args.action == 'run' else ['check']):
            for kind, job_id, model in (check_jobs if action == 'check' else run_jobs):
                python = interpreters.get(model, sys.executable)
                if not Path(python).is_file():
                    raise FileNotFoundError(f'missing environment: {python}; run the corresponding setup script')
                command = [python, str(root / 'scripts/run_akr_comprehensive.py'), action,
                           '--root', str(root), '--config', str(config_path), '--run-id', args.run_id,
                           '--models', args.models, '--sections', args.sections, '--profile', args.profile,
                           '--worker-kind', kind, '--worker-id', job_id]
                if args.enable_conditional: command.append('--enable-conditional')
                if action == 'run': command.append('--execute')
                if assets_path: command += ['--assets', str(assets_path)]
                print(f'{action}: {kind}/{job_id}', flush=True)
                code = _bounded(command, root, env, deadline)
                status[f'{action}/{kind}/{job_id}'] = 'passed' if code == 0 else f'exit_{code}'
                atomic_json(out / 'STATUS.json', status)
                if code:
                    atomic_json(out / 'FINAL_STATUS.json', {'complete': False, 'phase': action,
                                'failed_job': job_id, 'exit_code': code})
                    return code
    if args.action == 'check':
        result = {'preflight_complete': True, 'complete': False, 'gpu_experiments_started': False,
                  'checked_groups': len(check_jobs), 'checked_baseline_jobs': len(plan['baseline_jobs'])}
    else:
        result = report(root, out, plan, study, {**assets, 'profile': args.profile}, args.run_id)
    atomic_json(out / 'FINAL_STATUS.json', result)
    print(json.dumps(result, indent=2)); return 0 if args.action == 'check' or result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
