"""Registered, resumable baselines; importing/planning never loads a model.

``backend_factory(model_id, revision, **kwargs)`` can replace model loading in
CPU tests. AVES factories provide ``feature(Query, prompt)``; all others provide
``predict(Query, prompt, labels, **kwargs)``. Queries never carry truth labels.
Published specialist/LoRA supervision is external and is NOT support matched.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

from akr_closing.core import Journal, Query, atomic_json, digest, file_hash, metrics, read_manifest
from akr_closing.pipeline import prepare_audio, prompt_for
from akr_closing.repair import RidgeReadout


BZ_MANIFEST = 'data/manifests/beans_zero_targets_fullscan_cap10.csv'
DEFAULT_ASSETS = {
    'aves': 'external/aves/aves-base-bio.onnx',
    'marmaudio_code': 'external/marmaudio/Code_Usage',
    'marmaudio_checkpoint': 'external/marmaudio/Code_Usage/resnet50_logMel128_trainsplit_wo_brown_wo_vocs_epoch1.stdc',
    'marmaudio_labels': 'external/marmaudio/Code_Usage/resnet50_logMel128_trainsplit_wo_brown_wo_vocs_labels_used.tsv',
    'lora_dogs': 'external/adapters/dogs-k2',
    'lora_watkins': 'external/adapters/watkins',
}
_HASH_CACHE = {}


def _environment():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ['torch', 'transformers', 'accelerate', 'peft', 'numpy', 'scipy',
                 'scikit-learn', 'soundfile', 'librosa', 'onnxruntime', 'onnxruntime-gpu', 'qwen-omni-utils']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable, packages=packages)


def _hash(path):
    """Share hashing across jobs; changed inode/stat identity invalidates cache."""
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = file_hash(path)
    return _HASH_CACHE[key]


def _assets(root, assets):
    values = {**DEFAULT_ASSETS, **assets}
    return {k: (Path(v) if Path(v).is_absolute() else root / v).resolve()
            for k, v in values.items() if k in DEFAULT_ASSETS}


def _beans_rows(root, study):
    rows = read_manifest(root / BZ_MANIFEST, old_root=study['config'].get('old_root'), root=root)
    if len(rows) != 2950:
        raise ValueError('BEANS-Zero must contain exactly 2950 registered IDs')
    if any(not r.get('instruction_text') or not r.get('dataset_name') or not r['label'] for r in rows):
        raise ValueError('BEANS-Zero requires registered instructions, components, and truth')
    return rows


def baseline_jobs(root: Path, study: dict, models: dict, assets: dict, *, include_diagnostic=True) -> list[dict]:
    """Enumerate applicable jobs without inventing splits or reading outcomes.

    AVES/specialist jobs are shared across models on identical episodes. Published
    adapters apply only to the registered Qwen2.5-Omni-7B Dogs/Watkins episodes.
    BEANS-Zero jobs cover every explicitly selected model in ``models``.
    """
    root = Path(root).resolve()
    jobs = []
    for name, dataset in study['datasets'].items():
        if name not in {'marmaudio', 'dogs', 'watkins'} and dataset['plan'].get('dataset',{}).get('partition') != 'beans_zero_supervised':
            continue
        for episode in dataset['plan']['episodes']:
            shared = dict(dataset=name, episode=episode['id'], condition=episode['condition'])
            jobs.append(dict(id=f'aves_ridge__{name}__{episode["id"]}', kind='aves_ridge',
                             supervision='matched registered support_ids; frozen AVES; fixed ridge', **shared))
            if name == 'marmaudio':
                jobs.append(dict(id=f'marmaudio_specialist__{episode["id"]}', kind='marmaudio_specialist',
                                 supervision='external pretrained specialist; unmatched training supervision', **shared))
            elif name in {'dogs','watkins'} and 'qwen7b' in models and any(
                    j['dataset'] == name and j['model'] == 'qwen7b' and j['episode'] == episode['id']
                    for j in study['jobs']):
                jobs.append(dict(id=f'published_lora__{name}__qwen7b__{episode["id"]}',
                                 kind='published_lora', model='qwen7b', model_spec=models['qwen7b'],
                                 supervision='published adapter; unmatched external training supervision', **shared))
    if not include_diagnostic:
        return jobs
    registration = {}
    if (root / BZ_MANIFEST).is_file():
        rows = _beans_rows(root, study)
        registration = dict(registered_manifest_sha256=_hash(root / BZ_MANIFEST),
                            registered_ids_sha256=digest([r['event_id'] for r in rows]))
    for tag, spec in models.items():
        jobs.append(dict(id=f'beans_zero__{tag}', kind='beans_zero', dataset='beans_zero',
                         model=tag, model_spec=spec, condition='full',
                         supervision='instruction diagnostic; no fitted support or invented AKR split',
                         **registration))
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate baseline job IDs')
    return jobs


def _population(root, job, study):
    if job['kind'] == 'beans_zero':
        rows = _beans_rows(root, study)
        if job.get('registered_manifest_sha256', _hash(root / BZ_MANIFEST)) != _hash(root / BZ_MANIFEST):
            raise RuntimeError('registered BEANS-Zero manifest changed')
        ids = [r['event_id'] for r in rows]
        if job.get('registered_ids_sha256', digest(ids)) != digest(ids):
            raise RuntimeError('registered BEANS-Zero IDs changed')
        return rows, [], ids, [], {str(root / BZ_MANIFEST): _hash(root / BZ_MANIFEST)}
    plan = study['datasets'][job['dataset']]['plan']
    matches = [e for e in plan['episodes'] if e['id'] == job['episode']]
    if len(matches) != 1 or matches[0]['condition'] != job['condition']:
        raise ValueError('job does not match a registered episode/condition')
    ep = matches[0]
    support, query = list(ep['support_ids']), list(ep['query_ids'])
    rows, labels = plan['rows'], list(plan['labels'])
    if not query or not support or len(set(query)) != len(query) or len(set(support)) != len(support):
        raise ValueError('nonempty unique registered support/query IDs required')
    if set(support) & set(query):
        raise ValueError('registered support/query overlap')
    lookup = {r['event_id']: r for r in rows}
    if len(lookup) != len(rows) or not set(support + query) <= set(lookup):
        raise ValueError('duplicate or unknown registered event IDs')
    if any(lookup[i]['label'] not in labels for i in support + query):
        raise ValueError('registered truth outside label vocabulary')
    if job['kind'] == 'published_lora':
        if job['model_spec']['id'] != 'Qwen/Qwen2.5-Omni-7B' or not any(
                j['dataset'] == job['dataset'] and j['model'] == 'qwen7b'
                and j['episode'] == job['episode'] for j in study['jobs']):
            raise ValueError('published LoRA requires corresponding registered Qwen7B queries')
    used = support + query if job['kind'] == 'aves_ridge' else query
    return [lookup[i] for i in used], support if job['kind'] == 'aves_ridge' else [], query, labels, plan.get('source_hashes', {})


def _required_assets(job, assets):
    kind = job['kind']
    if kind == 'aves_ridge':
        return [assets['aves']]
    if kind == 'marmaudio_specialist':
        return [assets['marmaudio_checkpoint'], assets['marmaudio_labels'],
                assets['marmaudio_code']/'marmaudio/classifier.py',
                assets['marmaudio_code']/'marmaudio/filterbank.py']
    if kind == 'published_lora':
        directory = assets['lora_' + job['dataset']]
        return [directory / name for name in ['adapter_config.json', 'adapter_model.safetensors', 'task_config.yaml']]
    if kind == 'beans_zero':
        return []
    raise ValueError('unknown baseline kind')


def _specialist_labels(path):
    with path.open(encoding='utf-8') as stream:
        labels = list(dict.fromkeys(r['label'] for r in csv.DictReader(stream, delimiter='\t')))
    return ['Infant Cry' if x == 'Infant' else x for x in labels]


class _AVES:
    def __init__(self, checkpoint):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(checkpoint), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def feature(self, query, prompt):
        import soundfile as sf
        waveform, rate = sf.read(query.audio_path, always_2d=True, dtype='float32')
        if rate != 16000:
            raise ValueError('AVES requires prepared 16 kHz waveform')
        frames = self.session.run(None, {self.input_name: waveform.mean(1)[None]})[0]
        return frames.mean(1)[0].astype(np.float32)


class _Specialist:
    def __init__(self, assets):
        import torch
        sys.path.insert(0, str(assets['marmaudio_code']))
        from marmaudio.classifier import get_resnet50
        from marmaudio.filterbank import STFT, MelFilter, Log1p
        self.torch = torch
        self.labels = _specialist_labels(assets['marmaudio_labels'])
        self.model = torch.nn.Sequential(torch.nn.Sequential(
            STFT(1024, 368), MelFilter(96000, 1024, 128, 1000, 48000), Log1p(7)),
            get_resnet50(len(self.labels)))
        self.model.load_state_dict(torch.load(assets['marmaudio_checkpoint'], map_location='cpu', weights_only=True))
        self.model.eval()

    def predict(self, query, prompt, labels, **kwargs):
        import soundfile as sf
        signal, rate = sf.read(query.audio_path, always_2d=True)
        if rate != 96000:
            raise ValueError('specialist requires original 96 kHz audio')
        signal = signal.mean(1)
        signal = (signal - signal.mean()) / max(signal.std(), 1e-12)
        with self.torch.inference_mode():
            logits = self.model(self.torch.tensor(signal[None], dtype=self.torch.float32))
        return {'prediction': self.labels[int(logits.argmax(-1).item())]}


def _backend(job, assets, settings, factory):
    if job['kind'] == 'aves_ridge':
        return (factory('AVES-bio-ONNX', _hash(assets['aves']), checkpoint=str(assets['aves']))
                if factory else _AVES(assets['aves']))
    if job['kind'] == 'marmaudio_specialist':
        return (factory('MarmAudio-specialist', _hash(assets['marmaudio_checkpoint']), assets=assets)
                if factory else _Specialist(assets))
    if factory is None:
        from akr_closing.model_factory import create_backend
        factory = create_backend
    kwargs = {k: settings[k] for k in ['layer_fractions', 'feature_fraction', 'max_context_tokens'] if k in settings}
    model = job['model_spec']
    backend = factory(model['id'], model['revision'], profile=settings.get('profile', '80gb'), **kwargs)
    if job['kind'] == 'published_lora':
        from peft import PeftModel
        backend.thinker = PeftModel.from_pretrained(backend.thinker, str(assets['lora_' + job['dataset']]),
                                                   is_trainable=False)
        backend.thinker.eval()
    return backend


def _query(row, directory, job):
    """Materialize protocol audio without independent amplitude normalization."""
    import soundfile as sf
    target = directory / (digest(row['event_id']) + '.wav')
    # Derived audio is cheap and may have been left partial by an interruption.
    # Always regenerate it before a fresh model call; source identity is locked.
    if target.exists():
        target.unlink()
    if job['kind'] == 'marmaudio_specialist':
        if job['condition'] == 'full':
            target = Path(row['audio_path'])
        elif not target.exists():
            from animal_omni.audio import Intervention, apply_intervention
            signal, rate = sf.read(row['audio_path'], always_2d=True)
            if rate != 96000:
                raise ValueError('specialist requires original 96 kHz audio')
            signal = apply_intervention(signal.mean(1), rate, Intervention('lowpass', high_hz=1000))
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix('.tmp.wav')
            sf.write(tmp, signal, rate, subtype='FLOAT')
            tmp.replace(target)
    elif job['kind'] == 'beans_zero':
        target = Path(row['audio_path'])
    else:
        prepare_audio(row['audio_path'], target, job['condition'])
    return Query(row['event_id'], str(target), row.get('recording_id', ''))


def _canonical(text):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split())


def _journal_status(directory, fingerprint, ids, rows):
    if not (directory/'manifest.json').exists():
        return {'complete': False, 'completed': 0, 'expected': len(ids)}, None
    journal = Journal(directory, fingerprint, ids)
    lookup = {r['event_id']: r for r in rows}
    for key in ids:
        if journal.has(key):
            result = json.loads(journal._path(key).read_text())['result']
            if result.get('event_id') != key or result.get('target') != lookup[key]['label']:
                raise RuntimeError('prediction identity/truth changed')
            if not isinstance(result.get('prediction'), str):
                raise RuntimeError('prediction output is missing')
    return journal.status(), journal


def _scores(job, results, labels):
    if job['kind'] != 'beans_zero':
        return {'metrics': metrics([r['target'] for r in results], [r['prediction'] for r in results], labels)}
    return {
        'metrics': {'n': len(results), 'exact_match': float(np.mean(
            [r['prediction'] == _canonical(r['target']) for r in results]))},
        'components': {name: {'n': sum(r['dataset_name'] == name for r in results),
            'exact_match': float(np.mean([r['prediction'] == _canonical(r['target'])
                                         for r in results if r['dataset_name'] == name]))}
            for name in sorted({r['dataset_name'] for r in results})},
    }


def _run_baseline_job(root: Path, out: Path, job: dict, study: dict, assets: dict, *,
                      execute: bool = False, backend_factory=None, limit: int | None = None,
                      report_only: bool = False) -> dict:
    """Preflight or execute one job. ``limit`` is isolated, non-scientific smoke.

    Reports inspect actual per-query journal items, never trust an old summary.
    Input audio, assets, prompts, labels, config and implementation are locked.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    if report_only and execute:
        raise ValueError('report-only environment handling cannot execute models')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', job['id']):
        raise ValueError('unsafe baseline job ID')
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError('limit must be a positive integer')
    directory = (out/'smoke'/f'limit_{limit}' if limit is not None else out) / job['id']
    resolved_assets = _assets(root, assets)
    required = _required_assets(job, resolved_assets)
    if job['kind'] == 'beans_zero':
        required.append(root / BZ_MANIFEST)
    missing = [str(p) for p in required if not p.is_file()]
    base = dict(job=job['id'], kind=job['kind'], dataset=job['dataset'], output_dir=str(directory),
                supervision=job['supervision'], smoke=limit is not None, complete=False)
    if missing:
        if execute:
            raise FileNotFoundError('missing baseline assets: ' + ', '.join(missing))
        return dict(base, ready=False, missing=missing, completed=0, expected=0)
    rows, support_ids, query_ids, labels, source_hashes = _population(root, job, study)
    missing = [r['audio_path'] for r in rows if not Path(r['audio_path']).is_file()]
    missing += [str(p) for p in source_hashes if not Path(p).is_file()]
    if missing:
        if execute:
            raise FileNotFoundError('missing registered input: ' + ', '.join(missing[:8]))
        return dict(base, ready=False, missing=missing, completed=0, expected=len(query_ids))
    for path, expected in source_hashes.items():
        if _hash(path) != expected:
            raise RuntimeError('registered source changed: ' + str(path))
    if job['kind'] != 'beans_zero' and study['datasets'][job['dataset']]['plan'].get('dataset',{}).get('partition') == 'beans_zero_supervised':
        if any(r.get('sha256') != _hash(r['audio_path']) for r in rows):
            raise RuntimeError('BEANS-Zero waveform changed since supervised registration')
    prompt = prompt_for(labels) if labels else ''
    max_new_tokens = 32 if job['kind'] == 'beans_zero' else study['config']['settings'].get('max_new_tokens', 12)
    if job['kind'] == 'published_lora':
        import yaml
        adapter = resolved_assets['lora_' + job['dataset']]
        task = yaml.safe_load((adapter/'task_config.yaml').read_text())
        adapter_cfg = json.loads((adapter/'adapter_config.json').read_text())
        if set(task['dataset']['labels']) != set(labels) or task['model']['id'] != job['model_spec']['id']:
            raise ValueError('published adapter task/base model or label vocabulary mismatch')
        if adapter_cfg.get('peft_type') != 'LORA' or 'Qwen2.5-Omni-7B' not in adapter_cfg.get('base_model_name_or_path', ''):
            raise ValueError('published adapter is not a Qwen7B Thinker LoRA')
        prompt = task['evaluation']['prompt']
        max_new_tokens = task['evaluation'].get('max_new_tokens', max_new_tokens)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError('positive integer max_new_tokens required')
    if job['kind'] == 'marmaudio_specialist':
        import soundfile as sf
        if not set(labels) <= set(_specialist_labels(resolved_assets['marmaudio_labels'])):
            raise ValueError('specialist training label vocabulary does not cover registered labels')
        if any(sf.info(r['audio_path']).samplerate != 96000 for r in rows):
            raise ValueError('specialist requires original 96 kHz audio')
    settings = {**study['config']['settings'], 'profile': assets.get('profile', '80gb')}
    fingerprint = dict(version=1, job=job, settings=settings, config_sha256=study.get('config_sha256'),
                       source_hashes=source_hashes, rows=rows, support_ids=support_ids, query_ids=query_ids,
                       labels=labels, prompt=prompt, limit=limit, max_new_tokens=max_new_tokens,
                       audio={r['audio_path']: _hash(r['audio_path']) for r in rows},
                       assets={str(p): _hash(p) for p in required},
                       environment=_environment(),
                       code={str(p): _hash(p) for p in [Path(__file__),
                             *sorted((Path(__file__).parents[1]/'akr_closing').glob('*.py')),
                             Path(__file__).parents[1]/'src/animal_omni/audio.py',
                             Path(__file__).parents[1]/'src/animal_omni/metrics.py']})
    if report_only and (directory/'predictions/manifest.json').is_file():
        saved = json.loads((directory/'predictions/manifest.json').read_text())
        fingerprint['environment'] = saved['provenance']['environment']
    selected = query_ids if limit is None else query_ids[:limit]
    status, journal = _journal_status(directory/'predictions', fingerprint, selected, rows)
    report = dict(base, ready=True, missing=[], expected=len(query_ids), completed=status['completed'],
                  processed_expected=len(selected), processed_complete=status['complete'],
                  support_ids=support_ids, query_ids=query_ids, fingerprint=digest(fingerprint),
                  execution_environment=fingerprint['environment'])
    report['complete'] = status['complete'] and limit is None
    if not execute:
        if status['complete']:
            report.update(_scores(job, journal.results(), labels))
        return report
    if journal is None:
        journal = Journal(directory/'predictions', fingerprint, selected)
    lookup = {r['event_id']: r for r in rows}
    pending = [i for i in selected if not journal.has(i)]
    if pending:
        backend = _backend(job, resolved_assets, settings, backend_factory)
        if job['kind'] == 'aves_ridge':
            features = Journal(directory/'features', fingerprint, support_ids + selected)
            for event in support_ids + selected:
                if not features.has(event):
                    q = _query(lookup[event], directory/'audio', job)
                    vector = np.asarray(backend.feature(q, ''), dtype=np.float64)
                    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                        raise ValueError('AVES returned invalid feature vector')
                    features.add(event, dict(event_id=event, feature=vector.tolist()))
            vectors = {r['event_id']: np.asarray(r['feature']) for r in features.results()}
            readout = RidgeReadout.fit(np.stack([vectors[i] for i in support_ids]),
                                      [labels.index(lookup[i]['label']) for i in support_ids], len(labels),
                                      alpha=settings['ridge_alpha'])
        for event in pending:
            row = lookup[event]
            if job['kind'] == 'aves_ridge':
                result = {'prediction': labels[int(readout.predict(vectors[event])[0])]}
            else:
                q = _query(row, directory/'audio', job)
                result = backend.predict(q, row['instruction_text'] if job['kind'] == 'beans_zero' else prompt,
                                         labels, max_new_tokens=max_new_tokens)
            if job['kind'] == 'beans_zero':
                raw = result.get('raw_prediction')
                if not isinstance(raw, str):
                    raise ValueError('instruction diagnostic requires raw model output')
                result = dict(prediction=_canonical(raw), raw_prediction=raw,
                              canonical_target=_canonical(row['label']), dataset_name=row['dataset_name'],
                              instruction_text=row['instruction_text'])
            if not isinstance(result.get('prediction'), str):
                raise ValueError('missing baseline prediction')
            journal.add(event, dict(result, event_id=event, target=row['label']))
    results = journal.results()
    report.update(completed=len(results), processed_complete=True, complete=limit is None)
    report.update(_scores(job, results, labels))
    atomic_json(directory/'summary.json', report)
    return report


def run_baseline_job(root: Path, out: Path, job: dict, study: dict, assets: dict, *,
                     execute: bool = False, backend_factory=None, limit: int | None = None) -> dict:
    """Preflight/execute one registered job; resume requires identical environment.

    ``limit`` isolates a smoke run beneath ``out/smoke`` and never marks the full
    job complete. Injected factories use ``factory(model_id, revision, **kwargs)``.
    """
    return _run_baseline_job(root, out, job, study, assets, execute=execute,
                             backend_factory=backend_factory, limit=limit)


def report_baseline_job(root: Path, out: Path, job: dict, study: dict, assets: dict) -> dict:
    """Read-only validation using the recorded execution environment.

    The reporting interpreter may differ from the worker's isolated environment;
    assets, audio, code, config, query identities and outputs must still match.
    """
    return _run_baseline_job(root, out, job, study, assets, report_only=True)
