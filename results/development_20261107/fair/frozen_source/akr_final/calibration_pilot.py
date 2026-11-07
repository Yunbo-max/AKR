"""Single calibration pool pilot; development queries never select parameters."""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import yaml

from akr_closing.core import Query, Journal, atomic_json, digest, file_hash, read_manifest
from akr_closing.pipeline import prompt_for, prepare_audio
from akr_closing.repair import RidgeReadout
from .beans_zero_protocol import canonical_pcm_hash
from .rebuttal_runner import DatasetRunner, array_hash, measured

ALPHAS = (0., .003, .01, .03)
METHODS = ('native', 'zero', 'fixed_relative', 'fixed_cap', 'split_guarded', 'cv_refit', 'ridge')


def choose_alpha(losses):
    """Paired mean CE selection, deterministic ties; no improvement => native."""
    values = {float(a): np.asarray(v, dtype=float) for a, v in losses.items()}
    if 0. not in values or any(v.ndim != 1 or not len(v) for v in values.values()):
        raise ValueError('native and nonempty loss vectors required')
    if len({len(v) for v in values.values()}) != 1:
        raise ValueError('paired loss vectors must have equal length')
    if any(not np.isfinite(a) or a < 0 or not np.isfinite(v).all() for a, v in values.items()):
        raise ValueError('finite alpha and losses required')
    means = {a: float(v.mean()) for a, v in values.items()}
    best = min(means, key=lambda a: (means[a], a))
    accepted = means[0.] - means[best] > 1e-6
    return dict(alpha=best if accepted else 0., accepted=accepted,
                reason='heldout_ce_improved' if accepted else 'no_heldout_ce_improvement',
                mean_ce={str(a): v for a, v in means.items()})


def allocate(rows, labels, seed):
    """Select distinct source/decoded-audio identities before any model call."""
    if len({r['event_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate event IDs')
    groups, audio, selected = set(), set(), {}
    for split in ['train', 'valid']:
        for label in labels:
            choices = sorted((r for r in rows if r['split'] in
                              ([split] if split == 'train' else ['valid', 'validation'])
                              and r['label'] == label),
                             key=lambda r: digest([seed, r['event_id']]))
            ids = []
            for r in choices:
                g, h = r['source_filename'], r['pcm_sha256']
                if not g or not h: raise ValueError('source and audio identities required')
                if g in groups or h in audio: continue
                ids.append(r['event_id']); groups.add(g); audio.add(h)
                if len(ids) == 4: break
            if len(ids) != 4: raise ValueError(f'four distinct {split} examples required: {label}')
            selected[split, label] = ids
    fit = [i for c in labels for i in selected['train', c][:2]]
    cal = [selected['train', c][2] for c in labels]
    guard = [selected['train', c][3] for c in labels]
    pool = fit + cal + guard
    return dict(seed=seed, condition='lp:1000', id=f's{seed}_lp_1000',
                support_ids=pool, fit_ids=fit, calibration_ids=cal, guard_ids=guard,
                query_ids=[i for c in labels for i in selected['valid', c]],
                folds=[fit, cal + guard])


def make_entry(internal, settings, rows, labels, sources, seed=20261107):
    e = allocate(rows, labels, seed)
    fixed = dict(rank=4, ridge_alpha=10., alpha=.01)
    e.update(lock={'selection': fixed, 'support': e['support_ids'], 'prompt': prompt_for(labels)},
             meta={'condition': e['condition'], 'support_ids': e['support_ids']},
             lock_sha256=digest(fixed))
    used = set(e['support_ids'] + e['query_ids'])
    settings = {**settings, 'rank': 4, 'ridge_alpha': 10., 'relative_alpha': .01,
                'raw_eta': 300., 'max_new_tokens': 12, 'pilot_alphas': list(ALPHAS)}
    plan = dict(mode='standalone', config=internal, dataset={'name': 'dogs_pool',
                'partition': 'official_validation_development', 'group_key': 'source_filename'},
                labels=labels, rows=sorted([r for r in rows if r['event_id'] in used], key=lambda r:r['event_id']),
                episodes=[e], source_hashes=sources, selection_uses_outcomes=False,
                evaluation_status='bounded development pilot; official test excluded; one seed')
    return {'plan': plan, 'options': {'dataset': 'dogs_pool', 'settings': settings}}


def prepare(root):
    root = Path(root)
    cfgpath = root/'configs/akr_comprehensive.yaml'
    manifest = root/'data/manifests/beans_dogs_protocol.csv'
    labelpath = root/'configs/beans_dogs.yaml'
    cfg = yaml.safe_load(cfgpath.read_text())
    labels = yaml.safe_load(labelpath.read_text())['dataset']['labels']
    rows = read_manifest(manifest, old_root=cfg.get('old_root'), root=root)
    # Official test audio is never read for this development pilot.
    rows = [r for r in rows if r['split'] in ['train', 'valid', 'validation']]
    for r in rows:
        r['pcm_sha256'] = canonical_pcm_hash(Path(r['audio_path']))
        r['original_embedded_sha256'] = r.get('sha256')
        r['sha256'] = file_hash(Path(r['audio_path']))
    internal = {'models': [{'tag': k, **v} for k,v in cfg['models'].items()],
                'E9': cfg['models']['qwen3b'], **{k:cfg['settings'][k] for k in
                  ['layer_fractions', 'feature_fraction', 'max_context_tokens', 'ridge_alpha']}}
    sources = {str(p):file_hash(p) for p in [cfgpath, manifest, labelpath,
                root/'scripts/pilot_calibration_pool.py',
                root/'docs/superpowers/plans/2026-11-07-calibration-pool-pilot.md']}
    return make_entry(internal, cfg['settings'], rows, labels, sources)


class PilotRunner(DatasetRunner):
    def verify_caches(self):
        # Historical Journal permits missing checksums; this pilot requires them.
        for category in ['loss_cache', 'trials']:
            for path in (self.out/category).glob('**/items/*.json'):
                item = json.loads(path.read_text())
                if not item.get('result_sha256') or digest(item['result']) != item['result_sha256']:
                    raise RuntimeError('missing/mismatched pilot journal checksum')
        for path in (self.out/'audio'/self.options['dataset']).glob('*.wav'):
            if path.name.endswith('.partial.wav'): continue
            meta = Path(str(path)+'.json')
            if not meta.exists() or json.loads(meta.read_text()).get('sha256') != file_hash(path):
                raise RuntimeError('missing/mismatched processed audio checksum')

    def _query(self, event, condition):
        row = self.lookup[event]; source = Path(row['audio_path'])
        identity = dict(source=str(source), hash=self.wave_hashes[event], condition=condition)
        path = self.out/'audio'/self.options['dataset']/(digest(identity)+'.wav')
        meta = Path(str(path)+'.json')
        if path.exists():
            if not meta.exists() or json.loads(meta.read_text()) != dict(identity=identity, sha256=file_hash(path)):
                raise RuntimeError('missing/mismatched processed audio checksum')
        else:
            temporary = path.with_suffix('.partial.wav')
            # A prior interrupted temporary file is safe to regenerate.
            if temporary.exists(): temporary.unlink()
            prepare_audio(source, temporary, condition)
            temporary.replace(path)
            atomic_json(meta, dict(identity=identity, sha256=file_hash(path)))
        return Query(event, str(path), row.get('recording_id') or event)

    def preflight(self):
        super().preflight()
        if any(self.lookup[i].get('sha256') and self.lookup[i]['sha256'] != h
               for i,h in self.wave_hashes.items()):
            raise ValueError('waveform differs from pilot registration')

    def matrix(self, ids, e, tag, kind='feature'):
        values = super().matrix(ids, e, tag, kind)
        if not np.isfinite(values).all(): raise ValueError('nonfinite cached array')
        return values

    def loss(self, e, ids, mapping, alpha):
        if not set(ids) <= set(e['support_ids']) or set(ids) & set(e['query_ids']):
            raise ValueError('loss endpoint restricted to calibration pool')
        fields = None if alpha == 0 else mapping.predict(self.matrix(ids, e, 'qwen3b'))
        ident = dict(run=self.fingerprint, ids=ids, alpha=alpha,
                     fields=None if fields is None else array_hash(fields), mode='relative_cap')
        journal = Journal(self.out/'loss_cache'/digest(ident), ident, ids)
        for j, event in enumerate(ids):
            if journal.has(event): continue
            backend = self._model('qwen3b', e)
            value = measured(backend, lambda: backend.support_answer_loss(
                self._query(event, e['condition']), self.lookup[event]['label'], prompt_for(self.labels),
                direction=None if fields is None else fields[j], alpha=alpha,
                scale_mode='relative_cap', raw_eta=self.settings['raw_eta']))
            if not np.isfinite(value['answer_loss']): raise ValueError('nonfinite answer loss')
            journal.add(event, value)
        result = np.array([r['answer_loss'] for r in journal.results()])
        if not np.isfinite(result).all(): raise ValueError('nonfinite cached answer loss')
        return result

    def run_pilot(self):
        start = time.perf_counter()
        atomic_json(self.out/'STATUS.json', {'complete': False, 'phase': 'verifying'})
        self.verify_caches()
        atomic_json(self.out/'STATUS.json', {'complete': False, 'phase': 'calibration'})
        e, = self.plan['episodes']; pool = e['support_ids']; s = self.settings
        x = self.matrix(pool, e, 'qwen3b'); g = self.matrix(pool, e, 'qwen3b', 'gradient')
        def fit(ids):
            indices = [pool.index(i) for i in ids]
            return self.router_fit(x[indices], g[indices], rank=s['rank'], alpha=s['ridge_alpha'])
        allmap = fit(pool); splitmap = fit(e['fit_ids'])
        split = choose_alpha({a:self.loss(e, e['calibration_ids'], splitmap, a) for a in ALPHAS})
        if split['accepted']:
            native = self.loss(e, e['guard_ids'], splitmap, 0.)
            proposed = self.loss(e, e['guard_ids'], splitmap, split['alpha'])
            split['guard_mean_ce'] = dict(native=float(native.mean()), proposed=float(proposed.mean()))
            if proposed.mean() > native.mean():
                split.update(alpha=0., accepted=False, reason='guard_veto')
        oof = {a:[] for a in ALPHAS}
        folds = e['folds']
        if len(folds) != 2 or set(folds[0]) & set(folds[1]) or set(folds[0]+folds[1]) != set(pool):
            raise ValueError('invalid disjoint folds')
        for held in folds:
            training = [i for i in pool if i not in held]
            mapping = fit(training)
            for a in ALPHAS: oof[a].extend(self.loss(e, held, mapping, a).tolist())
        cv = choose_alpha(oof)
        policies = dict(split_guarded=split, cv_refit=cv)
        lock = {'run':self.fingerprint, 'policies':policies, 'methods':list(METHODS)}
        saved = self.out/'POLICY_LOCK.json'
        if saved.exists() and json.loads(saved.read_text()) != lock:
            raise RuntimeError('changed/corrupt policy lock')
        atomic_json(saved, lock)  # Before query features, generations or scoring.
        selection_seconds = time.perf_counter()-start
        print(json.dumps({'phase':'policy_locked', 'policies':policies}), flush=True)
        fq = self.matrix(e['query_ids'], e, 'qwen3b'); fields = allmap.predict(fq)
        records = {}
        records['native'] = self.trial(e, 'qwen3b', 'native', None, None)
        native = records['native']
        records['zero'] = self.trial(e, 'qwen3b', 'zero', fields, native, alpha=0.)
        self._no_op(native, records['zero'])
        for name, mode in [('fixed_relative','relative_norm'), ('fixed_cap','relative_cap')]:
            records[name] = self.trial(e, 'qwen3b', name, fields, native, mode=mode, labelled_count=len(pool))
        for name, mapping in [('split_guarded',splitmap), ('cv_refit',allmap)]:
            policy = policies[name]
            records[name] = self.trial(e, 'qwen3b', name, mapping.predict(fq), native,
                mode='relative_cap', alpha=policy['alpha'], policy=policy, labelled_count=len(pool))
            if policy['alpha'] == 0: self._no_op(native, records[name])
        y = np.array([self.labels.index(self.lookup[i]['label']) for i in pool])
        ridge = RidgeReadout.fit(x, y, len(self.labels), alpha=s['ridge_alpha'])
        predictions = [self.labels[k] for k in ridge.predict(fq)]
        records['ridge'] = self.trial(e, 'qwen3b', 'ridge', None, native, external=predictions, labelled_count=len(pool))
        rows = [json.loads((self.out/'trials/qwen3b'/e['id']/name/'METRICS.json').read_text()) for name in METHODS]
        # Re-open every loss journal so corrupt or incomplete evidence cannot be ignored.
        costs = []
        for path in sorted((self.out/'loss_cache').glob('*/manifest.json')):
            meta = json.loads(path.read_text()); journal = Journal(path.parent, meta['provenance'], meta['expected'])
            if not journal.complete: raise RuntimeError('incomplete loss journal')
            costs.extend(journal.results())
        cache_costs = [json.loads(p.read_text())['origin'] for p in (self.out/'cache').glob('*.npz.json')]
        result = dict(complete=True, no_op_exact=True, rows=rows, policies=policies,
            predictions={k:[r['prediction'] for r in v] for k,v in records.items()},
            calibration_pool_count=len(pool), development_count=len(e['query_ids']),
            query_labels_used_for_selection=False, official_test_used=False,
            selection_elapsed_seconds_this_invocation=selection_seconds,
            elapsed_seconds_this_invocation=time.perf_counter()-start,
            cached_loss_records=len(costs), loss_forward_calls=sum(r['new_model_calls'] for r in costs),
            loss_seconds=sum(r['wall_seconds'] for r in costs),
            peak_loss_allocated_bytes=max((v for r in costs for v in r['peak_allocated_bytes'].values()), default=0),
            shared_array_records=len(cache_costs),
            shared_array_seconds=sum(r.get('wall_seconds',0) for r in cache_costs),
            shared_feature_records=sum(r['kind']=='new_feature_forward' for r in cache_costs),
            shared_gradient_records=sum(r['kind']=='new_support_backward' for r in cache_costs),
            cost_note='Features/gradients shared across methods; costs are not compute matched. Timings on resume exclude reused work.',
            protocol=self.plan['evaluation_status'], fingerprint=self.fingerprint,
            roles={k:e[k] for k in ['support_ids','fit_ids','calibration_ids','guard_ids','query_ids','folds']})
        atomic_json(self.out/'REPORT.json', result)
        atomic_json(self.out/'STATUS.json', {'complete':True, 'phase':'complete', 'report_sha256':file_hash(self.out/'REPORT.json')})
        return result
