#!/usr/bin/env python3
"""Real-weight batch=2 smoke for the repaired OmniBackend batched paths.

Uses two real dogs test queries from the registered Train-50 manifest and
exercises feature_many + predict_many end to end, recording CUDA peaks.
"""
import json, sys, time
from pathlib import Path

ROOT = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR')
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch, yaml
from akr_closing.backend import OmniBackend
from akr_closing.core import Query, read_manifest
from akr_closing.pipeline import prepare_audio, prompt_for

OUT = ROOT / 'results/smoke_batched_20261108.json'
WORK = ROOT / 'results/smoke_batched_20261108'
(WORK / 'audio').mkdir(parents=True, exist_ok=True)


def main():
    cfg = yaml.safe_load((ROOT / 'configs/akr_train50.yaml').read_text())
    ds = cfg['datasets']['dogs']
    rows = read_manifest(ROOT / ds['manifest'], root=ROOT)
    tests = [r for r in rows if r.get('split') == 'test'][:2]
    assert len(tests) == 2
    labels = sorted({r['label'] for r in rows})
    prompt = prompt_for(labels)
    items = []
    for r in tests:
        dst = WORK / 'audio' / (r['event_id'] + '.wav')
        prepare_audio(Path(r['audio_path']), dst, 'full')
        q = Query(r['event_id'], str(dst), r.get('recording_id') or r['event_id'])
        items.append(dict(q=q, prompt=prompt, support=None, direction=None,
                          alpha=0.3, scale_mode='relative_cap', raw_eta=300.,
                          max_new_tokens=32, scope='audio'))
    model = cfg['models']['qwen7b']
    st = cfg['settings']
    backend = OmniBackend(model['id'], revision=model['revision'], profile='80gb',
                          layer_fractions=st['layer_fractions'],
                          feature_fraction=st['feature_fraction'],
                          max_context_tokens=st['max_context_tokens'])
    result = dict(status='ok', events=[r['event_id'] for r in tests],
                  labels=len(labels), batch=len(items),
                  torch=torch.__version__,
                  backend_sha256=__import__('hashlib').sha256(
                      (ROOT / 'akr_closing/backend.py').read_bytes()).hexdigest())
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    features = backend.feature_many([{k: it[k] for k in ('q', 'prompt')} for it in items])
    result['feature_wall_seconds'] = time.perf_counter() - t0
    result['feature_peak_bytes'] = int(torch.cuda.max_memory_allocated())
    result['feature_dim'] = int(len(features[0]))
    result['feature_count'] = len(features)
    print('SMOKE feature_many OK', result['feature_count'], result['feature_dim'],
          result['feature_peak_bytes'], flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    predictions = backend.predict_many(items, labels)
    result['predict_wall_seconds'] = time.perf_counter() - t0
    result['predict_peak_bytes'] = int(torch.cuda.max_memory_allocated())
    result['predictions'] = [{k: p.get(k) for k in
                              ('prediction', 'raw_prediction', 'generated_tokens',
                               'input_tokens', 'scale_mode', 'raw_eta')} for p in predictions]
    print('SMOKE predict_many OK', result['predictions'], result['predict_peak_bytes'], flush=True)
    assert len(predictions) == 2 and len(features) == 2
    assert result['feature_peak_bytes'] > 0 and result['predict_peak_bytes'] > 0
    OUT.write_text(json.dumps(result, indent=1))
    print('SMOKE_DONE', OUT, flush=True)


if __name__ == '__main__':
    main()
