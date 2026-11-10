#!/usr/bin/env python3
"""Is batched greedy generation deterministic for identical inputs?

Runs predict_many twice on the same two real dogs queries (direction=None,
alpha=0) and feature_many twice, comparing signatures and arrays bitwise.
This distinguishes a no-op path difference from CUDA nondeterminism.
"""
import json, sys, hashlib
from pathlib import Path

ROOT = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR')
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np, torch, yaml
from akr_closing.backend import OmniBackend
from akr_closing.core import Query, read_manifest
from akr_closing.pipeline import prepare_audio, prompt_for

WORK = ROOT / 'results/smoke_determinism_20261108'
(WORK / 'audio').mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'results/smoke_determinism_20261108.json'


def main():
    cfg = yaml.safe_load((ROOT / 'configs/akr_train50.yaml').read_text())
    rows = read_manifest(ROOT / cfg['datasets']['dogs']['manifest'], root=ROOT)
    tests = [r for r in rows if r.get('split') == 'test'][:2]
    labels = sorted({r['label'] for r in rows})
    prompt = prompt_for(labels)
    items = []
    for r in tests:
        dst = WORK / 'audio' / (r['event_id'] + '.wav')
        prepare_audio(Path(r['audio_path']), dst, 'full')
        q = Query(r['event_id'], str(dst), r.get('recording_id') or r['event_id'])
        items.append(dict(q=q, prompt=prompt, support=None, direction=None, alpha=0.,
                          scale_mode='relative_cap', raw_eta=300., max_new_tokens=32, scope='audio'))
    model = cfg['models']['qwen7b']; st = cfg['settings']
    backend = OmniBackend(model['id'], revision=model['revision'], profile='80gb',
                          layer_fractions=st['layer_fractions'],
                          feature_fraction=st['feature_fraction'],
                          max_context_tokens=st['max_context_tokens'])

    def sig(rows):
        return [(x.get('prediction'), x.get('raw_prediction'), x.get('generated_tokens')) for x in rows]

    torch.cuda.reset_peak_memory_stats()
    r1 = backend.predict_many(items, labels)
    p1 = int(torch.cuda.max_memory_allocated())
    r2 = backend.predict_many(items, labels)
    p2 = int(torch.cuda.max_memory_allocated())
    same = sig(r1) == sig(r2)
    print('SAME_INPUT_TWICE_EQUAL', same, flush=True)
    print('RUN1', sig(r1), flush=True)
    print('RUN2', sig(r2), flush=True)

    f1 = backend.feature_many([{k: it[k] for k in ('q', 'prompt')} for it in items])
    f2 = backend.feature_many([{k: it[k] for k in ('q', 'prompt')} for it in items])
    feq = all(np.array_equal(a, b) for a, b in zip(f1, f2))
    print('FEATURES_EQUAL', feq, flush=True)
    result = dict(same_input_twice_equal=same, features_equal=feq,
                  predict_peak_bytes=p1, predict_peak_bytes_second=p2,
                  run1=sig(r1), run2=sig(r2),
                  backend_train50_runner=hashlib.sha256((ROOT / 'akr_final/train50_runner.py').read_bytes()).hexdigest(),
                  events=[r['event_id'] for r in tests], batch=len(items))
    OUT.write_text(json.dumps(result, indent=1))
    print('DETERMINISM_SMOKE_DONE', OUT, flush=True)


if __name__ == '__main__':
    main()
