#!/usr/bin/env python3
"""GPU parity smoke test: legacy wrapper versus new direct Thinker, sequential loads.

A failed check stops the run-all launcher. It is not a benchmark result.
"""
from pathlib import Path
import argparse,gc,json,sys
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'src')]

def main():
    import yaml,torch
    from huggingface_hub import snapshot_download
    from akr_closing.core import read_manifest,Query,atomic_json,digest
    from akr_closing.pipeline import prompt_for,prepare_audio
    from akr_closing.backend import OmniBackend
    from animal_omni.qwen_runner import QwenThinkerRunner
    from animal_omni.metrics import normalize_label
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/akr_closing.yaml')
    ap.add_argument('--profile',default='24gb')
    ap.add_argument('--output',type=Path,default=ROOT/'results/akr_backend_parity.json')
    args=ap.parse_args(); cfg=yaml.safe_load(args.config.read_text()); ds=cfg['datasets'][0]
    rows=read_manifest(ROOT/ds['manifest'],old_root=cfg.get('old_root'),root=ROOT)
    records=[next(r for r in rows if r['label']==label and r.get('split','') not in {'test','valid'}) for label in ds['labels']]
    prompt=prompt_for(ds['labels']); queries=[]
    for r in records:
        path=ROOT/'results/akr_parity_audio'/(r['event_id']+'.wav')
        prepare_audio(Path(r['audio_path']),path,'full')
        queries.append(Query(r['event_id'],str(path),r.get('recording_id',r['event_id'])))
    results=[]
    for spec in cfg['models']:
        local=snapshot_download(spec['id'],revision=spec['revision'])
        legacy=QwenThinkerRunner(local)
        old=[normalize_label(legacy.predict(q.audio_path,prompt,max_new_tokens=12),ds['labels']) or '' for q in queries]
        del legacy; gc.collect(); torch.cuda.empty_cache()
        new=OmniBackend(spec['id'],spec['revision'],profile=args.profile,
             layer_fractions=cfg['layer_fractions'],feature_fraction=cfg['feature_fraction'],max_context_tokens=cfg['max_context_tokens'])
        modern=[new.predict(q,prompt,ds['labels'])['prediction'] for q in queries]
        result={'model':spec,'event_ids':[q.event_id for q in queries],'legacy':old,'new':modern,
                'passed':old==modern,'provenance':new.provenance,'scope':'interface parity, not empirical accuracy'}
        results.append(result); del new; gc.collect(); torch.cuda.empty_cache()
    payload={'passed':all(r['passed'] for r in results),'results':results,'config_fingerprint':digest(cfg)}
    atomic_json(args.output,payload)
    if not payload['passed']: raise SystemExit('Backend prediction parity failed. Investigate before comparing new and historical runs.')
    print('Backend parity passed on the declared smoke examples; not a full numerical-equivalence claim.')

if __name__=='__main__': main()
