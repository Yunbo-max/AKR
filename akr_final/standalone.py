"""Start F1/F2/F3 from raw data and a fixed YAML; no old RUN/LOCK inputs.

This creates a NEW fixed-configuration evaluation on previously inspected data.
It does not recover old selections or invent historical experiment metadata.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import signal

import numpy as np
import yaml
from akr_closing.core import digest, file_hash, make_split, read_manifest, support_ids
from akr_closing.pipeline import prompt_for


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def build_plan(root: Path, config_path: Path) -> tuple[dict, dict]:
    """Deterministic fresh plan; never searches for source-run artifacts."""
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('Standalone YAML must contain a mapping')
    if raw.get('dataset') != 'marmaudio' or raw.get('group_key') != 'recording_id':
        raise ValueError('Standalone scope is MarmAudio with real recording_id groups')
    if raw.get('conditions') != ['full', 'lp:1000']:
        raise ValueError('conditions must be [full, lp:1000]')
    seeds = raw.get('seeds', [])
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(isinstance(s,bool) or not isinstance(s,int) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Use one to three distinct, nonnegative fixed seeds')
    k = _positive_int(raw['k_per_class'], 'k_per_class')
    rank = _positive_int(raw['fixed_settings']['rank'], 'rank')
    alpha, penalty = float(raw['fixed_settings']['alpha']), float(raw['fixed_settings']['ridge_alpha'])
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Fixed alpha must be finite/nonnegative and ridge_alpha positive')
    labels = raw['labels']
    if not isinstance(labels,list) or not labels or len(labels)!=len(set(labels)) or not all(isinstance(l,str) and l for l in labels):
        raise ValueError('Nonempty distinct label strings required')
    ranks = raw.get('ranks', [1,2,4,8])
    if not isinstance(ranks,list) or not ranks or len(set(ranks)) != len(ranks):
        raise ValueError('Distinct diagnostic ranks required')
    for r in ranks: _positive_int(r,'diagnostic rank')
    folds = _positive_int(raw.get('folds',5),'folds')
    if folds < 2: raise ValueError('folds must be >=2')
    for name, expected in [('model_7b','Qwen/Qwen2.5-Omni-7B'),('model_3b','Qwen/Qwen2.5-Omni-3B')]:
        m=raw[name]
        if m['id'] != expected or not isinstance(m.get('revision'),str) or not m['revision']:
            raise ValueError('Only the existing Qwen2.5-Omni 7B/3B backend is supported; provide model revisions')
    fractions=raw.get('layer_fractions',[.25,.5,.75,.875])
    fraction=raw.get('feature_fraction',.875)
    if not fractions or any(not np.isfinite(f) or not 0 <= f <= 1 for f in fractions) or not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Layer fractions must be finite values in [0,1]')
    context_limit=_positive_int(raw.get('max_context_tokens',32768),'max_context_tokens')
    manifest=Path(raw['manifest']);manifest=manifest if manifest.is_absolute() else root/manifest
    rows=read_manifest(manifest,old_root=raw.get('old_root'),root=root)
    rows=sorted(rows,key=lambda r:r['event_id'])
    if set(r['label'] for r in rows)!=set(labels) or any(not r.get('recording_id') for r in rows):
        raise ValueError('Manifest needs the declared labels and real recording groups')
    lookup={r['event_id']:r for r in rows}
    settings={'rank':rank,'alpha':alpha,'ridge_alpha':penalty}
    ds={'name':'marmaudio','manifest':str(manifest),'labels':labels,'group_key':'recording_id'}
    cfg={'datasets':[ds],'seeds':seeds,'models':[{'tag':'qwen7b',**raw['model_7b']}],
         'E9':deepcopy(raw['model_3b']),'layer_fractions':fractions,'feature_fraction':fraction,
         'max_context_tokens':context_limit,'ridge_alpha':penalty,'old_root':raw.get('old_root')}
    episodes=[];splits={}
    for seed in seeds:
        split=make_split(rows,labels,seed=seed,group_key='recording_id')
        sid=support_ids(rows,split['train'],labels,k,seed=seed)
        qids=split['confirmation']
        sg={lookup[i]['recording_id'] for i in sid};qg={lookup[i]['recording_id'] for i in qids}
        if sg & qg or set(sid)&set(qids):raise ValueError('Support/query recording overlap')
        if len(sg)<2 or len(qids)<2:raise ValueError('Need at least two support groups and two queries')
        splits[str(seed)]=split
        for condition in raw['conditions']:
            context=f'qwen7b/marmaudio/s{seed}/semantic/{condition.replace(":","_")}'
            spec={'selection':settings,'support':sid,'prompt':prompt_for(labels),
                  'scale_type':'relative_active_state_norm',
                  'status':'standalone_fixed_configuration_not_validation_selected',
                  'settings_origin':'fixed_yaml_not_historical_LOCK'}
            # Internal legacy member names let the SAME numerical F1/F2/F3 code run.
            # No file named RUN.json, EPISODE.json or LOCK.json is fabricated/read.
            meta={'context':context,'condition':condition,'k_per_class':k,
                  'support_ids':sid,'support_labels':[lookup[i]['label'] for i in sid],
                  'split_fingerprint':split['fingerprint'],'settings_origin':spec['settings_origin']}
            episodes.append({'seed':seed,'condition':condition,'context':context,
                             'source_base':None,'support_ids':sid,'query_ids':qids,
                             'lock':spec,'meta':meta,'lock_sha256':digest(spec),
                             'id':f's{seed}_{condition.replace(":","_")}'})
    options={'dataset':'marmaudio','label_mode':'semantic','k_per_class':k,'max_seeds':len(seeds),
             'conditions':raw['conditions'],'anchor_condition':'lp:1000','ranks':ranks,'folds':folds}
    plan={'mode':'standalone','config':cfg,'dataset':ds,'labels':labels,'rows':rows,'episodes':episodes,
          'splits':splits,'fixed_settings':settings,
          'source_hashes':{str(config_path):file_hash(config_path),str(manifest):file_hash(manifest)},
          'selection_uses_outcomes':False,'historical_metadata_recovered':False,
          'query_partition':'new deterministic confirmation split; previously inspected source data',
          'evaluation_status':'standalone fixed-configuration re-evaluation; not an old-run continuation or untouched test'}
    return plan, options


def main(argv=None):
    from akr_final import runner as runtime
    from akr_closing.cli import run_lock
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--config',type=Path)
    p.add_argument('--run-id',default='akr_standalone_v1')
    p.add_argument('--tasks',default='F1,F2,F3')
    p.add_argument('--profile',choices=['16gb','24gb','48gb'],default='24gb')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--cache-only',action='store_true',help='Reuse this standalone run only; never load a model')
    args=p.parse_args(argv);root=args.root.resolve()
    config=args.config or root/'configs/akr_standalone.yaml'
    if not config.is_absolute():config=root/config
    plan,options=build_plan(root,config)
    signal.signal(signal.SIGINT,runtime._stop);signal.signal(signal.SIGTERM,runtime._stop)
    r=runtime.FinalRunner(root,root/'results/.no_historical_run_required',args.run_id,options,args.profile,
                          cache_only=args.cache_only,standalone_plan=plan)
    try:
        with run_lock(root/'results/akr_closing/.runner.lock'):
            r.run(args.tasks.split(','),args.execute)
    except InterruptedError:return 130
    finally:r.close()
    print(f'Output: {r.out}')
    return 0
