"""Outcome-blind official-train sampling and exact official-test registration."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from akr_closing.core import atomic_json, digest, file_hash, read_manifest


def allocate_train_support(rows, labels, budget=50, seed=20261107, group_key='source_filename'):
    """Balanced deterministic prefixes over distinct source/PCM connected groups.

    Test IDs never change to rescue a support allocation. Source identity and
    decoded PCM equivalence are composed transitively, including validation rows.
    Group metadata is used as supplied; this is not a claim of unseen individuals.
    """
    if isinstance(budget,bool) or not isinstance(budget,int) or budget < len(labels):
        raise ValueError('support budget must cover every class')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    if not labels or len(set(labels))!=len(labels) or set(labels)!={r['label'] for r in rows}:
        raise ValueError('label vocabulary mismatch')
    lookup={r['event_id']:r for r in rows}
    if not rows or len(lookup)!=len(rows) or any(not r['event_id'] for r in rows):
        raise ValueError('nonempty unique event identities required')
    if any(r.get('split') not in {'train','valid','validation','test'} for r in rows):
        raise ValueError('explicit official train/validation/test split required')
    if any(not r.get(group_key) or not r.get('pcm_sha256') for r in rows):
        raise ValueError('real source group and decoded PCM identity required')
    parent={i:i for i in lookup}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for r in sorted(rows,key=lambda r:r['event_id']):
        i=r['event_id']
        for token in [('source',r[group_key]),('pcm',r['pcm_sha256'])]:
            if token in seen:
                a,b=find(i),find(seen[token]);parent[max(a,b)]=min(a,b)
            seen[token]=i
    components={}
    for i in sorted(lookup):components.setdefault(find(i),[]).append(i)
    group_ids={i:digest(members) for members in components.values() for i in members}
    query=sorted(i for i,r in lookup.items() if r['split']=='test')
    if not query:raise ValueError('complete official test is empty')
    blocked={group_ids[i] for i in query}
    excluded=sorted(i for i,r in lookup.items() if r['split']=='train' and group_ids[i] in blocked)
    candidates=sorted((r for r in rows if r['split']=='train' and group_ids[r['event_id']] not in blocked),
                      key=lambda r:digest([seed,r['event_id']]))
    counts={c:0 for c in labels};chosen=set();support=[]
    for _ in range(budget):
        selected=None
        for label in sorted(labels,key=lambda c:(counts[c],digest([seed,c]))):
            selected=next((r for r in candidates if r['label']==label and group_ids[r['event_id']] not in chosen),None)
            if selected is not None:break
        if selected is None:raise ValueError(f'insufficient distinct official-train groups for budget {budget}')
        support.append(selected['event_id']);chosen.add(group_ids[selected['event_id']]);counts[selected['label']]+=1
    if any(n==0 for n in counts.values()):raise ValueError('support allocation missing a class')
    if chosen & blocked:raise ValueError('source/PCM overlap across support and test')
    return dict(support_ids=support,query_ids=query,support_class_counts=counts,groups=group_ids,
                excluded_overlap_train_ids=excluded,support_budget=budget,seed=seed,
                complete_official_test=True,query_group_count=len(blocked),
                selection='stratified official train; fixed seed; source/PCM disjoint; no model outcomes',
                grouping_scope=f'supplied {group_key} plus canonical PCM; no stronger identity claim')


def validate_population(expected, records):
    if not expected or len(set(expected))!=len(expected):raise ValueError('invalid registered query population')
    if [r.get('event_id') for r in records]!=list(expected):
        raise ValueError('predictions do not cover the exact registered query population in order')
    if any(not isinstance(r.get('prediction'),str) for r in records):
        raise ValueError('predictions must include strings; invalid model answers use the empty string')


def code_identity(root):
    root=Path(root)
    paths=list((root/'akr_final').glob('*.py'))+list((root/'akr_closing').glob('*.py'))
    paths += [root/'src/animal_omni/conditional_kv.py',root/'src/animal_omni/metrics.py',root/'scripts/fair_speech_lm.py']
    return {str(p.relative_to(root)):file_hash(p) for p in sorted(paths)}


def prepare_registration(root, config, dataset, seed, budgets=(50,), *, config_path=None, audio_root=None):
    """Read actual audio once before model loading; keep all official test rows."""
    from akr_final.beans_zero_protocol import canonical_pcm_hash
    root=Path(root).resolve();ds=config['datasets'][dataset]
    manifest=Path(ds['manifest']);manifest=manifest if manifest.is_absolute() else root/manifest
    rows=read_manifest(manifest,root=root,old_root=config.get('old_root'))
    if audio_root is not None:
        # Relocation is explicit and relative to the configured old root only.
        old=config.get('old_root')
        if not old:raise ValueError('audio-root relocation requires old_root in config')
        original=read_manifest(manifest)
        for row,raw in zip(rows,original):
            p=Path(raw['audio_path'])
            if not p.is_relative_to(old):raise ValueError('audio path is outside declared old_root')
            row['audio_path']=str(Path(audio_root).resolve()/p.relative_to(old))
    for row in rows:
        p=Path(row['audio_path'])
        if not p.is_file():raise FileNotFoundError(f'missing audio: {p}')
        sha=file_hash(p)
        if row.get('sha256') and row['sha256']!=sha:raise ValueError(f'manifest waveform hash mismatch: {row["event_id"]}')
        row['sha256']=sha;row['pcm_sha256']=canonical_pcm_hash(p)
    labels=ds.get('labels') or sorted({r['label'] for r in rows})
    splits={str(n):allocate_train_support(rows,labels,int(n),seed,ds.get('group_key','source_filename')) for n in sorted(set(budgets))}
    if '50' not in splits:raise ValueError('every registration retains the main Train-50 split')
    ntest=sum(r['split']=='test' for r in rows)
    if ds.get('expected_test_count') is not None and ntest!=ds['expected_test_count']:
        raise ValueError(f'{dataset}: official test count {ntest} differs from registered {ds["expected_test_count"]}')
    source={str(manifest):file_hash(manifest)}
    provenance=manifest.with_suffix('.provenance.json')
    if provenance.is_file():source[str(provenance)]=file_hash(provenance)
    if config_path is not None:source[str(Path(config_path).resolve())]=file_hash(Path(config_path))
    return dict(version=1,dataset=dataset,labels=labels,rows=rows,splits=splits,seed=seed,
                source_hashes=source,code=code_identity(root),source_group_key=ds.get('group_key','source_filename'),
                query_ids=splits['50']['query_ids'],complete_official_test=True,
                official_counts=dict(Counter(r['split'] for r in rows)),
                test_history='new registered evaluation of previously available datasets; not globally untouched',
                selection_uses_query_outcomes=False)


def lock_registration(path, value):
    path=Path(path)
    if path.exists() and json.loads(path.read_text())!=value:
        raise RuntimeError('registration changed; use a new run ID')
    atomic_json(path,value)


def validate_registered_inputs(root, registration):
    if code_identity(root)!=registration['code']:raise RuntimeError('registered code changed; use a new run ID')
    for p,sha in registration['source_hashes'].items():
        if not Path(p).is_file() or file_hash(Path(p))!=sha:raise RuntimeError(f'registered source changed: {p}')
    for row in registration['rows']:
        if not Path(row['audio_path']).is_file() or file_hash(Path(row['audio_path']))!=row['sha256']:
            raise RuntimeError(f'registered waveform changed: {row["event_id"]}')
