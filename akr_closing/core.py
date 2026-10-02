"""Protocol, immutable journals and paired statistics for closing experiments.

This module does not load models. All metrics are fractions unless suffixed _pp.
A split is a locked re-evaluation unless truly new data are explicitly documented.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp,path)


@dataclass(frozen=True)
class Query:
    """The inference API intentionally cannot receive ground-truth targets."""
    event_id: str
    audio_path: str
    recording_id: str


def read_manifest(path: Path, *, old_root: str | None=None, root: Path | None=None) -> list[dict]:
    with Path(path).open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    required={'event_id','audio_path','label'}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f'{path}: nonempty CSV with {sorted(required)} required')
    seen=set()
    for r in rows:
        if r['event_id'] in seen: raise ValueError('duplicate event ID; use a one-condition manifest')
        seen.add(r['event_id'])
        p=r['audio_path']
        if old_root and root and (p==old_root or p.startswith(old_root.rstrip('/')+'/')):
            p=str(root/Path(p).relative_to(old_root))
        elif not Path(p).is_absolute(): p=str((root or Path.cwd())/p)
        r['audio_path']=p
        r.setdefault('recording_id',r.get('source_filename',''))
        r.setdefault('split','')
    return rows


def make_split(rows: list[dict], labels: list[str], *, seed: int,
               group_key: str | None='recording_id') -> dict:
    """Keep official train/test, divide valid into selection/confirmation.

    Without official splits, divide whole source-recording groups 55/15/15/15.
    Class coverage of support is enforced separately (never repaired using test).
    Existing group IDs are used as supplied; event IDs are NOT fake recordings.
    """
    rng=np.random.default_rng(seed)
    if set(r['label'] for r in rows) != set(labels): raise ValueError('label vocabulary mismatch')
    official=all(r.get('split') in {'train','valid','validation','test'} for r in rows)
    if official:
        train=[r['event_id'] for r in rows if r['split']=='train']
        test=[r['event_id'] for r in rows if r['split']=='test']
        valid=[r for r in rows if r['split'] in {'valid','validation'}]
        sel=[]; conf=[]
        for label in labels:
            ids=[r['event_id'] for r in valid if r['label']==label]; rng.shuffle(ids)
            if len(ids)<2: raise ValueError('official validation needs >=2 examples per class')
            n=max(1,len(ids)//2); sel+=ids[:n]; conf+=ids[n:]
    else:
        if not group_key or any(not r.get(group_key) for r in rows):
            raise ValueError('non-official split requires real recording-group metadata')
        groups=sorted({r[group_key] for r in rows}); rng.shuffle(groups)
        if len(groups)<8: raise ValueError('need >=8 independent recording groups')
        n=len(groups); a=max(1,int(n*.55)); b=max(a+1,int(n*.70)); c=max(b+1,int(n*.85))
        buckets=[set(groups[:a]),set(groups[a:b]),set(groups[b:c]),set(groups[c:])]
        train,sel,conf,test=([r['event_id'] for r in rows if r[group_key] in bucket] for bucket in buckets)
    out={'train':sorted(train),'selection':sorted(sel),'confirmation':sorted(conf),'test':sorted(test),
         'seed':seed,'evaluation_status':'locked_reevaluation',
         'note':'These source datasets were already inspected. This split is NOT a new untouched benchmark.'}
    sets=[set(out[k]) for k in ['train','selection','confirmation','test']]
    if any(not s for s in sets) or any(sets[i]&sets[j] for i in range(4) for j in range(i+1,4)):
        raise ValueError('empty or overlapping split')
    if not official:
        lookup={r['event_id']:r for r in rows}
        gs=[{lookup[i][group_key] for i in s} for s in sets]
        if any(gs[i]&gs[j] for i in range(4) for j in range(i+1,4)): raise ValueError('group leakage')
    out['fingerprint']=digest(out)
    return out


def support_ids(rows: list[dict], pool: list[str], labels: list[str], k: int, *, seed: int) -> list[str]:
    if k<1: raise ValueError('k is examples per class and must be positive')
    allowed=set(pool); rng=np.random.default_rng(seed); by_class=[]
    for label in labels:
        ids=sorted(r['event_id'] for r in rows if r['event_id'] in allowed and r['label']==label)
        rng.shuffle(ids)
        if len(ids)<k: raise ValueError(f'class {label} has {len(ids)} support examples, needs {k}')
        by_class.append(ids[:k])
    # Interleaved, deterministic, and prefixes are nested as k increases.
    return [by_class[c][j] for j in range(k) for c in range(len(labels))]


class Journal:
    """Per-item atomic JSON avoids malformed append records after interruption."""
    def __init__(self, root: Path, fingerprint: dict, expected: Iterable[str]):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.expected=list(expected)
        if not self.expected or len(set(self.expected))!=len(self.expected):
            raise ValueError('expected keys must be nonempty and unique')
        self.meta={'fingerprint':digest(fingerprint),'provenance':fingerprint,'expected':self.expected}
        path=self.root/'manifest.json'
        if path.exists():
            if json.loads(path.read_text())!=self.meta:
                raise RuntimeError(f'configuration/provenance changed for {root}; choose a new run ID')
        else: atomic_json(path,self.meta)
        self.items=self.root/'items'; self.items.mkdir(exist_ok=True)
    def _path(self,key): return self.items/(digest(key)+'.json')
    def has(self,key):
        path=self._path(key)
        if not path.exists(): return False
        try: value=json.loads(path.read_text())
        except (ValueError,OSError) as exc: raise RuntimeError(f'corrupted journal item {path}') from exc
        if value.get('key')!=key or 'result' not in value: raise RuntimeError(f'misidentified journal item {path}')
        if value.get('result_sha256') and digest(value['result'])!=value['result_sha256']:
            raise RuntimeError(f'journal checksum mismatch {path}')
        return True
    def add(self,key,value):
        if key not in self.expected: raise ValueError('foreign result key')
        if self.has(key): raise ValueError('duplicate result')
        atomic_json(self._path(key),{'key':key,'result':value,'result_sha256':digest(value)})
    @property
    def complete(self): return all(self.has(key) for key in self.expected)
    def results(self):
        if not self.complete: raise RuntimeError('partial runs are not scientific summaries')
        return [json.loads(self._path(key).read_text())['result'] for key in self.expected]
    def status(self):
        done=sum(self.has(k) for k in self.expected)
        return {'complete':done==len(self.expected),'completed':done,'expected':len(self.expected)}


def metrics(target: list[str], prediction: list[str], labels: list[str]) -> dict:
    from sklearn.metrics import f1_score
    if not target or len(target)!=len(prediction): raise ValueError('aligned nonempty predictions needed')
    return {'n':len(target),'accuracy':float(np.mean(np.array(target)==np.array(prediction))),
            'macro_f1':float(f1_score(target,prediction,labels=labels,average='macro',zero_division=0)),
            'balanced_accuracy':float(np.mean([np.mean(np.asarray(prediction)[np.asarray(target)==c]==c) for c in set(target)])),
            'invalid_rate':float(np.mean([p not in labels for p in prediction]))}


def paired_stats(target, baseline, method, groups=None, *, seed=0,n_boot=5000) -> dict:
    from scipy.stats import binomtest
    y=np.array(target); a=(np.array(baseline)==y); b=(np.array(method)==y)
    if not len(y) or len(baseline)!=len(y) or len(method)!=len(y): raise ValueError('aligned predictions needed')
    groups=np.array(groups if groups is not None else [str(i) for i in range(len(y))])

    if len(groups)!=len(y): raise ValueError('group vector length mismatch')
    unique=np.unique(groups); blocks=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(seed); boot=[]; delta=b.astype(float)-a.astype(float)
    for _ in range(n_boot):
        idx=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])
        boot.append(float(delta[idx].mean()*100))
    win=int(np.sum(b&~a)); loss=int(np.sum(a&~b))
    return {'gain_pp':float(delta.mean()*100),'ci95_pp':[float(x) for x in np.quantile(boot,[.025,.975])],
            'mcnemar_p':float(binomtest(win,win+loss,.5).pvalue) if win+loss else 1.,
            'method_only_correct':win,'baseline_only_correct':loss,'n_clusters':len(blocks),
            'mcnemar_note':'Sample-level auxiliary test; cluster bootstrap is primary when recordings repeat.'}


def confirmation_gate(stats: dict, invalid_rate: float) -> bool:
    return stats['gain_pp']>0 and stats['ci95_pp'][0]>0 and invalid_rate<=.01
