"""Calibration-only proxy sweep; cannot select the primary AKR configuration."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.fair_speech_lm import load_registration,FairRunner,support_arrays,read
from akr_closing.core import atomic_json,file_hash
from akr_final.predictability import fit_repository

out=ROOT/'results/fair_speech_lm'/sys.argv[1]
runner=FairRunner(out,'sensitivity_extended',load_registration(out))
e,pool,x,g,y=support_arrays(runner)
rows=read(out/'PARAMETER_SENSITIVITY_PROXY.json')['rows']
for rank in [0,1,2,4,8,16]:
    for ridge in [3000.,10000.,100000.]:
        error=baseline=0.; cos=[]
        for held in e['folds']:
            h=[pool.index(i) for i in held]; t=[j for j in range(len(pool)) if j not in h]
            pred=fit_repository(x[t],g[t],rank=rank,alpha=ridge).predict(x[h])
            error+=float(np.square(pred-g[h]).sum()); baseline+=float(np.square(g[t].mean(0)-g[h]).sum())
            cos.extend((np.sum(pred*g[h],axis=1)/(np.linalg.norm(pred,axis=1)*np.linalg.norm(g[h],axis=1)).clip(1e-30)).tolist())
        rows.append(dict(rank=rank,ridge=ridge,relative_sse=error/baseline,mean_cosine=float(np.mean(cos))))
runner.close()
baseline=next(r for r in rows if r['rank']==4 and r['ridge']==10)
rank_effect=max(r['relative_sse'] for r in rows if r['ridge']==10)-min(r['relative_sse'] for r in rows if r['ridge']==10)
ridge_effect=max(r['relative_sse'] for r in rows if r['rank']==4)-min(r['relative_sse'] for r in rows if r['rank']==4)
result=dict(calibration_only=True,proxy_only=True,used_for_primary_selection=False,rows=rows,
    baseline=baseline,best_proxy=min(rows,key=lambda r:r['relative_sse']),
    one_factor_range_at_baseline={'rank':rank_effect,'ridge':ridge_effect},
    note='Extended upper ridge boundary after initial proxy best was at 1000. No development outcomes used; no change to main search.',
    script_sha256=file_hash(Path(__file__)))
atomic_json(out/'PARAMETER_SENSITIVITY_EXTENDED.json',result)
print({k:v for k,v in result.items() if k!='rows'})
