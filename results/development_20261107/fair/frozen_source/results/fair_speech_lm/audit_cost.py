"""Reconstruct top-level LoRA forward counts missed by PEFT module hooks."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from akr_closing.core import atomic_json,digest,file_hash

out=ROOT/'results/fair_speech_lm'/sys.argv[1]
paths=sorted((out/'training').glob('fold_*/lr_*/DONE.json'))+[out/'training/refit/DONE.json']
if len(paths)!=7: raise RuntimeError('all six CV stages plus refit required')
rows=[]
for path in paths:
    envelope=json.loads(path.read_text()); result=envelope['result']
    if envelope['sha256']!=digest(result) or not result['complete']: raise RuntimeError('invalid training receipt')
    expected=len(result['train_ids'])*result['identity']['max_epoch']
    if result['train_exposures']!=expected or result['history'][-1]['exposures']!=expected:
        raise RuntimeError('training exposure mismatch')
    count=expected+sum(len(v) for v in result['losses'].values())+3
    rows.append(dict(stage=str(path.parent.relative_to(out)),receipt_sha256=file_hash(path),
        original_module_hook_count=result['forward_calls'],reconstructed_top_level_forwards=count,
        train_exposures=expected,heldout_and_native_forwards=sum(len(v) for v in result['losses'].values()),
        parity_forwards=3,optimizer_steps=result['optimizer_steps'],wall_seconds=result['wall_seconds']))
result=dict(complete=True,stages=rows,cv_top_level_forwards=sum(r['reconstructed_top_level_forwards'] for r in rows[:-1]),
    refit_top_level_forwards=rows[-1]['reconstructed_top_level_forwards'],
    explanation='PEFT LoraModel.forward invokes the original model.forward directly, bypassing the original module hook. Saved hook counts undercount wrapped calls. Reconstructed counts use immutable exposure/heldout records plus three parity calls; original receipts are preserved.',
    excludes='checkpoint block recomputation and backward operations; these are not FLOP counts',
    independently_reproduced_on_tiny_peft=True,script_sha256=file_hash(Path(__file__)))
atomic_json(out/'COST_AUDIT.json',result)
print({k:v for k,v in result.items() if k!='stages'})
