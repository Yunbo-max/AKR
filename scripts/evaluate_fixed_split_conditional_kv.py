#!/usr/bin/env python3
"""Evaluate support-trained conditional KV on an untouched fixed test split."""
from __future__ import annotations
import argparse,csv,json,logging
from pathlib import Path
import numpy as np,torch,yaml
from animal_omni.conditional_kv import ConditionalGradientRouter,broadcast_audio_delta,flatten_gradient,unflatten_gradient
from animal_omni.kv_hooks import KVDeltaHooks
from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner

def rep(root,e,layer):
 with np.load(root/f'{e}.npz',allow_pickle=False) as x:return x['representation'][layer].astype('float32')
def write(path,rows):
 tmp=path.with_suffix(path.suffix+'.tmp');
 with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 tmp.replace(path)
def main():
 logging.getLogger().setLevel(logging.ERROR)
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--condition',default='lp_0-1000');p.add_argument('--query-split',default='test');p.add_argument('--limit-query',type=int);p.add_argument('--support-sizes',type=int,nargs='+');p.add_argument('--gradient-dir',type=Path,required=True);p.add_argument('--support-split',type=Path,required=True);p.add_argument('--representation-dir',type=Path,required=True);p.add_argument('--model-id',required=True);p.add_argument('--feature-layer',type=int,default=0);p.add_argument('--rank',type=int,default=4);p.add_argument('--alpha',type=float,default=10);p.add_argument('--eta',type=float,default=300);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());labels=cfg['dataset']['labels'];rows=[r for r in csv.DictReader(a.manifest.open()) if r['condition']==a.condition and r['split']==a.query_split];rows=rows[:a.limit_query] if a.limit_query else rows;split=json.loads(a.support_split.read_text());order=split['support_order'];grad={}
 for e in order:grad[e]=torch.load(a.gradient_dir/f'{e}.pt',map_location='cpu',weights_only=True)['pooled_audio_gradient']
 first,keys=flatten_gradient(grad[order[0]]);width=first.size//len(keys);models={}
 for k in (a.support_sizes or split['support_sizes']):
  ev=order[:k];x=np.stack([rep(a.representation_dir,e,a.feature_layer) for e in ev]);g=np.stack([flatten_gradient(grad[e])[0] for e in ev]);models[k]=ConditionalGradientRouter.fit(x,g,rank=a.rank,alpha=a.alpha)
 out=list(csv.DictReader(a.output.open())) if a.resume and a.output.exists() else [];done={(r['event_id'],int(r['support_k']),r['method']) for r in out};runner=QwenThinkerRunner(a.model_id);token=runner.model.thinker.config.audio_token_id;total=len(rows)*len(models)*2
 for row in rows:
  feature=rep(a.representation_dir,row['event_id'],a.feature_layer);inputs=runner.prepare_inputs(row['audio_path'],cfg['evaluation']['prompt']);mask=inputs['input_ids'].eq(token);n=inputs['input_ids'].shape[1]
  for k,router in models.items():
   for method,vector in [('fixed_mean',router.fixed_mean()),('conditional',router.predict(feature))]:
    if (row['event_id'],k,method) in done:continue
    pooled=unflatten_gradient(vector.astype('float32'),keys,width);d=broadcast_audio_delta(pooled,mask,a.eta)
    with torch.inference_mode(),KVDeltaHooks(runner.model.thinker,d):generated=runner.model.generate(**inputs,return_audio=False,do_sample=False,max_new_tokens=cfg['evaluation']['max_new_tokens'],use_audio_in_video=False)
    raw=runner.processor.batch_decode(generated[:,n:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip();z=normalize_label(raw,labels) or '';out.append({'event_id':row['event_id'],'target':row['label'],'support_k':k,'method':method,'eta':a.eta,'rank':router.rank,'prediction':z,'raw_prediction':raw,'correct':str(z==row['label']).lower()});write(a.output,out)
    if len(out)==1 or len(out)%100==0 or len(out)==total:print(f'[{len(out)}/{total}] {row["event_id"]} K={k} {method} -> {z}',flush=True)
if __name__=='__main__':main()
