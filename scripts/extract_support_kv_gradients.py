#!/usr/bin/env python3
"""Extract labeled degraded-audio gradients for nested fixed support sets."""
from __future__ import annotations
import argparse,csv,json,logging,random
from collections import defaultdict
from pathlib import Path
import torch,yaml
from animal_omni.kv_hooks import label_kv_gradients,pooled_audio_gradient,tokenwise_audio_gradient
from animal_omni.qwen_runner import QwenThinkerRunner

def nested_stratified_order(rows,seed):
 rng=random.Random(seed);by=defaultdict(list)
 for r in rows:by[r['label']].append(r)
 labels=sorted(by);rng.shuffle(labels)
 for values in by.values():rng.shuffle(values)
 out=[]
 while any(by.values()):
  for label in labels:
   if by[label]:out.append(by[label].pop())
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--condition',default='lp_0-1000');p.add_argument('--split',default='train');p.add_argument('--model-id',required=True);p.add_argument('--support-sizes',type=int,nargs='+',default=[1,5,10,20]);p.add_argument('--seed',type=int,default=20250813);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--split-output',type=Path,required=True);p.add_argument('--save-tokenwise',action='store_true');p.add_argument('--label-map',type=Path,help='JSON map from dataset labels to arbitrary output strings');p.add_argument('--registered-support',type=Path,help='JSON with a frozen support_sets mapping');p.add_argument('--support-k-per-class',type=int,help='support_sets key used with --registered-support');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());all_rows=[r for r in csv.DictReader(a.manifest.open()) if r['condition']==a.condition];rows=all_rows if a.registered_support else [r for r in all_rows if r.get('split','')==a.split];label_map={label:label for label in cfg['dataset']['labels']}
 if a.label_map:
  label_map=json.loads(a.label_map.read_text())
  if set(label_map)!=set(cfg['dataset']['labels']) or len(set(label_map.values()))!=len(label_map):raise ValueError('label map must bijectively cover configured labels')
  prompt=('Classify this sound into one of the registered arbitrary acoustic categories. '
          f"Choose exactly one label from: {', '.join(label_map[label] for label in cfg['dataset']['labels'])}. "
          'Answer with only the label.')
 else:prompt=cfg['evaluation']['prompt']
 if a.registered_support:
  if a.support_k_per_class is None:raise ValueError('--support-k-per-class is required with --registered-support')
  registered=json.loads(a.registered_support.read_text());support_ids=registered['support_sets'][str(a.support_k_per_class)];by_event={r['event_id']:r for r in rows};missing=set(support_ids)-set(by_event)
  if missing:raise ValueError(f'registered support missing from condition {a.condition}: {sorted(missing)}')
  selected=[by_event[event] for event in support_ids];selection='registered_nested_k_per_class_support';support_sizes=[len(selected)]
 else:
  order=nested_stratified_order(rows,a.seed);selected=order[:max(a.support_sizes)];selection='seeded_label_round_robin_without_query_access';support_sizes=a.support_sizes
 a.output_dir.mkdir(parents=True,exist_ok=True);a.split_output.write_text(json.dumps({'seed':a.seed,'condition':a.condition,'source_split':a.split if not a.registered_support else None,'support_sizes':support_sizes,'support_k_per_class':a.support_k_per_class,'registered_support':str(a.registered_support) if a.registered_support else None,'support_order':[r['event_id'] for r in selected],'labels':[r['label'] for r in selected],'output_labels':[label_map[r['label']] for r in selected],'label_map':label_map,'selection':selection,'gradient_granularity':'tokenwise_and_pooled' if a.save_tokenwise else 'pooled'},indent=2));runner=QwenThinkerRunner(a.model_id);logging.disable(logging.CRITICAL)
 for x in runner.model.parameters():x.requires_grad_(False)
 token=runner.model.thinker.config.audio_token_id
 for i,r in enumerate(selected,1):
  path=a.output_dir/f"{r['event_id']}.pt"
  if path.exists():continue
  output_label=label_map[r['label']];inputs=runner.teacher_forced_inputs(r['audio_path'],prompt,output_label);mask=inputs['input_ids'].eq(token);loss,d=label_kv_gradients(runner.model.thinker,inputs);pooled=pooled_audio_gradient(d,mask);record={'event_id':r['event_id'],'condition':a.condition,'target':r['label'],'target_output':output_label,'label_map':label_map,'base_loss':loss,'pooled_audio_gradient':pooled}
  if a.save_tokenwise:record['tokenwise_audio_gradient']=tokenwise_audio_gradient(d,mask)
  torch.save(record,path);print(f'[{i}/{len(selected)}] {r["event_id"]} {r["label"]} loss={loss:.4f} tokens={int(mask.sum())}',flush=True)
if __name__=='__main__':main()
