#!/usr/bin/env python3
"""Select a frozen linear probe on train/valid and report untouched test."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score,f1_score

def main():
 p=argparse.ArgumentParser();p.add_argument('--representation-dir',type=Path,required=True);p.add_argument('--output-predictions',type=Path,required=True);p.add_argument('--output-summary',type=Path,required=True);p.add_argument('--alphas',nargs='+',type=float,default=[.01,.1,1,10,100]);a=p.parse_args()
 rec=[np.load(x,allow_pickle=False) for x in sorted(a.representation_dir.glob('*.npz'))];x=np.stack([z['representation'].astype('float32') for z in rec]);y=np.array([str(z['label']) for z in rec]);s=np.array([str(z['split']) for z in rec]);e=np.array([str(z['event_id']) for z in rec]);tr=s=='train';va=s=='valid';te=s=='test'
 cand=[]
 for layer in range(x.shape[1]):
  for alpha in a.alphas:
   m=make_pipeline(StandardScaler(),RidgeClassifier(alpha=alpha)).fit(x[tr,layer],y[tr]);z=m.predict(x[va,layer]);cand.append((f1_score(y[va],z,average='macro',zero_division=0),layer,alpha))
 score,layer,alpha=max(cand,key=lambda z:(z[0],-z[1],-z[2]));fit=tr|va;m=make_pipeline(StandardScaler(),RidgeClassifier(alpha=alpha)).fit(x[fit,layer],y[fit]);z=m.predict(x[te,layer]);rows=[{'event_id':i,'target':t,'prediction':q,'correct':str(t==q).lower()} for i,t,q in zip(e[te],y[te],z)]
 a.output_predictions.parent.mkdir(parents=True,exist_ok=True)
 with a.output_predictions.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 out={'n_train':int(tr.sum()),'n_valid':int(va.sum()),'n_test':int(te.sum()),'selected_layer':layer,'alpha':alpha,'validation_macro_f1':score,'test_accuracy':accuracy_score(y[te],z),'test_macro_f1':f1_score(y[te],z,average='macro',zero_division=0),'protocol':'select_on_valid_refit_train_plus_valid_evaluate_test'};a.output_summary.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
