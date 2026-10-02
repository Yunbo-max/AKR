#!/usr/bin/env python3
"""Apply a transparent prefix duration cap without padding short examples."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import soundfile as sf

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--minimum-seconds',type=float,default=0.0);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--output-manifest',type=Path,required=True);a=p.parse_args()
    rows=list(csv.DictReader(a.manifest.open())); out=[]
    for r in rows:
        x,sr=sf.read(r['audio_path'],always_2d=True,dtype='float32'); x=x[:int(a.seconds*sr)]
        minimum=int(a.minimum_seconds*sr)
        if len(x)<minimum:
            import numpy as np
            x=np.pad(x,((0,minimum-len(x)),(0,0)))
        path=a.output_dir/r['dataset_name']/f"{r['event_id']}.wav";path.parent.mkdir(parents=True,exist_ok=True);sf.write(path,x,sr,subtype='FLOAT')
        out.append({**r,'audio_path':str(path.resolve()),'duration_cap_s':a.seconds,'minimum_duration_s':a.minimum_seconds,'duration_protocol':'prefix_cap_minimum_tail_zero_pad'})
    a.output_manifest.parent.mkdir(parents=True,exist_ok=True)
    with a.output_manifest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
    print('wrote',len(out))
if __name__=='__main__':main()
