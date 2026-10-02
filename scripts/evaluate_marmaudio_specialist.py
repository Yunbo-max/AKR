#!/usr/bin/env python3
"""Evaluate the authors' 96 kHz ResNet-50 MarmAudio specialist checkpoint."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from animal_omni.audio import Intervention, apply_intervention


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--code-root",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--training-labels",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--lowpass-hz",type=int)
    args=parser.parse_args(); sys.path.insert(0,str(args.code_root))
    from marmaudio.classifier import get_resnet50
    from marmaudio.filterbank import STFT,MelFilter,Log1p

    labels=[]
    with args.training_labels.open(encoding="utf-8") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            if row["label"] not in labels:labels.append(row["label"])
    frontend=torch.nn.Sequential(STFT(1024,368),MelFilter(96000,1024,128,1000,48000),Log1p(7))
    model=torch.nn.Sequential(frontend,get_resnet50(len(labels)))
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    model.load_state_dict(state); model.eval().to(args.device)
    with args.manifest.open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    output=[]
    with torch.inference_mode():
        for index,row in enumerate(rows,1):
            signal,sr=sf.read(row["audio_path"],always_2d=False)
            if sr!=96000:raise RuntimeError(f"specialist requires original 96 kHz audio, got {sr}")
            if signal.ndim==2:signal=signal.mean(axis=1)
            if args.lowpass_hz:
                signal=apply_intervention(signal,sr,Intervention("lowpass",high_hz=args.lowpass_hz))
            signal=(signal-signal.mean())/max(signal.std(),1e-12)
            logits=model(torch.tensor(signal[None],dtype=torch.float32,device=args.device))
            prediction=labels[int(logits.argmax(-1).item())]
            prediction="Infant Cry" if prediction=="Infant" else prediction
            spectrum=f"original_rate_lowpass_0-{args.lowpass_hz/1000:g}k" if args.lowpass_hz else "original_0-48k"
            output.append({"event_id":row["event_id"],"target":row["label"],"prediction":prediction,
                           "correct":str(prediction==row["label"]).lower(),"spectrum":spectrum})
            if index==1 or index%50==0 or index==len(rows):print(f"[{index}/{len(rows)}] {prediction}",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=output[0].keys());writer.writeheader();writer.writerows(output)


if __name__=="__main__":main()
