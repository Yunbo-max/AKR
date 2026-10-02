#!/usr/bin/env python3
"""Render the three pre-registered core figures from authoritative artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--frequency-summary",type=Path,required=True)
    parser.add_argument("--geometry",type=Path)
    parser.add_argument("--oracle",type=Path)
    parser.add_argument("--conditional",type=Path)
    parser.add_argument("--conditional-k",type=int,default=20)
    parser.add_argument("--conditional-eta",type=float,default=300.0)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)

    with args.frequency_summary.open(newline="",encoding="utf-8") as f:
        frequency={r["condition"]:r for r in csv.DictReader(f)}
    cutoffs=[1000,2000,4000,6000,8000]
    accuracy=[float(frequency[f"lp_0-{x}"]["accuracy"]) for x in cutoffs]
    full=float(frequency["full_0-8k"]["accuracy"])
    fig,ax=plt.subplots(figsize=(5.5,3.5)); ax.plot(np.array(cutoffs)/1000,accuracy,"o-",label="Low-pass")
    ax.axhline(full,color="black",ls="--",label="Full observable 0–8 kHz")
    ax.set(xlabel="Low-pass cutoff (kHz)",ylabel="Accuracy",ylim=(0,1)); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(args.output_dir/"fig1_frequency_recognition.png",dpi=220); plt.close(fig)

    if args.geometry and args.geometry.exists():
        geometry=json.loads(args.geometry.read_text())["layers"]
        keys=sorted(geometry,key=lambda k:(int(k.split("_")[0]),k.split("_")[1]))
        ranks=[1,2,4,8,16,32]
        matrix=np.array([[geometry[k]["explained_energy"].get(str(r),np.nan) for r in ranks] for k in keys])
        fig,(ax,cosine_ax)=plt.subplots(1,2,figsize=(11.5,8),gridspec_kw={"width_ratios":[1.4,1]})
        image=ax.imshow(matrix,aspect="auto",vmin=0,vmax=1,cmap="viridis")
        ax.set(xticks=range(len(ranks)),xticklabels=ranks,yticks=range(len(keys)),yticklabels=keys,
               xlabel="Subspace rank",ylabel="Thinker layer / projection")
        ax.set_title("a  Gradient singular spectrum")
        fig.colorbar(image,ax=ax,label="Explained gradient energy")
        layers=sorted({int(key.split("_")[0]) for key in keys})
        for kind,color in [("k","#3366aa"),("v","#cc6633")]:
            same=[geometry[f"{layer}_{kind}"].get("same_label_mean_cosine",np.nan) for layer in layers]
            different=[geometry[f"{layer}_{kind}"].get("different_label_mean_cosine",np.nan) for layer in layers]
            cosine_ax.plot(layers,same,color=color,label=f"{kind.upper()} same label")
            cosine_ax.plot(layers,different,color=color,ls="--",label=f"{kind.upper()} different label")
        cosine_ax.set(xlabel="Thinker layer",ylabel="Mean pairwise gradient cosine")
        cosine_ax.set_title("b  Label-conditioned cosine geometry")
        cosine_ax.axhline(0,color="black",lw=.7)
        cosine_ax.legend(frameon=False,fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir/"fig2_frequency_error_kv_geometry.png",dpi=220); plt.close(fig)

    if args.oracle and args.oracle.exists() and args.conditional and args.conditional.exists():
        with args.oracle.open(newline="",encoding="utf-8") as f: oracle_rows=list(csv.DictReader(f))
        with args.conditional.open(newline="",encoding="utf-8") as f: conditional_rows=list(csv.DictReader(f))
        selected=[r for r in conditional_rows if int(r["support_k"])==args.conditional_k
                  and float(r["eta"])==args.conditional_eta]
        fixed=[r for r in selected if r["method"]=="fixed_mean"]
        conditional=[r for r in selected if r["method"]=="conditional"]
        rates=[0.0, sum(r["correct"]=="true" for r in fixed)/len(fixed),
               sum(r["correct"]=="true" for r in conditional)/len(conditional),
               max(sum(r[k]=="true" for r in oracle_rows)/len(oracle_rows)
                   for k in oracle_rows[0] if k.startswith("correct_eta_"))]
        names=["Degraded",f"Fixed KV\nK={args.conditional_k}",
               f"Conditional KV\nK={args.conditional_k}","Oracle KV"]
        fig,ax=plt.subplots(figsize=(6.2,3.7)); bars=ax.bar(names,rates,color=["#888888","#5b8db8","#e0903c","#4a9b67"])
        ax.set(ylabel="Eligible failure recovery",ylim=(0,1.08))
        for bar,value in zip(bars,rates): ax.text(bar.get_x()+bar.get_width()/2,value+.025,f"{value:.1%}",ha="center")
        fig.tight_layout(); fig.savefig(args.output_dir/"fig3_kv_recovery.png",dpi=220); plt.close(fig)
    elif args.oracle and args.oracle.exists():
        with args.oracle.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        columns=sorted([k for k in rows[0] if k.startswith("correct_eta_")],key=lambda k:float(k.rsplit("_",1)[-1].replace("p",".")))
        etas=[float(k.rsplit("_",1)[-1].replace("p",".")) for k in columns]
        recovery=[sum(r[k]=="true" for r in rows)/len(rows) for k in columns]
        fig,ax=plt.subplots(figsize=(5.5,3.5)); ax.semilogx(etas,recovery,"o-")
        ax.set(xlabel="Oracle KV step size η",ylabel="Eligible failure recovery",ylim=(0,1))
        fig.tight_layout(); fig.savefig(args.output_dir/"fig3_kv_recovery.png",dpi=220); plt.close(fig)


if __name__=="__main__":main()
