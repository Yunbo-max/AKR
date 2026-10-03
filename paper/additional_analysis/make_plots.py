"""Source-grounded AKR exploratory figures; no model calls or new experiments.
Run: python make_plots.py. Dependencies: numpy, matplotlib.
Every plot uses one axes and Matplotlib's default color cycle / colormap.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT/'source_data.json').read_text(encoding='utf-8'))
OUT = ROOT/'figures'
OUT.mkdir(exist_ok=True)

def new_figure(w=8, h=6):
    fig=plt.figure(figsize=(w,h),layout='constrained')
    return fig,fig.add_subplot(111)

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=210)
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)

def csvwrite(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

# Each target is compared to ITS own condition-matched readout.
matrices={}
for name,s in D['transfer'].items():
    m=np.array(s['counts'],dtype=float)/s['n_test']*100
    matrices[name]=m
    deficit=np.diag(m)[None,:]-m
    assert np.allclose(np.diag(deficit),0)
    fig,ax=new_figure(7.4,6.3)
    im=ax.imshow(deficit,vmin=-5,vmax=75)
    ax.set_xticks(range(6),D['conditions'])
    ax.set_yticks(range(6),D['conditions'])
    ax.set_xlabel('Target recording bandwidth')
    ax.set_ylabel('Readout training bandwidth')
    ax.set_title(f'{name}: transfer deficit against target-matched readout\nPositive = transferred readout performs worse',fontsize=12,pad=14)
    for i in range(6):
        for j in range(6):
            ax.text(j,i,f'{deficit[i,j]:.1f}',ha='center',va='center',fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.15',alpha=.85))
    fig.colorbar(im,ax=ax,label='Accuracy difference (percentage points)',shrink=.86)
    fig.supxlabel('Source-selected readouts; full is not the LP8 filtered control.',fontsize=9)
    save(fig,f'01_{name.lower()}_transfer_deficit')
    csvwrite(f'{name.lower()}_transfer.csv',[{'source':D['conditions'][i],'target':D['conditions'][j], 'n_test':s['n_test'],'accuracy_percent':m[i,j],'target_reference_percent':m[j,j],'target_referenced_deficit_pp':deficit[i,j]} for i in range(6) for j in range(6)])

# Two-way transfer: reference on the SOURCE and changed input on the TARGET.
fig,ax=new_figure(8.5,4.4)
rows=[]
for name,m in matrices.items():
    for source,target in [(0,1),(1,0)]:
        rows.append({'dataset':name,'source':D['conditions'][source], 'target':D['conditions'][target], 'source_matched_percent':m[source,source],'transferred_percent':m[source,target]})
y=np.arange(len(rows))
for i,r in enumerate(rows):
    ax.plot([r['source_matched_percent'],r['transferred_percent']],[i,i],linewidth=2)
ax.scatter([r['source_matched_percent'] for r in rows],y,marker='o',s=55,label='Same-bandwidth input')
ax.scatter([r['transferred_percent'] for r in rows],y,marker='x',s=65,label='Changed-bandwidth input')
for i,r in enumerate(rows):
    ax.annotate(f"{r['source_matched_percent']:.2f}%",(r['source_matched_percent'],i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    ax.annotate(f"{r['transferred_percent']:.2f}%",(r['transferred_percent'],i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
ax.set_yticks(y,[f"{r['dataset']}: {r['source']} → {r['target']}" for r in rows]);ax.invert_yaxis()
ax.set_xlim(0,102);ax.set_ylim(3.6,-.6)
ax.set_xlabel('Frozen readout accuracy (%)');ax.set_title('Keeping the readout fixed: both directions can fail',pad=14)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False,fontsize=9)
fig.supxlabel('Within each row the fitted readout is unchanged; this is not AKR output accuracy.',fontsize=9)
save(fig,'02_bidirectional_readout_transfer');csvwrite('bidirectional_transfer.csv',rows)

# Measured accuracy differences, NOT a decomposition of mutual information.
rows=[]
marm=D['marmaudio']
for i,c in enumerate(marm['cutoffs']):
    rows.append({'dataset':'MarmAudio','cutoff_khz':c,'matched_readout_change_pp':100*(marm['full_matched_rounded']-marm['matched'][i]),'transfer_deficit_pp':100*(marm['matched'][i]-marm['transfer'][i])})
for name,m in matrices.items():
    for i,c in enumerate([1,2,4,6,8],1):
        rows.append({'dataset':name,'cutoff_khz':c,'matched_readout_change_pp':m[0,0]-m[i,i],'transfer_deficit_pp':m[i,i]-m[0,i]})
fig,ax=new_figure(7.6,6)
for name,marker in [('MarmAudio','o'),('Dogs','s'),('Watkins','^')]:
    rr=[r for r in rows if r['dataset']==name]
    ax.plot([r['matched_readout_change_pp'] for r in rr],[r['transfer_deficit_pp'] for r in rr],marker=marker,linewidth=1,label=name)
    for r in rr:
        if r['cutoff_khz'] in (1,2):
            ax.annotate(f"{r['cutoff_khz']} kHz",(r['matched_readout_change_pp'],r['transfer_deficit_pp']),xytext=(5,5),textcoords='offset points',fontsize=9)
ax.plot([-5,23],[-5,23],linestyle=':',linewidth=1,label='Equal measured differences')
ax.axhline(0,linewidth=.6);ax.axvline(0,linewidth=.6)
ax.set_xlim(-7,23);ax.set_ylim(-7,74)
ax.set_xlabel('Full matched − filtered matched accuracy (pp)')
ax.set_ylabel('Filtered matched − full-readout transfer accuracy (pp)')
ax.set_title('Bandwidth shift: matched readout change vs transfer deficit',fontsize=12)
ax.legend(frameon=False,fontsize=9,loc='upper left')
fig.supxlabel('Paired accuracy diagnostics, not additive amounts of acoustic information.',fontsize=9)
save(fig,'03_readout_change_vs_transfer_deficit');csvwrite('readout_phase.csv',rows)

# Exact paired transitions from complete cross-prompt summary.
p=D['cross_prompt'];fig,ax=new_figure(8,4.6)
y=np.arange(2)
ax.barh(y,[r['repaired'] for r in p['rows']],label='Native wrong → repair correct',height=.48)
ax.barh(y,[-r['damaged'] for r in p['rows']],label='Native correct → repair wrong',height=.48)
ax.axvline(0,linewidth=.8)
for i,r in enumerate(p['rows']):
    ax.text(r['repaired']+.7,i,f"+{r['repaired']}",va='center')
    ax.text(-r['damaged']-.7,i,f"−{r['damaged']}",va='center',ha='right')
    ax.annotate(f"Net: +{r['repaired']-r['damaged']} / 139 = +{100*(r['repaired']-r['damaged'])/139:.2f} pp",(0,i),xytext=(0,-38),textcoords='offset points',fontsize=10)
ax.set_yticks(y,[r['prompt'] for r in p['rows']]);ax.invert_yaxis();ax.set_ylim(1.65,-.55)
ax.set_xlim(-16,42);ax.set_xlabel('Number of recordings (negative side denotes damaged answers)')
ax.set_title('Prompt changes alter both correction benefit and collateral error',fontsize=12,pad=14)
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=9)
fig.supxlabel('Dogs A–J, class-routed pooled variant; not the continuous AKR method.',fontsize=9)
save(fig,'04_cross_prompt_repairs_and_harms');csvwrite('cross_prompt_transitions.csv',p['rows'])

# Paired model-size interaction already measured in original ledger.
s=D['size_crossover'];fig,ax=new_figure(7.5,4.6)
x=np.arange(6);ax.plot(x,s['gain_7b_minus_3b_pp'],marker='o',linewidth=2)
ax.axhline(0,linestyle='--',linewidth=.8)
for i,v in enumerate(s['gain_7b_minus_3b_pp']):
    ax.annotate(f'{v:+.2f}',(i,v),xytext=(0,9 if v>=0 else -17),textcoords='offset points',ha='center',fontsize=10)
ax.set_xticks(x,s['cutoffs']);ax.set_xlabel('Input condition');ax.set_ylabel('Accuracy difference: 7B − 3B (pp)')
ax.set_ylim(-7,12);ax.set_title('The model-size advantage reverses at 1 kHz',fontsize=12)
fig.supxlabel('Paired native generation on MarmAudio; not a comparison of AKR-trained models.',fontsize=9)
save(fig,'05_model_size_bandwidth_interaction')

# Historical development grid is kept separate from final tests.
s=D['rank_layer'];fig,ax=new_figure(7.6,4.9)
for layer,vals,marker in zip(s['feature_layers'],s['macro_f1'],['o','s','^','D']):
    ax.plot(s['ranks'],100*np.array(vals),marker=marker,label=f'Feature layer {layer}',linewidth=1.7)
ax.set_xscale('log',base=2);ax.set_xticks(s['ranks'],[str(r) for r in s['ranks']])
ax.set_ylim(0,12);ax.set_xlabel('Requested correction rank');ax.set_ylabel('Validation macro-F1 (%)')
ax.set_title('Correction rank × acoustic feature depth',fontsize=12)
ax.legend(frameon=False,fontsize=9,ncol=2)
fig.supxlabel('Historical Dogs validation: n=64, 20 supports. Feature layer ≠ intervention layer.',fontsize=9)
save(fig,'06_rank_feature_depth')

print(f'Created {len(list(OUT.glob("*.png")))} analysis figures (PNG + SVG); no GPU runs.')
