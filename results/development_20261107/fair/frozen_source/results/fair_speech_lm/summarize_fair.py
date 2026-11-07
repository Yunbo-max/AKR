"""Complete-only scientific summary and parameter-sensitivity plots."""
from pathlib import Path
import json
import sys
import importlib.metadata
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from akr_closing.core import file_hash,atomic_json

def read(p): return json.loads(Path(p).read_text())

def main(run):
    out=ROOT/'results/fair_speech_lm'/run
    report=read(out/'REPORT.json'); status=read(out/'STATUS.json')
    if not report['complete'] or not status['complete'] or status['report_sha256']!=file_hash(out/'REPORT.json'):
        raise RuntimeError('complete verified report required')
    proxy=read(out/'PARAMETER_SENSITIVITY_EXTENDED.json'); ce=read(out/'PARAMETER_CE.json')
    if not ce['complete']: raise RuntimeError('auxiliary sensitivity incomplete')
    source=ROOT/'results/calibration_pilot/pool_dogs_3b_20261107_v2/dogs/trials/qwen3b/s20261107_lp_1000'
    parity={}
    for new,old in [('native','native'),('akr_reference','cv_refit')]:
        previous=read(source/old/'SCORED_PREDICTIONS.json')
        current=report['predictions'][new]
        parity[new]=all(all(a[k]==b[k] for k in ['event_id','prediction','raw_prediction','generated_tokens'])
                        for a,b in zip(previous,current)) and len(previous)==len(current)
    if not all(parity.values()): raise RuntimeError('historical source parity failed')
    samples=[json.loads(line) for line in (out/'GPU_SAMPLES.jsonl').read_text().splitlines() if line.strip()]
    peaks={str(i):max(g['used_mib'] for s in samples for g in s['gpus'] if g['index']==i) for i in [0,1]}
    lora=report['policies']['lora']; akr=report['policies']['akr']
    refit=read(out/'training/refit/DONE.json')['result']
    cost=read(out/'COST_AUDIT.json')
    if not cost['complete']: raise RuntimeError('cost audit incomplete')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ranks=sorted({r['rank'] for r in proxy['rows']}); ridges=sorted({r['ridge'] for r in proxy['rows']})
    matrix=np.array([[next(v['relative_sse'] for v in proxy['rows'] if v['rank']==r and v['ridge']==a)
                      for a in ridges] for r in ranks])
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    image=axes[0].imshow(matrix,cmap='RdBu_r',vmin=.88,vmax=1.12,aspect='auto')
    axes[0].set(xticks=range(len(ridges)),xticklabels=[f'{v:g}' for v in ridges],
        yticks=range(len(ranks)),yticklabels=ranks,xlabel='Ridge penalty',ylabel='Gradient rank',
        title='Calibration gradient-error proxy\nRelative SSE; 1 = training-fold mean predictor')
    for i in range(len(ranks)):
        for j in range(len(ridges)): axes[0].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=7)
    fig.colorbar(image,ax=axes[0],shrink=.7)
    strengths=[.01,.03,.1]
    for kind in ['regression','shrunk','class_pooled']:
        values=[akr['mean_ce'][f'{kind}_a{a:g}'] for a in strengths]
        axes[1].plot(strengths,values,'o-',label=kind)
    axes[1].axhline(akr['mean_ce']['native'],ls='--',color='gray',label='native')
    axes[1].set(xscale='log',xlabel='Relative K/V correction strength',ylabel='OOF answer cross-entropy',
                title='Actual Speech LM calibration loss\nLower is better; NOT development accuracy')
    axes[1].legend(fontsize=8)
    fig.suptitle('Parameter sensitivity: two folds within the same 40-example calibration pool')
    fig.savefig(out/'parameter_sensitivity.png',dpi=180); fig.savefig(out/'parameter_sensitivity.pdf'); plt.close(fig)
    ce_lookup={(r['rank'],r['ridge']):r['mean_ce'] for r in ce['rows']}
    base=ce_lookup[4,10.]
    strength_range=max(akr['mean_ce'][f'regression_a{a:g}'] for a in strengths)-min(akr['mean_ce'][f'regression_a{a:g}'] for a in strengths)
    parameter_effects=dict(strength_ce_range_at_original_map=strength_range,
        ridge_ce_change_at_rank4=ce_lookup[4,3000.]-base,
        rank_ce_change_at_ridge10=ce_lookup[16,10.]-base,
        joint_ce_change=ce_lookup[16,3000.]-base,
        scope='This grid and calibration pool only; no global sensitivity claim.')
    lines=['# Fair Speech LM development comparison and parameter sensitivity','',
        '**One previously inspected development subset; not official-test confirmation or a global best-performance claim.**','',
        'Same pinned Qwen2.5-Omni-3B checkpoint, Dogs 10-class vocabulary, LP1 audio, 40 calibration IDs, two 20/20 folds, prompt, greedy 12-token output and 40 development IDs.',
        'The attempted fresh-validation design failed preflight because Rudy has only four validation examples, all previously evaluated. The same original development IDs were therefore registered for every method before this run. Official test remains unused.','',
        '| Speech LM method | Accuracy | Macro-F1 | Repaired vs native | Harmed vs native |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in report['rows']:
        lines.append(f"| {r['method']} | {100*r['accuracy']:.1f}% | {100*r['macro_f1']:.2f}% | {r['repaired']} | {r['harmed']} |")
    lines+=['','All final class answers are generated by the Speech LM. No external Ridge classifier appears as a main-result method. The class-pooled AKR candidate uses an internal acoustic readout only to select a gradient centroid for K/V injection.','',
        '## Matched conditions and remaining differences','',
        '- Both adaptation searches use exactly the same 40 labelled clips and folds, and nine nonzero configurations plus native. Selection minimizes pooled held-out answer-token CE; refitting uses all 40.',
        '- LoRA uses language-decoder q/v projections, rank 8, alpha 16, dropout .05, AdamW, effective batch 4, and LR {5e-5,2e-4,5e-4} x epoch {1,3,5}. Audio/vision/base weights remain frozen. This is a specified LoRA baseline, not exhaustive LoRA optimization.',
        '- AKR uses regression, residual-shrunk regression, or class-pooled gradient routing x strength {.01,.03,.1}. Original CV-cap AKR is retained as a reference.',
        '- Compute, adaptable parameter counts, and historical research effort are not matched. Additional AKR sensitivity diagnostics are disclosed and never enter primary model selection.',
        f"- Selected LoRA: {lora['candidate']}; OOF CE {min(lora['mean_ce'].values()):.6f}.",
        f"- Selected AKR: {akr['candidate']}; OOF CE {min(akr['mean_ce'].values()):.6f}.",
        '- Selection at the edge of the registered grid does not demonstrate convergence or a global optimum. No parameters were changed using this run\'s development predictions.','',
        '## Parameter evidence','',
        f"- Original regression: strength sweep .01/.03/.1 changes pooled OOF CE by a range of {strength_range:.6f}.",
        f"- At fixed strength .03: ridge 10 -> 3000 (rank 4) changes CE by {parameter_effects['ridge_ce_change_at_rank4']:+.6f}.",
        f"- At fixed strength .03: rank 4 -> 16 (ridge 10) changes CE by {parameter_effects['rank_ce_change_at_ridge10']:+.6f}.",
        f"- Joint rank16/ridge3000 vs rank4/ridge10 changes CE by {parameter_effects['joint_ce_change']:+.6f}.",
        '- Negative CE changes are improvements. The proxy heatmap measures gradient prediction, not recognition accuracy.',
        f"- Best proxy setting: {proxy['best_proxy']}. This was not used to pick the main AKR winner.",
        '- Next sweep priorities must follow actual CE sensitivity, with fresh confirmation after development; rank and ridge should be checked jointly. Layer locations, calibration-pool size and feature-layer selection were held fixed, so their sensitivity is unknown.','',
        '## Verification and resource use','',
        '- 36 relevant tests passed, including tiny real PEFT training/save/reload, frozen-base checks, foreign-fold receipt rejection, and concurrent policy-lock serialization. Independent review completed with material findings fixed.',
        f"- Real native and original-AKR predictions exactly reproduce historical text/classes/generated lengths: {parity}.",
        f"- LoRA cross-validation: {lora['cv_compute']['train_exposures']} training exposures, {lora['cv_compute']['optimizer_steps']} optimizer steps, {cost['cv_top_level_forwards']} reconstructed top-level forwards, {lora['cv_compute']['wall_seconds']:.1f} summed stage seconds.",
        '- Original immutable PEFT training receipts undercounted forward calls because the original-module hook was bypassed. COST_AUDIT.json reconstructs counts from exposure/loss records and three parity calls per stage. These exclude checkpoint recomputation/backward and are not FLOP counts.',
        f"- LoRA refit: {refit.get('train_exposures',0)} training-example exposures, {refit.get('optimizer_steps',0)} optimizer updates; {refit.get('wall_seconds',0):.1f} seconds.",
        f"- LoRA refit reconstructed top-level forwards: {cost['refit_top_level_forwards']}.",
        f"- LoRA trainable parameters: {refit.get('audit',{}).get('trainable_parameters',0)}; peak allocated memory {refit.get('peak_allocated_bytes',0)/1024**3:.2f} GiB.",
        f"- AKR primary selection compute: {akr['cv_compute']}; imported 40 support gradients were required from the prior frozen-backbone extraction, not free supervision.",
        f"- NVIDIA sampled total-device peaks (including unrelated workloads): GPU0 {peaks['0']/1024:.2f} GiB, GPU1 {peaks['1']/1024:.2f} GiB. CUDA allocator accounting can differ from NVIDIA total occupancy under WSL.",
        '- Training stages resume only when completed and checksummed. An interrupted stage restarts from deterministic initialization; optimizer-step resume is not claimed.','',
        f"Run: `{out.relative_to(ROOT)}`.",
        f"Report SHA256: `{file_hash(out/'REPORT.json')}`.",'']
    target=ROOT/'docs/FAIR_SPEECH_LM_RESULTS_20261107.md'; target.write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(out/'SUMMARY_EVIDENCE.json',dict(complete=True,report_sha256=file_hash(out/'REPORT.json'),
        summary_sha256=file_hash(target),historical_prediction_parity=parity,parameter_effects=parameter_effects,
        gpu_peak_mib=peaks,versions={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','numpy']},
        script_sha256=file_hash(Path(__file__))))
    print(json.dumps(dict(summary=str(target),parameter_effects=parameter_effects,peaks_mib=peaks)))

if __name__=='__main__': main(sys.argv[1])
