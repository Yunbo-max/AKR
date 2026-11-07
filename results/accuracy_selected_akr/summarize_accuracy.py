"""Summarize the completed, bounded second development iteration."""
from pathlib import Path
import json
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run):
    out = ROOT / 'results/accuracy_selected_akr' / run
    report = read(out / 'REPORT.json')
    status = read(out / 'STATUS.json')
    assert report['complete'] and status['complete']
    assert status['report_sha256'] == sha(out / 'REPORT.json')
    assert report['cached_inference_verified'] and not report['official_test_used']
    prior = ROOT / 'results/fair_speech_lm/fair_dogs_3b_20261107_v2'
    previous = read(prior / 'REPORT.json')
    sensitivity = read(prior / 'SUMMARY_EVIDENCE.json')['parameter_effects']
    choices = report['policy']['selection']['choices']
    samples = [json.loads(s) for s in (out / 'GPU_SAMPLES.jsonl').read_text().splitlines() if s.strip()]
    peaks = {str(i): max(g['used_mib'] for s in samples for g in s['gpus'] if g['index'] == i) for i in (0, 1)}
    lines = [
        '# Accuracy-selected AKR development experiment', '',
        'This is a second development iteration on the same previously inspected 40 Dogs validation clips. It is not an independent confirmation. The official test set remains unused.', '',
        'Qwen2.5-Omni-3B generates every final answer. Both methods use the same 40 labelled calibration clips, two 20/20 folds, prompt, class normalizer and greedy 12-token decoding. Each searches nine adaptation configurations plus native. Selection uses pooled calibration-fold accuracy, then macro-F1, invalid rate and a predeclared simpler-candidate order. Selected models refit on all 40 calibration clips before development evaluation.', '',
        '| Method | Development accuracy | Macro-F1 | Repaired / harmed vs native |',
        '| --- | ---: | ---: | ---: |',
    ]
    for row in report['rows']:
        label = row['method']
        if label in choices and choices[label]['candidate'] == 'native':
            label += ' selection policy (native fallback; no adapter)'
        lines.append(f"| {label} | {row['accuracy']:.1%} | {row['macro_f1']:.2%} | {row['repaired']} / {row['harmed']} |")
    lines += ['', '## Locked calibration choices', '']
    for method in ('akr', 'lora'):
        choice = choices[method]
        score = choice['scores'][choice['candidate']]
        lines.append(f"- {method}: `{choice['candidate']}`, pooled calibration-fold accuracy {score['accuracy']:.1%}, macro-F1 {score['macro_f1']:.2%}.")
    lines += [
        '', 'AKR compares residual-shrunk, centered-regression and centered-class gradient directions, each at relative K/V correction strength 0.03, 0.1 and 0.3. Centering subtracts the training-fold mean gradient. The class variant uses an internal ridge readout to choose a calibration gradient centroid; final answers still come from the Speech LM.',
        '', 'LoRA reuses strictly verified calibration-fold adapters trained at learning rates 5e-5, 2e-4 and 5e-4 for 1, 3 and 5 epochs. It adapts only language-decoder q/v projections (rank 8, alpha 16, dropout 0.05); the audio encoder, vision encoder and base weights remain frozen.',
        f"Full-pool LoRA refit: `{report['policy']['refit']['kind']}`.",
        '', '## Parameter sensitivity', '',
        '| AKR mapping | Accuracy at alpha 0.03 | At 0.1 | At 0.3 | Range |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    curves = {}
    for kind in ('shrunk', 'centered_regression', 'centered_class'):
        values = [choices['akr']['scores'][f'{kind}_a{a:g}']['accuracy'] for a in (.03, .1, .3)]
        curves[kind] = values
        lines.append(f"| {kind} | {values[0]:.1%} | {values[1]:.1%} | {values[2]:.1%} | {max(values)-min(values):.1%} |")
    lines += [
        '', 'These are calibration-fold recognition accuracies, not development scores. They describe this small pool and this grid only.',
        f"The earlier answer-CE sweep found an alpha effect range of {sensitivity['strength_ce_range_at_original_map']:.6f} for 0.01/0.03/0.1. At fixed alpha 0.03, rank 4→16 changed CE by {sensitivity['rank_ce_change_at_ridge10']:+.6f}; ridge 10→3000 changed CE by {sensitivity['ridge_ce_change_at_rank4']:+.6f}; changing both gave {sensitivity['joint_ce_change']:+.6f}. Lower CE is better. Gradient-prediction SSE improved under stronger ridge, but that did not imply better Speech LM CE or recognition.",
        '', 'Prioritize correction strength and direction construction in a future registered sweep; check rank and ridge jointly using held-out recognition. Layer positions, feature layer and calibration size were held fixed and their sensitivity remains unknown. This experiment does not establish a globally most sensitive parameter.',
        '', '## Prior result and limits', '',
        '| First iteration, CE-selected | Development accuracy |',
        '| --- | ---: |',
    ]
    for row in previous['rows']:
        lines.append(f"| {row['method']} | {row['accuracy']:.1%} |")
    if choices['lora']['candidate'] == 'native':
        lines += ['', '**The second-round LoRA policy selected native, so its development row is not an adapted LoRA model. The first-round CE-selected LoRA result remains the adapted comparator; do not claim AKR beats LoRA by comparing only against this native fallback.**']
    lines += [
        '', 'The first negative AKR result is preserved. This second iteration was designed after inspecting it. Label budget and current candidate counts are matched; historical search effort, compute and adaptable parameter counts are not matched. One model, one dataset, one seed and 40 inspected development examples cannot establish superiority across models or datasets. LoRA at the grid boundary is not proven converged.',
        '', '## Verification and GPU use', '',
        '- The current calibration/fair-comparison/centered-direction targeted test run passed 18 tests; TARGETED_TESTS.log preserves the output. The preceding instrumentation maintenance suite passed 39 tests (overlapping coverage; do not add the counts).',
        '- All selected inference records were checked through the cache-only report path, with model loading prohibited. Native calibration predictions agree exactly between the AKR and LoRA pipelines. Input, source, fold and adapter identities are checksummed.',
        f"- Sampled NVIDIA total-device peaks, including other workloads: GPU 0 {peaks['0']/1024:.2f} GiB ({peaks['0']*1024**2/1e9:.2f} GB); GPU 1 {peaks['1']/1024:.2f} GiB ({peaks['1']*1024**2/1e9:.2f} GB). The scheduling target was around 90 GiB; this is not a constant-occupancy or per-job-memory claim.",
        f"- Report SHA256: `{sha(out / 'REPORT.json')}`.",
        '',
    ]
    target = ROOT / 'docs/ACCURACY_SELECTED_AKR_RESULTS_20261107.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for kind, values in curves.items():
        axes[0].plot([.03, .1, .3], [v*100 for v in values], 'o-', label=kind)
    axes[0].axhline(choices['akr']['scores']['native']['accuracy']*100, color='gray', ls='--', label='native')
    axes[0].set(xscale='log', xlabel='Relative correction strength', ylabel='Calibration OOF accuracy (%)', title='Selection: same 40 calibration clips')
    from matplotlib.ticker import NullLocator
    axes[0].set_xticks([.03, .1, .3], labels=['0.03', '0.1', '0.3'])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].legend(fontsize=7)
    rows = report['rows']
    names = [r['method'] + ('\n(native fallback)' if r['method'] in choices and choices[r['method']]['candidate'] == 'native' else '') for r in rows]
    axes[1].bar(names, [100*r['accuracy'] for r in rows], color=['#777777','#56B4E9','#009E73','#E69F00'])
    prior_lora = next(r['accuracy'] for r in previous['rows'] if r['method'] == 'lora')
    axes[1].axhline(prior_lora*100, color='#E69F00', ls='--', label='Prior CE-selected LoRA')
    axes[1].legend(fontsize=7)
    axes[1].set(ylabel='Development accuracy (%)', title='Previously inspected 40-clip development set')
    axes[1].tick_params(axis='x', labelsize=8)
    fig.savefig(out/'accuracy_sensitivity.png', dpi=180)
    fig.savefig(out/'accuracy_sensitivity.pdf')
    plt.close(fig)
    evidence = dict(complete=True, report_sha256=sha(out/'REPORT.json'), summary_sha256=sha(target),
                    gpu_peak_mib=peaks, calibration_accuracy_curves=curves, prior_ce_sensitivity=sensitivity)
    (out/'SUMMARY_EVIDENCE.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(dict(summary=str(target), choices={k:v['candidate'] for k,v in choices.items()}, peaks=peaks)))


if __name__ == '__main__':
    main(sys.argv[1])
