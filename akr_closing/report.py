"""Complete-job-only tables and a separate coverage ledger (no missing = zero)."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from .core import Journal,atomic_json


def report_run(root:Path):
    root=Path(root);coverage=[];incomplete=set();corrupt=[]
    for manifest in sorted(root.rglob('manifest.json')):
        try:
            m=json.loads(manifest.read_text())
            if 'expected' not in m or 'provenance' not in m:continue
            j=Journal(manifest.parent,m['provenance'],m['expected'])
            state=j.status();job=str(manifest.parent.relative_to(root));coverage.append({'job':job,**state})
            if not state['complete']:incomplete.add(manifest.parent)
        except (ValueError,RuntimeError,OSError) as exc:
            corrupt.append({'path':str(manifest),'reason':str(exc)});incomplete.add(manifest.parent)
    rows=[]
    for path in sorted(root.rglob('METRICS.json')):
        if any(parent==path.parent or parent in path.parents for parent in incomplete):continue
        m=json.loads(path.read_text())
        if m.get('n',0)<=0:continue
        rows.append({'job':str(path.parent.relative_to(root)),**m})
    atomic_json(root/'COMPLETE_METRICS.json',rows);atomic_json(root/'COVERAGE_ALL.json',coverage)
    keys=['job','stage','n','accuracy','macro_f1','balanced_accuracy','invalid_rate','mean_wall_seconds','mean_model_forwards','peak_cuda_bytes']
    with (root/'complete_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    def esc(x):
        for a,b in [('\\',r'\textbackslash{}'),('_',r'\_'),('%',r'\%'),('&',r'\&'),('#',r'\#')]:x=x.replace(a,b)
        return x
    lines=['% Complete result cells only; missing results are NOT zeros.',r'\begin{tabular}{p{.65\linewidth}rrr}',
           r'\toprule Method / protocol & Acc. & Macro-F1 & Invalid \\',r'\midrule']
    for r in rows:
        lines.append(f"{esc(r['job'])} & {100*r['accuracy']:.2f} & {100*r['macro_f1']:.2f} & {100*r['invalid_rate']:.2f}" + r" \\")
    lines.extend([r'\bottomrule',r'\end{tabular}']);(root/'complete_metrics.tex').write_text('\n'.join(lines)+'\n')
    status=json.loads((root/'TASK_STATUS.json').read_text()) if (root/'TASK_STATUS.json').exists() else {}
    out={'complete_result_cells':len(rows),'incomplete_jobs':len(incomplete),'corrupt_artifacts':corrupt,'tasks':status,
         'all_requested_tasks_complete':bool(status) and all(v.get('status')=='completed' for v in status.values()) and not incomplete and not corrupt,
         'note':'A completed negative result is valid; gated-off test is not an accuracy of zero. E1/E7/E8/E6 have task-specific summaries.'}
    atomic_json(root/'FINAL_STATUS.json',out);return out
