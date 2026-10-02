"""Same-support external-readout comparison. No query target enters inference.

S1 is a new, explicitly labelled protocol extension, not a historical result.
It compares direct ridge, text rendering, and ridge evidence supplied to the LM
with native, fixed-mean and continuous-KV answers on identical query identities.
"""
from pathlib import Path
import time
import numpy as np
from akr_closing.core import Journal, atomic_json
from akr_closing.pipeline import prompt_for
from akr_closing.repair import RidgeReadout


def evidence_prompt(task_prompt: str, predicted_label: str, labels: list[str]) -> str:
    if predicted_label not in labels:
        raise ValueError('External prediction must be one declared label.')
    return (task_prompt + '\nExternal classifier prediction: ' + predicted_label
            + '. This prediction may be incorrect. Listen to the recording and '
              'return exactly one of the allowed labels.')


def run_simple_baselines(runner):
    from akr_final.runner import _check_stop
    tag='qwen7b'
    methods=['native','fixed_mean','continuous_pooled','ridge_direct','probe_to_text','probe_to_lm']
    try:
        for e in runner.plan['episodes']:
            _check_stop()
            base=Path('S1')/tag/e['id']; settings=e['lock']['selection']
            x,g=runner._support(e,tag); fq=runner._queries(e,tag)
            penalty=settings.get('ridge_alpha',runner.cfg['ridge_alpha'])
            mapping=runner.router_fit(x,g,rank=settings['rank'],alpha=penalty)
            field=mapping.predict(fq)
            native=runner._predict(base/'native',e,tag,'native',settings,None)
            runner._score_trial(base/'native',e,native,native)
            zero=runner._predict(base/'zero_alpha',e,tag,'zero_alpha',settings,field)
            runner._no_op(native,zero)
            for name,arr in [('fixed_mean',np.repeat(g.mean(0)[None,:],len(fq),axis=0)),
                             ('continuous_pooled',field)]:
                rows=runner._predict(base/name,e,tag,name,settings,arr)
                runner._score_trial(base/name,e,rows,native)
            # Ground-truth labels are read only for the labelled SUPPORT here.
            y=np.array([runner.labels.index(runner.lookup[i]['label']) for i in e['support_ids']])
            readout=RidgeReadout.fit(x,y,len(runner.labels),alpha=penalty)
            scores=readout.mapping.predict(fq)
            predicted=[runner.labels[int(k)] for k in scores.argmax(1)]
            direct=[{'prediction':label,'raw_prediction':label,
                     'class_scores':[float(v) for v in score],
                     'new_model_calls':0,'output_source':'external_ridge'}
                    for label,score in zip(predicted,scores)]
            # Rendering a class as text changes neither its identity nor accuracy.
            for name in ['ridge_direct','probe_to_text']:
                atomic_json(runner.out/base/name/'PREDICTIONS.json',direct)
                runner._score_trial(base/name,e,direct,native)
            prompts=[evidence_prompt(prompt_for(runner.labels),p,runner.labels) for p in predicted]
            ident={'run':runner.fingerprint,'episode':e['id'],'tag':tag,
                   'method':'probe_to_lm','prompts':prompts,'support_ids':e['support_ids']}
            journal=Journal(runner.out/base/'probe_to_lm',ident,e['query_ids'])
            for event,pred,prompt in zip(e['query_ids'],predicted,prompts):
                if journal.has(event): continue
                _check_stop()
                backend=runner._model(tag,e)
                q=runner._query(event,e['condition'])
                start=time.perf_counter(); before=getattr(backend,'forward_count',0)
                value=backend.predict(q,prompt,runner.labels,direction=None,alpha=0.,
                                      scope='audio',kind='kv',layer_group='all')
                journal.add(event,{**value,'external_prediction':pred,
                                   'wall_seconds':time.perf_counter()-start,
                                   'new_model_calls':getattr(backend,'forward_count',0)-before})
            rows=journal.results()
            runner._score_trial(base/'probe_to_lm',e,rows,native)
            # True QUERY labels are used only after all predictions for scoring.
            truth=[runner.lookup[i]['label'] for i in e['query_ids']]
            agrees=sum(r['prediction']==p for r,p in zip(rows,predicted))
            spoiled=sum(p==t and r['prediction']!=t for r,p,t in zip(rows,predicted,truth))
            rescued=sum(p!=t and r['prediction']==t for r,p,t in zip(rows,predicted,truth))
            atomic_json(runner.out/base/'COMPLETE.json',{
                'complete':True,'methods':methods,'support_ids':e['support_ids'],
                'query_ids':e['query_ids'],'query_labels_used_for_inference':False,
                'probe_to_lm_agreement':agrees,'correct_probe_spoiled':spoiled,
                'incorrect_probe_rescued':rescued,'settings_origin':'fixed standalone settings',
                'interpretation':'Compare identical supervision; direct ridge is a replacement readout, not native repair.',
                'new_protocol_not_historical_confirmation':True})
    finally:
        runner.close()
