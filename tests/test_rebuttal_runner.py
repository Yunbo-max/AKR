import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from test_rebuttal_plan import config_fixture
from test_akr_final_three import FakeBackend

class RebuttalFake(FakeBackend):
    loss_calls=[]
    def support_answer_loss(self,q,target,prompt,*,direction=None,**kwargs):
        self.loss_calls.append(q.event_id);self.forward_count+=1
        return {'answer_loss':3. if direction is None else 1.+.00001*float(np.square(direction).mean()),
                'answer_tokens':1,'applied_norms':{},'effective_relative_norms':{}}
    def predict(self,q,prompt,labels,*,direction=None,alpha=0,**kw):
        if direction is not None and not np.any(direction):direction=None
        return super().predict(q,prompt,labels,direction=direction,alpha=alpha,**kw)


def setup(tmp_path):
    root,path,cfg=config_fixture(tmp_path);cfg['seeds']=[11];cfg['settings']['control_alphas']=[.01]
    path.write_text(yaml.safe_dump(cfg));RebuttalFake.grad_calls=[];RebuttalFake.loss_calls=[]
    RebuttalFake.loads=0;RebuttalFake.predict_calls=0
    return root,path


def test_full_synthetic_rebuttal_with_complete_only_report_and_resume(tmp_path):
    from akr_final.rebuttal_runner import RebuttalStudy
    root,path=setup(tmp_path)
    study=RebuttalStudy(root,path,'t',['core','scale'],backend_factory=RebuttalFake)
    result=study.run(execute=True)
    assert result['complete'] and result['completed_jobs']==4
    rows=study.report()['rows'];assert rows
    names={r['method'] for r in rows}
    assert {'native','akr_relative','akr_cap','gain_cap','scalar_cap','ridge_all','probe_to_text','probe_to_lm','akr_all'}<=names
    assert all(r['accuracy']==r['accuracy'] for r in rows)
    e=study.study['datasets']['marmaudio']['plan']['episodes'][0]
    assert set(RebuttalFake.grad_calls)<=set(e['support_ids'])
    assert set(RebuttalFake.loss_calls)<=set(e['calibration_ids']+e['guard_ids'])
    assert not set(RebuttalFake.loss_calls)&set(e['query_ids'])
    for row in rows:
        if row['method']=='ridge_all':
            other=next(v for v in rows if all(v[k]==row[k] for k in ['model','episode']) and v['method']=='probe_to_text')
            assert row['accuracy']==other['accuracy']
    before=(RebuttalFake.loads,RebuttalFake.predict_calls,len(RebuttalFake.loss_calls))
    again=RebuttalStudy(root,path,'t',['core','scale'],backend_factory=RebuttalFake)
    assert again.run(execute=True)['complete']
    assert before==(RebuttalFake.loads,RebuttalFake.predict_calls,len(RebuttalFake.loss_calls))
    assert not list(root.rglob('RUN.json')) and not list(root.rglob('LOCK.json'))


def test_preflight_does_not_load_model_and_change_requires_new_run(tmp_path):
    from akr_final.rebuttal_runner import RebuttalStudy
    root,path=setup(tmp_path);s=RebuttalStudy(root,path,'p',['core'],backend_factory=RebuttalFake)
    s.run(execute=False);assert RebuttalFake.loads==0
    cfg=yaml.safe_load(path.read_text());cfg['settings']['raw_eta']=99.;path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError,match='changed'):RebuttalStudy(root,path,'p',['core'],backend_factory=RebuttalFake)


def test_missing_query_journal_is_incomplete_not_a_published_mean(tmp_path):
    from akr_final.rebuttal_runner import RebuttalStudy
    root,path=setup(tmp_path);s=RebuttalStudy(root,path,'p',['core'],backend_factory=RebuttalFake)
    s.run(execute=True)
    item=next((s.out/'marmaudio'/'trials').glob('**/native/items/*.json'));item.unlink()
    report=s.report();assert not report['complete'] and report['incomplete_trials']


def test_cross_role_duplicate_audio_preflight_rejects(tmp_path):
    from akr_final.rebuttal_runner import RebuttalStudy
    root,path=setup(tmp_path);s=RebuttalStudy(root,path,'p',['core'],backend_factory=RebuttalFake)
    data=s.study['datasets']['marmaudio']['plan'];e=data['episodes'][0];lookup={r['event_id']:r for r in data['rows']}
    Path(lookup[e['guard_ids'][0]]['audio_path']).write_bytes(Path(lookup[e['fit_ids'][0]]['audio_path']).read_bytes())
    with pytest.raises(ValueError,match='waveform'):s.run(execute=False)


def test_two_independent_noop_controls_and_incomplete_receipt(tmp_path):
    from akr_final.rebuttal_runner import RebuttalStudy
    root,path=setup(tmp_path);s=RebuttalStudy(root,path,'p',['core'],backend_factory=RebuttalFake)
    result=s.run(execute=True)
    assert {'zero','zero_field'} <= {r['method'] for r in result['rows']}
    receipt=next((s.out/'marmaudio'/'jobs').glob('*/*.json'))
    v=json.loads(receipt.read_text());v['methods'].remove('zero_field');receipt.write_text(json.dumps(v))
    assert not s.report()['complete']


def test_stale_metrics_not_accepted_as_recognition_evidence(tmp_path):
    from akr_final.rebuttal_runner import RebuttalStudy
    root,path=setup(tmp_path);s=RebuttalStudy(root,path,'p',['core'],backend_factory=RebuttalFake)
    s.run(execute=True)
    p=next((s.out/'marmaudio'/'trials').glob('**/native/METRICS.json'))
    v=json.loads(p.read_text());v['accuracy']=.123456789;p.write_text(json.dumps(v))
    with pytest.raises(RuntimeError,match='metric'):s.report()


def test_cli_execution_requires_authorization_before_files_or_model(tmp_path):
    from akr_final.rebuttal_cli import main
    assert main(['run','--root',str(tmp_path)])==2
