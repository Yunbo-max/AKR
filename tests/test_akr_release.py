"""Release contracts. All model calls below use a synthetic CPU backend."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from akr_final.release import load_registry, make_command, main, _bounded
from akr_final.simple_baselines import evidence_prompt
from test_akr_standalone import fixture_config, make_runner
from test_akr_final_three import FakeBackend

ROOT = Path(__file__).resolve().parents[1]

def test_public_entrypoint_plan_from_another_directory(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT/'scripts/run_akr_release.py'), 'plan'],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert list(plan['tasks']) == ['S1']
    assert not plan['gpu_started'] and not plan['old_RUN_LOCK_required']
    assert '--execute' not in plan['command']

@pytest.mark.parametrize('tasks', [[], ['S1','S1'], ['E1'], ['F4']])
def test_invalid_tasks_rejected(tasks):
    with pytest.raises(ValueError): make_command(ROOT,'release',tasks,'24gb',False)

@pytest.mark.parametrize('run_id', ['../outside', '/tmp/x', 'bad name', ''])
def test_invalid_run_id_rejected(run_id):
    with pytest.raises(ValueError): make_command(ROOT,run_id,['S1'],'24gb',False)

def test_explicit_execution_flag_required(capsys):
    assert main(['run','--root',str(ROOT)]) == 2
    assert '--execute' in capsys.readouterr().err

def test_check_never_forwards_execution_flag(monkeypatch):
    calls=[]
    monkeypatch.setattr('akr_final.release._bounded',lambda cmd,*args: calls.append(cmd) or 0)
    assert main(['check','--root',str(ROOT),'--execute']) == 0
    assert len(calls)==1 and '--execute' not in calls[0]

def test_exhausted_deadline_does_not_spawn():
    assert _bounded(['must-not-exist'],ROOT,{},0) == 124

@pytest.mark.parametrize('budget',['nan','inf','0','25'])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):main(['plan','--root',str(ROOT),'--budget-hours',budget])

def test_declared_dependencies_and_packages_present():
    try: import tomllib
    except ImportError: import tomli as tomllib
    cfg=tomllib.loads((ROOT/'pyproject.toml').read_text())
    pk=cfg['tool']['setuptools']['package-dir']
    assert {'animal_omni','akr_closing','akr_final'} <= set(pk)
    assert (ROOT/'requirements-cpu.txt').is_file()
    assert (ROOT/'requirements-qwen.txt').is_file()

def test_simple_baselines_complete_and_resume(tmp_path):
    root,path,_=fixture_config(tmp_path)
    FakeBackend.loads=0;FakeBackend.grad_calls=[];FakeBackend.predict_calls=0
    runner=make_runner(root,path,'s1')
    assert runner.run(['S1'],execute=True)['complete']
    allowed={i for e in runner.plan['episodes'] for i in e['support_ids']}
    assert set(FakeBackend.grad_calls)<=allowed
    for ep in runner.plan['episodes']:
        base=runner.out/'S1/qwen7b'/ep['id']
        assert json.loads((base/'COMPLETE.json').read_text())['complete']
        assert json.loads((base/'ridge_direct/PREDICTIONS.json').read_text()) == json.loads((base/'probe_to_text/PREDICTIONS.json').read_text())
    calls=(FakeBackend.loads,len(FakeBackend.grad_calls),FakeBackend.predict_calls)
    runner.close()
    resume=make_runner(root,path,'s1',cache_only=True)
    assert resume.run(['S1'],execute=True)['complete']
    assert calls==(FakeBackend.loads,len(FakeBackend.grad_calls),FakeBackend.predict_calls)

def test_evidence_prompt_only_accepts_declared_label():
    assert 'may be incorrect' in evidence_prompt('Recognize.','A',['A','B'])
    with pytest.raises(ValueError):evidence_prompt('Recognize.','OTHER',['A','B'])

def test_plan_checks_actual_yaml_against_registry(tmp_path):
    import shutil
    (tmp_path/'configs').mkdir()
    shutil.copy2(ROOT/'configs/akr_registry.json',tmp_path/'configs/akr_registry.json')
    cfg=(ROOT/'configs/akr_standalone.yaml').read_text().replace('ae9e1690543ffd5c0221dc27f79834d0294cba00','main')
    (tmp_path/'configs/akr_standalone.yaml').write_text(cfg)
    with pytest.raises(ValueError,match='revision'):
        main(['plan','--root',str(tmp_path)])
