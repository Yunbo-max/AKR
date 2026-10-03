"""Packaging contracts only. No checkpoint is downloaded or evaluated."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]

def invoke(root):
    p=subprocess.run([sys.executable,str(ROOT/'scripts/check_akr_release.py'),'--root',str(root)],capture_output=True,text=True)
    return p,json.loads(p.stdout) if p.stdout.startswith('{') else None

def test_source_release_is_self_contained():
    p,r=invoke(ROOT)
    assert p.returncode==0,p.stdout+p.stderr
    assert r['complete'] and r['gpu_started'] is False
    assert not r['raw_audio_required_for_this_check']

@pytest.fixture
def copy_release(tmp_path):
    # Restrict copying to assets checked by the integrity checker.
    for name in ['configs','paper','docs','scripts','akr_final','akr_closing','src']:
        shutil.copytree(ROOT/name,tmp_path/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['pyproject.toml','requirements-cpu.txt','requirements-qwen.txt']:
        shutil.copy2(ROOT/name,tmp_path/name)
    return tmp_path

def test_missing_manuscript_is_an_explicit_error(copy_release):
    (copy_release/'paper/main.tex').unlink()
    p,r=invoke(copy_release)
    assert p.returncode==1 and any('paper/main.tex' in s for s in r['errors'])

def test_nonimmutable_model_or_registry_drift_fails(copy_release):
    cfg=copy_release/'configs/akr_standalone.yaml'
    cfg.write_text(cfg.read_text().replace('ae9e1690543ffd5c0221dc27f79834d0294cba00','main'))
    p,r=invoke(copy_release)
    assert p.returncode==1 and any('revision' in s for s in r['errors'])

def test_missing_plot_is_not_silently_tolerated(copy_release):
    (copy_release/'paper/figures/intro.jpg').unlink()
    p,r=invoke(copy_release)
    assert p.returncode==1 and any('intro.jpg' in s for s in r['errors'])

def test_s1_cannot_be_called_completed_without_evidence(copy_release):
    path=copy_release/'configs/akr_registry.json'
    data=json.loads(path.read_text());data['experiments']['S1']['status']='completed'
    path.write_text(json.dumps(data))
    p,r=invoke(copy_release)
    assert p.returncode==1 and any('S1' in s for s in r['errors'])
