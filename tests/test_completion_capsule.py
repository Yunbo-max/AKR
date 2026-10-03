"""The publication transport must never accept a partial or conflicting capsule."""
import base64
import importlib.util
import json
import lzma
from pathlib import Path
import subprocess
import pytest

spec = importlib.util.spec_from_file_location('completion', Path(__file__).parents[1] / 'scripts/materialize_20261003.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(tmp):
    subprocess.run(['git','init','-q',str(tmp)],check=True)
    (tmp/'old.txt').write_text('before')
    subprocess.run(['git','add','.'],cwd=tmp,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','base'],cwd=tmp,check=True)
    ref=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tmp,text=True).strip()
    entries=[{'path':p,'content':s,'sha256':m.digest(s.encode()),'mode':'100644'} for p,s in [('old.txt','before-after'),('new.txt','new')]]
    entries[0]['content']='after';entries[0]['sha256']=m.digest(b'after')
    directory=tmp/'migration/final_20261003';(directory/'parts').mkdir(parents=True)
    manifest={'format':'json_utf8_lzma_base64_v1','base_ref':ref}
    encode(directory,manifest,entries)
    return directory,manifest,entries


def encode(directory,manifest,entries):
    data=json.dumps(entries).encode();compressed=lzma.compress(data);part=base64.b64encode(compressed)+b'\n'
    (directory/'parts/00.b64').write_bytes(part)
    manifest.update(file_count=len(entries),decoded_bytes=len(data),decoded_sha256=m.digest(data),compressed_sha256=m.digest(compressed),parts=[{'path':'00.b64','bytes':len(part),'sha256':m.digest(part)}])
    (directory/'manifest.json').write_text(json.dumps(manifest))


def test_materialization_idempotent(tmp_path):
    d,_,_=fixture(tmp_path)
    assert m.materialize(tmp_path,d,True)['complete']
    assert (tmp_path/'old.txt').read_text()=='before'
    m.materialize(tmp_path,d);m.materialize(tmp_path,d)
    assert (tmp_path/'old.txt').read_text()=='after'
    assert (tmp_path/'new.txt').read_text()=='new'


def test_truncated_transport_writes_nothing(tmp_path):
    d,_,_=fixture(tmp_path)
    (d/'parts/00.b64').write_text('AAAA\n')
    with pytest.raises(ValueError,match='Part hash'):m.materialize(tmp_path,d)
    assert (tmp_path/'old.txt').read_text()=='before'
    assert not (tmp_path/'new.txt').exists()


def test_conflict_writes_nothing(tmp_path):
    d,_,_=fixture(tmp_path)
    (tmp_path/'new.txt').write_text('someone else')
    with pytest.raises(ValueError,match='conflict'):m.materialize(tmp_path,d)
    assert (tmp_path/'old.txt').read_text()=='before'


def test_traversal_rejected(tmp_path):
    d,j,e=fixture(tmp_path);e[1]['path']='../outside.txt';encode(d,j,e)
    with pytest.raises(ValueError,match='Unsafe path'):m.materialize(tmp_path,d)
    assert (tmp_path/'old.txt').read_text()=='before'


def test_symlink_rejected(tmp_path):
    d,_,_=fixture(tmp_path);(tmp_path/'new.txt').symlink_to(tmp_path/'old.txt')
    with pytest.raises(ValueError,match='Symlink'):m.materialize(tmp_path,d)
    assert (tmp_path/'old.txt').read_text()=='before'
