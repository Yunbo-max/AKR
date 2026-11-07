import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from akr_final.train50_features import FeatureExtractor, extract_feature, feature_provenance, _vector


def tone(tmp_path, hz=440, rate=16000, seconds=0.6, name="tone.wav"):
    t = np.arange(int(seconds * rate)) / rate
    path = tmp_path / name
    sf.write(path, 0.3 * np.sin(2 * np.pi * hz * t), rate, subtype="FLOAT")
    return path


def test_mfcc_repeatable_finite_and_audio_responsive(tmp_path):
    a, b = tone(tmp_path), tone(tmp_path, 2000, name="other.wav")
    model = FeatureExtractor("mfcc")
    x, y = model.extract(a), model.extract(b)
    assert x.shape == y.shape == (78,)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    np.testing.assert_array_equal(x, model.extract(a))
    assert np.linalg.norm(x - y) > 1.0
    assert json.loads(json.dumps(model.provenance()))["settings"]["n_mfcc"] == 13


def test_mfcc_handles_stereo_resampling_short_and_silence(tmp_path):
    mono = tone(tmp_path, rate=8000)
    x = extract_feature(mono, "mfcc")
    assert x.shape == (78,) and np.isfinite(x).all()
    short = tmp_path / "short.wav"
    sf.write(short, np.zeros((1, 2)), 22050)
    assert np.isfinite(extract_feature(short, "mfcc")).all()
    stereo = tmp_path / "stereo.wav"
    values, sr = sf.read(mono)
    sf.write(stereo, np.c_[values, values], sr, subtype="FLOAT")
    np.testing.assert_allclose(x, extract_feature(stereo, "mfcc"), atol=1e-8)


def test_mfcc_provenance_fingerprints_code_and_all_settings():
    p = feature_provenance("mfcc")
    assert len(p["implementation_sha256"]) == 64
    assert p["settings"]["dimension"] == 78
    assert p["audio"]["duration_truncation"] is False
    p["settings"]["n_mfcc"] = 1
    assert feature_provenance("mfcc")["settings"]["n_mfcc"] == 13


@pytest.mark.parametrize("bad", [[], [[1, 2]], [np.nan], [np.inf]])
def test_invalid_feature_schema_rejected(bad):
    with pytest.raises(ValueError, match="finite 1-D"):
        _vector(bad)


def test_reject_unknown_backend_and_unregistered_settings():
    with pytest.raises(ValueError, match="unknown frozen"):
        FeatureExtractor("ast")
    with pytest.raises(ValueError, match="fixed settings"):
        FeatureExtractor("mfcc", {"n_mfcc": 2})
    with pytest.raises(ValueError, match="assets require"):
        FeatureExtractor("beats")
    with pytest.raises(ValueError, match="unknown BEATs"):
        FeatureExtractor("beats", dict(source_dir=".", source_revision="a"*40,
                         checkpoint="x", checkpoint_sha256="b"*64, fake=True))


def test_nonfinite_audio_rejected(tmp_path):
    path = tmp_path / "nan.wav"
    sf.write(path, np.array([np.nan, 0.0]), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="finite PCM"):
        extract_feature(path, "mfcc")


def test_missing_egemaps_dependency_never_substitutes(monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "opensmile":
            raise ImportError("test missing dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="no substitute"):
        FeatureExtractor("egemaps")


def test_real_opensmile_functionals_if_installed(tmp_path):
    pytest.importorskip("opensmile")
    extractor = FeatureExtractor("egemaps")
    vector = extractor.extract(tone(tmp_path, seconds=1.0))
    assert vector.shape == (88,) and np.isfinite(vector).all()
    assert len(extractor.provenance()["settings"]["feature_names"]) == 88
    assert extractor.provenance()["settings"]["package_assets_sha256"]


def test_real_beats_if_explicitly_provisioned(tmp_path):
    import os
    # This test never downloads weights or launches a GPU. The assets config is
    # supplied by the operator when the official checkpoint is locally present.
    config = os.environ.get("AKR_TEST_BEATS_ASSETS")
    if not config:
        pytest.skip("official BEATs checkpoint/source assets are not provisioned")
    assets = json.loads(Path(config).read_text())
    assets["device"] = "cpu"
    extractor = FeatureExtractor("beats", assets)
    x = extractor.extract(tone(tmp_path, seconds=1.0))
    y = extractor.extract(tone(tmp_path, hz=1500, seconds=1.0, name="second.wav"))
    assert x.ndim == 1 and len(x) == 768 and np.isfinite(x).all()
    assert not np.allclose(x, y)
    assert extractor._model.training is False
    assert all(not p.requires_grad for p in extractor._model.parameters())


def test_beats_assets_verify_clean_git_source_and_checkpoint_hash(tmp_path):
    import subprocess
    from akr_final.train50_features import _beats_assets
    source = tmp_path / "beats"
    source.mkdir()
    for name in ("BEATs.py", "backbone.py", "modules.py"):
        (source / name).write_text("# local fixture for provenance validation only\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "remote", "add", "origin", "https://github.com/microsoft/unilm.git"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "beats"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a real checkpoint: provenance fixture")
    import hashlib
    assets = dict(source_dir=str(source), source_revision=revision, checkpoint=str(checkpoint),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    _, _, provenance = _beats_assets(assets)
    assert set(provenance["source_files_sha256"]) == {"BEATs.py", "backbone.py", "modules.py"}
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _beats_assets(assets)
    (source / "BEATs.py").write_text("# modified\n")
    with pytest.raises(ValueError, match="modified"):
        _beats_assets(assets)
