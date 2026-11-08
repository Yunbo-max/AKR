"""Frozen acoustic baselines; optional backends never fall back to another feature."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping

import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly

RATE = 16000
MFCC_SETTINGS = dict(sample_rate=RATE, mono="channel_mean", resample="scipy.resample_poly",
                     preemphasis=0.97, frame_samples=400, hop_samples=160, n_fft=512,
                     window="numpy.hamming_symmetric", mel_scale="HTK", n_mels=40,
                     fmin=0, fmax=8000, mel_normalization=None, n_mfcc=13,
                     log="natural", log_floor=1e-10, dct_type=2, dct_norm="ortho",
                     delta_width=2, delta_edges="replicate", final_frame="zero_pad",
                     pooling="mean_std_population_of_mfcc_delta_delta2", dimension=78)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"{name} is required for this feature backend; install its optional dependencies") from exc


def _vector(value):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("feature backend must return a nonempty finite 1-D vector")
    return x


def _audio(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    if sr <= 0 or not len(x) or not np.isfinite(x).all():
        raise ValueError("audio must contain nonempty finite PCM and a positive sample rate")
    x = x.mean(axis=1)
    if sr != RATE:
        g = math.gcd(int(sr), RATE)
        x = resample_poly(x, RATE // g, int(sr) // g)
    return x


def _delta(x):
    padded = np.pad(x, ((2, 2), (0, 0)), mode="edge")
    return sum(i * (padded[2+i:2+i+len(x)] - padded[2-i:2-i+len(x)])
               for i in (1, 2)) / 10.0


def _mfcc(x):
    x = np.r_[x[0], x[1:] - 0.97 * x[:-1]]
    count = max(1, int(math.ceil(max(0, len(x) - 400) / 160)) + 1)
    x = np.pad(x, (0, (count - 1) * 160 + 400 - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    power = np.abs(np.fft.rfft(frames * np.hamming(400), n=512)) ** 2 / 512
    mel_max = 2595 * np.log10(1 + 8000 / 700)
    edges = 700 * (10 ** (np.linspace(0, mel_max, 42) / 2595) - 1)
    hz = np.fft.rfftfreq(512, 1 / RATE)
    filters = np.maximum(0, np.minimum((hz[None, :] - edges[:-2, None]) /
                                      (edges[1:-1, None] - edges[:-2, None]),
                                      (edges[2:, None] - hz[None, :]) /
                                      (edges[2:, None] - edges[1:-1, None])))
    cepstra = dct(np.log(np.maximum(power @ filters.T, 1e-10)), type=2,
                  norm="ortho", axis=1)[:, :13]
    delta = _delta(cepstra)
    features = np.concatenate((cepstra, delta, _delta(delta)), axis=1)
    return _vector(np.r_[features.mean(axis=0), features.std(axis=0)])


def _beats_assets(assets):
    required = {"source_dir", "source_revision", "checkpoint", "checkpoint_sha256"}
    if not isinstance(assets, Mapping) or required - set(assets):
        raise ValueError(f"BEATs assets require {sorted(required)}")
    unknown = set(assets) - required - {"device"}
    if unknown:
        raise ValueError(f"unknown BEATs asset settings: {sorted(unknown)}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(assets["source_revision"])):
        raise ValueError("BEATs source_revision must be a full immutable Git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", str(assets["checkpoint_sha256"])):
        raise ValueError("BEATs checkpoint_sha256 must be a full SHA-256")
    source, ckpt = Path(assets["source_dir"]).resolve(), Path(assets["checkpoint"]).resolve()
    if not (source / "BEATs.py").is_file() or not (source / "backbone.py").is_file():
        raise ValueError("source_dir must be the official microsoft/unilm beats directory")
    try:
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        origin = subprocess.check_output(["git", "-C", str(source), "remote", "get-url", "origin"], text=True).strip()
        root = Path(subprocess.check_output(["git", "-C", str(source), "rev-parse", "--show-toplevel"], text=True).strip())
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", str(source)], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("BEATs requires a clean pinned official Git checkout") from exc
    if head != assets["source_revision"] or dirty.strip():
        raise ValueError("BEATs source revision differs or source tree is modified")
    if not re.fullmatch(r"(?:https://github\.com/|git@github\.com:)microsoft/unilm(?:\.git)?/?", origin):
        raise ValueError("BEATs checkout origin must be microsoft/unilm")
    if _sha(ckpt) != assets["checkpoint_sha256"]:
        raise ValueError("BEATs checkpoint SHA-256 mismatch")
    files = {str(p.relative_to(source)): _sha(p) for p in sorted(source.rglob("*.py"))
             if "__pycache__" not in p.parts}
    return source, ckpt, dict(source_repository="https://github.com/microsoft/unilm",
                             source_revision=head, source_files_sha256=files,
                             checkpoint_sha256=assets["checkpoint_sha256"],
                             device=str(assets.get("device", "cpu")), sample_rate=RATE,
                             pooling="last_hidden_mean_over_unpadded_tokens", dtype="float32",
                             fbank_mean=15.41663, fbank_std=6.55582,
                             minimum_samples=4000, short_audio="right_zero_pad",
                             finetuned_model=False)


class FeatureExtractor:
    """Reuse one frozen backend across support and all test examples."""
    def __init__(self, kind, assets=None):
        self.kind = str(kind).lower()
        if self.kind == "egemapsv02":
            self.kind = "egemaps"
        if assets is not None and not isinstance(assets, Mapping):
            raise ValueError("feature assets must be a mapping")
        self.assets = dict(assets) if assets is not None else None
        self._model = None
        common = dict(schema_version=1, kind=self.kind, implementation_sha256=_sha(__file__),
                      numpy=_version("numpy"), scipy=_version("scipy"), soundfile=_version("soundfile"),
                      audio=dict(sample_rate=RATE, mono="channel_mean", resample="scipy.resample_poly",
                                 peak_normalization=False, duration_truncation=False))
        if self.kind == "mfcc":
            if assets:
                raise ValueError("MFCC has fixed settings and accepts no assets")
            common["settings"] = dict(MFCC_SETTINGS)
        elif self.kind == "egemaps":
            if assets:
                raise ValueError("eGeMAPSv02 has fixed settings and accepts no assets")
            try:
                import opensmile
            except ImportError as exc:
                raise RuntimeError("eGeMAPSv02 requires opensmile; no substitute is permitted") from exc
            self._model = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                           feature_level=opensmile.FeatureLevel.Functionals,
                                           num_workers=1)
            package = Path(opensmile.__file__).resolve().parent
            configs = {str(p.relative_to(package)): _sha(p) for p in sorted(package.rglob("*"))
                       if p.is_file() and p.suffix in {".conf", ".inc", ".so", ".dll", ".dylib"}}
            if not configs:
                raise RuntimeError("cannot fingerprint installed openSMILE configs/binaries")
            common.update(opensmile=_version("opensmile"), settings=dict(
                feature_set="eGeMAPSv02", feature_level="Functionals", dimension=88,
                feature_names=list(self._model.feature_names), package_assets_sha256=configs))
        elif self.kind == "beats":
            self._source, self._checkpoint, settings = _beats_assets(assets)
            common.update(torch=_version("torch"), torchaudio=_version("torchaudio"), settings=settings)
        else:
            raise ValueError(f"unknown frozen feature kind: {kind!r}")
        self._provenance = common

    def provenance(self):
        return json.loads(json.dumps(self._provenance, sort_keys=True, allow_nan=False))

    def _load_beats(self):
        import torch
        # Official code uses an absolute `from backbone import ...`. Refuse an
        # existing conflicting module rather than importing another implementation.
        _, _, current = _beats_assets(self.assets)
        if current != self._provenance["settings"]:
            raise ValueError("BEATs assets changed after provenance registration")
        for name in ("backbone", "modules"):
            existing = sys.modules.get(name)
            if existing is not None and Path(existing.__file__).resolve() != self._source / (name + ".py"):
                raise RuntimeError(f"a conflicting {name} module is already loaded")
        spec = importlib.util.spec_from_file_location("_akr_official_beats", self._source / "BEATs.py")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(self._source))
        previous_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous_bytecode
            sys.path.pop(0)
        # Explicit weights_only=False matches the official config+model checkpoint
        # API. The bytes have already been checked against the supplied hash.
        checkpoint = torch.load(self._checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("cfg"), dict) or "model" not in checkpoint:
            raise ValueError("BEATs checkpoint must contain official cfg and model mappings")
        cfg = module.BEATsConfig(checkpoint["cfg"])
        if cfg.finetuned_model:
            raise ValueError("use a pretrained BEATs checkpoint, not AudioSet classifier probabilities")
        model = module.BEATs(cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
        self._model = model.to(device=self.assets.get("device", "cpu"), dtype=torch.float32).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    def extract(self, path):
        x = _audio(path)
        if self.kind == "mfcc":
            return _mfcc(x)
        if self.kind == "egemaps":
            values = self._model.process_signal(x, RATE).to_numpy()
            if values.shape != (1, 88):
                raise ValueError(f"unexpected eGeMAPSv02 functionals shape {values.shape}")
            return _vector(values[0])
        import torch
        if self._model is None:
            self._load_beats()
        # A 16-frame patch needs at least 175 ms in the official Kaldi frontend.
        # Fixed 250 ms minimum is recorded; no batch padding enters mean pooling.
        if len(x) < 4000:
            x = np.pad(x, (0, 4000 - len(x)))
        wave = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.assets.get("device", "cpu"))
        mask = torch.zeros_like(wave, dtype=torch.bool)
        with torch.inference_mode():
            hidden, token_mask = self._model.extract_features(wave, padding_mask=mask,
                                                              fbank_mean=15.41663, fbank_std=6.55582)
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("BEATs must return hidden representations [1, tokens, width]")
        if token_mask is not None:
            hidden = hidden[0, ~token_mask[0]]
        else:
            hidden = hidden[0]
        return _vector(hidden.mean(dim=0).cpu().numpy())


    def extract_many(self, paths):
        """Batched extraction; each item keeps its own valid-token mean pool."""
        paths = list(paths)
        if self.kind in ("mfcc", "egemaps"):
            return [self.extract(path) for path in paths]
        import torch
        if self._model is None:
            self._load_beats()
        waves = []
        for path in paths:
            x = _audio(path)
            # Same 250 ms minimum as the per-file path; padding never enters
            # any item's mean pool.
            if len(x) < 4000:
                x = np.pad(x, (0, 4000 - len(x)))
            waves.append(x)
        max_len = max(len(w) for w in waves)
        padded = torch.zeros((len(waves), max_len), dtype=torch.float32)
        pmask = torch.zeros((len(waves), max_len), dtype=torch.bool)
        for i, w in enumerate(waves):
            padded[i, :len(w)] = torch.from_numpy(w.astype(np.float32))
            pmask[i, len(w):] = True
        device = self.assets.get("device", "cpu")
        padded, pmask = padded.to(device), pmask.to(device)
        with torch.inference_mode():
            hidden, token_mask = self._model.extract_features(
                padded, padding_mask=pmask, fbank_mean=15.41663, fbank_std=6.55582)
        if hidden.ndim != 3 or hidden.shape[0] != len(paths):
            raise ValueError("BEATs must return one hidden representation per waveform")
        values = []
        for i in range(len(paths)):
            if token_mask is not None:
                hidden_i = hidden[i, ~token_mask[i]]
            else:
                hidden_i = hidden[i]
            values.append(_vector(hidden_i.mean(dim=0).cpu().numpy()))
        return values


def extract_feature(path, kind, assets=None):
    return FeatureExtractor(kind, assets).extract(path)


def feature_provenance(kind, assets=None):
    return FeatureExtractor(kind, assets).provenance()
