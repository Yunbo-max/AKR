"""Local, frozen NatureLM-audio native inference for the C15 specialist case.

Importing this module never imports NatureLM/torch, downloads assets, or loads a
model. Construction requires explicit local snapshots; no AKR adapter is implied.
The official model API was inspected at AUDITED_SOURCE_REVISION (see scope doc).
"""
from __future__ import annotations

from copy import deepcopy
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from akr_closing.core import Query, digest, file_hash


AUDITED_SOURCE_REVISION = "a4c65ad7d558f902852e20a299eafe927ca0921d"
SOURCE_URL = "https://github.com/earthspecies/NatureLM-audio"
MODEL_URL = "https://huggingface.co/EarthSpeciesProject/NatureLM-audio"
_SOURCE_FILES = (
    "NatureLM/models/NatureLM.py", "NatureLM/models/__init__.py",
    "NatureLM/models/Qformer.py", "NatureLM/models/utils.py",
    "NatureLM/models/beats/BEATs.py", "NatureLM/models/beats/backbone.py",
    "NatureLM/models/beats/modules.py", "NatureLM/models/beats/quantizer.py",
    "NatureLM/config.py", "NatureLM/processors.py", "NatureLM/utils.py",
    "NatureLM/checkpoint_utils.py", "NatureLM/storage_utils.py", "pyproject.toml",
)


def _file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required local NatureLM asset missing: {path}")
    return path


def _revision(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{name} requires the actual immutable 40-character revision")
    return value


def _weight_files(directory):
    """Inspect only explicitly indexed weights, never recursively hash a tree."""
    directory = Path(directory).expanduser().resolve()
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text())
        names = sorted(set(data.get("weight_map", {}).values()))
        if not names or not all(isinstance(n, str) for n in names):
            raise ValueError(f"Empty/invalid safetensors weight index: {index}")
        files = [index]
        for name in names:
            p = (directory / name).resolve()
            if p.parent != directory or not name.endswith(".safetensors"):
                raise ValueError("Weight index must name local top-level safetensors shards")
            files.append(_file(p))
        return files
    return [_file(directory / "model.safetensors")]


def _official_factory(assets):
    """Load the pinned official implementation, replacing only remote paths.

    Upstream init_audio_Qformer hard-codes a remote bert-base-uncased config.
    A subclass substitutes a hash-recorded local config, retaining the official
    BertLMHeadModel and exact initialization, with no global monkeypatch.
    """
    import torch

    code = Path(assets["naturelm_code"]).resolve()
    existing = sys.modules.get("NatureLM.models")
    if existing and not Path(existing.__file__).resolve().is_relative_to(code):
        raise RuntimeError("Another NatureLM source is already imported; use an isolated process")
    sys.path.insert(0, str(code))
    models = importlib.import_module("NatureLM.models")
    official = importlib.import_module("NatureLM.models.NatureLM")
    config_module = importlib.import_module("NatureLM.config")
    processors = importlib.import_module("NatureLM.processors")
    for module in (models, official, config_module, processors):
        if not Path(module.__file__).resolve().is_relative_to(code):
            raise RuntimeError("NatureLM import did not resolve to registered source")

    bert_config = str(Path(assets["naturelm_bert_config"]).resolve())

    class LocalNatureLM(models.NatureLM):
        @staticmethod
        def init_audio_Qformer(num_query_token, audio_width, num_hidden_layers=2):
            cfg = official.BertConfig.from_json_file(bert_config)
            cfg.num_hidden_layers = num_hidden_layers
            cfg.encoder_width = audio_width
            cfg.add_cross_attention = True
            cfg.cross_attention_freq = 1
            cfg.query_length = num_query_token
            qformer = official.BertLMHeadModel(config=cfg)
            tokens = torch.nn.Parameter(torch.zeros(1, num_query_token, cfg.hidden_size))
            tokens.data.normal_(mean=0.0, std=cfg.initializer_range)
            return qformer, tokens

    checkpoint = Path(assets["naturelm_checkpoint"]).resolve()
    # HF mixin loads only the local checkpoint, while the model constructor loads
    # its explicit local Llama snapshot. Q-Former requires no BERT weights.
    model = LocalNatureLM.from_pretrained(
        str(checkpoint), local_files_only=True,
        llama_path=str(Path(assets["naturelm_llama"]).resolve()),
        beats_path=None, audio_llama_proj_model=None,
        device=assets.get("naturelm_device", "cuda"),
        audio_encoding_cache_size=0,
    )
    model = model.to(assets.get("naturelm_device", "cuda")).eval()
    model.llama_tokenizer.pad_token_id = model.llama_tokenizer.eos_token_id
    model.llama_model.generation_config.pad_token_id = model.llama_tokenizer.pad_token_id
    cfg = config_module.Config.from_sources(Path(assets["naturelm_config"]).resolve())
    if cfg.generate is None:
        raise ValueError("NatureLM config requires generate settings")
    cfg.generate.do_sample = False
    cfg.generate.max_new_tokens = assets.get("naturelm_max_new_tokens", 32)
    cfg.generate.merging_alpha = assets.get("naturelm_merging_alpha", 1.0)
    return SimpleNamespace(model=model, torch=torch, generate_cfg=cfg.generate,
                           processor=processors.NatureLMAudioProcessor(
                               sample_rate=16000, max_length_seconds=10))


class NatureLMBackend:
    """Frozen native specialist; factory injection is for contract tests only.

    Assets: naturelm_code, naturelm_source_revision, naturelm_checkpoint,
    naturelm_checkpoint_revision, naturelm_llama, naturelm_llama_revision,
    naturelm_bert_config, naturelm_config. All paths must already exist locally.
    Calls consume the Query waveform and supplied prompt, never support labels.
    """

    def __init__(self, assets, *, factory=None):
        self.assets = dict(assets)
        required = ("naturelm_code", "naturelm_source_revision", "naturelm_checkpoint",
                    "naturelm_checkpoint_revision", "naturelm_llama", "naturelm_llama_revision",
                    "naturelm_bert_config", "naturelm_config")
        missing = [key for key in required if not self.assets.get(key)]
        if missing:
            raise ValueError("Missing local NatureLM assets: " + ", ".join(missing))
        for key in ("naturelm_source_revision", "naturelm_checkpoint_revision", "naturelm_llama_revision"):
            _revision(self.assets[key], key)
        if self.assets["naturelm_source_revision"] != AUDITED_SOURCE_REVISION:
            raise ValueError("NatureLM source revision differs from inspected API; audit before use")
        code = Path(self.assets["naturelm_code"]).resolve()
        proc = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
        if proc.returncode or proc.stdout.strip() != self.assets["naturelm_source_revision"]:
            raise ValueError("NatureLM source checkout HEAD does not match registered revision")
        dirty = subprocess.run(["git", "-C", str(code), "diff", "--quiet", "HEAD", "--", "NatureLM", "pyproject.toml"],
                               capture_output=True, check=False)
        if dirty.returncode:
            raise ValueError("NatureLM inspected source has tracked modifications")
        files = [_file(code / p) for p in _SOURCE_FILES]
        for key in ("naturelm_checkpoint", "naturelm_llama"):
            directory = Path(self.assets[key]).resolve()
            files.extend([_file(directory / "config.json"), *_weight_files(directory)])
        llama = Path(self.assets["naturelm_llama"]).resolve()
        files.append(_file(llama / "tokenizer_config.json"))
        tokenizer = [llama / "tokenizer.json", llama / "tokenizer.model"]
        if not any(p.is_file() for p in tokenizer):
            raise FileNotFoundError("Local Llama tokenizer.json or tokenizer.model is required")
        files.extend(p for p in tokenizer if p.is_file())
        files.extend(p for p in (llama / "special_tokens_map.json", llama / "generation_config.json") if p.is_file())
        files.extend(_file(self.assets[k]) for k in ("naturelm_bert_config", "naturelm_config"))
        max_tokens = self.assets.get("naturelm_max_new_tokens", 32)
        alpha = self.assets.get("naturelm_merging_alpha", 1.0)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("naturelm_max_new_tokens must be positive integer")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("naturelm_merging_alpha must be finite in [0,1]")
        self._provenance = {
            "backend": "NatureLM-audio", "mode": "frozen_native_specialist",
            "official_source": SOURCE_URL, "official_checkpoint": MODEL_URL,
            "source_revision": self.assets["naturelm_source_revision"],
            "checkpoint_revision_declared": self.assets["naturelm_checkpoint_revision"],
            "llama_revision_declared": self.assets["naturelm_llama_revision"],
            "snapshot_revision_note": "Declared download revisions; actual local bytes pinned below",
            "files": {str(p): {"sha256": file_hash(p), "size_bytes": p.stat().st_size}
                      for p in sorted(set(files))},
            "window_seconds": 10, "tail_policy": "include_final_partial_window_and_pad",
            "aggregation": "all_windows_same_valid_label_else_invalid",
            "do_sample": False, "max_new_tokens": max_tokens, "merging_alpha": alpha,
            "device": self.assets.get("naturelm_device", "cuda"),
            "audio_cache": "disabled",
            "validation": "contract_test_injected" if factory is not None else "real_assets_loaded_not_benchmark_validated",
            "akr_kv_adapter": False, "task_adaptation_support_used": 0,
            "pretraining_overlap": "Watkins named in model card; exact clip overlap unaudited",
        }
        self._provenance["runtime_packages"] = {}
        for package in ("torch", "transformers", "peft", "safetensors", "huggingface-hub",
                        "resampy", "soundfile", "pydantic", "NatureLM-audio"):
            try:
                self._provenance["runtime_packages"][package] = version(package)
            except PackageNotFoundError:
                self._provenance["runtime_packages"][package] = None
        self.runtime = (factory or _official_factory)(self.assets)
        for name in ("model", "torch", "generate_cfg", "processor"):
            if not hasattr(self.runtime, name):
                raise TypeError(f"NatureLM runtime factory missing {name}")
        self.torch=self.runtime.torch
        self.forward_count=0
        self.runtime.model.eval()
        for parameter in self.runtime.model.parameters():
            parameter.requires_grad_(False)
        # Capture effective config, including fields inherited from pinned YAML.
        cfg = self.runtime.generate_cfg
        cfg.do_sample = False
        cfg.max_new_tokens = max_tokens
        cfg.merging_alpha = alpha
        self._provenance["effective_generate"] = (
            cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else vars(cfg).copy())
        self._provenance["fingerprint"] = digest(self._provenance)

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def predict(self, query: Query, prompt: str, labels: list[str]):
        from animal_omni.metrics import normalize_label
        import numpy as np
        import soundfile as sf

        if type(query) is not Query:
            raise TypeError("NatureLM inference accepts label-free akr_closing.core.Query only")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Nonempty registered prompt required")
        if not labels or len(set(labels)) != len(labels) or any(not isinstance(x, str) or not x.strip() for x in labels):
            raise ValueError("Nonempty distinct label vocabulary required")
        audio = _file(query.audio_path)
        signal, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
        if sample_rate < 1 or not len(signal) or not np.isfinite(signal).all():
            raise ValueError("Nonempty finite query waveform required")
        signal = signal.mean(axis=1)
        window_size = 10 * sample_rate
        runtime = self.runtime
        windows = []
        # The official Pipeline can drop a trailing partial window. Explicitly
        # retain every sample; processor pads each final/short window to 10s.
        with runtime.torch.inference_mode():
            for start in range(0, len(signal), window_size):
                end = min(start + window_size, len(signal))
                tensor, prompts = runtime.processor([signal[start:end]], [prompt], [sample_rate])
                tensor = tensor.to(self.assets.get("naturelm_device", "cuda"))
                batch = {"raw_wav": tensor, "prompt": prompts[0], "audio_chunk_sizes": 1,
                         "padding_mask": runtime.torch.zeros_like(tensor).bool()}
                self.forward_count+=1
                result = runtime.model.generate(batch, runtime.generate_cfg, prompts)
                if not isinstance(result, (list, tuple)) or len(result) != 1 or not isinstance(result[0], str):
                    raise RuntimeError("NatureLM generate must return exactly one raw string")
                raw = result[0]
                windows.append({"start_seconds": start / sample_rate, "end_seconds": end / sample_rate,
                                "raw_prediction": raw, "prediction": normalize_label(raw, labels) or ""})
        predictions = {row["prediction"] for row in windows}
        # No cherry-picking a window; disagreement or invalid text stays invalid.
        prediction = next(iter(predictions)) if len(predictions) == 1 and "" not in predictions else ""
        raw = windows[0]["raw_prediction"] if len(windows) == 1 else "\n".join(
            f"#{w['start_seconds']:.2f}s - {w['end_seconds']:.2f}s#: {w['raw_prediction']}" for w in windows)
        return {"prediction": prediction, "raw_prediction": raw, "windows": windows,
                "model_call_unit": "NatureLM generate calls per audio window; not decoder forwards or FLOPs",
                "event_id": query.event_id, "audio_sha256": file_hash(audio),
                "prompt_sha256": digest(prompt), "labels_sha256": digest(labels),
                "backend_provenance": self.provenance}
