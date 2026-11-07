"""Adapter contracts with a tiny injected runtime, not NatureLM validation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from akr_closing.core import Query
from akr_final import train50_naturelm as module


@pytest.fixture
def assets(tmp_path, monkeypatch):
    code = tmp_path / "official-code"
    for name in module._SOURCE_FILES:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# isolated adapter test fixture\n")
    for name in ("checkpoint", "llama"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"contract fixture, not model weights")
    (tmp_path / "llama/tokenizer_config.json").write_text("{}")
    (tmp_path / "llama/tokenizer.json").write_text("{}")
    (tmp_path / "bert_config.json").write_text("{}")
    (tmp_path / "inference.yml").write_text("generate: {}\n")

    def git(args, **kwargs):
        assert args[:3] == ["git", "-C", str(code)]
        return SimpleNamespace(returncode=0, stdout=module.AUDITED_SOURCE_REVISION + "\n")

    monkeypatch.setattr(module.subprocess, "run", git)
    return {
        "naturelm_code": str(code), "naturelm_source_revision": module.AUDITED_SOURCE_REVISION,
        "naturelm_checkpoint": str(tmp_path / "checkpoint"), "naturelm_checkpoint_revision": "a" * 40,
        "naturelm_llama": str(tmp_path / "llama"), "naturelm_llama_revision": "b" * 40,
        "naturelm_bert_config": str(tmp_path / "bert_config.json"),
        "naturelm_config": str(tmp_path / "inference.yml"), "naturelm_device": "cpu",
    }


def runtime_factory(outputs):
    torch = pytest.importorskip("torch")
    seen = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.outputs = iter(outputs)

        def generate(self, batch, config, prompts):
            assert not torch.is_grad_enabled()
            assert not self.training and not self.weight.requires_grad
            assert config.do_sample is False
            assert batch["audio_chunk_sizes"] == 1
            assert batch["padding_mask"].dtype == torch.bool
            return [next(self.outputs)]

    def processor(audios, prompts, rates):
        assert len(audios) == len(prompts) == len(rates) == 1
        seen.append({"n": len(audios[0]), "prompt": prompts[0], "rate": rates[0]})
        return torch.from_numpy(audios[0][None].copy()), list(prompts)

    runtime = SimpleNamespace(model=Model(), torch=torch, processor=processor,
                              generate_cfg=SimpleNamespace(num_beams=2, do_sample=True))
    return lambda _assets: runtime, runtime, seen


def query(tmp_path, seconds=1.0):
    path = tmp_path / "q.wav"
    sf.write(path, np.zeros(int(seconds * 16000), dtype=np.float32), 16000)
    return Query("registered-id", str(path), "recording-1")


def test_frozen_native_uses_registered_prompt_and_authoritative_normalizer(assets, tmp_path):
    factory, runtime, seen = runtime_factory(["False Killer Whale"])
    backend = module.NatureLMBackend(assets, factory=factory)
    result = backend.predict(query(tmp_path), "Registered prompt", ["Killer Whale", "False Killer Whale"])
    assert result["prediction"] == "False Killer Whale"
    assert result["raw_prediction"] == "False Killer Whale"
    assert result["event_id"] == "registered-id"
    assert seen == [{"n": 16000, "prompt": "Registered prompt", "rate": 16000}]
    assert result["backend_provenance"]["validation"] == "contract_test_injected"
    assert result["backend_provenance"]["akr_kv_adapter"] is False
    assert result["backend_provenance"]["task_adaptation_support_used"] == 0
    assert runtime.generate_cfg.max_new_tokens == 32
    assert runtime.generate_cfg.merging_alpha == 1.0
    saved = backend.provenance
    saved["files"].clear()
    assert backend.provenance["files"]  # property is detached from locked identity


def test_complete_audio_tail_coverage_and_disagreement_invalid(assets, tmp_path):
    factory, _, seen = runtime_factory(["cat", "dog", "dog"])
    backend = module.NatureLMBackend(assets, factory=factory)
    result = backend.predict(query(tmp_path, seconds=21), "Choose a label", ["cat", "dog"])
    assert [row["n"] for row in seen] == [160000, 160000, 16000]
    assert [(w["start_seconds"], w["end_seconds"]) for w in result["windows"]] == [(0, 10), (10, 20), (20, 21)]
    assert result["prediction"] == ""
    assert "#20.00s - 21.00s#: dog" in result["raw_prediction"]


def test_ambiguous_label_is_not_replaced_with_valid_fallback(assets, tmp_path):
    factory, _, _ = runtime_factory(["cat or dog"])
    backend = module.NatureLMBackend(assets, factory=factory)
    assert backend.predict(query(tmp_path), "Prompt", ["cat", "dog"])["prediction"] == ""


def test_missing_assets_and_unpinned_revision_rejected_before_factory(assets):
    with pytest.raises(ValueError, match="Missing local NatureLM assets"):
        module.NatureLMBackend({})
    with pytest.raises(ValueError, match="immutable"):
        module.NatureLMBackend({**assets, "naturelm_checkpoint_revision": "main"})
    with pytest.raises(ValueError, match="differs from inspected"):
        module.NatureLMBackend({**assets, "naturelm_source_revision": "c" * 40})


def test_bad_source_and_local_checkpoint_requirements_fail_closed(assets, monkeypatch):
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(ValueError, match="HEAD"):
        module.NatureLMBackend(assets)


def test_weight_index_cannot_escape_snapshot(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"key": "../elsewhere.safetensors"}}))
    with pytest.raises(ValueError, match="local top-level"):
        module._weight_files(tmp_path)


def test_invalid_query_contract_and_generation_shape_fail(assets, tmp_path):
    factory, _, _ = runtime_factory(["cat"])
    backend = module.NatureLMBackend(assets, factory=factory)
    with pytest.raises(TypeError, match="label-free"):
        backend.predict({"audio_path": "q.wav", "label": "cat"}, "Prompt", ["cat"])
    with pytest.raises(ValueError, match="prompt"):
        backend.predict(query(tmp_path), "", ["cat"])
    with pytest.raises(ValueError, match="distinct"):
        backend.predict(query(tmp_path), "Prompt", ["cat", "cat"])
    backend.runtime.model.generate = lambda *args: "cat"
    with pytest.raises(RuntimeError, match="exactly one raw string"):
        backend.predict(query(tmp_path), "Prompt", ["cat"])


def test_only_explicit_model_assets_hashed_and_content_changes_identity(assets):
    factory, _, _ = runtime_factory([])
    first = module.NatureLMBackend(assets, factory=factory)
    Path(assets["naturelm_code"], "unrelated_large_artifact.bin").write_bytes(b"not a model input")
    factory, _, _ = runtime_factory([])
    assert module.NatureLMBackend(assets, factory=factory).provenance == first.provenance
    Path(assets["naturelm_checkpoint"], "model.safetensors").write_bytes(b"changed")
    factory, _, _ = runtime_factory([])
    assert module.NatureLMBackend(assets, factory=factory).provenance["fingerprint"] != first.provenance["fingerprint"]


def test_measured_specialist_counts_windows_with_explicit_unit(assets,tmp_path):
    from akr_final.rebuttal_runner import measured
    factory,_,_=runtime_factory(['cat','cat','cat'])
    backend=module.NatureLMBackend(assets,factory=factory)
    result=measured(backend,lambda:backend.predict(query(tmp_path,seconds=21),'Prompt',['cat','dog']))
    assert result['new_model_calls']==3
    assert result['model_call_unit']=='NatureLM generate calls per audio window; not decoder forwards or FLOPs'
