"""Local-only acceptance of native MTP semantics using a released train question.

This is software/interface qualification, never a benchmark result. It performs
generation and backpropagation but no optimizer step, and retains actual evidence.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
import traceback
import uuid
from pathlib import Path

from .assets import atomic_json, check_assets, sha256_file
from .config import HARNESS_REVISION, MODEL_REVISION, REPO_ROOT


def source_identity() -> dict[str, str]:
    paths = list((REPO_ROOT / "mtp_opsd").rglob("*.py"))
    paths += list((REPO_ROOT / "configs" / "mtp_opsd").glob("*.json"))
    paths.append(REPO_ROOT / "requirements-mtp-opsd.txt")
    return {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in sorted(paths)}


def source_digest() -> str:
    return hashlib.sha256(json.dumps(source_identity(), sort_keys=True).encode()).hexdigest()


def check_environment() -> dict:
    required = {
        "torch": "2.10.0", "transformers": "5.16.1", "accelerate": "1.15.0",
        "datasets": "4.8.0", "huggingface-hub": "1.7.2", "safetensors": "0.8.0",
        "math-verify": "0.9.0", "antlr4-python3-runtime": "4.11.0",
    }
    versions = {}
    for package, expected in required.items():
        actual = importlib.metadata.version(package)
        versions[package] = actual
        if actual.split("+", 1)[0] != expected:
            raise RuntimeError(f"Expected {package}=={expected}, found {actual}; use the isolated pinned environment")
    harness = importlib.metadata.distribution("lm_eval")
    source = json.loads(harness.read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != HARNESS_REVISION:
        raise RuntimeError("Install lm_eval from the exact Git revision in requirements-mtp-opsd.txt")
    return {"python": platform.python_version(), "packages": versions,
            "harness_revision": HARNESS_REVISION, "harness_version": harness.version}


def structural_contract(config: dict) -> dict:
    model = config["model"]
    return {
        "core": {key: model[key] for key in (
            "id", "revision", "dtype", "attn_implementation",
            "max_context_tokens", "enable_thinking",
        )},
        "training": {
            "gradient_checkpointing": model["gradient_checkpointing"],
            "lora": {**model["lora"], "target_modules": sorted(model["lora"]["target_modules"])},
        },
    }


def require_qualification(path: str | Path, config: dict) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Native MTP qualification is missing: {path}. Run the qualify stage first.")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "passed":
        raise RuntimeError("Native MTP qualification has not passed; inspect its recorded failures")
    if receipt.get("source_digest") != source_digest() or receipt.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("Qualification is for different source/model bytes; qualify this revision")
    current = check_environment()
    if receipt.get("environment") != current:
        raise RuntimeError("Qualification environment differs from the current pinned environment")
    expected, actual = structural_contract(config), receipt.get("structural_contract", {})
    if actual.get("core") != expected["core"]:
        raise RuntimeError("Model runtime settings differ from qualification; qualify the same overrides")
    if config["kind"] == "train" and actual.get("training") != expected["training"]:
        raise RuntimeError("Trainable structure differs from qualification; qualify the same overrides")
    return receipt


def _adapter_digest(backend) -> dict:
    return {
        name: hashlib.sha256(p.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()
        for name, p in backend.model.named_parameters()
        if name.endswith(("lora_A", "lora_B"))
    }


def _selected_logprobs(backend, prefix: list[int], response: list[int], mode: str) -> list[float]:
    import torch
    with torch.no_grad():
        states = backend.prepare_hidden(prefix + response, include_mtp=mode == "mtp")
        values = []
        for j, token in enumerate(response):
            head = "head1" if mode == "mtp" and j % 2 == 1 else "head0"
            row = len(prefix) + j - (2 if head == "head1" else 1)
            logits = backend.project_logits(states[head][0, row:row + 1]).float()
            values.append(float(logits.log_softmax(-1)[0, token].item()))
        return values


def run_qualification(config: dict, assets_root: str | Path, output: str | Path) -> dict:
    import torch
    from .backend import load_backend
    from .data import load_training_examples
    from .losses import full_vocab_forward_kl
    from .train import _backward_trajectory

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    invocation = uuid.uuid4().hex
    started = time.monotonic()
    result = {
        "invocation_id": invocation, "status": "running", "source_digest": source_digest(),
        "source_files": source_identity(), "model_revision": MODEL_REVISION,
        "structural_contract": structural_contract(config),
        "purpose": "native software semantics, not a benchmark or method-effectiveness result",
        "optimizer_steps": 0,
    }
    backend = None
    try:
        result["environment"] = check_environment()
        result["assets"] = check_assets(assets_root, full_hashes=False)
        checks = {}
        result["checks"] = checks
        # Exercise the actual offline native task/scorer interfaces before any
        # expensive model operation; this does not generate benchmark answers.
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE"):
            os.environ[key] = "1"
        from .evaluate import TASKS, _native_task, _source_identity
        harness_root, evaluator_sources = _source_identity()
        checks["native_evaluator_sources"] = evaluator_sources
        checks["native_task_documents"] = {
            name: len(_native_task(harness_root, name, config).eval_docs)
            for name in TASKS
        }
        if not torch.cuda.is_available():
            raise RuntimeError("This 4B native qualification requires the actual CUDA execution host")
        torch.cuda.set_device(torch.device(config["model"]["device"]))
        torch.manual_seed(config["training"]["seed"])
        torch.cuda.reset_peak_memory_stats()
        backend = load_backend(config["model"], trainable=True)
        example = load_training_examples(config)[0]
        result["native_input_id"] = example["id"]
        result["backend"] = backend.provenance
        prompt = backend.make_prompt(example["question"])
        generated = {}
        # Both cache paths must agree with the all-prefix scorer used by KD.
        for mode in ("ntp", "mtp"):
            generation = backend.generate(
                prompt, mode=mode, max_new_tokens=32, temperature=0.8, top_p=0.95,
                seed=config["training"]["seed"], record_logprobs=True,
            )
            if len(generation.token_ids) < 4:
                raise RuntimeError("Native calibration completion is too short to qualify multi-step cache alignment")
            recomputed = _selected_logprobs(backend, prompt, generation.token_ids, mode)
            errors = [abs(a - b) for a, b in zip(generation.raw_logprobs, recomputed, strict=True)]
            maximum = max(errors)
            checks[f"{mode}_cache_full_sequence"] = {
                "max_abs_selected_logprob_error": maximum,
                "mean_abs_selected_logprob_error": sum(errors) / len(errors),
                "tolerance": 0.15, "head_counts": generation.head_counts,
                "cached_logprobs": generation.raw_logprobs, "full_sequence_logprobs": recomputed,
            }
            if maximum > 0.15:
                raise RuntimeError(f"{mode} cache/all-prefix parity failed ({maximum}); inspect positions/cache before training")
            if mode == "mtp" and generation.head_counts["head1"] == 0:
                raise RuntimeError("Native MTP generation never used its pretrained future head")
            generated[mode] = generation
        result["native_generations"] = {key: value.to_dict() for key, value in generated.items()}

        response = generated["mtp"].token_ids[:16]
        backend.model.eval()
        with torch.no_grad():
            states = backend.prepare_hidden(prompt + response, include_mtp=True)
            same_logits = backend.project_logits(states["head1"][0, len(prompt)-1:len(prompt)+3])
            self_kl = float(full_vocab_forward_kl(same_logits, same_logits).item())
        del states, same_logits
        checks["identical_teacher_student_kl"] = self_kl
        if abs(self_kl) > 1e-5:
            raise RuntimeError("Identical-distribution FKL must be approximately zero")

        # A real self-generated snippet supplies different teacher context. It is
        # not treated as a correct answer or used to compute a benchmark score.
        teacher_prompt = backend.make_prompt(example["question"], generated["ntp"].text)
        before = dict(backend.counters)
        backend.model.zero_grad(set_to_none=True)
        stats = _backward_trajectory(
            backend, prompt, teacher_prompt, response, head_weights=[1.0, 1.0],
            chunk_size=config["training"]["logit_chunk_size"], temperature=1.0, target_count=1,
        )
        gradient_sums = {"backbone": 0.0, "mtp": 0.0}
        for name, parameter in backend.model.named_parameters():
            if parameter.grad is None:
                continue
            if not parameter.requires_grad or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Frozen or nonfinite gradient: {name}")
            gradient_sums["mtp" if name.startswith("mtp.") else "backbone"] += float(parameter.grad.abs().sum().item())
        if min(gradient_sums.values()) <= 0:
            raise RuntimeError(f"The KL must update both main-model and MTP adapters: {gradient_sums}")
        mtp_calls = backend.counters["mtp_forward"] - before.get("mtp_forward", 0)
        if mtp_calls != 2:
            raise RuntimeError("Distillation must execute the native MTP module for BOTH teacher and student")
        checks["self_generated_reference_kd"] = {"losses": stats, "gradient_abs_sums": gradient_sums,
                                                 "teacher_plus_student_mtp_calls": mtp_calls}
        backend.model.zero_grad(set_to_none=True)
        backend.model.eval()
        original_parameters = _adapter_digest(backend)
        original_logprobs = _selected_logprobs(backend, prompt, response, "mtp")
        with tempfile.TemporaryDirectory(prefix="native_mtp_roundtrip_", dir=output.parent) as temporary:
            backend.save_adapter(Path(temporary) / "adapter")
            del backend
            backend = None
            gc.collect()
            torch.cuda.empty_cache()
            backend = load_backend(config["model"], adapter_dir=Path(temporary) / "adapter", trainable=False)
            if _adapter_digest(backend) != original_parameters:
                raise RuntimeError("Adapter save/reload changed parameter bytes")
            restored = _selected_logprobs(backend, prompt, response, "mtp")
            difference = max(abs(a-b) for a, b in zip(restored, original_logprobs, strict=True))
            if difference > 1e-5:
                raise RuntimeError(f"Adapter save/reload changed model predictions: {difference}")
            checks["adapter_roundtrip"] = {"parameter_bytes_equal": True,
                                            "max_abs_selected_logprob_error": difference}
        result.update(status="passed", checks=checks,
                      gpu={"name": torch.cuda.get_device_name(),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated()},
                      runtime_acceptance_scope="offline native task/scorer construction, 32-token cache, two-head KD gradient, exact adapter restoration")
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output.parent / "qualification_history" / f"{invocation}.json", result)
        atomic_json(output, result)
        if backend is not None:
            del backend
    return result
