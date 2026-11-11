"""Online, multi-sample self-distillation using pretrained native MTP heads.

This is a migration baseline, not the proposed multi-partition block teacher.
The teacher and student share parameters.  A majority-consistent self-generated
reference is supplied only to the teacher.  All optimization for a rollout group
uses one unchanged parameter state, then one optimizer step is taken.

Source delivery is generated/unexecuted.  Runtime receipts below are produced
only when the user executes this module through the delivered CLI/runbook.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from .answers import select_reference
from .backend import load_backend
from .data import load_training_examples
from .losses import chunk_ranges, full_vocab_forward_kl, response_slices


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with tmp.open("x", encoding="utf-8") as f:
            json.dump(_json_safe(value), f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_canonical_json(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _git_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--", "mtp_opsd", "configs/mtp_opsd", "scripts/mtp_opsd"], cwd=root, check=True, capture_output=True, text=True).stdout
        return {"commit": sha, "scoped_dirty_patch": diff, "scoped_dirty_patch_sha256": hashlib.sha256(diff.encode()).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "scoped_dirty_patch": None, "scoped_dirty_patch_sha256": None}


def _source_identity() -> dict[str, str]:
    """Hash the actual loaded package source, including untracked local repairs."""
    root = Path(__file__).resolve().parent
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def _capture_rng() -> dict[str, Any]:
    result: dict[str, Any] = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state = np.random.get_state()
        result["numpy"] = [state[0], state[1].tolist(), state[2], state[3], state[4]]
    except ImportError:
        pass
    return result


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Resume requires the same visible CUDA device topology")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        import numpy as np
        saved = state["numpy"]
        np.random.set_state((saved[0], np.asarray(saved[1], dtype=np.uint32), saved[2], saved[3], saved[4]))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass


def _sample_seed(seed: int, source_id: str, group_index: int) -> int:
    value = f"{seed}|{source_id}|{group_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63 - 1)


def _generation_dict(result: Any, index: int, seed: int) -> dict[str, Any]:
    data = _json_safe(result)
    if not isinstance(data, dict):
        raise TypeError("backend.generate must return a dataclass or mapping")
    if not isinstance(data.get("token_ids"), list) or not isinstance(data.get("text"), str):
        raise TypeError("Generation must contain token_ids:list and text:str")
    if data.get("finish_reason") not in {"eos", "stop", "length"}:
        raise ValueError("Generation must distinguish eos/stop from length truncation")
    data.update(index=index, seed=seed, generated_tokens=len(data["token_ids"]))
    return data


def _prompt_ids(backend: Any, question: str, reference: str | None = None) -> list[int]:
    ids = backend.make_prompt(question, reference=reference)
    if isinstance(ids, torch.Tensor):
        ids = ids.squeeze(0).tolist()
    if not isinstance(ids, list) or not ids or not all(isinstance(i, int) for i in ids):
        raise TypeError("backend.make_prompt must return a nonempty token-ID list")
    return ids


def _align_hidden(hidden: Mapping[str, torch.Tensor], head: str, start: int, length: int) -> torch.Tensor:
    if head not in hidden:
        raise RuntimeError(f"Native backend omitted required {head}; NTP fallback is forbidden")
    tensor = hidden[head]
    if tensor.ndim != 3 or tensor.shape[0] != 1 or start < 0 or start + length > tensor.shape[1]:
        raise RuntimeError(f"Invalid {head} alignment: tensor={tuple(tensor.shape)}, start={start}, length={length}")
    return tensor[0, start : start + length]


def _backward_trajectory(
    backend: Any,
    student_prompt: list[int],
    teacher_prompt: list[int],
    response_ids: list[int],
    *,
    head_weights: list[float],
    chunk_size: int,
    temperature: float,
    target_count: int,
) -> dict[str, Any]:
    """Accumulate exact hidden-state VJPs, then backpropagate through the LM once.

    The frozen vocabulary projector allows each small logit chunk to accumulate
    dL/dhidden on detached leaves.  One final multi-root backward applies the
    chain rule to the original head0/head1 states.  Calling backward through the
    whole LM for every vocabulary chunk would repeat expensive checkpoint
    recomputation O(response_length/chunk_size) times.
    """
    if any(parameter.requires_grad for parameter in backend.model.base.lm_head.parameters()):
        raise RuntimeError("Chunked hidden-state VJP requires the shared vocabulary projector to remain frozen")
    include_mtp = head_weights[1] > 0
    slices = response_slices(len(student_prompt), len(teacher_prompt), len(response_ids), include_mtp=include_mtp)
    active_slices = [s for s in slices if head_weights[int(s.head[-1])] > 0]
    weight_denominator = sum(head_weights[int(s.head[-1])] for s in active_slices)
    if weight_denominator <= 0 or target_count <= 0:
        raise ValueError("Distillation needs positive head weights and target count")
    device = backend.device
    teacher_ids = torch.tensor([teacher_prompt + response_ids], dtype=torch.long, device=device)
    backend.model.eval()
    teacher_mtp_before = int(backend.counters.get("mtp_forward", 0))
    with torch.no_grad():
        full_teacher = backend.prepare_hidden(teacher_ids, include_mtp=include_mtp)
        teacher_states = {
            s.head: _align_hidden(full_teacher, s.head, s.teacher_start, s.length).detach().cpu()
            for s in active_slices
        }
    teacher_mtp_forwards = int(backend.counters.get("mtp_forward", 0)) - teacher_mtp_before
    if teacher_mtp_forwards != int(include_mtp):
        raise RuntimeError("Teacher native-MTP call count does not match the requested distillation mode")
    del full_teacher, teacher_ids

    student_ids = torch.tensor([student_prompt + response_ids], dtype=torch.long, device=device)
    backend.model.train()
    student_mtp_before = int(backend.counters.get("mtp_forward", 0))
    full_student = backend.prepare_hidden(student_ids, include_mtp=include_mtp)
    student_mtp_forwards = int(backend.counters.get("mtp_forward", 0)) - student_mtp_before
    if student_mtp_forwards != int(include_mtp):
        raise RuntimeError("Student native-MTP call count does not match the requested distillation mode")
    student_states = {
        s.head: _align_hidden(full_student, s.head, s.student_start, s.length)
        for s in active_slices
    }
    projection_leaves = {
        name: tensor.detach().requires_grad_(True)
        for name, tensor in student_states.items()
    }
    del student_ids
    stats: dict[str, Any] = {
        "head0": {"kl_sum": 0.0, "positions": 0}, "head1": {"kl_sum": 0.0, "positions": 0},
        "objective_contribution": 0.0,
        "teacher_mtp_forwards": teacher_mtp_forwards,
        "student_mtp_forwards": student_mtp_forwards,
    }
    for spec in active_slices:
        head_weight = head_weights[int(spec.head[-1])]
        scale = head_weight / (weight_denominator * spec.length * target_count)
        for start, end in chunk_ranges(spec.length, chunk_size):
            with torch.no_grad():
                teacher_h = teacher_states[spec.head][start:end].to(device=device)
                teacher_logits = backend.project_logits(teacher_h)
            student_logits = backend.project_logits(projection_leaves[spec.head][start:end])
            raw_loss = full_vocab_forward_kl(student_logits, teacher_logits, temperature=temperature)
            value = float(raw_loss.detach().item())
            stats[spec.head]["kl_sum"] += value
            stats["objective_contribution"] += value * scale
            stats[spec.head]["positions"] += end - start
            (raw_loss * scale).backward()
            del raw_loss, student_logits, teacher_logits, teacher_h
    gradients = [projection_leaves[s.head].grad for s in active_slices]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("Missing gradient from a distillation head to its projection state")
    torch.autograd.backward(
        [student_states[s.head] for s in active_slices],
        grad_tensors=gradients,
    )
    del full_student, student_states, projection_leaves, teacher_states, gradients
    return stats


def _checkpoint_files(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*")) if p.is_file() and p.name != "manifest.json"
    }


def _save_checkpoint(
    backend: Any, optimizer: torch.optim.Optimizer, run_dir: Path,
    state: dict[str, Any], identity: dict[str, Any], selection_rng: random.Random,
) -> Path:
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    name = f"update_{state['updates']:06d}_prompt_{state['cursor']:07d}_{uuid.uuid4().hex[:8]}"
    tmp = checkpoints / (".writing_" + name)
    final = checkpoints / name
    tmp.mkdir()
    try:
        backend.save_adapter(tmp / "adapter")
        torch.save(
            {"optimizer": optimizer.state_dict(), "rng": _capture_rng(), "selection_rng": selection_rng.getstate()},
            tmp / "training_state.pt",
        )
        _write_json(tmp / "state.json", state)
        _write_json(tmp / "identity.json", identity)
        _write_json(tmp / "manifest.json", {"files": _checkpoint_files(tmp), "created_at": _utc_now()})
        os.replace(tmp, final)
        _write_json(run_dir / "latest_checkpoint.json", {"path": str(final.relative_to(run_dir)), "updates": state["updates"], "cursor": state["cursor"]})
    except BaseException:
        # Preserve a partial checkpoint for diagnosis; it is never advertised
        # as latest or used by resume.
        raise
    return final


def _resolve_checkpoint(run_dir: Path) -> Path:
    pointer = json.loads((run_dir / "latest_checkpoint.json").read_text(encoding="utf-8"))
    candidate = (run_dir / pointer["path"]).resolve()
    if run_dir.resolve() not in candidate.parents:
        raise ValueError("Checkpoint pointer escapes the run directory")
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    actual = _checkpoint_files(candidate)
    if actual != manifest["files"]:
        raise RuntimeError("Checkpoint checksum mismatch; refusing an incomplete or changed resume")
    return candidate


def _validate_settings(config: dict[str, Any]) -> None:
    sampling, training = config["sampling"], config["training"]
    if sampling["mode"] not in {"ntp", "mtp"} or sampling["group_size"] not in {4, 8}:
        raise ValueError("Delivered cases use mode ntp/mtp and group_size 4/8")
    if sampling.get("top_k", 0) != 0:
        raise ValueError("The delivered sampler uses nucleus sampling; top_k must be 0")
    if training.get("supervision", "all_prefixes") != "all_prefixes":
        raise ValueError("Only the audited all_prefixes positional baseline is implemented")
    weights = training["head_weights"]
    if len(weights) != 2 or any(not math.isfinite(w) or w < 0 for w in weights) or weights[0] <= 0:
        raise ValueError("head_weights must have two nonnegative values and positive head0")
    if (sampling["mode"] == "mtp") != (weights[1] > 0):
        raise ValueError("MTP cases must supervise both native heads; NTP cases must use head0 only")
    for key in ("max_prompts", "max_updates", "logit_chunk_size", "save_every_updates"):
        if training[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    for key in ("learning_rate", "gradient_clip", "max_hours", "kd_temperature"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if config["selection"].get("target_selection", "longest") != "longest":
        raise ValueError("The delivered target selection is longest disagreeing")
    if config["selection"].get("reference", "longest") != "longest" or config["selection"].get("distill", "disagreeing") != "disagreeing":
        raise ValueError("The baseline requires longest agreeing reference and disagreeing targets")
    if config["selection"].get("max_targets", 1) < 1:
        raise ValueError("selection.max_targets must be positive")


def run_training(
    config: dict[str, Any], run_dir: str | Path, *, resume: bool = False, backend: Any = None,
) -> dict[str, Any]:
    """Run one independently configured case; never silently overwrite another run."""
    _validate_settings(config)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".training.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Training lock exists at {lock}; inspect owner/liveness before manual recovery") from exc
    with os.fdopen(lock_fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "hostname": platform.node(), "created_at": _utc_now()}, f)

    invocation_id = uuid.uuid4().hex
    invocation_start = time.monotonic()
    elapsed_before = 0.0
    budget_totals: dict[str, Any] = {"elapsed_train_seconds": 0.0, "sampled_tokens": 0, "sampled_head0_tokens": 0, "sampled_head1_tokens": 0}
    checkpoint: Path | None = None
    optimizer: torch.optim.Optimizer | None = None
    state: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None
    selection_rng = random.Random(int(config["training"]["seed"]))
    reached_update_boundary = True
    budget_write_allowed = False
    optimizer_commit_in_progress = False
    checkpoint_ready = False
    active_prompt_cursor: int | None = None
    selection_rng_before_prompt = None
    try:
        config_digest = _digest(config)
        budget_path = run_dir / "budget_ledger.json"
        if budget_path.exists():
            budget_totals = json.loads(budget_path.read_text(encoding="utf-8"))
            if not resume or budget_totals.get("config_sha256") != config_digest:
                raise RuntimeError("Existing budget ledger belongs to this run; use matching resume or a new run")
        source_identity = _source_identity()
        training = config["training"]
        sampling = config["sampling"]
        selection_cfg = config["selection"]
        dataset = load_training_examples(config)
        if not dataset:
            raise RuntimeError("Training dataset is empty")
        if any(set(example) != {"id", "question", "source_index"} for example in dataset):
            raise RuntimeError("Training loader exposed unexpected fields; labels must not reach self-teaching")
        order = list(range(len(dataset)))
        random.Random(int(training["seed"])).shuffle(order)
        order = order[: min(len(order), int(training["max_prompts"]))]
        order_digest = _digest([{key: dataset[i][key] for key in ("id", "question", "source_index")} for i in order])
        _seed_everything(int(training["seed"]))
        identity = {
            "config_sha256": config_digest,
            "source_files_sha256": source_identity,
            "ordered_training_prompts_sha256": order_digest,
            "training_dataset_size": len(dataset),
            "prompt_budget": len(order),
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "mode": sampling["mode"],
            "group_size": sampling["group_size"],
        }
        if resume:
            checkpoint = _resolve_checkpoint(run_dir)
            saved_identity = json.loads((checkpoint / "identity.json").read_text(encoding="utf-8"))
            # Trainable names are checked after loading the adapter below.
            for key, value in identity.items():
                if saved_identity.get(key) != value:
                    raise RuntimeError(f"Resume identity mismatch for {key}; start a child run instead")
            state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
            elapsed_before = max(float(state["elapsed_train_seconds"]), float(budget_totals["elapsed_train_seconds"]))
        else:
            if (run_dir / "training_config.json").exists() or (run_dir / "latest_checkpoint.json").exists():
                raise FileExistsError("Run already contains training evidence; use --resume or a new run ID")
            _write_json(run_dir / "training_config.json", config)
            _write_json(run_dir / "source_identity.json", {**_git_identity(), "source_files_sha256": source_identity})
            state = {
                "cursor": 0, "updates": 0, "sampled_tokens": 0,
                "sampled_head0_tokens": 0, "sampled_head1_tokens": 0,
                "supervised_head0_positions": 0, "supervised_head1_positions": 0,
                "teacher_mtp_forwards": 0, "student_mtp_forwards": 0,
                "skip_counts": {}, "elapsed_train_seconds": 0.0,
                "created_at": _utc_now(),
            }
        if not 0 <= state["cursor"] <= len(order) or state["updates"] > training["max_updates"]:
            raise RuntimeError("Saved cursor/update count exceeds this run's fixed budget")
        budget_write_allowed = True
        if backend is not None and resume:
            raise ValueError("Resume loads its own verified adapter; do not inject a backend")
        if backend is None:
            backend = load_backend(
                config["model"], adapter_dir=(checkpoint / "adapter") if checkpoint else None,
                trainable=True, device=config["model"].get("device", "cuda:0"),
            )
        trainable_named = [(name, parameter) for name, parameter in backend.model.named_parameters() if parameter.requires_grad]
        if not trainable_named:
            raise RuntimeError("No trainable model parameters")
        identity["trainable_parameters"] = [{"name": name, "shape": list(parameter.shape)} for name, parameter in trainable_named]
        if resume and saved_identity.get("trainable_parameters") != identity["trainable_parameters"]:
            raise RuntimeError("Resume trainable parameter names/shapes changed")
        parameters = [parameter for _, parameter in trainable_named]
        optimizer = torch.optim.AdamW(parameters, lr=training["learning_rate"], weight_decay=training.get("weight_decay", 0.0))
        if resume:
            saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(saved["optimizer"])
            _restore_rng(saved["rng"])
            selection_rng.setstate(saved["selection_rng"])
            del saved
        optimizer.zero_grad(set_to_none=True)
        _write_json(run_dir / "environment.json", {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "device": str(backend.device),
            "trainable_parameter_count": sum(p.numel() for p in parameters),
            "invocation_id": invocation_id,
        })
        _write_json(run_dir / "backend_provenance.json", backend.provenance)
        if not resume:
            state["elapsed_train_seconds"] = elapsed_before + time.monotonic() - invocation_start
            checkpoint = _save_checkpoint(backend, optimizer, run_dir, state, identity, selection_rng)
        checkpoint_ready = True
        status = "running"
        _write_json(run_dir / "training_status.json", {"status": status, "invocation_id": invocation_id, "started_at": _utc_now(), "resume_from": str(checkpoint) if resume else None})
        stop_reason = "prompt_budget_exhausted"
        while state["cursor"] < len(order):
            elapsed = elapsed_before + time.monotonic() - invocation_start
            if elapsed >= float(training["max_hours"]) * 3600:
                stop_reason = "wall_time_budget_exhausted"
                break
            if state["updates"] >= int(training["max_updates"]):
                stop_reason = "update_budget_exhausted"
                break
            cursor = int(state["cursor"])
            active_prompt_cursor = cursor
            selection_rng_before_prompt = selection_rng.getstate()
            example = dataset[order[cursor]]
            prompt = _prompt_ids(backend, example["question"])
            record: dict[str, Any] = {
                "invocation_id": invocation_id, "prompt_cursor": cursor,
                "sample_id": example["id"], "source_index": example["source_index"],
                "question": example["question"], "updates_before": state["updates"],
                "student_prompt_tokens": len(prompt), "rollouts": [], "created_at": _utc_now(),
            }
            prompt_record = run_dir / "rollouts" / invocation_id / f"prompt_{cursor:07d}.json"
            if len(prompt) + sampling["max_new_tokens"] > int(config["model"]["max_context_tokens"]):
                skip = "student_context_limit"
                record["status"] = skip
                state["skip_counts"][skip] = state["skip_counts"].get(skip, 0) + 1
                state["cursor"] += 1
                _write_json(prompt_record, record)
                _append_jsonl(run_dir / "metrics.jsonl", {key: value for key, value in record.items() if key not in {"question", "rollouts"}})
                continue
            backend.model.eval()
            interrupted_group = False
            for group_index in range(int(sampling["group_size"])):
                if elapsed_before + time.monotonic() - invocation_start >= float(training["max_hours"]) * 3600:
                    interrupted_group = True
                    break
                seed = _sample_seed(int(sampling.get("seed", training["seed"])), str(example["id"]), group_index)
                with torch.no_grad():
                    result = backend.generate(
                        prompt, mode=sampling["mode"], max_new_tokens=int(sampling["max_new_tokens"]),
                        temperature=float(sampling["temperature"]), top_p=float(sampling["top_p"]), seed=seed,
                    )
                rollout = _generation_dict(result, group_index, seed)
                record["rollouts"].append(rollout)
                state["sampled_tokens"] += rollout["generated_tokens"]
                counts = rollout.get("head_counts", {})
                state["sampled_head0_tokens"] += int(counts.get("head0", 0))
                state["sampled_head1_tokens"] += int(counts.get("head1", 0))
                budget_totals["sampled_tokens"] += rollout["generated_tokens"]
                budget_totals["sampled_head0_tokens"] += int(counts.get("head0", 0))
                budget_totals["sampled_head1_tokens"] += int(counts.get("head1", 0))
                budget_totals.update(config_sha256=config_digest, elapsed_train_seconds=elapsed_before + time.monotonic() - invocation_start, updated_at=_utc_now())
                _write_json(budget_path, budget_totals)
                _write_json(prompt_record, {**record, "status": "sampling"})
            if interrupted_group:
                record["status"] = "wall_time_interrupted_group"
                _write_json(prompt_record, record)
                stop_reason = "wall_time_budget_exhausted"
                break
            selection = select_reference(record["rollouts"], threshold=float(selection_cfg["threshold"]), rng=selection_rng)
            record["selection"] = _json_safe(selection)
            if selection.reference_index is None or not selection.target_indices:
                skip = selection.status
                record["status"] = skip
                state["skip_counts"][skip] = state["skip_counts"].get(skip, 0) + 1
                state["cursor"] += 1
                _write_json(prompt_record, record)
                _append_jsonl(run_dir / "metrics.jsonl", {key: value for key, value in record.items() if key not in {"question", "rollouts"}})
                continue
            reference = record["rollouts"][selection.reference_index]
            teacher_prompt = _prompt_ids(backend, example["question"], reference=reference["text"])
            target_indices = sorted(selection.target_indices, key=lambda i: (-len(record["rollouts"][i]["token_ids"]), i))[: int(selection_cfg.get("max_targets", 1))]
            eligible: list[int] = []
            excluded: list[dict[str, Any]] = []
            for index in target_indices:
                response = record["rollouts"][index]["token_ids"]
                if len(teacher_prompt) + len(response) > int(config["model"]["max_context_tokens"]):
                    excluded.append({"index": index, "reason": "teacher_context_limit"})
                elif not response or (sampling["mode"] == "mtp" and len(response) < 2):
                    excluded.append({"index": index, "reason": "insufficient_response_tokens"})
                else:
                    eligible.append(index)
            record.update(
                teacher_prompt_tokens=len(teacher_prompt), selected_target_indices=eligible, excluded_targets=excluded,
                teacher_forward_input_tokens=sum(len(teacher_prompt) + len(record["rollouts"][index]["token_ids"]) for index in eligible),
                student_forward_input_tokens=sum(len(prompt) + len(record["rollouts"][index]["token_ids"]) for index in eligible),
            )
            if not eligible:
                skip = "no_eligible_target"
                record["status"] = skip
                state["skip_counts"][skip] = state["skip_counts"].get(skip, 0) + 1
                state["cursor"] += 1
                _write_json(prompt_record, record)
                _append_jsonl(run_dir / "metrics.jsonl", {key: value for key, value in record.items() if key not in {"question", "rollouts"}})
                continue
            reached_update_boundary = False
            optimizer.zero_grad(set_to_none=True)
            head_stats = {"head0": {"kl_sum": 0.0, "positions": 0}, "head1": {"kl_sum": 0.0, "positions": 0}}
            objective = 0.0
            teacher_mtp_forwards = 0
            student_mtp_forwards = 0
            for index in eligible:
                statistics = _backward_trajectory(
                    backend, prompt, teacher_prompt, record["rollouts"][index]["token_ids"],
                    head_weights=training["head_weights"], chunk_size=int(training["logit_chunk_size"]),
                    temperature=float(training["kd_temperature"]), target_count=len(eligible),
                )
                for head in head_stats:
                    head_stats[head]["kl_sum"] += statistics[head]["kl_sum"]
                    head_stats[head]["positions"] += statistics[head]["positions"]
                objective += statistics["objective_contribution"]
                teacher_mtp_forwards += statistics["teacher_mtp_forwards"]
                student_mtp_forwards += statistics["student_mtp_forwards"]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]), error_if_nonfinite=True)
            if not any(parameter.grad is not None for parameter in parameters):
                raise RuntimeError("No student gradient reached trainable parameters")
            optimizer_commit_in_progress = True
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["updates"] += 1
            state["cursor"] += 1
            state["teacher_mtp_forwards"] += teacher_mtp_forwards
            state["student_mtp_forwards"] += student_mtp_forwards
            for head in head_stats:
                state[f"supervised_{head}_positions"] += head_stats[head]["positions"]
                head_stats[head]["mean_kl"] = head_stats[head]["kl_sum"] / head_stats[head]["positions"] if head_stats[head]["positions"] else None
            optimizer_commit_in_progress = False
            reached_update_boundary = True
            record.update(
                status="updated", updates_after=state["updates"], objective=objective,
                head_statistics=head_stats, gradient_norm=float(grad_norm.detach().item()),
                teacher_mtp_forwards=teacher_mtp_forwards,
                student_mtp_forwards=student_mtp_forwards,
            )
            _write_json(prompt_record, record)
            _append_jsonl(run_dir / "metrics.jsonl", {key: value for key, value in record.items() if key not in {"question", "rollouts"}})
            state["elapsed_train_seconds"] = elapsed_before + time.monotonic() - invocation_start
            if state["updates"] % int(training["save_every_updates"]) == 0:
                checkpoint = _save_checkpoint(backend, optimizer, run_dir, state, identity, selection_rng)
            _write_json(run_dir / "training_status.json", {"status": "running", "invocation_id": invocation_id, **state})
        state["elapsed_train_seconds"] = elapsed_before + time.monotonic() - invocation_start
        checkpoint = _save_checkpoint(backend, optimizer, run_dir, state, identity, selection_rng)
        budget_totals.update(config_sha256=config_digest, elapsed_train_seconds=elapsed_before + time.monotonic() - invocation_start, updated_at=_utc_now())
        summary = {
            "status": "completed" if state["updates"] > 0 else "completed_no_updates",
            "learning_updates_observed": state["updates"] > 0,
            "stop_reason": stop_reason, "invocation_id": invocation_id,
            "checkpoint": str(checkpoint.relative_to(run_dir)), "adapter_dir": str((checkpoint / "adapter").relative_to(run_dir)),
            "config_sha256": config_digest, "completed_at": _utc_now(), **state,
            "cumulative_resource_counts_including_replays": budget_totals,
            "scope": "budgeted development baseline; no claim of convergence or benchmark improvement",
        }
        _write_json(run_dir / "training_summary.json", summary)
        _write_json(run_dir / "training_status.json", summary)
        return summary
    except BaseException as exc:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        emergency_checkpoint_error = None
        if checkpoint_ready and not optimizer_commit_in_progress and identity is not None and state is not None:
            # A failed forward/backward has not changed model weights.  Preserve
            # all completed updates and replay only this unfinished prompt.
            if state["cursor"] == active_prompt_cursor and selection_rng_before_prompt is not None:
                selection_rng.setstate(selection_rng_before_prompt)
            state["elapsed_train_seconds"] = elapsed_before + time.monotonic() - invocation_start
            try:
                checkpoint = _save_checkpoint(backend, optimizer, run_dir, state, identity, selection_rng)
            except BaseException as checkpoint_exc:
                emergency_checkpoint_error = f"{type(checkpoint_exc).__name__}: {checkpoint_exc}"
        failure = {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "invocation_id": invocation_id, "at": _utc_now(),
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "last_complete_checkpoint": str(checkpoint.relative_to(run_dir)) if checkpoint else None,
            "uncommitted_prompt_evidence_is_retained": True,
            "safe_update_boundary": reached_update_boundary,
            "optimizer_commit_in_progress": optimizer_commit_in_progress,
            "emergency_checkpoint_error": emergency_checkpoint_error,
        }
        if state is not None:
            state["elapsed_train_seconds"] = elapsed_before + time.monotonic() - invocation_start
            failure["state_at_failure"] = state
        # In-memory weights are not saved if an optimizer commit may have been
        # partial.  Resume always uses a complete, checksummed checkpoint.
        _write_json(run_dir / "failures" / f"{invocation_id}.json", failure)
        _write_json(run_dir / "training_status.json", failure)
        raise
    finally:
        if identity is not None and budget_write_allowed:
            budget_totals.update(
                config_sha256=identity["config_sha256"],
                elapsed_train_seconds=elapsed_before + time.monotonic() - invocation_start,
                updated_at=_utc_now(),
            )
            _write_json(run_dir / "budget_ledger.json", budget_totals)
        lock.unlink(missing_ok=True)
