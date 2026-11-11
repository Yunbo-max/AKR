"""Full native lm-evaluation-harness scoring for both decoding policies.

The released task prompts, filters, and scorers are loaded from the pinned
harness installation.  Explicit protocol overrides are zero shots, the model
chat template, and the common sampling budget.  This is not a reproduction of
the tasks' default greedy/few-shot leaderboard settings.

Each repetition is a separate complete native evaluation with ``repeats=1``.
This matters: the released tasks' ``take_first`` filters would otherwise discard
extra generations.  ``avg@k`` and ``pass@k`` are derived only from the retained
native per-document scores, with the complete released test denominator.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import HARNESS_REVISION, config_digest
from .data import dataset_load_spec, file_digest


TASKS = {
    "gsm8k_cot_zeroshot": {
        "dataset": "gsm8k", "num_documents": 1319,
        "yaml": "tasks/gsm8k/gsm8k-cot-zeroshot.yaml",
        "metric": "exact_match", "filter": "flexible-extract",
    },
    "minerva_math500": {
        "dataset": "MATH-500", "num_documents": 500,
        "yaml": "tasks/minerva_math/minerva_math500.yaml",
        "metric": "math_verify", "filter": "none",
    },
}

HARNESS_SOURCES = (
    "tasks/gsm8k/gsm8k-cot-zeroshot.yaml",
    "tasks/minerva_math/minerva_math500.yaml",
    "tasks/minerva_math/minerva_math_algebra.yaml",
    "tasks/minerva_math/utils.py", "tasks/_yaml_loader.py",
    "api/task.py", "api/metrics.py", "api/registry.py",
    "filters/extraction.py", "filters/selection.py", "evaluator.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plain(value: Any) -> Any:
    """JSON-normalize native NumPy scalars without silently stringifying scores."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return _plain(value.item())
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    raise TypeError(f"Unsupported native evaluation JSON value: {type(value).__name__}")


def _json_text(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(_json_text(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, values) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(_json_text(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_text(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


@contextmanager
def _evaluation_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".evaluation.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluator owns {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _source_identity() -> tuple[Path, dict]:
    """Require the declared VCS install and record actual native scorer sources."""
    import lm_eval
    import math_verify
    from lm_eval.api import metrics
    from lm_eval.api.registry import get_metric

    distribution = importlib.metadata.distribution("lm_eval")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != HARNESS_REVISION:
        raise RuntimeError(
            "Install the exact lm_eval VCS requirement from requirements-mtp-opsd.txt; "
            "a version string alone does not establish the native evaluator revision."
        )
    expected_versions = {
        "math-verify": "0.9.0", "antlr4-python3-runtime": "4.11.0",
        "latex2sympy2_extended": "1.11.0",
    }
    for name, expected in expected_versions.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Native math scorer requires {name}=={expected}, found {actual}")
    # At this commit exact_match is bundled in api/metrics.py.  Reject a plugin
    # substitution; do not fall back to unpinned evaluate.load() Hub code.
    if get_metric("exact_match") is not metrics.exact_match_fn:
        raise RuntimeError("The released local exact_match scorer was substituted")
    harness_root = Path(lm_eval.__file__).resolve().parent
    math_root = Path(math_verify.__file__).resolve().parent
    source_hashes = {name: file_digest(harness_root / name) for name in HARNESS_SOURCES}
    math_hashes = {
        path.name: file_digest(path)
        for path in sorted(math_root.glob("*.py"))
    }
    packages = {}
    for name in (
        "lm_eval", "math-verify", "antlr4-python3-runtime", "latex2sympy2_extended",
        "transformers", "torch", "datasets", "evaluate", "numpy", "sympy",
        "huggingface_hub", "safetensors",
    ):
        packages[name] = importlib.metadata.version(name)
    return harness_root, {
        "harness_revision": HARNESS_REVISION, "harness_direct_url": direct_url,
        "harness_source_sha256": source_hashes,
        "math_verify_source_sha256": math_hashes, "packages": packages,
        "python": platform.python_version(), "platform": platform.platform(),
        "exact_match_source": "bundled lm_eval.api.metrics.exact_match_fn",
    }


def _checkpoint_identity(config: dict, adapter_dir: str | Path | None) -> dict:
    if config["kind"] == "native":
        if adapter_dir is not None or config["model"]["lora"]["enabled"]:
            raise ValueError("Native evaluation requires disabled adapters and no adapter_dir")
        return {"kind": "native", "adapter_weights_sha256": None, "adapter_config_sha256": None}
    if adapter_dir is None:
        raise FileNotFoundError("Trained evaluation requires an explicit completed adapter checkpoint")
    directory = Path(adapter_dir).expanduser().resolve()
    metadata_path, weights_path = directory / "adapter_config.json", directory / "adapter.safetensors"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete adapter checkpoint: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    weights_sha = file_digest(weights_path)
    if metadata.get("weights_sha256") != weights_sha:
        raise ValueError("Adapter weights do not match their saved checksum")
    if (metadata.get("base_model_id"), metadata.get("base_revision")) != (
        config["model"]["id"], config["model"]["revision"]
    ):
        raise ValueError("Adapter belongs to a different base model revision")
    return {
        "kind": "trained", "adapter_directory": str(directory),
        "adapter_weights_sha256": weights_sha,
        "adapter_config_sha256": file_digest(metadata_path),
    }


def _native_task(harness_root: Path, name: str, config: dict):
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks._yaml_loader import load_yaml

    spec = TASKS[name]
    task_config = load_yaml(harness_root / spec["yaml"], resolve_func=True, recursive=True)
    if task_config.get("task") != name or task_config.get("output_type") != "generate_until":
        raise RuntimeError(f"Unexpected upstream task contract: {name}")
    source = dataset_load_spec(
        spec["dataset"], local_root=config["data"]["local_root"],
        cache_dir=config["data"].get("cache_dir"),
    )
    # Apply provenance BEFORE construction: ConfigurableTask loads its dataset
    # inside __init__, so overriding a previously constructed task is too late.
    task_config["dataset_path"] = source.pop("path")
    task_config["dataset_name"] = source.pop("name")
    task_config["dataset_kwargs"] = source
    task_config["num_fewshot"] = 0
    task_config["repeats"] = 1
    evaluation = config["evaluation"]
    task_config["generation_kwargs"].update({
        "do_sample": True, "temperature": evaluation["temperature"],
        "top_p": evaluation["top_p"], "top_k": 0,
        "max_gen_toks": evaluation["max_new_tokens"],
    })
    task = ConfigurableTask(config=task_config)
    task.EVAL_HARNESS_NAME = name
    if len(task.eval_docs) != spec["num_documents"]:
        raise ValueError(f"Incomplete released test split for {name}: {len(task.eval_docs)}")
    return task


def _document_seed(base_seed: int, sample_index: int, task: str, doc_id: int) -> int:
    # Exclude model/case/mode so all comparisons use paired sampling seeds.
    key = f"{base_seed}|{sample_index}|{task}|{doc_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def _make_adapter(backend, config: dict, mode: str, task_name: str,
                  sample_index: int, directory: Path):
    from lm_eval.api.model import LM

    class NativeMTPHarnessLM(LM):
        def __init__(self):
            super().__init__()
            self._device = backend.device
            self.tokenizer = backend.tokenizer
            self.max_length = backend.max_context_length
            self.eot_token_id = backend.tokenizer.eos_token_id
            self.seen: set[int] = set()

        @property
        def tokenizer_name(self):
            model = config["model"]
            return f"{model['id']}@{model['revision']}:thinking={model['enable_thinking']}"

        def chat_template(self, chat_template=False):
            if not chat_template:
                return None
            return self.tokenizer.get_chat_template()

        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            return self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
                enable_thinking=bool(config["model"]["enable_thinking"]),
            )

        def loglikelihood(self, requests):
            raise ValueError("The declared native tasks are generation-only; scoring requests are unsupported")

        def loglikelihood_rolling(self, requests):
            raise ValueError("The declared native tasks are generation-only; rolling likelihood is unsupported")

        def generate_until(self, requests, disable_tqdm=False):
            outputs = []
            evaluation = config["evaluation"]
            for request in requests:
                prompt, kwargs = request.args
                doc_id = int(request.doc_id)
                if request.task_name != task_name or doc_id in self.seen:
                    raise ValueError("Unexpected task or duplicate native generation request")
                if not 0 <= doc_id < TASKS[task_name]["num_documents"] or not isinstance(prompt, str):
                    raise ValueError("Unexpected native document identity or non-text prompt")
                self.seen.add(doc_id)
                if not kwargs.get("do_sample") or kwargs.get("top_k", 0) != 0:
                    raise ValueError("Native generation kwargs do not match the declared sampling protocol")
                stops = kwargs.get("until", [])
                if isinstance(stops, str):
                    stops = [stops]
                options = {
                    "mode": mode, "max_new_tokens": kwargs["max_gen_toks"],
                    "temperature": kwargs["temperature"], "top_p": kwargs["top_p"],
                    "seed": _document_seed(evaluation["seed"], sample_index, task_name, doc_id),
                    "stop": stops,
                }
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                entry = {
                    "task": task_name, "mode": mode, "sample_index": sample_index,
                    "source_index": doc_id, "prompt": prompt, "prompt_sha256": prompt_hash,
                    "generation_options": options,
                }
                try:
                    result = backend.generate(prompt, **options)
                    if result.finish_reason not in {"eos", "stop", "length"}:
                        raise RuntimeError(f"Unexpected generation finish: {result.finish_reason}")
                    if len(result.token_ids) > options["max_new_tokens"]:
                        raise RuntimeError("Backend exceeded the declared generation token budget")
                    if sum(result.head_counts.values()) != len(result.token_ids):
                        raise RuntimeError("Generation token/head count mismatch")
                    if mode == "ntp" and result.head_counts.get("head1", 0) != 0:
                        raise RuntimeError("NTP control unexpectedly used a future-token head")
                    entry.update({
                        "status": "generated", "text": result.text, "token_ids": result.token_ids,
                        "finish_reason": result.finish_reason, "head_counts": result.head_counts,
                        "prompt_tokens": getattr(result, "prompt_tokens", None),
                        "elapsed_seconds": getattr(result, "elapsed_seconds", None),
                    })
                    _append_jsonl(directory / "generations.jsonl", entry)
                    outputs.append(result.text)
                except BaseException as error:
                    _append_jsonl(directory / "errors.jsonl", {
                        **entry, "status": "generation_failed", "error_type": type(error).__name__,
                        "error": str(error), "time": _now(),
                    })
                    # A failed request leaves the whole pass incomplete.  Never
                    # drop it, substitute a correct answer, or shrink the test.
                    raise
            return outputs

        def get_model_info(self):
            return {
                "model_revision": config["model"]["revision"],
                "model_dtype": config["model"]["dtype"],
            }

    return NativeMTPHarnessLM()


def _scores(native_results: dict, samples: list[dict], name: str) -> dict[int, dict]:
    spec = TASKS[name]
    expected = spec["num_documents"]
    counts = native_results.get("n-samples", {}).get(name, {})
    if counts.get("original") != expected or counts.get("effective") != expected:
        raise RuntimeError(f"Native evaluator changed the full denominator for {name}: {counts}")
    rows = {}
    for sample in samples:
        if sample.get("filter") != spec["filter"]:
            continue
        source_index = sample["doc_id"]
        if not isinstance(source_index, int) or source_index in rows:
            raise RuntimeError("Duplicate or non-integer native document ID")
        score = float(sample[spec["metric"]])
        if not math.isfinite(score) or score not in {0.0, 1.0}:
            raise RuntimeError(f"Native task did not return a binary correctness score: {score}")
        if not sample.get("doc_hash") or not sample.get("target_hash"):
            raise RuntimeError("Native result is missing source/target identity hashes")
        rows[source_index] = {
            "score": score, "doc_hash": sample["doc_hash"],
            "target_hash": sample["target_hash"], "prompt_hash": sample["prompt_hash"],
        }
    if set(rows) != set(range(expected)):
        raise RuntimeError(f"Native scores do not cover every released {name} document exactly once")
    key = f"{spec['metric']},{spec['filter']}"
    native_score = float(native_results["results"][name][key])
    derived = sum(row["score"] for row in rows.values()) / expected
    if not math.isclose(native_score, derived, rel_tol=0, abs_tol=1e-10):
        raise RuntimeError("Native aggregate does not equal the retained per-document scores")
    return rows


def _load_complete_pass(directory: Path, identity_sha: str, sample_identity: dict,
                        task_name: str) -> tuple[dict, dict[int, dict]]:
    completion = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if completion.get("identity_sha256") != identity_sha or completion.get("sample") != sample_identity:
        raise RuntimeError("Completed evaluation pass belongs to a different config/checkpoint/protocol")
    expected_files = {"results.json", "samples.jsonl", "generations.jsonl"}
    if set(completion.get("files", {})) != expected_files:
        raise RuntimeError("Incomplete evaluation completion manifest")
    for name, checksum in completion["files"].items():
        if file_digest(directory / name) != checksum:
            raise RuntimeError(f"Modified evaluation artifact: {directory / name}")
    native = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    rows = _scores(native, _read_jsonl(directory / "samples.jsonl"), task_name)
    generations = _read_jsonl(directory / "generations.jsonl")
    indices = [entry["source_index"] for entry in generations]
    if len(indices) != len(rows) or set(indices) != set(rows):
        raise RuntimeError("Generation journal and native scoring denominator differ")
    return native, rows


def _run_pass(backend, config: dict, harness_root: Path, directory: Path,
              mode: str, task_name: str, sample_index: int, identity_sha: str):
    from lm_eval.evaluator import simple_evaluate

    task = _native_task(harness_root, task_name, config)
    adapter = _make_adapter(backend, config, mode, task_name, sample_index, directory)
    call_seed = (int(config["evaluation"]["seed"]) + sample_index) % (2**32)
    sample_identity = {"mode": mode, "task": task_name, "sample_index": sample_index, "seed": call_seed}
    _write_json(directory / "protocol.json", {
        "identity_sha256": identity_sha, **sample_identity, "started_at": _now(),
        "full_split": True, "limit": None, "num_fewshot": 0, "repeats": 1,
        "apply_chat_template": True, "generation_kwargs": task.get_config("generation_kwargs"),
        "document_seed_rule": "sha256(eval_seed|sample_index|task|source_index)[:8] modulo (2**63-1)",
    })
    started = time.monotonic()
    try:
        native = simple_evaluate(
            model=adapter, tasks=[task], num_fewshot=0, batch_size=1,
            device=str(backend.device), use_cache=None, cache_requests=False,
            limit=None, samples=None, bootstrap_iters=0, check_integrity=False,
            log_samples=True, predict_only=False, apply_chat_template=True,
            fewshot_as_multiturn=False, random_seed=call_seed,
            numpy_random_seed=call_seed, torch_random_seed=call_seed,
            fewshot_random_seed=call_seed,
        )
        if native is None:
            raise RuntimeError("Native evaluator returned no results; run one process per case")
        samples = native.pop("samples")[task_name]
        rows = _scores(native, samples, task_name)
        if adapter.seen != set(rows):
            raise RuntimeError("Generated and scored native document IDs differ")
        # Native per-filter results are intact; only their on-disk row order is
        # normalized to the released source order for stable comparisons.
        samples.sort(key=lambda item: (item["doc_id"], item["filter"]))
        _write_jsonl(directory / "samples.jsonl", samples)
        _write_json(directory / "results.json", native)
        _write_json(directory / "complete.json", {
            "status": "completed", "identity_sha256": identity_sha,
            "sample": sample_identity, "completed_at": _now(),
            "elapsed_seconds": time.monotonic() - started,
            "files": {name: file_digest(directory / name) for name in
                      ("results.json", "samples.jsonl", "generations.jsonl")},
        })
        return native, rows
    except BaseException as error:
        _write_json(directory / "failure.json", {
            "status": "incomplete", "error_type": type(error).__name__, "error": str(error),
            "failed_at": _now(), "denominator": TASKS[task_name]["num_documents"],
        })
        raise


def run_evaluation(config: dict, run_dir: str | Path, adapter_dir=None,
                   backend=None, *, resume: bool = False) -> dict:
    """Score every released test example; return the saved evaluation summary.

    ``resume`` reuses only checksummed complete native passes.  An interrupted
    pass is archived and rerun in full; its partial records are never overwritten
    or mixed into a score.  Changing the config/checkpoint requires a new run ID.
    """
    evaluation = config["evaluation"]
    if evaluation.get("harness_revision") != HARNESS_REVISION:
        raise ValueError("Evaluation harness revision mismatch")
    names, modes = evaluation["tasks"], evaluation["generation_modes"]
    if set(names) != set(TASKS) or len(names) != len(TASKS):
        raise ValueError("This baseline requires both complete GSM8K and MATH-500 native tasks")
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"ntp", "mtp"}:
        raise ValueError("Invalid or repeated evaluation generation mode")
    k = evaluation["num_samples"]
    if not isinstance(k, int) or k < 1 or evaluation["batch_size"] != 1:
        raise ValueError("Use positive integer num_samples and batch_size=1")
    if evaluation.get("top_k", 0) != 0 or not 0 < evaluation["top_p"] <= 1:
        raise ValueError("Evaluation uses nucleus sampling with top_k=0")
    if not math.isfinite(evaluation["temperature"]) or evaluation["temperature"] <= 0:
        raise ValueError("Evaluation sampling temperature must be finite and positive")
    if not config["data"].get("local_root"):
        raise ValueError("Acquire pinned datasets first; evaluation requires data.local_root and its manifest")
    if not config["model"].get("path"):
        raise ValueError("Acquire the pinned model first; evaluation requires model.path and its manifest")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        raise ValueError("Launch one process per independent case; distributed evaluation is not implemented")
    # All model/data files and metric code must already be acquired.  Dataset
    # builders receive local paths and the model loader uses local_files_only.
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE"):
        os.environ[key] = "1"
    checkpoint = _checkpoint_identity(config, adapter_dir)
    harness_root, sources = _source_identity()
    for spec in TASKS.values():
        dataset_load_spec(spec["dataset"], local_root=config["data"]["local_root"])
    data_manifest = Path(config["data"]["local_root"]).expanduser().resolve() / "dataset_manifest.json"
    model_path = Path(config["model"]["path"]).expanduser().resolve()
    model_manifest = model_path / "model_manifest.json"
    identity = {
        "config_sha256": config_digest(config), "checkpoint": checkpoint,
        "dataset_manifest_sha256": file_digest(data_manifest),
        "model_manifest_sha256": file_digest(model_manifest), "sources": sources,
        "runner_source_sha256": {
            name: file_digest(Path(__file__).parent / name)
            for name in ("evaluate.py", "backend.py", "data.py", "config.py")
        },
        "protocol": "native_task_zero_shot_chat_sampling_separate_full_passes_v1",
    }
    identity_sha = config_digest(identity)
    root = Path(run_dir).expanduser().resolve() / "evaluation"
    with _evaluation_lock(root):
        identity_path = root / "identity.json"
        if identity_path.exists():
            if not resume:
                raise FileExistsError("Evaluation evidence already exists; use --resume or a new run ID")
            if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise RuntimeError("Evaluation identity changed; use a new run ID")
        else:
            _write_json(identity_path, identity)
            _write_json(root / "resolved_config.json", config)

        if backend is None:
            from .backend import NativeMTPBackend
            backend = NativeMTPBackend.from_pretrained(
                config, adapter_dir=adapter_dir, trainable=False,
                device=config["model"]["device"],
            )
        for key in ("adapter_config_sha256", "adapter_weights_sha256"):
            if backend.provenance.get(key) != checkpoint[key]:
                raise RuntimeError("Loaded backend does not match the requested evaluation checkpoint")
        if backend.provenance.get("model_revision") != config["model"]["revision"]:
            raise RuntimeError("Loaded backend base model revision mismatch")
        _write_json(root / "backend_provenance.json", backend.provenance)
        summaries = []
        for mode in modes:
            for name in names:
                directory = root / mode / name
                directory.mkdir(parents=True, exist_ok=True)
                per_pass, native_per_sample = [], []
                for index in range(k):
                    sample_dir = directory / f"sample_{index:03d}"
                    call_seed = (int(evaluation["seed"]) + index) % (2**32)
                    sample_id = {"mode": mode, "task": name, "sample_index": index, "seed": call_seed}
                    if (sample_dir / "complete.json").is_file():
                        native, rows = _load_complete_pass(sample_dir, identity_sha, sample_id, name)
                    else:
                        if sample_dir.exists():
                            archived = directory / f"incomplete_sample_{index:03d}_{uuid.uuid4().hex[:12]}"
                            sample_dir.rename(archived)
                        sample_dir.mkdir()
                        native, rows = _run_pass(
                            backend, config, harness_root, sample_dir, mode, name, index, identity_sha,
                        )
                    per_pass.append(rows)
                    key = f"{TASKS[name]['metric']},{TASKS[name]['filter']}"
                    native_per_sample.append({
                        "sample_index": index, "seed": call_seed,
                        "score": float(native["results"][name][key]),
                        "native_metrics": native["results"][name],
                        "results_path": str((sample_dir / "results.json").relative_to(root)),
                    })
                n = TASKS[name]["num_documents"]
                derived = []
                for source_index in range(n):
                    first = per_pass[0][source_index]
                    for rows in per_pass[1:]:
                        if any(rows[source_index][key] != first[key] for key in ("doc_hash", "target_hash", "prompt_hash")):
                            raise RuntimeError("Source, target, or prompt changed between native repetitions")
                    scores = [rows[source_index]["score"] for rows in per_pass]
                    derived.append({
                        "source_index": source_index, "doc_hash": first["doc_hash"],
                        "target_hash": first["target_hash"], "scores": scores,
                        "any_correct": int(any(scores)),
                    })
                scores_path = directory / "derived_sample_scores.jsonl"
                # Recreate only deterministic derived summaries on resume.  Raw
                # native calls and their completion manifests remain immutable.
                if scores_path.exists():
                    if _read_jsonl(scores_path) != derived:
                        raise RuntimeError("Existing derived sample scores disagree with native evidence")
                else:
                    _write_jsonl(scores_path, derived)
                summary = {
                    "status": "completed", "task": name, "mode": mode,
                    "num_documents": n, "num_samples": k,
                    "primary_metric": f"{TASKS[name]['metric']},{TASKS[name]['filter']}",
                    "metrics": {
                        f"avg@{k}": sum(sum(row["scores"]) for row in derived) / (n * k),
                        f"pass@{k}": sum(row["any_correct"] for row in derived) / n,
                    },
                    "native_per_sample": native_per_sample,
                    "summary_path": str((directory / "summary.json").relative_to(root)),
                    "derived_scores_path": str(scores_path.relative_to(root)),
                    "denominator_policy": "all released test documents; truncations scored by native task",
                    "pass_at_k_definition": "fraction of documents with any native-scored correct answer among all k samples",
                }
                _write_json(directory / "summary.json", summary)
                summaries.append(summary)
        summary = {
            "status": "completed", "config_sha256": config_digest(config),
            "identity_sha256": identity_sha, "num_samples": k,
            "protocol": identity["protocol"], "results": summaries,
            "completed_at": _now(),
            "leaderboard_reproduction": False,
            "resume_policy": "reuse checksummed complete passes; archive and rerun an interrupted pass in full",
        }
        _write_json(root / "summary.json", summary)
        return summary
