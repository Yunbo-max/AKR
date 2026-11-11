"""Separate launch/save entry points for native NTP/MTP and migrated OPSD cases."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .assets import atomic_json, check_assets, download_assets, exclusive_lock, sha256_file
from .config import REPO_ROOT, available_cases, config_digest, load_case, run_directory


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)
    def flush(self):
        self.log.flush()
        self.stream.flush()
    def isatty(self):
        return False


@contextlib.contextmanager
def _logs(directory: Path, action: str):
    directory.mkdir(parents=True, exist_ok=True)
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    with (directory / f"{action}_{identity}.stdout.log").open("x") as out, \
         (directory / f"{action}_{identity}.stderr.log").open("x") as err:
        with contextlib.redirect_stdout(_Tee(sys.stdout, out)), contextlib.redirect_stderr(_Tee(sys.stderr, err)):
            try:
                yield
            except BaseException:
                traceback.print_exc()
                raise


def _provenance() -> dict:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            check=False, capture_output=True, text=True)
    return {"git_commit": result.stdout.strip() if result.returncode == 0 else None,
            "python": sys.version, "argv": sys.argv,
            "recorded_at": datetime.now(timezone.utc).isoformat()}


def _prepare_run(directory: Path, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "resolved_config.json"
    if target.exists():
        if json.loads(target.read_text()) != config:
            raise ValueError("The same run ID/case/seed already has a different resolved config; use a new run ID")
    else:
        atomic_json(target, config)
        atomic_json(directory / "run_identity.json", {
            **_provenance(), "case_id": config["case_id"], "config_sha256": config_digest(config),
            "stage": "development_transfer_baseline",
        })


def _checkpoint(directory: Path, selection: str, config: dict) -> Path | None:
    if config["kind"] == "native":
        if selection != "latest":
            raise ValueError("Native cases cannot load trained adapters")
        return None
    if selection == "latest":
        pointer = directory / "latest_checkpoint.json"
        if not pointer.is_file():
            raise FileNotFoundError("No trained checkpoint exists; run this case's train stage first")
        checkpoint = (directory / json.loads(pointer.read_text())["path"]).resolve()
    else:
        checkpoint = Path(selection).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint}")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    actual = {str(p.relative_to(checkpoint)): sha256_file(p)
              for p in sorted(checkpoint.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    if actual != manifest["files"]:
        raise RuntimeError("Checkpoint is incomplete or changed")
    identity = json.loads((checkpoint / "identity.json").read_text())
    if identity.get("config_sha256") != config_digest(config):
        raise ValueError("Checkpoint belongs to a different case/configuration")
    from .train import _source_identity
    if identity.get("source_files_sha256") != _source_identity():
        raise ValueError("Checkpoint belongs to different training sources; use its original checkout")
    adapter = checkpoint / "adapter"
    if not (adapter / "adapter.safetensors").is_file():
        raise FileNotFoundError("Checkpoint has no native-MTP adapter weights")
    return adapter


def _completed_training(directory: Path, config: dict) -> dict | None:
    summary_path = directory / "training_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("config_sha256") != config_digest(config):
        raise RuntimeError("Existing training summary has a different configuration")
    status_path, pointer_path = directory / "training_status.json", directory / "latest_checkpoint.json"
    if not status_path.is_file() or not pointer_path.is_file():
        return None
    status, pointer = json.loads(status_path.read_text()), json.loads(pointer_path.read_text())
    if summary.get("status") not in {"completed", "completed_no_updates"}:
        return None
    if any(summary.get(key) != status.get(key) for key in ("status", "invocation_id", "config_sha256")):
        return None
    if summary.get("checkpoint") != pointer.get("path"):
        return None
    _checkpoint(directory, "latest", config)
    state = json.loads((directory / pointer["path"] / "state.json").read_text())
    if any(summary.get(key) != state.get(key) for key in ("updates", "cursor")):
        raise RuntimeError("Completed training summary does not match the saved state")
    return summary


def _collect(outputs_root: Path, run_id: str) -> dict:
    from .config import SAFE_NAME
    if not SAFE_NAME.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    root = outputs_root.expanduser().resolve() / run_id
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    inventory = []
    for case in available_cases():
        case_dirs = sorted((root / case).glob("seed_*"))
        if not case_dirs:
            rows.append({"case_id": case, "status": "not_started"})
        for directory in case_dirs:
            row = {"case_id": case, "seed_directory": directory.name}
            for name, filename in (("training", "training_summary.json"),
                                   ("evaluation", "evaluation/summary.json")):
                file = directory / filename
                if file.is_file():
                    row[name] = json.loads(file.read_text())
                    inventory.append({"path": str(file.relative_to(root)), "sha256": sha256_file(file)})
            row["status"] = "evaluated" if row.get("evaluation", {}).get("status") == "completed" else "incomplete"
            if row.get("training", {}).get("status") == "completed_no_updates":
                row["learning_status"] = "no_optimizer_updates"
            rows.append(row)
    report = {"run_id": run_id, "cases": rows, "summary_file_identities": inventory,
              "claim_scope": "development baseline; costs, updates, seeds and complete native metrics must be compared",
              "generated_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(root / "comparison.json", report)
    lines = [f"# MTP OPSD baseline — {run_id}", "", "Native scorer results; incomplete cases are retained.", "",
             "| Case | Seed | Train updates | Decode | Task | Avg@N | Pass@N | Documents |",
             "|---|---|---:|---|---|---:|---:|---:|"]
    for row in rows:
        outcomes = row.get("evaluation", {}).get("results", [])
        if not outcomes:
            lines.append(f"| {row['case_id']} | {row.get('seed_directory', '—')} | — | — | {row['status']} | — | — | — |")
        for item in outcomes:
            n = item["num_samples"]
            metrics = item["metrics"]
            lines.append(f"| {row['case_id']} | {row.get('seed_directory', '—')} | {row.get('training', {}).get('updates', 0)} | "
                         f"{item['mode']} | {item['task']} | {metrics[f'avg@{n}']:.6f} | "
                         f"{metrics[f'pass@{n}']:.6f} | {item['num_documents']} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="List independent cases without loading models or datasets")
    acquire = commands.add_parser("download", help="Explicitly acquire the pinned inputs on the execution host")
    acquire.add_argument("--assets-root", default="assets/mtp_opsd")
    acquire.add_argument("--max-workers", type=int, default=4)
    check = commands.add_parser("check", help="Check environment/input identity; does not load a model")
    check.add_argument("--assets-root", default="assets/mtp_opsd")
    check.add_argument("--full-hashes", action="store_true")
    qualify = commands.add_parser("qualify", help="Run actual native-GPU interface and gradient qualification")
    qualify.add_argument("--assets-root", default="assets/mtp_opsd")
    qualify.add_argument("--output", default="artifacts/mtp_opsd/qualification.json")
    qualify.add_argument("--device", default="cuda:0")
    qualify.add_argument("--overrides", type=Path, help="Qualify the same structural overrides used by train/evaluate")
    for action in ("train", "evaluate", "run"):
        command = commands.add_parser(action)
        command.add_argument("--case", required=True, choices=available_cases())
        command.add_argument("--run-id", required=True)
        command.add_argument("--assets-root", default="assets/mtp_opsd")
        command.add_argument("--outputs-root", default="runs/mtp_opsd")
        command.add_argument("--qualification", default="artifacts/mtp_opsd/qualification.json")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--overrides", type=Path, help="Explicit JSON overrides; changing them needs a new run ID")
        command.add_argument("--resume", action="store_true")
        if action != "train":
            command.add_argument("--checkpoint", default="latest", help="Complete checkpoint directory, or latest")
    report = commands.add_parser("report", help="Collect all independently completed/pending cases")
    report.add_argument("--run-id", required=True)
    report.add_argument("--outputs-root", type=Path, default=Path("runs/mtp_opsd"))
    args = parser.parse_args(argv)

    if args.command == "plan":
        result = {"source_status": "generated_unexecuted", "cases": [load_case(c) for c in available_cases()]}
    elif args.command == "download":
        if not 1 <= args.max_workers <= 16:
            raise ValueError("max-workers must be between 1 and 16")
        result = download_assets(args.assets_root, max_workers=args.max_workers)
    elif args.command == "check":
        from .qualification import check_environment
        result = {"environment": check_environment(), "assets": check_assets(args.assets_root, full_hashes=args.full_hashes)}
    elif args.command == "qualify":
        from .qualification import run_qualification
        overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
        overrides.setdefault("model", {})["device"] = args.device
        config = load_case("c03_mtp_opsd_g4", assets_root=args.assets_root, overrides=overrides)
        with exclusive_lock(Path(args.output).expanduser().resolve().with_suffix(".lock")):
            with _logs(Path(args.output).expanduser().resolve().parent / "logs", "qualify"):
                result = run_qualification(config, args.assets_root, args.output)
    elif args.command == "report":
        result = _collect(args.outputs_root, args.run_id)
    else:
        from .qualification import check_environment, require_qualification
        overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
        overrides.setdefault("model", {})["device"] = args.device
        config = load_case(args.case, assets_root=args.assets_root, seed=args.seed, overrides=overrides)
        check_environment()
        check_assets(args.assets_root)
        require_qualification(args.qualification, config)
        directory = run_directory(args.outputs_root, args.run_id, args.case, args.seed)
        with exclusive_lock(directory / ".case.lock"):
            _prepare_run(directory, config)
            with _logs(directory / "logs", args.command):
                result = {"run_directory": str(directory)}
                if args.command in {"train", "run"} and config["kind"] == "train":
                    from .train import run_training
                    # A resumed composite run reuses its completed training and
                    # resumes evaluator samples, rather than taking extra steps.
                    completed = _completed_training(directory, config) if args.command == "run" and args.resume else None
                    if completed is not None:
                        result["training"] = completed
                    else:
                        result["training"] = run_training(config, directory, resume=args.resume)
                elif args.command == "train":
                    raise ValueError("Native cases have no train stage; use evaluate or run")
                if args.command in {"evaluate", "run"}:
                    from .evaluate import run_evaluation
                    result["evaluation"] = run_evaluation(
                        config, directory, adapter_dir=_checkpoint(directory, args.checkpoint, config),
                        resume=args.resume,
                    )
                print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
