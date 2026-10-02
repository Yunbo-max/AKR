#!/usr/bin/env python3
"""Record local, Git-ignored experiment assets without copying large binaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def summarize(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size
    files = 0
    size = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            files += 1
            size += item.stat().st_size
    return files, size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("LOCAL_ARTIFACTS.md"))
    args = parser.parse_args()
    root = args.root.resolve()
    candidates = [
        root / ".hf-cache",
        root / ".venv",
        root / "data",
        root / "external",
    ]
    results = root / "results"
    if results.exists():
        candidates.extend(sorted(
            path for path in results.iterdir()
            if path.is_dir() and path.name.startswith(
                ("gradients", "reps", "token_reps", "lora")
            )
        ))
    rows = []
    for path in candidates:
        files, size = summarize(path)
        if files:
            rows.append((path.relative_to(root), files, size))
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Local artifact inventory",
        "",
        f"Generated at `{timestamp}`. These assets remain in the workspace but are ",
        "excluded from Git because they are downloaded or reproducible high-volume files.",
        "Compact manifests, final CSV/JSON results, figures, source, tests, and reports are tracked.",
        "",
        "| Path | Files | Bytes |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| `{path}` | {files} | {size} |" for path, files, size in rows)
    lines.extend([
        "",
        "The exact model IDs, dataset revisions, commands, and protocols are recorded in ",
        "`REPRODUCE.md`, `RESULTS.md`, and the manifests under `data/manifests/`.",
        "",
    ])
    output = args.output if args.output.is_absolute() else root / args.output
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output} with {len(rows)} artifact groups")


if __name__ == "__main__":
    main()
