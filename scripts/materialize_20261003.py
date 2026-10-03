#!/usr/bin/env python3
"""One-time, hash-checked migration; no imports or execution of payload code."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

LIMIT = 2_000_000


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    p = PurePosixPath(name)
    if (not name or '\\' in name or p.is_absolute() or
            any(x in {'..', '.', '.git', ''} for x in name.split('/'))):
        raise ValueError('Unsafe path: ' + name)
    dest = root.joinpath(*p.parts)
    if any(x.is_symlink() for x in [dest, *dest.parents] if x != root.parent):
        raise ValueError('Symlink path: ' + name)
    if root not in dest.resolve().parents:
        raise ValueError('Outside repository: ' + name)
    return dest


def load_entries(root: Path, directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != 'json_utf8_lzma_base64_v1':
        raise ValueError('Unsupported capsule format')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest.get('base_ref', '')):
        raise ValueError('Base revision must be immutable')
    parts = manifest['parts']
    if not 0 < len(parts) <= 100:
        raise ValueError('Invalid part count')
    chunks = []
    for index, part in enumerate(parts):
        if part['path'] != f'{index:02}.b64':
            raise ValueError('Missing, reordered or duplicate part')
        path = safe_path(directory, 'parts/' + part['path'])
        raw = path.read_bytes()
        if len(raw) != part['bytes'] or digest(raw) != part['sha256']:
            raise ValueError('Part hash/length mismatch: ' + part['path'])
        chunks.append(raw.strip())
    compressed = base64.b64decode(b''.join(chunks), validate=True)
    if digest(compressed) != manifest['compressed_sha256']:
        raise ValueError('Compressed checksum mismatch')
    dec = lzma.LZMADecompressor(memlimit=128 * 1024 * 1024)
    data = dec.decompress(compressed, max_length=LIMIT + 1)
    if not dec.eof or dec.unused_data or len(data) > LIMIT:
        raise ValueError('Incomplete, concatenated or oversized LZMA stream')
    if len(data) != manifest['decoded_bytes'] or digest(data) != manifest['decoded_sha256']:
        raise ValueError('Decoded checksum/length mismatch')
    entries = json.loads(data)
    if not isinstance(entries, list) or len(entries) != manifest['file_count']:
        raise ValueError('Wrong decoded file count')
    seen = set()
    for entry in entries:
        name = entry['path']
        safe_path(root, name)
        if name in seen or entry['mode'] not in {'100644', '100755'}:
            raise ValueError('Duplicate path or unsupported mode')
        seen.add(name)
        if digest(entry['content'].encode('utf-8')) != entry['sha256']:
            raise ValueError('File checksum mismatch: ' + name)
    return manifest, entries


def materialize(root: Path, directory: Path, check_only: bool = False) -> dict:
    root = root.resolve()
    manifest, entries = load_entries(root, directory)
    base = manifest['base_ref']
    subprocess.run(['git', 'merge-base', '--is-ancestor', base, 'HEAD'], cwd=root, check=True)
    # Validate EVERY file before writing any file. Preserve concurrent changes.
    for entry in entries:
        name = entry['path']; dest = safe_path(root, name)
        old = subprocess.run(['git', 'show', f'{base}:{name}'], cwd=root, capture_output=True)
        if dest.exists():
            if not dest.is_file():
                raise ValueError('Not a regular file: ' + name)
            current = dest.read_bytes()
            if digest(current) != entry['sha256'] and (old.returncode or current != old.stdout):
                raise ValueError('Concurrent/untracked content conflict: ' + name)
        elif old.returncode == 0:
            raise ValueError('Previously present file is missing: ' + name)
    receipt = {
        'complete': True, 'base_ref': base, 'file_count': len(entries),
        'decoded_sha256': manifest['decoded_sha256'],
        'files': {x['path']: x['sha256'] for x in entries},
        'model_execution': False, 'check_only': check_only,
    }
    if not check_only:
        for entry in entries:
            dest = safe_path(root, entry['path'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as handle:
                tmp = Path(handle.name)
                handle.write(entry['content'].encode('utf-8'))
            tmp.chmod(0o755 if entry['mode'] == '100755' else 0o644)
            tmp.replace(dest)
        receipt_path = root / 'migration/COMPLETION_VERIFIED_20261003.json'
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = materialize(args.root, args.root / 'migration/final_20261003', args.check_only)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
