#!/usr/bin/env python3
"""One-time hash-pinned recovery; normal experiments never call this script.
Only complete tar members and two SHA256-verified image fragments are restored.
The paper text and present audit are supplied separately. Never invent missing bytes.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import tarfile

CAPSULES = {
    'transport': ('migration/transport', '*.b64', '6d951d770fd7a39a1dfcaca53875c62e6451e200db99d47bec2fd0da6df53ab0'),
    'completion': ('migration/completion', '*.b64', '0671dca857f40e82d61ddd768a66ae3ffe5a5b476cfcbee2890a999edfec8b02'),
    'release-code': ('migration/release_ready', 'code*.b64', '5d4c07b5858437a7f573d8f10b34874f0edf43b34d9dd668d65541bc824f5fcd'),
    'release-paper': ('migration/release_ready', 'paper*.b64', '238249a7f52ef85affb847a19f62e91e2a661a53ffc319e96fcbb217fbd91706'),
}

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def members(data: bytes):
    stream = io.BytesIO(data)
    complete, partial = {}, {}
    while stream.tell() + 512 <= len(data):
        header = stream.read(512)
        if not header.strip(b'\x00'):
            break
        info = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
        path = PurePosixPath(info.name)
        if path.is_absolute() or '..' in path.parts or '\\' in info.name:
            raise ValueError('Unsafe archive path: ' + info.name)
        if not info.isfile():
            raise ValueError('Only regular files are recoverable: ' + info.name)
        if info.name in complete:
            raise ValueError('Duplicate archive member: ' + info.name)
        payload = stream.read(info.size)
        if len(payload) != info.size:
            partial[info.name] = payload
            break
        complete[info.name] = payload
        stream.seek((512 - info.size % 512) % 512, 1)
    return complete, partial

def restore(root: Path):
    root = root.resolve()
    receipt = root / 'migration/RESTORED_FINAL_FILES.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        for item in old['files']:
            if not (root / item['path']).is_file():
                raise RuntimeError('Recovered file was subsequently removed: ' + item['path'])
        return {'status': 'already_restored_edits_preserved', 'files': len(old['files'])}
    records, loaded = [], {}
    for tag, (directory, pattern, expected) in CAPSULES.items():
        files = sorted((root / directory).glob(pattern))
        if not files:
            raise FileNotFoundError(directory + '/' + pattern)
        encoded = ''.join(p.read_text(encoding='ascii').strip() for p in files)
        raw = base64.b64decode(encoded, validate=True)
        if sha(raw) != expected:
            raise ValueError('Capsule content mismatch: ' + tag)
        decoder = lzma.LZMADecompressor(memlimit=256 * 1024 * 1024)
        data = decoder.decompress(raw, max_length=32 * 1024 * 1024)
        if len(data) >= 32 * 1024 * 1024:
            raise ValueError('Capsule exceeds size bound')
        loaded[tag] = members(data)
        records.append({'capsule': tag, 'compressed_sha256': sha(raw),
                        'complete_stream': decoder.eof,
                        'complete_members': len(loaded[tag][0]),
                        'partial_member_names': list(loaded[tag][1])})
    selected = {}
    for tag in ['transport', 'completion', 'release-code', 'release-paper']:
        for path, payload in loaded[tag][0].items():
            if path.startswith(('paper/', 'tests/', 'scripts/', 'akr_final/')) or path in {'requirements-cpu.txt', 'requirements-qwen.txt', 'pyproject.toml'}:
                selected[path] = payload
    fragments = [
        ('paper/figures/intro.jpg', 'transport', 'completion', '__transport__/intro_jpeg_tail',
         '13cdc8202389321fe1eb54ee00d9bde6681e911c6e09094e89a89e816be45a96'),
        ('paper/figures/pipeline.jpg', 'completion', 'release-paper', '__transport__/pipeline_jpeg_tail',
         '6e1ce9a264f70c4321fabf60bf4f92ee6b68f73e128bd4ef89362721b7c41c0f'),
    ]
    for path, first, second, suffix, expected in fragments:
        value = loaded[first][1][path] + loaded[second][0][suffix]
        if sha(value) != expected:
            raise ValueError('Fragment reconstruction mismatch: ' + path)
        selected[path] = value
    written = []
    for path, payload in sorted(selected.items()):
        target = root / path
        if target.exists() and target.read_bytes() != payload:
            backup = root / 'migration/before_final_restore' / path
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(target.read_bytes())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if path.endswith('.sh'):
            target.chmod(0o755)
        written.append({'path': path, 'bytes': len(payload), 'sha256': sha(payload)})
    report = {'status': 'verified_complete_members_only', 'capsules': records,
              'files': written, 'historical_private_HEAD_verified': False,
              'incomplete_archives_claimed_complete': False,
              'raster_figures': 'Portable JPEG derivatives; original PNG hashes are in figure_renderings.json.'}
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    return {'status': report['status'], 'files': len(written)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(restore(args.root), indent=2))
