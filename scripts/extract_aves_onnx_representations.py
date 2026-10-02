#!/usr/bin/env python3
"""Extract official AVES-bio ONNX last-layer representations."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="only equal-length waveforms are batched; otherwise falls back to one")
    parser.add_argument(
        "--provider",
        choices=("CPUExecutionProvider", "CUDAExecutionProvider"),
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider",
    )
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.provider == "CUDAExecutionProvider":
        # Load CUDA/cuDNN wheels installed alongside onnxruntime-gpu before
        # constructing the session. This is harmless when the libraries are
        # already discoverable through LD_LIBRARY_PATH.
        ort.preload_dlls(directory="")
    session = ort.InferenceSession(str(args.checkpoint), providers=[args.provider])
    if session.get_providers()[0] != args.provider:
        raise RuntimeError(
            f"requested {args.provider}, but session providers are {session.get_providers()}"
        )
    print(f"provider={session.get_providers()[0]} batch_size={args.batch_size}", flush=True)
    input_name = session.get_inputs()[0].name
    pending = [r for r in rows if not (args.output_dir / f'{r["event_id"]}.npz').exists()]
    def load_audio(row):
        waveform, rate = sf.read(row["audio_path"], always_2d=True, dtype="float32")
        mono = waveform.mean(1)
        if rate != 16000:
            from math import gcd
            divisor = gcd(rate, 16000)
            mono = resample_poly(mono, 16000 // divisor, rate // divisor).astype(np.float32)
        return mono

    completed = 0
    while completed < len(pending):
        candidate = pending[completed:completed + args.batch_size]
        waveforms = [load_audio(row) for row in candidate]
        if len({len(waveform) for waveform in waveforms}) != 1:
            candidate, waveforms = candidate[:1], waveforms[:1]
        frames = session.run(None, {input_name: np.stack(waveforms)})[0]
        representations = frames.mean(1).astype(np.float32)
        for row, representation in zip(candidate, representations):
            output = args.output_dir / f'{row["event_id"]}.npz'
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, representation=representation[None].astype(np.float16),
                                event_id=row["event_id"], label=row["label"],
                                recording_id=row.get("recording_id", row["event_id"]),
                                split=row.get("split", ""), model_id="AVES-bio-ONNX")
            temporary.replace(output)
        completed += len(candidate)
        if completed == len(candidate) or completed % 50 < len(candidate) or completed == len(pending):
            print(f"[{completed}/{len(pending)}] {candidate[-1]['event_id']} {frames.shape}", flush=True)


if __name__ == "__main__": main()
