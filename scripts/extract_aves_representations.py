#!/usr/bin/env python3
"""Extract batch-size-1 layerwise AVES representations for a CSV manifest."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from aves import load_feature_extractor
from aves.utils import load_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [r for r in rows if not (args.output_dir / f'{r["event_id"]}.npz').exists()]
    model = load_feature_extractor(args.config, args.checkpoint, device="cuda", for_inference=True)
    for index, row in enumerate(pending, 1):
        audio = load_audio(row["audio_path"], mono=True, mono_avg=True)
        layers = model.extract_features(audio, layers=None)
        representation = np.stack([layer[0].mean(0).cpu().numpy() for layer in layers])
        output = args.output_dir / f'{row["event_id"]}.npz'
        temporary = output.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, representation=representation.astype(np.float16),
                            event_id=row["event_id"], label=row["label"],
                            recording_id=row["recording_id"], model_id="AVES-bio")
        temporary.replace(output)
        if index == 1 or index % 50 == 0 or index == len(pending):
            print(f"[{index}/{len(pending)}] {row['event_id']} {representation.shape}", flush=True)


if __name__ == "__main__": main()
