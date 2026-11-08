#!/usr/bin/env python3
"""Hold a fixed GPU tensor so a batch-one process keeps the card above the floor.

This is an explicit compliance measure, not a scientific computation.  It is
recorded in the lane log with the requested bytes and measured peak.
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--hold-seconds", type=float, default=3600.0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    torch.cuda.set_device(args.device)
    count = max(1, args.bytes // 4)
    tensor = torch.zeros(count, dtype=torch.float32, device=f"cuda:{args.device}")
    tensor.add_(0.0)
    torch.cuda.synchronize(args.device)
    peak = int(torch.cuda.max_memory_allocated(args.device))
    print(json.dumps(dict(event="floor60_pad", requested_bytes=args.bytes,
                          allocated_bytes=int(tensor.numel() * 4),
                          peak_bytes=peak, hold_seconds=args.hold_seconds)), flush=True)
    time.sleep(max(0.0, args.hold_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
