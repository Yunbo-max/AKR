#!/usr/bin/env python3
"""Invoke the C16 historical-readout entry with the historical 80gb profile.

`experiments/rebuttal_cases/c16_historical_readouts.py` accepts only
16gb/24gb/48gb on its CLI even though `execute()` forwards any profile to
FinalRunner and the recovered historical run registered `profile: 80gb`.
This wrapper is otherwise identical to the C16 CLI: action plan/check/run/report,
explicit source RUN registration, isolated output under
results/akr_historical_readouts/<run_id>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from experiments.rebuttal_cases.c16_historical_readouts import execute


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'check', 'run', 'report'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile', default='80gb')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args(argv)
    result = execute(args.action, args.root, args.source_run, args.config, args.run_id,
                     profile=args.profile, cache_only=args.cache_only)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
