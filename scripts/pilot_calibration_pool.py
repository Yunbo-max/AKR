#!/usr/bin/env python3
"""Bounded real 3B single-pool pilot; no official test or LoRA training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from akr_closing.cli import run_lock, validate_run_id
from akr_closing.core import atomic_json, file_hash
from akr_final.calibration_pilot import prepare, PilotRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='pool_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--cached-only', action='store_true', help='verify completed caches without loading weights or requiring CUDA')
    parser.add_argument('--max-seconds', type=int, default=3600)
    args = parser.parse_args(argv)
    validate_run_id(args.run_id)
    if not 1 <= args.max_seconds <= 7200: raise ValueError('budget must be 1..7200 seconds')
    out = ROOT/'results/calibration_pilot'/args.run_id
    def expired(*_): raise TimeoutError('pilot runtime budget exhausted; same-source resume supported')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    started = time.perf_counter()
    runner = None
    with run_lock(ROOT/'results/calibration_pilot/.runner.lock'):
        atomic_json(out/'dogs/STATUS.json', dict(complete=False, phase='preflight_verification'))
        try:
            entry = prepare(ROOT)
            def no_reload(*a, **kw): raise AssertionError('complete resume attempted model reload')
            runner = PilotRunner(ROOT, out, 'dogs', entry, '24gb', no_reload if args.cached_only else None)
            runner.preflight()
            atomic_json(out/'REGISTRATION.json', entry)
            print(json.dumps({'phase':'preflight_complete', 'directory':str(out),
                              'calibration_count':40, 'development_count':40}), flush=True)
            if not args.execute and not args.cached_only: return 0
            if args.execute:
                import torch
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError('select exactly one available CUDA device')
                free, total = torch.cuda.mem_get_info()
                # Conservative floor for this shared 96 GiB workstation pilot.
                if free < 24 * 1024**3: raise RuntimeError('less than 24 GiB free GPU memory')
                atomic_json(out/'EXECUTION.json', dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    free_gpu_bytes_before=free, total_gpu_bytes=total, runtime_budget_seconds=args.max_seconds))
            result = runner.run_pilot(); runner.close()
            again = PilotRunner(ROOT, out, 'dogs', entry, '24gb', no_reload)
            try:
                again.preflight(); resumed = again.run_pilot()
                if result['policies'] != resumed['policies'] or result['predictions'] != resumed['predictions']:
                    raise AssertionError('resume changed results')
            finally: again.close()
            atomic_json(out/('CACHED_VALIDATION.json' if args.cached_only else 'VALIDATION.json'), dict(complete=True, resume_without_model_reload=True,
                elapsed_seconds=time.perf_counter()-started,
                initial_selection_seconds=result['selection_elapsed_seconds_this_invocation'],
                initial_run_seconds=result['elapsed_seconds_this_invocation'],
                report_sha256=file_hash(runner.out/'REPORT.json'), fingerprint=runner.fingerprint))
            print(json.dumps({'complete':True, 'directory':str(out), 'rows':[
                {k:r[k] for k in ['method','accuracy','macro_f1','repaired','harmed']} for r in result['rows']]}), flush=True)
        except BaseException as exc:
            atomic_json(out/'dogs/STATUS.json', dict(complete=False, phase='failed', error=str(exc)))
            atomic_json(out/'ERROR.json', dict(complete=False, error_type=type(exc).__name__, error=str(exc)))
            raise
        finally:
            if runner is not None: runner.close()
            signal.alarm(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
