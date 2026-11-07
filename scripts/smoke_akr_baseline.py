#!/usr/bin/env python3
"""Run one isolated baseline smoke on registered data; never a full result."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from akr_final.comprehensive import make_plan
from akr_final.comprehensive_baselines import run_baseline_job
from akr_closing.model_factory import validate_backend_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=['aves_ridge', 'marmaudio_specialist', 'published_lora', 'beans_zero'])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', default='qwen7b')
    parser.add_argument('--enable-conditional', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'results/comprehensive_smoke')
    args = parser.parse_args()
    study, plan = make_plan(ROOT, ROOT/'configs/akr_comprehensive.yaml', [args.model],
                           ['baselines', 'diagnostic'], args.enable_conditional)
    matches = [j for j in plan['baseline_jobs'] if j['kind'] == args.kind and j['dataset'] == args.dataset]
    if not matches: parser.error('no applicable registered job')
    job = matches[0]
    if 'model_spec' in job:
        validate_backend_environment(job['model_spec']['id'])
    result = run_baseline_job(ROOT, args.output, job, study, {}, execute=True, limit=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ['support_ids', 'query_ids', 'execution_environment']}, indent=2))
    if result['complete'] or not result['processed_complete']:
        raise RuntimeError('invalid smoke completion state')


if __name__ == '__main__':
    main()
