#!/usr/bin/env python3
"""Materialize the derived supervised BEANS registration without model work."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from akr_final.beans_zero_protocol import DEFAULT_OUTPUT, prepare_registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--outdir', type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = prepare_registration(args.root, args.outdir)
    print(json.dumps({k: v for k, v in result.items() if k != 'components'}, indent=2))
    for component in result['components'].values():
        print(json.dumps({k: component[k] for k in
            ['name', 'dataset_key', 'manifest', 'labels', 'eligible_rows', 'exclusion_counts']}))


if __name__ == '__main__':
    main()
