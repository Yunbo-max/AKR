#!/usr/bin/env python3
"""Audit final_v1 cell receipts: completeness, query counts, native-bw ledger."""
import json, hashlib, sys
from pathlib import Path

ROOT = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR')
EXPECT = {'dogs': 139, 'watkins': 339, 'bats': 2000, 'humbugdb': 1859}
RUN_ID = 'final_v1'
SEED = 20261107


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def fhash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    rows = json.load(open(ROOT / 'results/run_commands_final_v1_20261108.json'))
    rows = [r for r in rows if r['case'] != 'c00']
    out = []
    for r in rows:
        case, ds, model, cond = r['case'], r['dataset'], r['model'], r['condition']
        cell = ROOT / 'results/akr_train50' / RUN_ID / case / f'{ds}__{model}__{cond.replace(":","_")}__s{SEED}'
        status = cell / 'STATUS.json'
        report = cell / 'REPORT.json'
        cached = cell / 'CACHED_VALIDATION.json'
        rec = dict(case=case, dataset=ds, model=model, condition=cond,
                   cell=str(cell.relative_to(ROOT)), status='MISSING', ok=False, reason='')
        try:
            st = json.load(open(status)) if status.exists() else {}
            rp = json.load(open(report)) if report.exists() else {}
            cv = json.load(open(cached)) if cached.exists() else {}
            rec['status'] = st.get('status', 'MISSING')
            issues = []
            if st.get('status') != 'complete' or not st.get('complete'):
                issues.append('status-not-complete')
            if not report.exists() or not rp.get('complete'):
                issues.append('report-incomplete')
            if rp.get('scored_queries') != EXPECT[ds]:
                issues.append(f"scored={rp.get('scored_queries')}!={EXPECT[ds]}")
            want_bw = 13 if (ds == 'watkins' and cond == 'lp:1000') else 0
            if rp.get('native_bw_queries') != want_bw:
                issues.append(f"native_bw={rp.get('native_bw_queries')}!={want_bw}")
            if not cv.get('complete'):
                issues.append('cached-validation-missing')
            if report.exists() and cv.get('report_sha256') != fhash(report):
                issues.append('cached-report-hash-mismatch')
            reg = cell / 'REGISTRATION.json'
            if reg.exists() and rp.get('registration_sha256') != digest(json.load(open(reg))):
                issues.append('registration-hash-mismatch')
            rec['ok'] = not issues
            rec['reason'] = ';'.join(issues)
        except Exception as exc:
            rec['reason'] = f'{type(exc).__name__}:{exc}'
        out.append(rec)
    ok = sum(1 for r in out if r['ok'])
    for rec in out:
        print(f"{'OK' if rec['ok'] else 'FAIL'}\t{rec['case']}\t{rec['dataset']}\t{rec['model']}\t{rec['condition']}\t{rec['status']}\t{rec['reason']}")
    print(f'SUMMARY ok={ok}/{len(out)}')
    return 0 if ok == len(out) else 1


if __name__ == '__main__':
    sys.exit(main())
