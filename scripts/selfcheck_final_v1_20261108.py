#!/usr/bin/env python3
"""Independent watchdog for the AKR final_v1 submitter.

Every 5 minutes: restart the submitter loop if no submitter process exists.
Every 30 minutes: append an in-flight/progress heartbeat to the status log.
"""
import json, subprocess, time
from collections import Counter
from pathlib import Path

ROOT = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR')
STATE = ROOT / 'results/run_specs_20261108/serial_state.json'
LOG = ROOT / 'run_final_v1_20261108.log'


def log(msg):
    with open(LOG, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def submitter_alive():
    ps = subprocess.run(['ps', '-eo', 'pid,comm,args'], capture_output=True, text=True)
    for line in ps.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        _, comm, args = parts
        if comm.startswith('python') and 'scripts/serial_submit_final_v1_20261108.py' in args:
            return True
    return False


def main():
    cycle = 0
    while True:
        time.sleep(300)
        cycle += 1
        if not submitter_alive():
            log('SELFCHECK submitter missing -> restart loop')
            subprocess.Popen(
                ['setsid', 'nohup', 'bash',
                 str(ROOT / 'scripts/run_serial_submitter_loop_20261108.sh')],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, start_new_session=True)
        if cycle % 6 == 0:
            try:
                d = json.load(open(STATE))
                cnt = Counter(e['status'] for e in d['cells'].values())
                log(f"SELFCHECK in_flight={cnt.get('SUBMITTED', 0)} ok={cnt.get('OK', 0)} "
                    f"failed={cnt.get('FAILED', 0)} pending={cnt.get('PENDING', 0)} "
                    f"c16={d.get('c16', {}).get('status')}")
            except Exception as exc:
                log(f"SELFCHECK state-read-error {type(exc).__name__}:{exc}")


if __name__ == '__main__':
    main()
