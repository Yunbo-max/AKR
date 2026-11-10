#!/usr/bin/env python3
"""Bounded-concurrency (<=4) single-GPU submitter for AKR final_v1.

One cell per qz job (spec 79fe954a, 1xH100), at most MAX_ACTIVE jobs in a
non-terminal state.  Submission order follows the case waves.  Non-OOM
failures retry up to 2 times (3 attempts total).  CUDA OOM failures follow a
strict batch-halving ladder using a derived per-cell batch map (receipt +
sha256 recorded); the ladder may exceed the standard retry cap until batch
reaches 1, where a floor pad is added.  Derived-map retries archive the
identity-bound CASE/STATUS/caches but keep REGISTRATION.json.
"""
import hashlib, json, subprocess, time, sys, os, shutil
from pathlib import Path

ROOT = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR')
QZ = '/inspire/hdd/global_user/zhangjiaquan-253108540222/bin/qz'
STATUS_LOG = ROOT / 'run_final_v1_20261108.log'
STATE = ROOT / 'results/run_specs_20261108/serial_state.json'
SERIAL_DIR = ROOT / 'results/run_specs_20261108/serial'
MAP_DIR = ROOT / 'results/run_specs_20261108/batch_maps'
SPEC_ID = '79fe954a-be92-4772-ac0b-94ad8a79b7bb'
RUN_ID, SEED = 'final_v1', 20261107
EXPECT = {'dogs': 139, 'watkins': 339, 'bats': 2000, 'humbugdb': 1859}
NONTERM = {'job_queuing', 'job_running', 'job_pending', 'job_scheduling',
           'job_creating', 'job_stopping', 'job_restarting', 'job_paused'}
WAVES = [('W1', ['c01', 'c02', 'c03', 'c14']),
         ('W2', ['c05', 'c06', 'c07', 'c08', 'c09', 'c10', 'c11', 'c13', 'c15']),
         ('W3', ['c12', 'c04'])]
WAVE_OF = {c: w for w, cs in WAVES for c in cs}
MAP_V2 = ROOT / 'results/batch_floor60_runner_map_20261108.json'
PROBE = ROOT / 'results/probe_mem_70gb_20261108.json'
MAX_ACTIVE = 8
PRIORITY_NEW_CASES = ['c04', 'c11', 'c12', 'c13', 'c14', 'c15']


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(msg):
    line = f'{now()} {msg}'
    with open(STATUS_LOG, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def fsha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qz(args, body=None, timeout=180):
    cmd = [QZ] + args
    if body is not None:
        cmd += ['--data', json.dumps(body)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return {'_exc': type(exc).__name__, '_error': str(exc)[:300]}
    try:
        return json.loads(r.stdout)
    except Exception:
        return {'_raw': r.stdout, '_err': r.stderr, '_rc': r.returncode}


def list_ours():
    body = {'workspace_id': 'ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6',
            'project_ids': ['project-160ccb20-98ab-4538-a847-01d1f83d5b0f'],
            'keyword': 'akr-fv1', 'page_size': 200, 'page_number': 1}
    d = qz(['train', 'ListJobs'], body)
    r = d.get('Result')
    if not isinstance(r, dict):
        return None
    return r.get('jobs', [])


def tag_of(c):
    return f"{c['case']}-{c['dataset']}-{c['model']}-{c['condition'].replace(':','-')}"


def name_of(c):
    return f"akr-fv1-s-{tag_of(c)}-20261108"


def cell_dir(c):
    return (ROOT / 'results/akr_train50' / RUN_ID / c['case'] /
            f"{c['dataset']}__{c['model']}__{c['condition'].replace(':','_')}__s{SEED}")


def verify_cell(c):
    d = cell_dir(c)
    try:
        st = json.load(open(d / 'STATUS.json'))
        rp = json.load(open(d / 'REPORT.json'))
        cv = json.load(open(d / 'CACHED_VALIDATION.json'))
    except Exception as exc:
        return False, f'missing:{type(exc).__name__}'
    if st.get('status') != 'complete' or not st.get('complete'):
        return False, f"status:{st.get('status')}"
    if not rp.get('complete'):
        return False, 'report-incomplete'
    if rp.get('scored_queries') != EXPECT[c['dataset']]:
        return False, f"scored:{rp.get('scored_queries')}"
    want_bw = 13 if (c['dataset'] == 'watkins' and c['condition'] == 'lp:1000') else 0
    if rp.get('native_bw_queries') != want_bw:
        return False, f"native_bw:{rp.get('native_bw_queries')}!={want_bw}"
    if not cv.get('complete'):
        return False, 'cached-validation-missing'
    return True, ''


def current_code_identity():
    paths = sorted((ROOT / 'akr_final').glob('*.py')) + sorted((ROOT / 'akr_closing').glob('*.py'))
    paths += [ROOT / 'src/animal_omni/conditional_kv.py', ROOT / 'src/animal_omni/metrics.py',
              ROOT / 'scripts/fair_speech_lm.py']
    return {str(p.relative_to(ROOT)): fsha(p) for p in paths if p.exists()}


CURRENT_CODE = current_code_identity()


def cell_code_hash(c):
    reg = cell_dir(c) / 'REGISTRATION.json'
    if not reg.exists():
        return None
    try:
        return json.load(open(reg)).get('code', {}).get('akr_final/train50_runner.py')
    except Exception:
        return None


def cell_code_matches(c):
    reg = cell_dir(c) / 'REGISTRATION.json'
    if not reg.exists():
        return True
    try:
        return json.load(open(reg)).get('code') == CURRENT_CODE
    except Exception:
        return False


def full_archive_cell(c, attempt):
    """Move the whole cell dir (incl. stale REGISTRATION) before a new-code retry."""
    d = cell_dir(c)
    if not d.exists():
        return None
    base = ROOT / 'results/akr_train50/final_v1_codefix_archive' / c['case'] / f'{tag_of(c)}_a{attempt}'
    dest = base
    n = 0
    while dest.exists():
        n += 1
        dest = base.parent / f'{base.name}_r{n}'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest))
    return str(dest.relative_to(ROOT))


def archive_cell_for_retry(c, attempt):
    """Keep REGISTRATION; move identity-bound artifacts aside for a derived map.

    Collision-proof and exception-tolerant: archiving must never kill the
    submitter (a repeated OOM retry targets the same attempt directory).
    """
    d = cell_dir(c)
    if not d.exists():
        return None
    base = ROOT / 'results/akr_train50/final_v1_oom_archive' / f'{tag_of(c)}_a{attempt}'
    dest = base
    n = 0
    while dest.exists():
        n += 1
        dest = base.parent / f'{base.name}_r{n}'
    dest.mkdir(parents=True, exist_ok=True)
    for p in sorted(d.iterdir()):
        if p.name in ('REGISTRATION.json', '.run.lock'):
            continue
        target = dest / p.name
        try:
            if target.exists():
                target = dest / f'{p.name}.{int(time.time()*1000)}'
            shutil.move(str(p), str(target))
        except Exception as exc:
            log(f'ARCHIVE-WARN {tag_of(c)} move {p.name} failed: {type(exc).__name__}:{exc}')
    return str(dest.relative_to(ROOT))


def map_batch(c):
    """Effective runner batch for a cell under batch-map v2 (same guard as runner)."""
    if c['case'] in ('c04', 'c16') or c['model'] in ('glm4voice9b', 'naturelm', 'qwen3omni'):
        return 1
    try:
        m = json.load(open(MAP_V2))
    except Exception:
        return 1
    cells = m.get('cells') or {}
    for key in (f"{c['case']}|{c['model']}|{c['condition']}|{c['dataset']}",
                f"{c['case']}|{c['model']}|{c['condition'].replace(':','_')}|{c['dataset']}"):
        if key in cells:
            return max(1, int(cells[key]))
    return max(1, int((m.get('datasets') or {}).get(c['dataset'], m.get('default', 1))))


def lookup_peak_b1(c):
    """Forward-only b=1 peak from the memory probe (generation adds KV-cache)."""
    try:
        d = json.load(open(PROBE))
    except Exception:
        return None
    best = None
    for m in d.get('measurements', []):
        if (m.get('case') == c['case'] and m.get('dataset') == c['dataset']
                and m.get('model') == c['model'] and m.get('batch') == 1
                and m.get('status') == 'ok'):
            val = m.get('max_memory_allocated_bytes')
            if val:
                best = max(best or 0, int(val))
    return best


def pad_at_b1(c):
    peak = lookup_peak_b1(c)
    if peak is None:
        return 12_000_000_000
    return max(0, 30_000_000_000 - peak) + 1_000_000_000


def derived_map(c, to_batch, from_batch, reason, attempt, base_path=None):
    """Create a per-cell derived batch map, receipt and sha256.

    base_path defaults to batch-map v2; pass the cell's current derived map so
    extractor-batch mitigations survive the OOM ladder.
    """
    base = Path(base_path) if base_path else MAP_V2
    if not base.is_absolute():
        base = ROOT / base
    m = json.load(open(base))
    key = f"{c['case']}|{c['model']}|{c['condition']}|{c['dataset']}"
    m.setdefault('cells', {})[key] = int(to_batch)
    m['cells'][key.replace(':', '_')] = int(to_batch)
    m['description'] = (str(m.get('description', '')) +
                        f" | DERIVED {reason} {key} {from_batch}->{to_batch} attempt {attempt}")
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    tag = tag_of(c)
    path = MAP_DIR / f'{tag}_b{to_batch}.json'
    path.write_text(json.dumps(m, indent=1))
    sha = fsha(path)
    receipt = dict(cell=key, reason=reason, from_batch=from_batch, to_batch=to_batch,
                   attempt=attempt, source_map=str(base.relative_to(ROOT)),
                   source_map_sha256=fsha(base),
                   derived_map=str(path.relative_to(ROOT)), derived_map_sha256=sha,
                   created=now(),
                   note='batch-map v2 is forward-only; generation KV-cache triggered this halving')
    Path(str(path) + '.receipt.json').write_text(json.dumps(receipt, indent=1))
    return str(path.relative_to(ROOT)), sha


def is_oom(c, job_name):
    """Detect CUDA OOM from the cell STATUS reason or the lane log tail."""
    d = cell_dir(c)
    texts = []
    st = d / 'STATUS.json'
    if st.exists():
        try:
            texts.append(str(json.load(open(st)).get('reason', '')))
        except Exception:
            pass
    lane = ROOT / 'logs/run' / job_name / f"lane0_{c['case']}_{c['dataset']}_{c['model']}_{c['condition'].replace(':','_')}.log"
    if lane.exists():
        try:
            with open(lane, 'rb') as f:
                f.seek(max(0, lane.stat().st_size - 200_000))
                texts.append(f.read().decode('utf-8', 'replace'))
        except Exception:
            pass
    blob = '\n'.join(texts).lower()
    return ('out of memory' in blob) or ('outofmemoryerror' in blob) or ('cuda oom' in blob)


def build_job(c, e=None):
    e = e or {}
    case, ds, model, cond = c['case'], c['dataset'], c['model'], c['condition']
    tag = tag_of(c)
    name = name_of(c)
    pad = 11_500_000_000 if case == 'c12' else (5_000_000_000 if case == 'c04' else 0)
    batch_map = c['batch_map']
    if e.get('batch_map_path'):
        batch_map = e['batch_map_path']
    if e.get('batch_override') == 1 and case not in ('c12', 'c04'):
        pad = pad_at_b1(c)
    cell = dict(case=case, dataset=ds, model=model, condition=cond,
                profile=c['profile'], batch_map=batch_map)
    if case in ('c03', 'c15'):
        cell['assets'] = e.get('assets_path') or 'results/run_assets_20261108.json'
    if pad:
        cell['pad_bytes'] = pad
    spec = dict(name=name, lanes=[cell])
    spec_path = SERIAL_DIR / f'spec_{tag}.json'
    SERIAL_DIR.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=1))
    tpl = json.load(open(ROOT / 'scripts/qz_akr_batchverify_1gpu_20261108.json'))
    job = dict(tpl)
    job['name'] = name
    job['description'] = (f'AKR final_v1 serial single-GPU cell {case}/{ds}/{model}/{cond}; '
                          f'pad={pad}; batch_map={batch_map}')
    job['command'] = (
        f'mkdir -p {ROOT}/logs/run/{name} && cd {ROOT} && '
        f'export HF_HOME={ROOT}/.hf-cache ANIMAL_HF_HOME={ROOT}/.hf-cache HF_HUB_OFFLINE=1 && '
        f'MAX_ATTEMPTS=1 bash scripts/run_wave_cells_20261108.sh '
        f'results/run_specs_20261108/serial/spec_{tag}.json > logs/run/{name}/driver.log 2>&1')
    job['max_running_time_ms'] = '21600000' if (case == 'c04' or e.get('long_cap')) else '5400000'
    job['framework_config'][0]['spec_id'] = SPEC_ID
    job_path = SERIAL_DIR / f'job_{tag}.json'
    job_path.write_text(json.dumps(job, indent=1))
    return name, job, pad, batch_map


def build_c16_job():
    name = 'akr-fv1-s-c16-historical-20261108'
    tpl = json.load(open(ROOT / 'scripts/qz_akr_batchverify_1gpu_20261108.json'))
    job = dict(tpl)
    job['name'] = name
    job['description'] = 'AKR final_v1 Wave4 serial single-GPU: historical_rerun_v1 core + C16 readout, pad 12e9'
    job['command'] = (
        f'mkdir -p {ROOT}/logs/run/{name} && cd {ROOT} && '
        f'bash scripts/run_c16_serial_20261108.sh > logs/run/{name}/driver.log 2>&1')
    job['max_running_time_ms'] = '93600000'
    job['framework_config'][0]['spec_id'] = SPEC_ID
    (SERIAL_DIR / 'job_c16-historical.json').write_text(json.dumps(job, indent=1))
    return name, job


def main():
    rows = [r for r in json.load(open(ROOT / 'results/run_commands_final_v1_20261108.json'))
            if r['case'] != 'c00']
    queue = []
    for wave, cases in WAVES:
        for case in cases:
            queue += [r for r in rows if r['case'] == case]
    assert len(queue) == 112, len(queue)
    state = json.load(open(STATE)) if STATE.exists() else {'cells': {}, 'c16': {'status': 'PENDING'}}
    state.setdefault('cells', {})
    state.setdefault('c16', {'status': 'PENDING'})
    state.setdefault('started_at', now())
    log(f'SERIAL SUBMITTER START cells={len(queue)} spec={SPEC_ID} max_active={MAX_ACTIVE}')

    def save():
        STATE.write_text(json.dumps(state, indent=1))

    def find(c):
        tag = f"{c['case']}|{c['dataset']}|{c['model']}|{c['condition']}"
        return state['cells'].setdefault(tag, {'status': 'PENDING', 'attempts': 0, 'job_ids': [], 'last': ''})

    poll_fail = 0
    while True:
        jl = list_ours()
        if jl is None:
            poll_fail += 1
            if poll_fail == 5 or poll_fail % 10 == 0:
                log(f'POLL-FAIL consecutive={poll_fail} qz ListJobs unparseable; exit at 15')
            if poll_fail >= 15:
                log('POLL-FAIL exit for wrapper restart')
                sys.exit(1)
            time.sleep(30); continue
        poll_fail = 0
        jobs = {j['job_id']: j for j in jl}
        changed = False
        # 1) resolve every in-flight cell job
        for tag, e in list(state['cells'].items()):
            if e.get('status') != 'SUBMITTED':
                continue
            jid = e['job_ids'][-1]
            j = jobs.get(jid)
            if j is None:
                log(f'RETRY {tag} job-missing {jid}')
                e['status'] = 'PENDING'; changed = True; continue
            if j['status'] in NONTERM:
                continue
            c = e['command']
            ok = (j['status'] == 'job_succeeded')
            vok, why = verify_cell(c)
            if vok:
                e['status'] = 'OK'; e['last'] = f"{j['status']}:verified-complete"; changed = True
                ch = cell_code_hash(c)
                log(f'CELL OK {tag} job={jid} attempts={e["attempts"]} job_status={j["status"]} '
                    f'code_train50_runner={(ch or "unknown")[:12]}')
            else:
                e['last'] = f"{j['status']}:{why}"
                cap_stop = (j['status'] in ('job_stopped', 'job_timeout', 'job_failed')
                            and int(j.get('running_time_ms') or 0) >= 5_300_000)
                if cap_stop:
                    arch = None if cell_code_matches(c) else full_archive_cell(c, e['attempts'])
                    e['status'] = 'PENDING'; e['long_cap'] = True; e['attempts'] = 1
                    e['last'] = f"cap-resubmit {j['status']} at {j.get('running_time_ms')}ms"
                    changed = True
                    log(f'CELL CAP-RESUBMIT {tag} job={jid} status={j["status"]} '
                        f'running_ms={j.get("running_time_ms")} cap=21600000 archive={arch}')
                elif is_oom(c, name_of(c)):
                    cur = e.get('batch_override') or map_batch(c)
                    nxt = max(1, cur // 2)
                    if nxt == cur:
                        e['status'] = 'FAILED'; changed = True
                        log(f'CELL FAILED {tag} attempts={e["attempts"]} reason=oom-at-batch1 {e["last"]}')
                    else:
                        rel, sha = derived_map(c, nxt, cur, 'cuda_oom_halve', e['attempts'],
                                               e.get('batch_map_path'))
                        arch = (archive_cell_for_retry(c, e['attempts']) if cell_code_matches(c)
                                else full_archive_cell(c, e['attempts']))
                        e['status'] = 'PENDING'
                        e['batch_override'] = nxt
                        e['batch_map_path'] = rel
                        e['batch_map_sha256'] = sha
                        e['halve_from'] = cur
                        changed = True
                        log(f'CELL OOM-HALVE {tag} batch {cur}->{nxt} job={jid} '
                            f'map={rel} sha256={sha} archive={arch}')
                elif e['attempts'] >= 3:
                    e['status'] = 'FAILED'; changed = True
                    log(f'CELL FAILED {tag} attempts={e["attempts"]} reason={e["last"]}')
                else:
                    arch = (archive_cell_for_retry(c, e['attempts']) if cell_code_matches(c)
                            else full_archive_cell(c, e['attempts']))
                    e['status'] = 'PENDING'; changed = True
                    log(f'CELL RETRY {tag} attempts={e["attempts"]}/3 reason={e["last"]} archive={arch}')
        if changed:
            save()

        active = sum(1 for e in state['cells'].values() if e.get('status') == 'SUBMITTED')
        # 2) resolve C16 if it is the only thing left
        c16 = state['c16']
        if c16.get('status') == 'SUBMITTED':
            jid = c16['job_ids'][-1]
            j = jobs.get(jid)
            if j is None:
                c16['status'] = 'FAILED'; c16['last'] = 'missing'; save()
                log(f'C16 FAILED job={jid} status=missing')
            elif j['status'] not in NONTERM:
                ok = (j['status'] == 'job_succeeded')
                cap_stop = (not ok and j['status'] in ('job_stopped', 'job_timeout', 'job_failed')
                            and int(j.get('running_time_ms') or 0) >= 5_300_000)
                if cap_stop:
                    c16['status'] = 'PENDING'; c16['last'] = f"cap-stop {j['status']}"; save()
                    log(f'C16 CAP-RESUBMIT job={jid} status={j["status"]} '
                        f'running_ms={j.get("running_time_ms")} cap=93600000')
                else:
                    c16['status'] = 'OK' if ok else 'FAILED'; c16['last'] = j['status']; save()
                    log(f'C16 {"OK" if ok else "FAILED"} job={jid} status={c16["last"]}')
            # C16 running must NOT block cell refills; falls through to the fill loop.

        c16 = state['c16']

        def is_started(case):
            if case == 'c16':
                return c16.get('status') in ('OK', 'SUBMITTED')
            return any(e['status'] in ('OK', 'SUBMITTED')
                       for tag, e in state['cells'].items() if tag.split('|')[0] == case)

        qidx = {id(c): i for i, c in enumerate(queue)}
        pending_new = [c for c in queue
                       if find(c).get('status') == 'PENDING' and c['case'] in PRIORITY_NEW_CASES]
        seed = []
        for case in PRIORITY_NEW_CASES:
            if not is_started(case):
                for c in pending_new:
                    if c['case'] == case:
                        seed.append(c); break
        seed_ids = {id(c) for c in seed}
        rest_new = sorted([c for c in pending_new if id(c) not in seed_ids],
                          key=lambda c: (PRIORITY_NEW_CASES.index(c['case']), qidx[id(c)]))
        seq = seed + (['C16'] if c16.get('status') == 'PENDING' else []) + rest_new
        rerun = [c for c in queue
                 if find(c).get('status') == 'PENDING' and c['case'] not in PRIORITY_NEW_CASES]

        if not seq and not rerun and active == 0:
            if c16.get('status') in ('OK', 'FAILED'):
                log(f'SERIAL SUBMITTER DONE cells_ok={sum(1 for e in state["cells"].values() if e.get("status")=="OK")} '
                    f'cells_failed={sum(1 for e in state["cells"].values() if e.get("status")=="FAILED")} c16={c16.get("status")}')
                state['finished_at'] = now(); save(); return

        def submit_cell(c):
            nonlocal active
            e = find(c)
            tag = f"{c['case']}|{c['dataset']}|{c['model']}|{c['condition']}"
            name, job, pad, bmap = build_job(c, e)
            r = qz(['train', 'CreateJob'], job)
            jid = r.get('Result', {}).get('job_id') if isinstance(r.get('Result'), dict) else None
            if not jid:
                log(f'SUBMIT ERROR {tag} {json.dumps(r)[:300]}')
                return False
            e['attempts'] += 1
            e['status'] = 'SUBMITTED'
            e['job_ids'].append(jid)
            e['command'] = c
            e['wave'] = WAVE_OF.get(c['case'], '?')
            save()
            active += 1
            log(f'SUBMIT {tag} attempt={e["attempts"]} job={jid} name={name} pad={pad} '
                f'batch={e.get("batch_override", "")} map={bmap} active={active}')
            return True

        def submit_c16():
            nonlocal active
            name, job = build_c16_job()
            r = qz(['train', 'CreateJob'], job)
            jid = r.get('Result', {}).get('job_id') if isinstance(r.get('Result'), dict) else None
            if not jid:
                log(f'C16 SUBMIT ERROR {json.dumps(r)[:300]}')
                return False
            c16.update(status='SUBMITTED', job_ids=c16.get('job_ids', []) + [jid]); save()
            active += 1
            log(f'SUBMIT C16 {name} {jid} active={active}')
            return True

        reserve = len(PRIORITY_NEW_CASES) + 1
        while active + (1 if c16.get('status') == 'SUBMITTED' else 0) < MAX_ACTIVE:
            new_active = sum(1 for e in state['cells'].values()
                             if e['status'] == 'SUBMITTED' and e['command']['case'] in PRIORITY_NEW_CASES)
            c16_active = 1 if c16.get('status') == 'SUBMITTED' else 0
            if new_active + c16_active < reserve:
                if seq:
                    item = seq.pop(0)
                    if item == 'C16':
                        if not submit_c16(): break
                    elif not submit_cell(item):
                        break
                elif rerun:
                    if not submit_cell(rerun.pop(0)): break
                else:
                    break
            else:
                if rerun:
                    if not submit_cell(rerun.pop(0)): break
                elif seq:
                    item = seq.pop(0)
                    if item == 'C16':
                        if not submit_c16(): break
                    elif not submit_cell(item):
                        break
                else:
                    break
        time.sleep(60)

        time.sleep(60)


if __name__ == '__main__':
    while True:
        try:
            main()
            break
        except SystemExit:
            raise
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            try:
                with open(STATUS_LOG, 'a') as f:
                    f.write(f'{now()} SUBMITTER-EXCEPTION {type(exc).__name__}: {exc}; continuing in 60s\n')
            except Exception:
                pass
            time.sleep(60)
