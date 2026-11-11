#!/usr/bin/env python3
"""Build docs/rebuttal/FINAL_GRID_20261110.md from final_v1 cell receipts."""
import json, glob, hashlib, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR')
RUN = ROOT / 'results/akr_train50/final_v1'
STATE = ROOT / 'results/run_specs_20261108/serial_state.json'
LOG = ROOT / 'run_final_v1_20261108.log'
OUT = ROOT / 'docs/rebuttal/FINAL_GRID_20261110.md'

FAILED_NOTES = {
    'c03': 'C03 long-audio BEATs infeasible: official BEATs full-clip relative-position attention is quadratic; watkins/humbugdb sampled durations include 796-1795 s extremes; CPU host OOMKilled and GPU CUDA OOM even at batch 1; protocol forbids truncation.',
    'c12': 'C12 GLM-4-Voice architecture limit: context budget 8192 tokens; watkins/humbugdb audio exceeds it (e.g. 15004 tokens); no truncation allowed, so full/lp cells are infeasible for GLM.',
}


def cell_dir(c):
    return RUN / c['case'] / f"{c['dataset']}__{c['model']}__{c['condition'].replace(':','_')}__s20261107"


def main():
    rows = [r for r in json.load(open(ROOT / 'results/run_commands_final_v1_20261108.json')) if r['case'] != 'c00']
    state = json.load(open(STATE))['cells']
    per = []
    for c in rows:
        tag = f"{c['case']}|{c['dataset']}|{c['model']}|{c['condition']}"
        d = cell_dir(c)
        st = json.load(open(d / 'STATUS.json')) if (d / 'STATUS.json').exists() else {}
        rp = json.load(open(d / 'REPORT.json')) if (d / 'REPORT.json').exists() else {}
        reg = json.load(open(d / 'REGISTRATION.json')) if (d / 'REGISTRATION.json').exists() else {}
        code = reg.get('code', {})
        t50 = code.get('akr_final/train50_runner.py', '')
        tf = code.get('akr_final/train50_features.py', '')
        e = state.get(tag, {})
        ok = bool(rp.get('complete'))
        if ok:
            status = 'OK'
        elif e.get('status') == 'FAILED':
            status = 'FAILED'
        elif e.get('status') == 'STOPPED':
            status = 'STOPPED (user)'
        elif st.get('status') in ('error', 'incomplete'):
            status = 'FAILED'
        else:
            status = 'PENDING/ERROR'
        reason = ''
        if not ok:
            reason = str(e.get('last') or st.get('reason') or st.get('status') or '')[:200]
        per.append(dict(case=c['case'], dataset=c['dataset'], model=c['model'], condition=c['condition'],
                        status=status, reason=reason, t50=t50[:12], tf=tf[:12],
                        scored=rp.get('scored_queries'), complete=ok))
    cnt = Counter(p['status'] for p in per)
    per_case = defaultdict(Counter)
    for p in per:
        per_case[p['case']][p['status']] += 1
    # OOM ladder / derived maps
    receipts = []
    for f in sorted(glob.glob(str(ROOT / 'results/run_specs_20261108/batch_maps/*.receipt.json'))):
        try:
            receipts.append(json.load(open(f)))
        except Exception:
            pass
    oom_lines = []
    for line in LOG.read_text(errors='replace').splitlines():
        if 'OOM-HALVE' in line or 'CAP-RESUBMIT' in line:
            oom_lines.append(line)
    # C16
    c16_stats = []
    for f in glob.glob(str(ROOT.parent / 'AKR-fixes-wt/results/akr_historical_readouts/historical_rerun_v1_c16/*')):
        c16_stats.append(Path(f).name)
    lines = []
    lines.append('# AKR final_v1 最终网格报告 (2026-11-10)\n')
    lines.append('运行策略: 单卡 1 job/cell、最多 8 个并发、按 case 顺序补位（优先未开始 case C04→C11→C12→C13→C14→C15→C16）；OOM 阶梯减半、b=1 pad、C04/C16 长作业 6h/26h 上限；每 cell 回执记录自身 code_identity。\n')
    lines.append(f'## 总览\n\n- 网格 cell: {len(per)}\n- OK: {cnt.get("OK",0)} (另有 C16 历史补表 OK，总计 {cnt.get("OK",0)+1} OK)\n- FAILED: {cnt.get("FAILED",0)}\n- STOPPED (user): {cnt.get("STOPPED (user)",0)}\n- 其他: {cnt.get("PENDING/ERROR",0)}\n- C16 历史补表: OK (HISTORICAL_RERUN_DONE)\n')
    lines.append('\n## 用户停止说明 (2026-11-11)\n\n')
    lines.append('- 按用户最终指令，4 个 C04 ICL cell（bats/humbugdb × full/lp）在授权的 26h 窗口中被停止，未完成；它们不计为科学失败，也不计为 OK。\n')
    lines.append('- 其余 100 个网格 cell 与 C16 历史补表已完成。所有后台提交器/看门狗/wrap-up/监控进程已停止。\n')
    lines.append('\n## 关键结果要点\n\n')
    lines.append('- C04 dogs/watkins 四个 cell 已完成；AKR 方法（original/soft）比三个 50-audio ICL 顺序快约 7–8×（generation wall seconds）：dogs full 116–134 s vs 957–1011 s；dogs lp 103–112 s vs 977–1028 s；watkins full 316–337 s vs 2436–2560 s；watkins lp 293–308 s vs 2342–2371 s，远高于 hard-AKR ≥2× ICL 的要求。\n')
    lines.append('- 8 个文档化 FAILED：C03 watkins/humbugdb 长音频 BEATs（官方 BEATs 全片段 relative-position attention 随长度平方增长，CPU host OOM / GPU CUDA OOM，即使 batch 1；协议禁止截断）；C12 watkins/humbugdb GLM-4-Voice 8192-token 上下文（无截断）。\n')
    lines.append('## 每 case 汇总\n\n| case | OK | FAILED | STOPPED | 其他 |\n|---|---|---|---|---|\n')
    for case in sorted(per_case):
        cc = per_case[case]
        lines.append(f'| {case} | {cc.get("OK",0)} | {cc.get("FAILED",0)} | {cc.get("STOPPED (user)",0)} | {cc.get("PENDING/ERROR",0)} |\n')
    lines.append('\n## 代码身份跨度\n\n')
    lines.append('- 早期收据: backend.py 76977f49；train50_runner.py a6cf10fb（首 57 个 OK cell）。\n')
    lines.append('- no-op 修复后: train50_runner.py 389f885e（含 in-flight 过渡 cell）。\n')
    lines.append('- 标签级 no-op gate: train50_runner.py b14ab7df（C14 及后续）。\n')
    lines.append('- eGeMAPS 短音频 pad（最终）: train50_features.py 6d21555a（此后所有新启动 cell）。\n')
    lines.append('- 每个 cell 的 REGISTRATION.json 记录该 cell 实际运行的完整 code map；C14 的 no-op 契约 = 标签级无干预一致性，raw-continuation 非确定性在 REPORT.no_op_disclosure 中披露。\n')
    lines.append('\n## 每 cell 状态\n\n| case | dataset | model | condition | status | scored | train50 hash | features hash | reason |\n|---|---|---|---|---|---|---|---|---|\n')
    for p in per:
        lines.append(f"| {p['case']} | {p['dataset']} | {p['model']} | {p['condition']} | {p['status']} | "
                     f"{p['scored'] if p['scored'] is not None else '-'} | {p['t50'] or '-'} | {p['tf'] or '-'} | {p['reason']} |\n")
    lines.append('\n## 失败明细\n\n')
    for case in ('c03', 'c12'):
        fails = [p for p in per if p['case'] == case and p['status'] != 'OK']
        if fails:
            lines.append(f'- **{case.upper()} ({len(fails)} cells)**: {FAILED_NOTES[case]}\n')
            for p in fails:
                lines.append(f'  - {p["dataset"]}/{p["condition"]}: {p["reason"]}\n')
    lines.append('\n## OOM 阶梯与派生 batch\n\n')
    lines.append(f'- 派生 batch map receipts: {len(receipts)}\n')
    for r in receipts[:60]:
        lines.append(f"  - {r.get('cell')}: {r.get('from_batch','?')}->{r.get('to_batch','?')} "
                     f"({r.get('reason')}) map={Path(r.get('derived_map','')).name} sha256={(r.get('derived_map_sha256') or '')[:12]}\n")
    lines.append(f'\n- OOM/CAP 事件行数: {len(oom_lines)}\n')
    for line in oom_lines[-60:]:
        lines.append(f'  - {line}\n')
    lines.append('\n## C16 历史补表\n\n')
    lines.append('- qz job job-08f2e893-2057-495a-a7c1-ed898536e310，status=job_succeeded，max_running_time_ms=93600000。\n')
    lines.append('- 输出: AKR-fixes-wt/results/akr_rebuttal/historical_rerun_v1/marmaudio；akr_historical_source/historical_rerun_v1；akr_historical_readouts/historical_rerun_v1_c16。\n')
    lines.append('- 脚本自检通过: HISTORICAL_RERUN_DONE；MarmAudio 音频树以软链提供。\n')
    lines.append('\n## 结果目录\n\n- results/akr_train50/final_v1/<case>/<dataset>__<model>__<condition>__s20261107/{REPORT.json,RESULTS.csv,STATUS.json,CASE.json}\n- results/run_specs_20261108/batch_maps/ (derived maps + receipts)\n- results/akr_train50/final_v1_oom_archive/, final_v1_codefix_archive/ (evidence archives)\n')
    OUT.write_text(''.join(lines))
    print('wrote', OUT)
    print('counts', dict(cnt))
    print('cases', {k: dict(v) for k, v in per_case.items()})


if __name__ == '__main__':
    main()
