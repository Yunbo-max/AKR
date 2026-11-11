# AKR final_v1 最终网格报告 (2026-11-10)
运行策略: 单卡 1 job/cell、最多 8 个并发、按 case 顺序补位（优先未开始 case C04→C11→C12→C13→C14→C15→C16）；OOM 阶梯减半、b=1 pad、C04/C16 长作业 6h/26h 上限；每 cell 回执记录自身 code_identity。
## 总览

- 网格 cell: 112
- OK: 100 (另有 C16 历史补表 OK，总计 101 OK)
- FAILED: 8
- STOPPED (user): 4
- 其他: 0
- C16 历史补表: OK (HISTORICAL_RERUN_DONE)

## 用户停止说明 (2026-11-11)

- 按用户最终指令，4 个 C04 ICL cell（bats/humbugdb × full/lp）在授权的 26h 窗口中被停止，未完成；它们不计为科学失败，也不计为 OK。
- 其余 100 个网格 cell 与 C16 历史补表已完成。所有后台提交器/看门狗/wrap-up/监控进程已停止。

## 关键结果要点

- C04 dogs/watkins 四个 cell 已完成；AKR 方法（original/soft）比三个 50-audio ICL 顺序快约 7–8×（generation wall seconds）：dogs full 116–134 s vs 957–1011 s；dogs lp 103–112 s vs 977–1028 s；watkins full 316–337 s vs 2436–2560 s；watkins lp 293–308 s vs 2342–2371 s，远高于 hard-AKR ≥2× ICL 的要求。
- 8 个文档化 FAILED：C03 watkins/humbugdb 长音频 BEATs（官方 BEATs 全片段 relative-position attention 随长度平方增长，CPU host OOM / GPU CUDA OOM，即使 batch 1；协议禁止截断）；C12 watkins/humbugdb GLM-4-Voice 8192-token 上下文（无截断）。
## 每 case 汇总

| case | OK | FAILED | STOPPED | 其他 |
|---|---|---|---|---|
| c01 | 16 | 0 | 0 | 0 |
| c02 | 8 | 0 | 0 | 0 |
| c03 | 4 | 4 | 0 | 0 |
| c04 | 4 | 0 | 4 | 0 |
| c05 | 4 | 0 | 0 | 0 |
| c06 | 4 | 0 | 0 | 0 |
| c07 | 4 | 0 | 0 | 0 |
| c08 | 4 | 0 | 0 | 0 |
| c09 | 8 | 0 | 0 | 0 |
| c10 | 8 | 0 | 0 | 0 |
| c11 | 8 | 0 | 0 | 0 |
| c12 | 4 | 4 | 0 | 0 |
| c13 | 8 | 0 | 0 | 0 |
| c14 | 8 | 0 | 0 | 0 |
| c15 | 8 | 0 | 0 | 0 |

## 代码身份跨度

- 早期收据: backend.py 76977f49；train50_runner.py a6cf10fb（首 57 个 OK cell）。
- no-op 修复后: train50_runner.py 389f885e（含 in-flight 过渡 cell）。
- 标签级 no-op gate: train50_runner.py b14ab7df（C14 及后续）。
- eGeMAPS 短音频 pad（最终）: train50_features.py 6d21555a（此后所有新启动 cell）。
- 每个 cell 的 REGISTRATION.json 记录该 cell 实际运行的完整 code map；C14 的 no-op 契约 = 标签级无干预一致性，raw-continuation 非确定性在 REPORT.no_op_disclosure 中披露。

## 每 cell 状态

| case | dataset | model | condition | status | scored | train50 hash | features hash | reason |
|---|---|---|---|---|---|---|---|---|
| c01 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c01 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c01 | watkins | qwen7b | full | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c01 | watkins | qwen7b | lp:1000 | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c01 | bats | qwen7b | full | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c01 | bats | qwen7b | lp:1000 | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c01 | humbugdb | qwen7b | full | OK | 1859 | a6cf10fb74df | 31d5669926ec |  |
| c01 | humbugdb | qwen7b | lp:1000 | OK | 1859 | a6cf10fb74df | 31d5669926ec |  |
| c01 | dogs | qwen3b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c01 | dogs | qwen3b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c01 | watkins | qwen3b | full | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c01 | watkins | qwen3b | lp:1000 | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c01 | bats | qwen3b | full | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c01 | bats | qwen3b | lp:1000 | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c01 | humbugdb | qwen3b | full | OK | 1859 | a6cf10fb74df | 31d5669926ec |  |
| c01 | humbugdb | qwen3b | lp:1000 | OK | 1859 | a6cf10fb74df | 31d5669926ec |  |
| c02 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c02 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c02 | watkins | qwen7b | full | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c02 | watkins | qwen7b | lp:1000 | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c02 | bats | qwen7b | full | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c02 | bats | qwen7b | lp:1000 | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c02 | humbugdb | qwen7b | full | OK | 1859 | a6cf10fb74df | 31d5669926ec |  |
| c02 | humbugdb | qwen7b | lp:1000 | OK | 1859 | a6cf10fb74df | 31d5669926ec |  |
| c03 | dogs | qwen7b | full | OK | 139 | b14ab7df0b28 | 31d5669926ec |  |
| c03 | dogs | qwen7b | lp:1000 | OK | 139 | b14ab7df0b28 | 31d5669926ec |  |
| c03 | watkins | qwen7b | full | FAILED | - | 0ff7d36ae330 | 6d21555aa744 | beats_long_duration_infeasible: watkins sampled durations up to 1260s; official BEATs quadratic relative-position attention; CPU host OOM and GPU CUDA OOM at batch 1; no truncation |
| c03 | watkins | qwen7b | lp:1000 | FAILED | - | 1fd316a02062 | 6d21555aa744 | beats_long_duration_infeasible: watkins sampled durations up to 1260s; official BEATs quadratic relative-position attention; CPU host OOM and GPU CUDA OOM at batch 1; no truncation |
| c03 | bats | qwen7b | full | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c03 | bats | qwen7b | lp:1000 | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c03 | humbugdb | qwen7b | full | FAILED | - | - | - | beats_long_duration_infeasible: humbugdb sampled durations up to 796s; official BEATs quadratic relative-position attention; CPU host OOM and GPU CUDA OOM at batch 1; no truncation |
| c03 | humbugdb | qwen7b | lp:1000 | FAILED | - | - | - | beats_long_duration_infeasible: humbugdb sampled durations up to 796s; official BEATs quadratic relative-position attention; CPU host OOM and GPU CUDA OOM at batch 1; no truncation |
| c04 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c04 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c04 | watkins | qwen7b | full | OK | 339 | 389f885efa81 | 31d5669926ec |  |
| c04 | watkins | qwen7b | lp:1000 | OK | 339 | 389f885efa81 | 31d5669926ec |  |
| c04 | bats | qwen7b | full | STOPPED (user) | - | 1fd316a02062 | 6d21555aa744 | USER STOP 2026-11-11: stopped incomplete by user instruction |
| c04 | bats | qwen7b | lp:1000 | STOPPED (user) | - | 1fd316a02062 | 6d21555aa744 | USER STOP 2026-11-11: stopped incomplete by user instruction |
| c04 | humbugdb | qwen7b | full | STOPPED (user) | - | 1fd316a02062 | 6d21555aa744 | USER STOP 2026-11-11: stopped incomplete by user instruction |
| c04 | humbugdb | qwen7b | lp:1000 | STOPPED (user) | - | 1fd316a02062 | 6d21555aa744 | USER STOP 2026-11-11: stopped incomplete by user instruction |
| c05 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c05 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c05 | watkins | qwen7b | full | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c05 | watkins | qwen7b | lp:1000 | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c06 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c06 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c06 | watkins | qwen7b | full | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c06 | watkins | qwen7b | lp:1000 | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c07 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c07 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c07 | watkins | qwen7b | full | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c07 | watkins | qwen7b | lp:1000 | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c08 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c08 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c08 | watkins | qwen7b | full | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c08 | watkins | qwen7b | lp:1000 | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c09 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c09 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c09 | watkins | qwen7b | full | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c09 | watkins | qwen7b | lp:1000 | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c09 | bats | qwen7b | full | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c09 | bats | qwen7b | lp:1000 | OK | 2000 | a6cf10fb74df | 31d5669926ec |  |
| c09 | humbugdb | qwen7b | full | OK | 1859 | b14ab7df0b28 | 6d21555aa744 |  |
| c09 | humbugdb | qwen7b | lp:1000 | OK | 1859 | b14ab7df0b28 | 6d21555aa744 |  |
| c10 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c10 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c10 | watkins | qwen7b | full | OK | 339 | a6cf10fb74df | 31d5669926ec |  |
| c10 | watkins | qwen7b | lp:1000 | OK | 339 | b14ab7df0b28 | 6d21555aa744 |  |
| c10 | bats | qwen7b | full | OK | 2000 | 1fd316a02062 | 6d21555aa744 |  |
| c10 | bats | qwen7b | lp:1000 | OK | 2000 | 1fd316a02062 | 6d21555aa744 |  |
| c10 | humbugdb | qwen7b | full | OK | 1859 | 1fd316a02062 | 6d21555aa744 |  |
| c10 | humbugdb | qwen7b | lp:1000 | OK | 1859 | 1fd316a02062 | 6d21555aa744 |  |
| c11 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c11 | dogs | qwen7b | lp:1000 | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c11 | watkins | qwen7b | full | OK | 339 | b14ab7df0b28 | 31d5669926ec |  |
| c11 | watkins | qwen7b | lp:1000 | OK | 339 | b14ab7df0b28 | 31d5669926ec |  |
| c11 | bats | qwen7b | full | OK | 2000 | b14ab7df0b28 | 31d5669926ec |  |
| c11 | bats | qwen7b | lp:1000 | OK | 2000 | b14ab7df0b28 | 6d21555aa744 |  |
| c11 | humbugdb | qwen7b | full | OK | 1859 | b14ab7df0b28 | 6d21555aa744 |  |
| c11 | humbugdb | qwen7b | lp:1000 | OK | 1859 | b14ab7df0b28 | 6d21555aa744 |  |
| c12 | dogs | glm4voice9b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c12 | dogs | glm4voice9b | lp:1000 | OK | 139 | b14ab7df0b28 | 6d21555aa744 |  |
| c12 | watkins | glm4voice9b | full | FAILED | - | 1fd316a02062 | 6d21555aa744 | glm_context_infeasible: GLM-4-Voice 8192-token budget; watkins audio exceeds it (e.g. 15004 tokens); no truncation |
| c12 | watkins | glm4voice9b | lp:1000 | FAILED | - | 1fd316a02062 | 6d21555aa744 | glm_context_infeasible: GLM-4-Voice 8192-token budget; watkins audio exceeds it; no truncation |
| c12 | bats | glm4voice9b | full | OK | 2000 | b14ab7df0b28 | 6d21555aa744 |  |
| c12 | bats | glm4voice9b | lp:1000 | OK | 2000 | b14ab7df0b28 | 6d21555aa744 |  |
| c12 | humbugdb | glm4voice9b | full | FAILED | - | 0ff7d36ae330 | 6d21555aa744 | glm_context_infeasible: GLM-4-Voice 8192-token budget; humbugdb durations up to 1795s; no truncation |
| c12 | humbugdb | glm4voice9b | lp:1000 | FAILED | - | 1fd316a02062 | 6d21555aa744 | glm_context_infeasible: GLM-4-Voice 8192-token budget; humbugdb durations up to 1795s; no truncation |
| c13 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c13 | dogs | qwen7b | lp:1000 | OK | 139 | b14ab7df0b28 | 6d21555aa744 |  |
| c13 | watkins | qwen7b | full | OK | 339 | b14ab7df0b28 | 6d21555aa744 |  |
| c13 | watkins | qwen7b | lp:1000 | OK | 339 | b14ab7df0b28 | 6d21555aa744 |  |
| c13 | bats | qwen7b | full | OK | 2000 | b14ab7df0b28 | 6d21555aa744 |  |
| c13 | bats | qwen7b | lp:1000 | OK | 2000 | b14ab7df0b28 | 6d21555aa744 |  |
| c13 | humbugdb | qwen7b | full | OK | 1859 | 1fd316a02062 | 6d21555aa744 |  |
| c13 | humbugdb | qwen7b | lp:1000 | OK | 1859 | b14ab7df0b28 | 6d21555aa744 |  |
| c14 | dogs | qwen7b | full | OK | 139 | 1fd316a02062 | 6d21555aa744 |  |
| c14 | dogs | qwen7b | lp:1000 | OK | 139 | 1fd316a02062 | 6d21555aa744 |  |
| c14 | watkins | qwen7b | full | OK | 339 | 1fd316a02062 | 6d21555aa744 |  |
| c14 | watkins | qwen7b | lp:1000 | OK | 339 | 1fd316a02062 | 6d21555aa744 |  |
| c14 | dogs | qwen3b | full | OK | 139 | 1fd316a02062 | 6d21555aa744 |  |
| c14 | dogs | qwen3b | lp:1000 | OK | 139 | 1fd316a02062 | 6d21555aa744 |  |
| c14 | watkins | qwen3b | full | OK | 339 | 1fd316a02062 | 6d21555aa744 |  |
| c14 | watkins | qwen3b | lp:1000 | OK | 339 | 1fd316a02062 | 6d21555aa744 |  |
| c15 | dogs | qwen7b | full | OK | 139 | a6cf10fb74df | 31d5669926ec |  |
| c15 | dogs | qwen7b | lp:1000 | OK | 139 | 1fd316a02062 | 6d21555aa744 |  |
| c15 | watkins | qwen7b | full | OK | 339 | 1fd316a02062 | 6d21555aa744 |  |
| c15 | watkins | qwen7b | lp:1000 | OK | 339 | 1fd316a02062 | 6d21555aa744 |  |
| c15 | bats | qwen7b | full | OK | 2000 | 1fd316a02062 | 6d21555aa744 |  |
| c15 | bats | qwen7b | lp:1000 | OK | 2000 | 1fd316a02062 | 6d21555aa744 |  |
| c15 | humbugdb | qwen7b | full | OK | 1859 | 1fd316a02062 | 6d21555aa744 |  |
| c15 | humbugdb | qwen7b | lp:1000 | OK | 1859 | 1fd316a02062 | 6d21555aa744 |  |

## 失败明细

- **C03 (4 cells)**: C03 long-audio BEATs infeasible: official BEATs full-clip relative-position attention is quadratic; watkins/humbugdb sampled durations include 796-1795 s extremes; CPU host OOMKilled and GPU CUDA OOM even at batch 1; protocol forbids truncation.
  - watkins/full: beats_long_duration_infeasible: watkins sampled durations up to 1260s; official BEATs quadratic relative-position attention; CPU host OOM and GPU CUDA OOM at batch 1; no truncation
  - watkins/lp:1000: beats_long_duration_infeasible: watkins sampled durations up to 1260s; official BEATs quadratic relative-position attention; CPU host OOM and GPU CUDA OOM at batch 1; no truncation
  - humbugdb/full: beats_long_duration_infeasible: humbugdb sampled durations up to 796s; official BEATs quadratic relative-position attention; CPU host OOM and GPU CUDA OOM at batch 1; no truncation
  - humbugdb/lp:1000: beats_long_duration_infeasible: humbugdb sampled durations up to 796s; official BEATs quadratic relative-position attention; CPU host OOM and GPU CUDA OOM at batch 1; no truncation
- **C12 (4 cells)**: C12 GLM-4-Voice architecture limit: context budget 8192 tokens; watkins/humbugdb audio exceeds it (e.g. 15004 tokens); no truncation allowed, so full/lp cells are infeasible for GLM.
  - watkins/full: glm_context_infeasible: GLM-4-Voice 8192-token budget; watkins audio exceeds it (e.g. 15004 tokens); no truncation
  - watkins/lp:1000: glm_context_infeasible: GLM-4-Voice 8192-token budget; watkins audio exceeds it; no truncation
  - humbugdb/full: glm_context_infeasible: GLM-4-Voice 8192-token budget; humbugdb durations up to 1795s; no truncation
  - humbugdb/lp:1000: glm_context_infeasible: GLM-4-Voice 8192-token budget; humbugdb durations up to 1795s; no truncation

## OOM 阶梯与派生 batch

- 派生 batch map receipts: 69
  - c01|qwen3b|full|watkins: 11->5 (cuda_oom_halve) map=c01-watkins-qwen3b-full_b5.json sha256=9d5bcddbfbac
  - c01|qwen3b|lp:1000|watkins: 11->5 (cuda_oom_halve) map=c01-watkins-qwen3b-lp-1000_b5.json sha256=bb7e97508341
  - c01|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c01-watkins-qwen7b-full_b2.json sha256=4fc398f5bd32
  - c01|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c01-watkins-qwen7b-full_b5.json sha256=aab324712678
  - c01|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c01-watkins-qwen7b-lp-1000_b2.json sha256=99cc833d8e46
  - c01|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c01-watkins-qwen7b-lp-1000_b5.json sha256=8526e188d97f
  - c02|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c02-watkins-qwen7b-full_b2.json sha256=01977452a893
  - c02|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c02-watkins-qwen7b-full_b5.json sha256=779abd929d9e
  - c02|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c02-watkins-qwen7b-lp-1000_b2.json sha256=30070ce25f6c
  - c02|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c02-watkins-qwen7b-lp-1000_b5.json sha256=b110d4603748
  - c03|dogs|full: ?->? (host_oom_beats_extractor_reduction) map=c03-dogs-full-beats4.json sha256=4210169e4f0a
  - c03|dogs|lp:1000: ?->? (host_oom_beats_extractor_reduction) map=c03-dogs-lp1000-beats4.json sha256=b49683692571
  - c03|humbugdb|qwen7b|full: ?->? (host_oom_beats_extractor_reduction_1) map=c03-humbugdb-full-beats1.json sha256=a7be400c629c
  - c03|humbugdb|full: ?->? (host_oom_beats_extractor_reduction) map=c03-humbugdb-full-beats4.json sha256=dba1a7365af9
  - c03|humbugdb|qwen7b|lp:1000: ?->? (host_oom_beats_extractor_reduction_1) map=c03-humbugdb-lp1000-beats1.json sha256=8c99aa919f24
  - c03|humbugdb|lp:1000: ?->? (host_oom_beats_extractor_reduction) map=c03-humbugdb-lp1000-beats4.json sha256=f43b7e76c3bf
  - c03|watkins|qwen7b|full: ?->? (host_oom_beats_extractor_reduction_1) map=c03-watkins-full-beats1.json sha256=f2b5c03eb33b
  - c03|watkins|full: ?->? (host_oom_beats_extractor_reduction) map=c03-watkins-full-beats4.json sha256=6b7a01fa485b
  - c03|watkins|qwen7b|lp:1000: ?->? (host_oom_beats_extractor_reduction_1) map=c03-watkins-lp1000-beats1.json sha256=e73148220bb4
  - c03|watkins|lp:1000: ?->? (host_oom_beats_extractor_reduction) map=c03-watkins-lp1000-beats4.json sha256=17fd0a91381a
  - c03|qwen7b|full|watkins: 2->1 (cuda_oom_halve) map=c03-watkins-qwen7b-full_b1.json sha256=821bc8f42460
  - c03|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c03-watkins-qwen7b-full_b2.json sha256=42823e6910b3
  - c03|qwen7b|full|watkins: 10->5 (cuda_oom_halve_beats4) map=c03-watkins-qwen7b-full_b5.json sha256=ad12897a1a70
  - c03|qwen7b|lp:1000|watkins: 2->1 (cuda_oom_halve) map=c03-watkins-qwen7b-lp-1000_b1.json sha256=699a1c99e3d2
  - c03|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c03-watkins-qwen7b-lp-1000_b2.json sha256=ab5c20fe9cb3
  - c03|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c03-watkins-qwen7b-lp-1000_b5.json sha256=040ad68c9c18
  - c05|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c05-watkins-qwen7b-full_b2.json sha256=96c3669f71cc
  - c05|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c05-watkins-qwen7b-full_b5.json sha256=ac68fe99a3d2
  - c05|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c05-watkins-qwen7b-lp-1000_b2.json sha256=4b9609d7bde0
  - c05|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c05-watkins-qwen7b-lp-1000_b5.json sha256=02fa03c1bde6
  - c06|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c06-watkins-qwen7b-full_b2.json sha256=f656027e98d2
  - c06|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c06-watkins-qwen7b-full_b5.json sha256=8b8627fccce7
  - c06|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c06-watkins-qwen7b-lp-1000_b2.json sha256=aa865b029b58
  - c06|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c06-watkins-qwen7b-lp-1000_b5.json sha256=286a9acf7cb3
  - c07|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c07-watkins-qwen7b-full_b2.json sha256=9a9930f40269
  - c07|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c07-watkins-qwen7b-full_b5.json sha256=7240a0d4922d
  - c07|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c07-watkins-qwen7b-lp-1000_b2.json sha256=f4118ad9ab1f
  - c07|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c07-watkins-qwen7b-lp-1000_b5.json sha256=1fd376692f8c
  - c08|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c08-watkins-qwen7b-full_b2.json sha256=5811b73d8ccd
  - c08|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c08-watkins-qwen7b-full_b5.json sha256=110d02f2887c
  - c08|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c08-watkins-qwen7b-lp-1000_b2.json sha256=aaa80b5ed7f4
  - c08|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c08-watkins-qwen7b-lp-1000_b5.json sha256=e09d2201663d
  - c09|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c09-watkins-qwen7b-full_b2.json sha256=c17c652115aa
  - c09|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c09-watkins-qwen7b-full_b5.json sha256=60f19b6a8f82
  - c09|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c09-watkins-qwen7b-lp-1000_b2.json sha256=d9dcecaca71d
  - c09|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c09-watkins-qwen7b-lp-1000_b5.json sha256=c8e9aa88c7f5
  - c10|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c10-watkins-qwen7b-full_b2.json sha256=c46859942790
  - c10|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c10-watkins-qwen7b-full_b5.json sha256=8632fe19d284
  - c10|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c10-watkins-qwen7b-lp-1000_b2.json sha256=3451a016ae76
  - c10|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c10-watkins-qwen7b-lp-1000_b5.json sha256=4d3e49313558
  - c11|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c11-watkins-qwen7b-full_b2.json sha256=0d60aadcaf12
  - c11|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c11-watkins-qwen7b-full_b5.json sha256=43894d0652d6
  - c11|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c11-watkins-qwen7b-lp-1000_b2.json sha256=2c6a020a5921
  - c11|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c11-watkins-qwen7b-lp-1000_b5.json sha256=45f0153b2917
  - c13|qwen7b|full|watkins: 5->2 (cuda_oom_halve) map=c13-watkins-qwen7b-full_b2.json sha256=70b74459d4b2
  - c13|qwen7b|full|watkins: 10->5 (cuda_oom_halve) map=c13-watkins-qwen7b-full_b5.json sha256=a00d77bd1638
  - c13|qwen7b|lp:1000|watkins: 5->2 (cuda_oom_halve) map=c13-watkins-qwen7b-lp-1000_b2.json sha256=fa1cb5d0918c
  - c13|qwen7b|lp:1000|watkins: 10->5 (cuda_oom_halve) map=c13-watkins-qwen7b-lp-1000_b5.json sha256=475b0cc0106c
  - c14|qwen7b|full|dogs: 9->1 (noop_check_batch1) map=c14-dogs-qwen7b-full_b1.json sha256=32f3d9a51a78
  - c14|qwen3b|full|watkins: 11->5 (cuda_oom_halve) map=c14-watkins-qwen3b-full_b5.json sha256=1cb958e656aa

- OOM/CAP 事件行数: 90
  - 2026-11-09T17:42:07Z CELL OOM-HALVE c08|watkins|qwen7b|lp:1000 batch 10->5 job=job-03c58628-31d0-49ef-8842-912c0dde1bed map=results/run_specs_20261108/batch_maps/c08-watkins-qwen7b-lp-1000_b5.json sha256=e09d2201663d6e5a795aab271ff82d0cb750db82b880827eb695ac3746dd31a2 archive=results/akr_train50/final_v1_oom_archive/c08-watkins-qwen7b-lp-1000_a1
  - 2026-11-09T17:44:09Z CELL OOM-HALVE c07|watkins|qwen7b|full batch 5->2 job=job-9de0829b-ce5a-400f-9d03-93437a3400ba map=results/run_specs_20261108/batch_maps/c07-watkins-qwen7b-full_b2.json sha256=9a9930f40269e591c555358d352ae581122b220d956f3415f6f951898fbea226 archive=results/akr_train50/final_v1_oom_archive/c07-watkins-qwen7b-full_a2
  - 2026-11-09T17:44:09Z CELL OOM-HALVE c07|watkins|qwen7b|lp:1000 batch 5->2 job=job-3b104a2e-9c1d-47a4-a680-cdbd1082c0da map=results/run_specs_20261108/batch_maps/c07-watkins-qwen7b-lp-1000_b2.json sha256=f4118ad9ab1f24935cee5b5c9ba077c2c0ae1011c0347bb4ea63a8364846423b archive=results/akr_train50/final_v1_oom_archive/c07-watkins-qwen7b-lp-1000_a2
  - 2026-11-09T17:46:10Z CELL OOM-HALVE c08|watkins|qwen7b|full batch 5->2 job=job-290c7247-ea3e-4657-b999-b94d47de6999 map=results/run_specs_20261108/batch_maps/c08-watkins-qwen7b-full_b2.json sha256=5811b73d8ccd24aae32eea033252fbee3c78dc8476e6bfc2b47a5ba040619a80 archive=results/akr_train50/final_v1_oom_archive/c08-watkins-qwen7b-full_a2
  - 2026-11-09T17:46:10Z CELL OOM-HALVE c08|watkins|qwen7b|lp:1000 batch 5->2 job=job-21c89500-1c27-4f58-9ac1-745e6db39eca map=results/run_specs_20261108/batch_maps/c08-watkins-qwen7b-lp-1000_b2.json sha256=aaa80b5ed7f427b976e0b3c091e198557dcd577870981fa03c35041cc418e92f archive=results/akr_train50/final_v1_oom_archive/c08-watkins-qwen7b-lp-1000_a2
  - 2026-11-09T17:53:14Z CELL OOM-HALVE c09|watkins|qwen7b|full batch 10->5 job=job-3f3f4f94-84c8-4967-9e4d-203e5a78a1f7 map=results/run_specs_20261108/batch_maps/c09-watkins-qwen7b-full_b5.json sha256=60f19b6a8f822692e942d63f54579e270695ccf1947a591ac74723f4888c84b5 archive=results/akr_train50/final_v1_oom_archive/c09-watkins-qwen7b-full_a1
  - 2026-11-09T17:57:19Z CELL OOM-HALVE c09|watkins|qwen7b|full batch 5->2 job=job-8d2c3a9c-649c-49ac-87f5-20ace43a1c21 map=results/run_specs_20261108/batch_maps/c09-watkins-qwen7b-full_b2.json sha256=c17c652115aa4211bb32581ddebea4d4354ab004b17e57e306e6c3a3a751f3de archive=results/akr_train50/final_v1_oom_archive/c09-watkins-qwen7b-full_a2
  - 2026-11-09T18:03:21Z CELL OOM-HALVE c09|watkins|qwen7b|lp:1000 batch 10->5 job=job-e1877280-946c-4af3-a7a4-7d33a6838509 map=results/run_specs_20261108/batch_maps/c09-watkins-qwen7b-lp-1000_b5.json sha256=c8e9aa88c7f59beb09d06949c593bc04b33578948576e493069014f8a093c5f1 archive=results/akr_train50/final_v1_oom_archive/c09-watkins-qwen7b-lp-1000_a1
  - 2026-11-09T18:08:23Z CELL OOM-HALVE c09|watkins|qwen7b|lp:1000 batch 5->2 job=job-190f2219-2ee4-4997-a397-db6c484eddbf map=results/run_specs_20261108/batch_maps/c09-watkins-qwen7b-lp-1000_b2.json sha256=d9dcecaca71d84e4e0695ea7c6a21f0a1c2f4865ce335379e00d7eddc2615bdd archive=results/akr_train50/final_v1_oom_archive/c09-watkins-qwen7b-lp-1000_a2
  - 2026-11-09T18:30:33Z CELL OOM-HALVE c10|watkins|qwen7b|full batch 10->5 job=job-08548e1a-6527-4e60-82e5-2447f6a91dee map=results/run_specs_20261108/batch_maps/c10-watkins-qwen7b-full_b5.json sha256=8632fe19d2843127e2e897b7c2fd70256845e5dc0b2717ff24e347bb136e3f82 archive=results/akr_train50/final_v1_oom_archive/c10-watkins-qwen7b-full_a1
  - 2026-11-09T18:35:40Z CELL OOM-HALVE c10|watkins|qwen7b|full batch 5->2 job=job-2001f4da-53c5-4edc-9851-205e5e9fce47 map=results/run_specs_20261108/batch_maps/c10-watkins-qwen7b-full_b2.json sha256=c468599427908d7c19971176686d111eccbba8b2ab60dcaa7b3fa9c1cf38fed3 archive=results/akr_train50/final_v1_oom_archive/c10-watkins-qwen7b-full_a2
  - 2026-11-09T20:57:26Z CELL OOM-HALVE c10|watkins|qwen7b|lp:1000 batch 10->5 job=job-9d2edff1-80aa-4ce1-90ef-2796b66f3aab map=results/run_specs_20261108/batch_maps/c10-watkins-qwen7b-lp-1000_b5.json sha256=4d3e49313558d9bb8e90b07bf32fc26300ddf4b279e027edfb09a04775f793d1 archive=results/akr_train50/final_v1_oom_archive/c10-watkins-qwen7b-lp-1000_a1
  - 2026-11-09T21:02:16Z CELL OOM-HALVE c10|watkins|qwen7b|lp:1000 batch 5->2 job=job-f40c9af3-8f5f-4be6-9f2a-1048ddc28628 map=results/run_specs_20261108/batch_maps/c10-watkins-qwen7b-lp-1000_b2.json sha256=3451a016ae76656c5bb27ffbec1d2f9d3ef1493e9112b1bb4853dd9872b5cf90 archive=results/akr_train50/final_v1_oom_archive/c10-watkins-qwen7b-lp-1000_a2
  - 2026-11-10T11:07:26Z CAP POLICY: pre-authorized long caps applied; C04 cells resubmit with max_running_time_ms=21600000 (6h); C16 pipeline=93600000 (26h); cells that hit the cap get automatic CAP-RESUBMIT with 6h; current c04 retries and C16 stopped and requeued under the new caps (cell caches kept for resume)
  - 2026-11-10T11:07:50Z CAP-RESUBMIT jobs: c16 job-08f2e893 cap=93600000; c04 watkins full job-d8c0273a cap=21600000; c04 watkins lp job-5c1d4afb cap=21600000; c04 bats full job-864ec6d5 cap=21600000; c04 bats lp job-7592fd86 cap=21600000; c04 humbugdb cells queued next at cap=21600000
  - 2026-11-10T11:50:05Z CELL OOM-HALVE c11|watkins|qwen7b|full batch 10->5 job=job-6c958920-2b2e-47ae-9c6a-72876b92728d map=results/run_specs_20261108/batch_maps/c11-watkins-qwen7b-full_b5.json sha256=43894d0652d62d0b4ed711a5f15b08eb58359224cd8749094a2494b38dd3221d archive=results/akr_train50/final_v1_oom_archive/c11-watkins-qwen7b-full_a2
  - 2026-11-10T11:56:06Z CELL OOM-HALVE c11|watkins|qwen7b|full batch 5->2 job=job-93c6543f-a378-4170-b227-2009accfdf1f map=results/run_specs_20261108/batch_maps/c11-watkins-qwen7b-full_b2.json sha256=0d60aadcaf1279585a3f0d0679b7d182d9880bbd82b9c6e5615578b49a735a0f archive=results/akr_train50/final_v1_oom_archive/c11-watkins-qwen7b-full_a3
  - 2026-11-10T12:28:13Z CELL OOM-HALVE c11|watkins|qwen7b|lp:1000 batch 10->5 job=job-d7afdc65-3cf3-4939-8670-ce287e561a9c map=results/run_specs_20261108/batch_maps/c11-watkins-qwen7b-lp-1000_b5.json sha256=45f0153b29175411001c4078e527b52c3a14e5c8fec7464df8730aaf236bb43f archive=results/akr_train50/final_v1_oom_archive/c11-watkins-qwen7b-lp-1000_a2
  - 2026-11-10T12:32:14Z CELL OOM-HALVE c11|watkins|qwen7b|lp:1000 batch 5->2 job=job-3d4ce50e-c6f3-4656-8242-b04f5d554269 map=results/run_specs_20261108/batch_maps/c11-watkins-qwen7b-lp-1000_b2.json sha256=2c6a020a5921a0e04a38837f75fe594f8b103ba57691f6ce212e8eff2e67de95 archive=results/akr_train50/final_v1_oom_archive/c11-watkins-qwen7b-lp-1000_a3
  - 2026-11-10T12:46:18Z CELL OOM-HALVE c03|watkins|qwen7b|full batch 10->5 job=job-84296bc7-ba02-400e-bc47-2e0b3eaf7cb2 map=results/run_specs_20261108/batch_maps/c03-watkins-qwen7b-full_b5.json sha256=b480e98586552e9623c01a574d5590a163ea70a3e8b825c6cc74f59c480e3c7d archive=results/akr_train50/final_v1_oom_archive/c03-watkins-qwen7b-full_a2_r1
  - 2026-11-10T12:52:52Z CELL OOM-HALVE c03|watkins|qwen7b|full batch 5->2 job=job-4232d0df-b96f-4afb-af3c-32572251f59c map=results/run_specs_20261108/batch_maps/c03-watkins-qwen7b-full_b2.json sha256=42823e6910b32f30764761de52f8b61ea0dcfcf8278e87cb5a2cfbb64d0b8082 archive=results/akr_train50/final_v1_oom_archive/c03-watkins-qwen7b-full_a2_r2
  - 2026-11-10T13:14:41Z CELL OOM-HALVE c03|watkins|qwen7b|lp:1000 batch 10->5 job=job-467b3d53-fbae-4607-b84f-fdaed4be8757 map=results/run_specs_20261108/batch_maps/c03-watkins-qwen7b-lp-1000_b5.json sha256=040ad68c9c18e8e86148c05404444cd7d3425f264346e9a68627f2572553227a archive=results/akr_train50/final_v1_codefix_archive/c03/c03-watkins-qwen7b-lp-1000_a2
  - 2026-11-10T14:58:39Z CELL OOM-HALVE c03|watkins|qwen7b|full batch 2->1 job=job-426ce0ad-0aaf-4629-8194-8f007a594ef8 map=results/run_specs_20261108/batch_maps/c03-watkins-qwen7b-full_b1.json sha256=821bc8f424603b9e942251c88e674975c77fe5cfe8efc52bbd5697f84f8cf721 archive=results/akr_train50/final_v1_oom_archive/c03-watkins-qwen7b-full_a2_r5
  - 2026-11-10T15:26:46Z CELL OOM-HALVE c03|watkins|qwen7b|lp:1000 batch 2->1 job=job-1d0e1ca9-5e5f-4548-91a9-55fe1dfffbf3 map=results/run_specs_20261108/batch_maps/c03-watkins-qwen7b-lp-1000_b1.json sha256=699a1c99e3d2519955bb737ee531bfc95ef53d0f4fecc1217606e6ad5531d033 archive=results/akr_train50/final_v1_oom_archive/c03-watkins-qwen7b-lp-1000_a2_r1
  - 2026-11-10T16:27:05Z CELL CAP-RESUBMIT c12|bats|glm4voice9b|full job=job-472c089f-8bd8-48b6-b888-7f21dbb6638b status=job_stopped running_ms=5428000 cap=21600000 archive=None
  - 2026-11-10T16:31:07Z CELL CAP-RESUBMIT c12|bats|glm4voice9b|lp:1000 job=job-eba2c970-3f0f-4fb5-8403-1b98d849cf5e status=job_stopped running_ms=5417000 cap=21600000 archive=None
  - 2026-11-10T16:33:53Z CELL OOM-HALVE c13|watkins|qwen7b|full batch 10->5 job=job-bacf344b-d71b-4d4f-97ef-cb297fcd8d2d map=results/run_specs_20261108/batch_maps/c13-watkins-qwen7b-full_b5.json sha256=a00d77bd16380c1a702a491adafcc7506255cece716ce8c00775154b7fd0e29c archive=results/akr_train50/final_v1_oom_archive/c13-watkins-qwen7b-full_a2
  - 2026-11-10T16:38:39Z CELL OOM-HALVE c13|watkins|qwen7b|full batch 5->2 job=job-1f322117-87a5-4326-999d-ede3dd97e6d3 map=results/run_specs_20261108/batch_maps/c13-watkins-qwen7b-full_b2.json sha256=70b74459d4b277061f09ba3d55be2a4da364976ecbc1bc8202976007058cbdf2 archive=results/akr_train50/final_v1_oom_archive/c13-watkins-qwen7b-full_a3
  - 2026-11-10T17:10:45Z CELL CAP-RESUBMIT c04|bats|qwen7b|lp:1000 job=job-7592fd86-945d-4857-a889-210b0b8ba0db status=job_stopped running_ms=21632000 cap=21600000 archive=results/akr_train50/final_v1_codefix_archive/c04/c04-bats-qwen7b-lp-1000_a2_r1
  - 2026-11-10T17:12:46Z CELL CAP-RESUBMIT c04|humbugdb|qwen7b|full job=job-da977a57-007e-48c0-b17c-31c47694aee2 status=job_stopped running_ms=21619000 cap=21600000 archive=results/akr_train50/final_v1_codefix_archive/c04/c04-humbugdb-qwen7b-full_a2_r1
  - 2026-11-10T17:14:47Z CELL CAP-RESUBMIT c09|humbugdb|qwen7b|full job=job-f4e31b07-372c-4331-99ec-a28ec3a4c500 status=job_stopped running_ms=5453000 cap=21600000 archive=None
  - 2026-11-10T17:14:47Z CELL OOM-HALVE c13|watkins|qwen7b|lp:1000 batch 10->5 job=job-09d687c9-4f56-4196-ad15-097a254ba1c3 map=results/run_specs_20261108/batch_maps/c13-watkins-qwen7b-lp-1000_b5.json sha256=475b0cc0106cd9bd5684951bf564f829d664165692856a9b451f3a9088e46d73 archive=results/akr_train50/final_v1_oom_archive/c13-watkins-qwen7b-lp-1000_a2
  - 2026-11-10T17:22:49Z CELL CAP-RESUBMIT c04|bats|qwen7b|full job=job-7632a021-9c2c-4461-b837-f3d7e17c1af1 status=job_stopped running_ms=21651000 cap=21600000 archive=results/akr_train50/final_v1_codefix_archive/c04/c04-bats-qwen7b-full_a3
  - 2026-11-10T17:24:50Z CELL CAP-RESUBMIT c04|humbugdb|qwen7b|lp:1000 job=job-3c302c56-fe78-4270-aa81-6ce578d3ec82 status=job_stopped running_ms=21646000 cap=21600000 archive=results/akr_train50/final_v1_codefix_archive/c04/c04-humbugdb-qwen7b-lp-1000_a3
  - 2026-11-10T17:26:50Z CELL OOM-HALVE c13|watkins|qwen7b|lp:1000 batch 5->2 job=job-fc73491d-aafc-49e8-9533-7427b7336360 map=results/run_specs_20261108/batch_maps/c13-watkins-qwen7b-lp-1000_b2.json sha256=fa1cb5d0918c62dd57cd54caff2e82760a645a32db28ab7f479e8078d8174215 archive=results/akr_train50/final_v1_oom_archive/c13-watkins-qwen7b-lp-1000_a3
  - 2026-11-10T19:17:11Z CELL CAP-RESUBMIT c09|humbugdb|qwen7b|lp:1000 job=job-ceff23ec-300f-4bf2-b0ee-53292ecbc292 status=job_stopped running_ms=5467000 cap=21600000 archive=None
  - 2026-11-10T20:58:36Z CELL OOM-HALVE c14|watkins|qwen7b|full batch 10->5 job=job-df6c3094-8add-4697-8430-038e2ada8f19 map=results/run_specs_20261108/batch_maps/c14-watkins-qwen7b-full_b5.json sha256=302b65568413c52b746142fc4475dc8c8c0ffc22eb936f546a5cafcc96613e81 archive=results/akr_train50/final_v1_oom_archive/c14-watkins-qwen7b-full_a2_r1
  - 2026-11-10T21:00:37Z CELL OOM-HALVE c14|watkins|qwen7b|lp:1000 batch 10->5 job=job-9526d157-f001-4c41-8d9b-642e586870ee map=results/run_specs_20261108/batch_maps/c14-watkins-qwen7b-lp-1000_b5.json sha256=2df97379f555e3e2c1a17f69d9ff93fcd6a3d974922ac27b4f5e2fb3dd177856 archive=results/akr_train50/final_v1_oom_archive/c14-watkins-qwen7b-lp-1000_a2_r1
  - 2026-11-10T21:04:38Z CELL OOM-HALVE c14|watkins|qwen7b|full batch 5->2 job=job-b58624d7-ca19-4d52-b99b-54e1d835b971 map=results/run_specs_20261108/batch_maps/c14-watkins-qwen7b-full_b2.json sha256=93569735f599dd4af59dfa29cd5e589014901b275b5372759814513a904a066c archive=results/akr_train50/final_v1_oom_archive/c14-watkins-qwen7b-full_a3
  - 2026-11-10T21:12:41Z CELL OOM-HALVE c14|watkins|qwen7b|lp:1000 batch 5->2 job=job-5a1d5f55-67f8-43f7-9781-89fab8614dbf map=results/run_specs_20261108/batch_maps/c14-watkins-qwen7b-lp-1000_b2.json sha256=67c79f4f60dac9a10713a0cd1353a46a6654847f4300cacdeb9173b1e49b340f archive=results/akr_train50/final_v1_oom_archive/c14-watkins-qwen7b-lp-1000_a3
  - 2026-11-10T21:42:47Z CELL OOM-HALVE c14|watkins|qwen3b|full batch 11->5 job=job-878bb014-67f9-444e-9ca2-c8827d2f7fc1 map=results/run_specs_20261108/batch_maps/c14-watkins-qwen3b-full_b5.json sha256=1cb958e656aa0956a65099ababc7a33881695835275a2757dee531ab8ca34ed9 archive=results/akr_train50/final_v1_oom_archive/c14-watkins-qwen3b-full_a2_r1
  - 2026-11-10T21:52:50Z CELL OOM-HALVE c14|watkins|qwen3b|lp:1000 batch 11->5 job=job-eacc6066-5af2-487a-ac9f-699613ef0c38 map=results/run_specs_20261108/batch_maps/c14-watkins-qwen3b-lp-1000_b5.json sha256=8a1c515a1eb2a2bab41af1e71734f780f39d31018bf49cb8dbdbcf232d59343e archive=results/akr_train50/final_v1_oom_archive/c14-watkins-qwen3b-lp-1000_a2_r1
  - 2026-11-10T22:10:55Z CELL OOM-HALVE c15|watkins|qwen7b|full batch 10->5 job=job-bcfcfdf9-4d28-4a02-8184-145737a17de7 map=results/run_specs_20261108/batch_maps/c15-watkins-qwen7b-full_b5.json sha256=fd12fb56d120790b0a525eaaafeec6cfb56b6b275cff092459f81b67c0d7b6c8 archive=results/akr_train50/final_v1_oom_archive/c15-watkins-qwen7b-full_a2
  - 2026-11-10T22:12:56Z CELL OOM-HALVE c15|watkins|qwen7b|lp:1000 batch 10->5 job=job-c163c6ad-182a-435d-b51b-927b37cf6492 map=results/run_specs_20261108/batch_maps/c15-watkins-qwen7b-lp-1000_b5.json sha256=26157dd3472d3dc324687f093714ef9b5660acf98d4f0b0579fda27229c5f4f6 archive=results/akr_train50/final_v1_oom_archive/c15-watkins-qwen7b-lp-1000_a2
  - 2026-11-10T22:14:57Z CELL OOM-HALVE c15|watkins|qwen7b|full batch 5->2 job=job-2509e934-adc1-4e20-96d6-c74df0ab3c11 map=results/run_specs_20261108/batch_maps/c15-watkins-qwen7b-full_b2.json sha256=f53969fdc045ca0be09b6b1bbeac1f9b1deeae6a14aa133e3fce92d9a491a94b archive=results/akr_train50/final_v1_oom_archive/c15-watkins-qwen7b-full_a3
  - 2026-11-10T22:18:59Z CELL OOM-HALVE c15|watkins|qwen7b|lp:1000 batch 5->2 job=job-ed073ed3-850b-4d13-ade7-4f1b8a1c1793 map=results/run_specs_20261108/batch_maps/c15-watkins-qwen7b-lp-1000_b2.json sha256=761ef26d17394df3d4f2ccb1dbf5e6ccaccc73c6c6c55bf6c2dce27c7c16b562 archive=results/akr_train50/final_v1_oom_archive/c15-watkins-qwen7b-lp-1000_a3
  - 2026-11-10T22:27:01Z CELL CAP-RESUBMIT c10|bats|qwen7b|full job=job-cad2d9b6-091f-489f-936c-47c99d2a8eb2 status=job_stopped running_ms=5439000 cap=21600000 archive=None
  - 2026-11-10T23:13:11Z CELL CAP-RESUBMIT c04|bats|qwen7b|lp:1000 job=job-54d08ba9-284b-40ea-ae22-a5a57c2b095c status=job_stopped running_ms=21631000 cap=21600000 archive=results/akr_train50/final_v1_codefix_archive/c04/c04-bats-qwen7b-lp-1000_a2_r2
  - 2026-11-10T23:15:12Z CELL CAP-RESUBMIT c04|humbugdb|qwen7b|full job=job-a911e129-233d-4327-8b98-46163b352df5 status=job_stopped running_ms=21641000 cap=21600000 archive=results/akr_train50/final_v1_codefix_archive/c04/c04-humbugdb-qwen7b-full_a2_r2
  - 2026-11-10T23:25:15Z CELL CAP-RESUBMIT c04|bats|qwen7b|full job=job-81278ef4-8388-47c1-a4d9-8b6db1346eb8 status=job_stopped running_ms=21646000 cap=21600000 archive=results/akr_train50/final_v1_codefix_archive/c04/c04-bats-qwen7b-full_a2_r1
  - 2026-11-10T23:27:16Z CELL CAP-RESUBMIT c04|humbugdb|qwen7b|lp:1000 job=job-96951aeb-057a-4270-bb60-5a6cbb2ca127 status=job_stopped running_ms=21637000 cap=21600000 archive=results/akr_train50/final_v1_codefix_archive/c04/c04-humbugdb-qwen7b-lp-1000_a2_r1
  - 2026-11-11T00:23:26Z CELL CAP-RESUBMIT c10|bats|qwen7b|lp:1000 job=job-dbb46ab8-03b3-46cb-9446-913d2fd31343 status=job_stopped running_ms=5419000 cap=21600000 archive=None
  - 2026-11-11T00:33:28Z CELL CAP-RESUBMIT c10|humbugdb|qwen7b|full job=job-4de7b8a7-13ab-47b7-984f-5697af5519f5 status=job_stopped running_ms=5432000 cap=21600000 archive=None
  - 2026-11-11T00:51:33Z CELL CAP-RESUBMIT c10|humbugdb|qwen7b|lp:1000 job=job-bf7bb615-3efc-430d-9ae3-45f4e02cfc6e status=job_stopped running_ms=5421000 cap=21600000 archive=None
  - 2026-11-11T05:16:16Z CELL CAP-RESUBMIT c04|bats|qwen7b|lp:1000 job=job-a6a3e7e2-b8a8-45a0-adaf-d10db86a9fe1 status=job_stopped running_ms=21643000 cap=21600000 archive=None
  - 2026-11-11T05:18:18Z CELL CAP-RESUBMIT c04|humbugdb|qwen7b|full job=job-afbdc5cc-a1bb-4e44-965a-93bce700595e status=job_stopped running_ms=21635000 cap=21600000 archive=None
  - 2026-11-11T05:27:18Z CELL CAP-RESUBMIT c04|bats|qwen7b|full job=job-39b769d4-221a-4adf-8414-d023fc1be90d status=job_stopped running_ms=21609000 cap=93600000 archive=None
  - 2026-11-11T05:29:19Z CELL CAP-RESUBMIT c04|humbugdb|qwen7b|lp:1000 job=job-8c9d7c63-9edc-4f33-b5cf-8f44b761435f status=job_stopped running_ms=21625000 cap=93600000 archive=None
  - 2026-11-11T11:27:03Z CELL CAP-RESUBMIT c04|bats|qwen7b|lp:1000 job=job-3c12454f-9ed7-47d3-bff2-33617a938a5b status=job_stopped running_ms=21635000 cap=93600000 det=False archive=None
  - 2026-11-11T11:27:04Z CELL CAP-RESUBMIT c04|humbugdb|qwen7b|full job=job-4f49bcb9-5a5d-4fa7-80d2-dc718c54c800 status=job_stopped running_ms=21640000 cap=93600000 det=False archive=None

## C16 历史补表

- qz job job-08f2e893-2057-495a-a7c1-ed898536e310，status=job_succeeded，max_running_time_ms=93600000。
- 输出: AKR-fixes-wt/results/akr_rebuttal/historical_rerun_v1/marmaudio；akr_historical_source/historical_rerun_v1；akr_historical_readouts/historical_rerun_v1_c16。
- 脚本自检通过: HISTORICAL_RERUN_DONE；MarmAudio 音频树以软链提供。

## 结果目录

- results/akr_train50/final_v1/<case>/<dataset>__<model>__<condition>__s20261107/{REPORT.json,RESULTS.csv,STATUS.json,CASE.json}
- results/run_specs_20261108/batch_maps/ (derived maps + receipts)
- results/akr_train50/final_v1_oom_archive/, final_v1_codefix_archive/ (evidence archives)
