# MTP OPSD baseline：Local 执行手册

**分支：`research/mtp-opsd-baseline`。交付状态：`generated_unexecuted`。**
代码已经准备；本轮 Web 交付未安装训练环境、下载模型数据、运行模型、训练或评测。
[方法定义](rounds/mtp_opsd_baseline_20261111/METHOD.md) ·
[交接范围](rounds/mtp_opsd_baseline_20261111/WEB_HANDOFF.md) ·
[配置](configs/mtp_opsd/common.json)

## 1. 先进入独立分支和环境

首次独立 checkout：

```bash
git clone --single-branch --branch research/mtp-opsd-baseline https://github.com/Yunbo-max/AKR.git AKR-mtp-opsd
cd AKR-mtp-opsd
git rev-parse HEAD
python3.11 -m venv .venv-mtp-opsd
source .venv-mtp-opsd/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-mtp-opsd.txt
python -m pip check
```

PyTorch 2.10.0 的 cu128 wheel 来自[官方版本说明](https://pytorch.org/get-started/previous-versions/#v2-10-0)。
实际主机需具有兼容驱动和 BF16 CUDA GPU。若 cu128 与主机不符，先按官方说明选用
同为 2.10.0 的兼容 CUDA build，再完成整套 qualification，记录实际 build。
不要复用历史声学 AKR 的 Transformers 4.x 环境。

本轮没有确认可用的 SSH host 或 GPU，也没有实测最低显存。建议先在一张显存充足的
GPU 上验收；不能承诺 24 GB 可训练、80 GB 必然达到某吞吐，或全部 case 24 小时跑完。
单进程、batch=1；没有实现 torchrun/FSDP/DDP。多个 GPU 可以各自运行不同 case。

## 2. 显式下载官方模型与数据

```bash
python -m mtp_opsd.cli plan
python -m mtp_opsd.cli download --assets-root assets/mtp_opsd
python -m mtp_opsd.cli check --assets-root assets/mtp_opsd --full-hashes
```

`plan` 只列配置，不读取模型数据。只有 `download` 获取外部输入。
下载可续传，随后核对完整文件、上游 LFS/Git hash 和本地 SHA256。
不要只下载普通 Transformers 会使用的权重；必须保留原生 `mtp.*` tensors。

| 输入 | 固定来源 / revision | 用途 |
|---|---|---|
| 模型 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) · `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 同模型学生和教师，含一个原生 MTP 层 |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) · `740312add88f781978c0658806c59bc2815b9866` | train 7,473 中固定洗牌取最多 512 个问题；完整 test 1,319 |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be) · `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | 完整 test 500，不参与训练 |
| Native evaluator | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520) · `d6de81643928d653435c431bae19945d41d32520` | 原生任务 YAML、prompts、filters、scorers |

生成 `assets/mtp_opsd/asset_manifest.json`、模型和数据 manifest。
后续训练、评测读取这些本地输入，不在缺失时悄悄下载替代品。

## 3. 真实 GPU 验收：必须先通过

```bash
CUDA_VISIBLE_DEVICES=0 python -m mtp_opsd.cli qualify \
  --assets-root assets/mtp_opsd \
  --output artifacts/mtp_opsd/qualification.json \
  --device cuda:0
```

该命令在真实模型上执行少量生成和反传，**没有 optimizer step，不产生 benchmark 分数**。
它依次检查：

- 固定环境版本、官方输入完整性，以及原生 evaluator/scorer 能离线构造完整任务。
- 两种解码的 32-token cached logits 与 full-sequence logits 对齐。
- 真实原生 MTP 头参与生成；教师和学生的 KD 各调用一次 MTP。
- 相同分布 KL≈0；参考条件化 KL 给 backbone 与 MTP adapters 有限非零梯度。
- adapter 保存再加载后的参数字节与输出一致。

所有记录保存到 qualification receipt、历史 receipt 和 stdout/stderr 日志。
32-token 验收不等于 1,536-token 训练显存/速度已验证；第一条真实训练仍需观察峰值资源。
未通过时修复实际接口/数值原因，再验收；不要放宽阈值以制造通过记录或移除原生头。

训练入口会验证 receipt 对应相同源代码、环境、模型 revision、核心运行设置和
训练结构。native cases 只需要共享核心设置，不要求启用训练 adapter。
修改 LoRA rank/target、dtype、attention 或 checkpointing 后，先用相同
`--overrides FILE.json` 重新 qualify，并使用新 run ID。
`--device` 可变化；到不同 GPU/驱动环境仍应实际重新验收。

## 4. 每个 case 单独启动

所有命令都在仓库根目录、上述 Python 环境下执行。下面给出同一个实验组 ID，
但每条命令是独立进程、独立 checkpoint 和独立输出目录。每张卡同一时刻只安排
一个 case；按需要依次执行，或在不同卡上启动，不要直接整批挤到一张卡。

| Case | 训练 rollout / 监督 | 每题自采样 G | 评测解码 |
|---|---|---:|---|
| c00_native_ntp | 无训练 | — | NTP |
| c01_native_mtp | 无训练 | — | MTP |
| c02_ntp_opsd_g4 | NTP / 主头 KL | 4 | NTP + MTP |
| c03_mtp_opsd_g4 | MTP / 主头与 MTP 头 KL | 4 | NTP + MTP |
| c04_ntp_opsd_g8 | NTP / 主头 KL | 8 | NTP + MTP |
| c05_mtp_opsd_g8 | MTP / 主头与 MTP 头 KL | 8 | NTP + MTP |

```bash
bash scripts/mtp_opsd/c00_native_ntp.sh run --run-id mtp_opsd_pilot_v1
bash scripts/mtp_opsd/c01_native_mtp.sh run --run-id mtp_opsd_pilot_v1
bash scripts/mtp_opsd/c02_ntp_opsd_g4.sh run --run-id mtp_opsd_pilot_v1
bash scripts/mtp_opsd/c03_mtp_opsd_g4.sh run --run-id mtp_opsd_pilot_v1
bash scripts/mtp_opsd/c04_ntp_opsd_g8.sh run --run-id mtp_opsd_pilot_v1
bash scripts/mtp_opsd/c05_mtp_opsd_g8.sh run --run-id mtp_opsd_pilot_v1
```

推荐首次顺序：c00、c01、c02、c03，再 c04、c05。
先看原生解码是否正常，再看 G=4 成本和实际 update 率，最后完成 G=8 对照。
没有自动扩展训练预算或自动启动更多实验。

每个 trained case 默认最多 512 个固定洗牌训练问题、128 次有效更新，
累计 24 小时训练预算。预算跨 resume/replay 累计，达到时在安全操作边界停止；
正在执行的单次生成/反传/保存可能越过时限，不是强杀式墙钟限制。
没有达到 128 步不等于代码失败，需要报告实际停止原因与 skip 比例。

首轮 **评测 n=1**，与训练 G=4/8 是两个不同概念。
每次评测完整 GSM8K 1,319 + MATH-500 500；六 case 全完成时，
共 10 个解码条件，即 18,190 条评测 response。
每条最多 1,536 tokens，完整评测时间额外计算，不在训练的 24 小时预算内。
n>1 可通过新 run 的配置改变，每个 repetition 都保留完整 native pass。

## 5. 单独训练、单独评测与恢复

```bash
bash scripts/mtp_opsd/c03_mtp_opsd_g4.sh train --run-id mtp_opsd_pilot_v1
bash scripts/mtp_opsd/c03_mtp_opsd_g4.sh evaluate --run-id mtp_opsd_pilot_v1 --checkpoint latest
```

这是 `run` 的替代用法；不要对已有完成结果再不带 resume 地重新训练。
中断后：

```bash
bash scripts/mtp_opsd/c03_mtp_opsd_g4.sh run --run-id mtp_opsd_pilot_v1 --resume
```

恢复会校验 config、源代码、训练问题顺序、checkpoint 文件 checksum、
模型 revision 和 trainable parameter names/shapes。已完成训练只有在 summary、
status 与 latest checkpoint 一致时才复用。

评测以完整 task/mode/repetition 为恢复单位：完成且 checksum 一致的 pass 复用；
中断 pass 保留为 `incomplete_sample_...`，完整重跑该 pass，不拼接半次结果。

如果进程被 SIGKILL，可能残留 `.training.lock`。先读取其 PID/hostname 并在对应
主机确认进程已不存在，再仅清理该 case 的残留锁。不能为了启动新进程删除仍被使用的锁。
checkpoint 不完整时只能恢复最后一个完整记录；不能手工把半成品改成 latest。

## 6. 每个 case 保存在哪里

例如 c03：

```text
runs/mtp_opsd/mtp_opsd_pilot_v1/c03_mtp_opsd_g4/seed_42/
  resolved_config.json
  run_identity.json
  logs/
  environment.json
  backend_provenance.json
  rollouts/<invocation_id>/prompt_*.json
  metrics.jsonl
  budget_ledger.json
  training_status.json
  training_summary.json
  latest_checkpoint.json
  checkpoints/update_*_prompt_*_<id>/
    adapter/adapter.safetensors
    adapter/adapter_config.json
    training_state.pt
    state.json
    identity.json
    manifest.json
  evaluation/<ntp-or-mtp>/<native-task>/sample_000/
    generations.jsonl
    samples.jsonl
    results.json
    complete.json
  evaluation/summary.json
```

每次训练保存原始 response IDs、解析结果、参考/target 选择、skip 原因、
两头 KL、MTP 实际调用计数及资源记录。checkpoint 含 adapter、optimizer、
Python/NumPy/Torch/CUDA RNG 与 selection RNG。默认每 16 updates 保存一次，
开始、正常结束及可安全恢复的失败边界也保存。
native case 没有 training/checkpoint 文件是正常情况。

汇总：

```bash
python -m mtp_opsd.cli report --run-id mtp_opsd_pilot_v1
```

生成该实验组的 `comparison.json` 和 `comparison.md`，保留未开始/未完成 case。
`completed_no_updates` 会标记为零次优化；不能写成训练收益。

## 7. 原生评测口径

| Native task | 原生主要指标 | 分母 |
|---|---|---:|
| gsm8k_cot_zeroshot | exact_match,flexible-extract | 1,319 |
| minerva_math500 | math_verify,none | 500 |

使用固定 upstream YAML/filters/scorers，明确设置 zero-shot、模型 chat template、
non-thinking、temperature=0.8、top_p=0.95、max_new_tokens=1,536。
这是声明的统一比较协议，不是官方默认 greedy/few-shot leaderboard 配置复现。
所有原生附加 filter 指标仍保存。

评测中的长度截断照样计入完整分母并按原生 scorer 评分；生成异常中止该 pass，
不能删掉失败题后汇总。原生 aggregate 必须与保存的逐题分数均值一致。
n=1 的 avg@1 与 pass@1 完全相同。

## 8. Local 首次接手需要回填的事实

记录实际 GPU/驱动/CUDA、Python/package build、模型与数据 hashes、qualification
receipt、第一条完整真实 train update 的峰值内存/耗时、全量 evaluator 输出路径。
如有问题，回填真实 traceback 和最小修复，不提前宣称 Gate 通过。

要继续远端运行，Local agent 只需依本手册完成验收和这六个有界 case，
不需要等待新增方法设计或用户再次许可。用户未指定远端 host；
本手册不提供虚构 SSH 命令。
