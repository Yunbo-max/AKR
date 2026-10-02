# AKR 最后三项：两项新消融 + 一项收窄的跨规模确认

本轮只交付代码、协议和测试，不生成新的科学结果。以已有连续 AKR 为主方法，不另开 tokenwise/factorized 搜索。旧 `akr_closing/`、E1–E9 配置、旧结果与 gate 均不改。新增包为 `akr_final/`，避免改变旧运行器的源码指纹。

## 为什么是这三项

**F1：声音能否预测修正？** 已有低秩/类别几何只说明目标有结构。对已保存 support 做录音分组交叉验证，将“低秩重建误差”与“声音回归预测误差”分开。每折重学标准化、SVD、ridge，不在全部样本上先拟合再假称留出验证。比较真实特征、训练特征打乱、固定均值；额外报告真实类别质心作为明确标注的 label-oracle diagnostic，不作为可部署方法。主指标为相对折内固定均值的 NMSE、余弦、投影/系数误差。不需要对 confirmation/test query 求梯度。

**F2：修正器能否跨频谱使用？** 同一批 support/query，在 full 与 1 kHz 之间做 2×2 源条件→目标条件交叉。每个 seed 只用已有 1-kHz LOCK 的 rank/penalty/relative alpha，四格共同固定，不在各格另搜参数。每个源条件拟合自己的标准化、基和回归器并原样应用于目标特征。原模型、固定均值、连续 AKR、无自身配对的 query-field 打乱均报告。直接 ridge 是读出参照。此实验测试修正映射的条件适用性，不把 probe 漂移自动解释成 native 错误的原因；跨频谱下降也是可报告结果。

**F3：同模型族的 3B 复现。** 不新增一个模型家族，不把它写成 cross-family generalization。7B/3B 使用相同 support 与 confirmation IDs；每个条件复用对应 7B LOCK，3B 不搜索超参数。3B 用自己的特征/梯度重拟合自己的 basis/router，不复制维度不兼容的 7B KV 张量。层位置继承源运行的相对深度。比较 native/fixed/continuous/ridge；已经完整且身份一致的 E9 预测优先复用。

## 严格控制工作量

仅 MarmAudio，语义标签，每类 8 条（共 48 条 support），full 与 1 kHz，原 source config 的前至多 3 个登记 seeds。选 seed 只看登记顺序，不看 accuracy、p-value 或 GATE.json。所有对应 LOCK 必须存在；缺锁就明确阻塞，不能把失败 seed 静默删掉。query 使用各 seed 原来的全部 confirmation IDs，不取易例前缀、不把已观察数据重命名为 untouched。

F1 为 6 个 support episode 的 CPU 分析；F2 为 3 个 seed 的 2×2 迁移；F3 为两个规模在相同 6 个 episode 上的比较。若源配置少于三个 seeds，使用所有登记 seeds，并明确输出数量。不会自动开启 Dogs/Watkins 长音频、E1/E7 长 ICL、LoRA 或新参数搜索。

## 一条命令

在现有 GPU 工作目录中保留数据、模型缓存、`.venv`：

```bash
git pull --ff-only origin main
SOURCE_RUN=akr_closing_all_v4 RUN_ID=akr_final_three_v1 PROFILE=24gb \
  bash scripts/run_akr_final_three.sh
```

`SOURCE_RUN` 是本地 E2 已跑出的运行目录名，不是 GitHub 分支。程序读取其真实 RUN.json、splits、EPISODE.json、LOCK.json，不假定 README 默认值就是实际设置。目录不在 `results/akr_closing/` 下时，可传绝对路径。

先只检查数据和登记信息，不加载模型：

```bash
.venv/bin/python scripts/run_akr_final_three.py \
  --source-run akr_closing_all_v4 --run-id akr_final_three_v1 --profile 24gb
```

仅用已有数组做 F1，禁止任何模型加载：

```bash
.venv/bin/python scripts/run_akr_final_three.py \
  --source-run akr_closing_all_v4 --run-id akr_final_three_v1 \
  --tasks F1 --cache-only --execute
```

若源数组因为路径、代码或身份不一致而无法安全复用，该模式报告缺项，不伪造成功。默认 `--execute` 模式可补提取本次范围内缺少的 support 特征/梯度；不重跑整个 E1–E9。24GB/16GB 是权重放置预算，不是任何上下文都能装下的保证；不自动量化、不偷偷截音频。48GB 卡可显式用 `PROFILE=48gb`，同一 RUN_ID 不更改 profile。

## 复用和结果身份

- 只读取旧运行目录，结果写入 `results/akr_final/<RUN_ID>/`。
- 优先导入身份匹配的 feature/audio-gradient NPZ；来源、SHA256 和复用记录保留。
- 7B confirmation 与 3B `<SOURCE_RUN>_3b_locked` E9 的逐项预测，只有配置/模型 revision/代码/锁/支持集/顺序/作用范围和完整 journal 校验符合时才复用。不能从一个平均准确率补出预测。
- 同一新运行中 F2/F3 完全一致的前向条件通过逐项缓存共享；零强度仍单独验证，不能用 native 输出冒充 no-op 检查。
- 训练梯度接口只允许登记 support IDs。Query 对象没有标签字段，标签在预测后计分。
- 所有新比较都保留负结果；无“显著才允许报告”的规则。不更改旧实验已登记的 gate。
- 出现无效输出计为错误；零强度若改变原始文本或标签，相关任务阻塞。

## 输出

`PLAN.json` 固定 episode 与 source hashes；`IDENTITY.json` 固定代码/环境；`WAVEFORM_HASHES.json` 固定音频。

`F1/<episode>/RESULT.json` 包含录音分组 folds、每 rank 的 NMSE、固定/打乱/label-oracle 对照及误差分解。

`F2/<target-episode>/from_<source-condition>/<method>/` 包含 journal、完整逐项预测和指标，可直接形成 2×2 迁移图。

`F3/<model>/<episode>/<method>/` 可形成 7B/3B 两行对照表。

`RESULTS.csv` 汇总已完整的试验格；`FINAL_STATUS.json` 明确全部请求任务是否完成。单格完整不等于所有任务完整。METRICS 中 accuracy/F1/invalid 是比例，gain_pp 是百分点。修复/改错计数、生成 token 数和实际增量范数保留；缓存复用不冒充新的运行耗时。

中断后重用同一命令恢复。任一任务失败会保存原因并允许执行其他任务，最终返回非零退出码；失败不能被改写为 completed。变更配置/源文件/环境使用新 RUN_ID，不删除身份文件绕过检查。

## 写进论文哪里

F1：方法分析“可压缩不等于可预测”；画 rank–重建误差/预测误差及打乱对照。
F2：Spectral Decision Drift 与方法结果之间的直接连接；四格共同锁参，报告所有迁移方向。
F3：补同族跨规模证据，报告效应大小、macro-F1、invalid 与逐 seed 方向，不声称已覆盖不同模型家族。

这三项结束即冻结范围，不因结果不理想继续扩 rank、prompt、alpha 或数据集。只按实际结果修改文字，不预设三项都必须提高性能。
