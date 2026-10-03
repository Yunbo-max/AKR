# 没有旧 RUN/LOCK：独立启动最后三项实验

## 这条入口解决什么

旧 `scripts/run_akr_final_three.sh` 是接续已有 selection 的入口，保留原行为。
新 `scripts/run_akr_standalone.sh` 直接读取 `configs/akr_standalone.yaml` 和原始
MarmAudio manifest/音频。**不读取旧 RUN.json、EPISODE.json 或 LOCK.json，也不
为了骗过检查伪造它们。** 旧缓存不是必需品，缺失的 support 特征/梯度按需重算。
新的计划、样本身份、固定设置、波形哈希、逐样本预测会自动保存；以后恢复同一
运行要保留这些新输出，不要通过删除校验文件强行沿用已改变配置的运行编号。

## 已限定的实验范围

- F1：仅 support 的分组梯度可预测性分析；每折重拟合 SVD/标准化/ridge。
- F2：Full 与 LP1 的 2×2 修正器迁移；两种源条件各自拟合，所有格子同固定参数。
- F3：Qwen2.5-Omni-7B/3B 的同族跨规模比较；各自提取特征/梯度并拟合修正器。
- MarmAudio 六类，每类 8 条（48 total），一个 seed=20260914，全部分组 confirmation query。
- 固定 r=4、ridge=10、relative alpha=.01。这是明确固定的新设置，**不是声称恢复了旧调参最优值**。
- 不自动新增 Qwen3/GLM 后端，也不扩大到其他数据集、LoRA、长音频 ICL。

重新划分只使用 manifest/seed/组信息，不根据模型表现筛选。原数据可能早已观察过，
故不声称是全新 untouched benchmark。分组不足、support 每类不足、原始音频缺失、
support/query 波形重复会明确失败，不借用 query 填 support，不截取音频蒙混过关。

## 运行

```bash
# 保留原数据目录和 .venv
.venv/bin/python scripts/run_akr_standalone.py --run-id akr_standalone_v1
RUN_ID=akr_standalone_v1 PROFILE=24gb bash scripts/run_akr_standalone.sh
```

脚本依次执行 CPU 单元测试、原始数据预检、正式模型实验。需要准备模型访问、
原始音频、manifest、相容的 Torch/Transformers/Qwen 环境以及足够内存。
16/24/48GB profile 是权重放置预算，不是所有录音均不 OOM 的保证。

只跑已收窄的一天核心范围 F1+F3：

```bash
TASKS=F1,F3 RUN_ID=akr_standalone_v1 PROFILE=24gb bash scripts/run_akr_standalone.sh
```

原始音频仍在，而中间缓存没保存：普通 `--execute` 会补算。
`--cache-only --execute` 禁止模型加载，因此首次空缓存会明确阻塞；它不是自动从头跑的选项。

## 自动输出

`results/akr_final/<RUN_ID>/` 包括：
- `STANDALONE_PROTOCOL.json`：真实新划分、support/query IDs、固定参数、配置来源。
- `IDENTITY.json`、`WAVEFORM_HASHES.json`：恢复及缓存检查。
- `cache/`、`prediction_cache/`：本次运行复用，不强制导入不明旧缓存。
- `F1/`、`F2/`、`F3/`：各实验产物；`RESULTS.csv`、`FINAL_STATUS.json`：完整率。

该入口复用原 continuous AKR 数学实现与模型后端；额外的小回归器照常拟合，
Qwen 主干不更新，只有 support 可以请求标签梯度。
运行记录中的内部 `lock` 字段只是复用既有数值执行代码的固定设置容器；
其 `settings_origin` 明确写 `fixed_yaml_not_historical_LOCK`，不对应任何旧锁文件。

## 论文与绘图

已有论文/历史结果不改。新结果单独按固定配置报告；不拿它替换另一套 support/
query 或选参协议的分数。新图使用 `paper/figure_prompts_standalone/` 下的文件，
未完成的 F1/F2/F3 图不填预测值，不把 CPU 测试当作模型性能实验。
