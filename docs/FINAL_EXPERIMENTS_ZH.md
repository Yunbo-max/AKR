# 最后实验：任务、数据、模型与命令

当前方法是 **Acoustic-to-KV Regression (AKR)**。本次代码整合不启动GPU实验，不重新训练Speech LM。

## 已完成与待跑

| ID | 内容 | 数据/模型 | 状态 |
|---|---|---|---|
| RQ1 historical | 原生识别修正 | MA-CT/LP1，Qwen2.5-Omni-7B，20条错误富集support | 公开原始预测和统计已导入；不默认重跑 |
| RQ2 historical | 条件化/范数/结构控制 | BEANS Dogs，Qwen2.5-Omni-7B | 完整历史强度比较保留 |
| F1 | 留出支持集梯度预测 | MA-CT Full/LP1，7B | 已完成；从交付包恢复标量导出，不伪称完整大梯度已迁移 |
| F2 | 修正器跨频谱迁移 | MA-CT Full/LP1，7B | 已完成，混合/有限结果保留 |
| F3 | 同族跨规模 | MA-CT Full/LP1，7B/3B | 已完成；不是跨模型家族泛化 |
| **S1** | **最简单替代比较** | **MA-CT Full/LP1，7B** | **本轮实现、真实模型未跑；默认新任务** |

S1在同一support与query上比较Native、Fixed mean、Continuous AKR、直接Ridge、Probe-to-text、Probe-to-LM。Probe-to-text只是同一类别预测的文字映射，准确率必须与Ridge一致。Probe-to-LM把预测作为明确可能出错的外部文字证据，交给原冻结模型听音回答。query真标签仅用于最终评分。

固定配置：六类MarmAudio，每类8条support（共48），配置seed20260914，rank4、ridge penalty10、relative alpha0.01，Full与1-kHz两条件。query使用确定划分中的全部录音组分离成员；不能按结果挑选、不能将partial作为完整结果。新配置是独立固定设置，不是原20条error-enriched、feature0、raw eta300协议。

指标：accuracy、macro-F1、invalid、repaired/harmed、probe服从率、正确probe被改错/错误probe被修复计数。不因为S1结果弱就反复看query调参。

## 模型具体规格

| | 主模型 | 规模参考 |
|---|---|---|
| Hub ID | Qwen/Qwen2.5-Omni-7B | Qwen/Qwen2.5-Omni-3B |
| Revision | ae9e1690543ffd5c0221dc27f79834d0294cba00 | f75b40e3da2003cdd6e1829b1f420ca70797c34e |
| 后端 | Qwen2_5OmniThinkerForConditionalGeneration | 同实现，独立拟合 |
| 模式 | Thinker冻结，Talker关闭 | 同左 |
| 精度/batch/decoding | BF16 / 1 / greedy | 同左 |
| 固定扩展feature index | 24 | 32 |
| 固定扩展K/V层 | 7,14,21,24 | 9,18,27,31 |

GLM-4-Voice与Qwen3-Omni目前没有本仓库已验证后端，不加入默认队列；不能仅换模型名就声称跨模型实验可运行。

## 数据

MA-CT：MarmAudio六类call type，546条四专家一致录音，93个source recording groups；没有clip-level caller IDs。BD-ID：BEANS Dogs十个个体，415/139/139 train/valid/test。BW-SP：BEANS Watkins31物种，1017/339/339。BZ-12：2950条、12组件、10秒前缀上限的BEANS-Zero历史诊断，不是官方完整时长成绩，也不在本轮默认队列。

仓库保留manifests、准备脚本、统计和预测，不镜像许可不明的原始音频或基础模型权重。没有旧RUN.json/LOCK.json不妨碍独立入口；没有真实音频或模型访问仍然无法执行模型实验。

```bash
python scripts/prepare_marmaudio_expert_validation.py \
  --archive /path/to/Technical_Validation_Data.zip \
  --annotations /path/to/Annotations.tsv \
  --output-dir data/marmaudio/expert_validation \
  --manifest data/manifests/marmaudio_expert_validation.csv
```

## 独立启动

完成本分支整合后使用以下入口；合并前以MIGRATION_STATUS.md为准。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
PYTHONPATH=src:. python -m pytest -q
python scripts/run_akr_release.py plan
```

真实运行先安装匹配GPU驱动的PyTorch/torchvision，再安装requirements-qwen.txt。plan不加载模型，check检查真实文件和分组。

```bash
python scripts/run_akr_release.py check --tasks S1 --run-id akr_s1_v1
python scripts/run_akr_release.py run --tasks S1 --run-id akr_s1_v1 \
  --profile 24gb --budget-hours 24 --execute
```

输出到results/akr_final/akr_s1_v1。相同代码/数据/配置/环境/run ID可续跑，变更需新ID。明确复评旧F1/F2/F3才写--tasks F1,F2,F3；全部独立入口用--tasks S1,F1,F2,F3。24小时是墙钟上限，不是未知GPU的完成时长保证。超时保留逐样本checkpoint，不把未完成格子填0。
