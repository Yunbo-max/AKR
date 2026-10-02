# 最后实验：任务、数据、模型与命令

当前主方法名称：**Acoustic-to-KV Regression (AKR)**。代码不包含训练Speech LM的新目标；只拟合support修正回归器。机器可读规格在 `configs/akr_registry.json`，独立启动固定配置在 `configs/akr_standalone.yaml`。

## 先看哪些完成，哪些待跑

| ID | 研究问题 | Benchmark / 模型 | 状态 |
|---|---|---|---|
| RQ1 historical | 原生识别是否改善 | MA-CT/LP1，Omni-7B；20 total错误富集support | 旧公开预测和统计已迁移；不默认重跑 |
| RQ2 historical | 条件化是否重要 | BD-ID，Omni-7B；fixed/random/pooled/ordered/permuted | 旧完整比较已迁移；不默认重跑 |
| F1 / RQ3 | 声音能否预测修正梯度 | MA-CT Full/LP1，Omni-7B；5折支持组留出 | 已完成标量导出迁移；大梯度未随本轮文件提供 |
| F2 | 修正器频谱迁移 | MA-CT，Omni-7B；Full/LP1四格 | 已完成导出迁移；跨条件结果有限 |
| F3 | 同族跨规模 | MA-CT Full/LP1，Omni-7B/3B | 已完成导出迁移；不能称跨家族泛化 |
| **S1** | **为何不用简单readout** | **MA-CT Full/LP1，Omni-7B** | **代码已实现，真实模型尚未运行；唯一默认待跑** |

BZ-12保留在历史结果/材料中，是10秒前缀上限的12组件诊断，不是官方全量时长得分。Watkins的31类诊断不等于完成了31类AKR修复。GLM-4-Voice和Qwen3-Omni当前没有已接通并验证的后端，不加入自动队列。

## S1具体设计

同一支持集、同一query、同一冻结模型，比较：Native generation、Fixed mean KV、Continuous AKR、Ridge直接分类、Probe-to-text（同一预测直接转文字）、Probe-to-LM（将预测作为明确可能出错的外部文字证据，再由原模型听音回答）。

固定：MA-CT六类；每类8条support，共48；配置seed20260914；rank4、ridge penalty10、relative alpha.01；Full与1kHz；query使用所有确定划分成员，不能根据成功/失败裁剪。按canonical manifest得到的样本身份会写入新的STANDALONE_PROTOCOL.json；材料改变后不保证仍是旧97条。

指标：accuracy、macro-F1、invalid、相对native的repaired/harmed；Probe-to-LM另报同意外部分类器的次数、把probe正确改错的次数、把probe错误改对的次数。Ridge-to-text应与Ridge类别准确率严格一致。真query标签只在预测结束后用于评分。

## 模型specification

| | 主模型 | 跨规模参考 |
|---|---|---|
| Hub ID | Qwen/Qwen2.5-Omni-7B | Qwen/Qwen2.5-Omni-3B |
| Revision | ae9e1690543ffd5c0221dc27f79834d0294cba00 | f75b40e3da2003cdd6e1829b1f420ca70797c34e |
| 后端 | Qwen2_5OmniThinkerForConditionalGeneration | 同实现，独立拟合 |
| 使用模式 | Thinker-only，Talker关闭 | 同左 |
| 精度 / batch / decoding | BF16 / 1 / greedy | 同左 |
| 本轮feature index | 24 | 32 |
| 本轮K/V层 | 7,14,21,24 | 9,18,27,31 |

原RQ1使用feature0和raw eta300；这些设置不应偷偷替换成表内固定扩展设置。支持集有监督反向；新query是特征前向加干预前向，无query梯度；不更新backbone。软件历史版本在旧记录中保留，新机器用真实环境指纹登记，不能伪称完全相同。

## 数据准备

代码和manifest已经在仓库；**音频仍需来自原数据源**。没有保存features/gradients没关系，可以重算。没有旧RUN/LOCK也可以启动；运行器会创建本次身份和新划分。没有原始音频则不能跑模型实验。

MA-CT使用 `scripts/prepare_marmaudio_expert_validation.py` 从原Technical_Validation_Data.zip及Annotations.tsv生成；示例：

```bash
python scripts/prepare_marmaudio_expert_validation.py \
  --archive /path/to/Technical_Validation_Data.zip \
  --annotations /path/to/Annotations.tsv \
  --output-dir data/marmaudio/expert_validation \
  --manifest data/manifests/marmaudio_expert_validation.csv
```

这会生成新的本地路径。原始数据请按来源许可获取，不从本仓库获取未经授权镜像。BD-ID/BW-SP的既有materialization与duration脚本保留在REPRODUCE.md及DATASETS.md。

## 新工作目录启动

```bash
git clone https://github.com/Yunbo-max/AKR.git
cd AKR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
# CPU tests / inventory: no model download
PYTHONPATH=src:. python -m pytest -q
python scripts/run_akr_release.py plan
```

GPU运行前先安装适配GPU驱动的PyTorch和torchvision，再 `pip install -r requirements-qwen.txt`。不要把旧README中的某一CUDA wheel强装到不匹配的驱动。实际backward兼容性必须在该机器验证。

```bash
# 只预检真实数据、身份、分组；不加载模型
python scripts/run_akr_release.py check --tasks S1 --run-id akr_s1_v1

# 明确执行；包含预检在内的24小时墙钟上限
python scripts/run_akr_release.py run --tasks S1 --run-id akr_s1_v1 \
  --profile 24gb --budget-hours 24 --execute
```

输出：`results/akr_final/akr_s1_v1/`。断点后使用相同配置、代码、环境和run ID续跑。变更这些条件需新ID；不是通过删除锁来强行复用不匹配缓存。

需要复评已有F1/F2/F3时，显式写 `--tasks F1,F2,F3`；要运行所有新独立入口则 `--tasks S1,F1,F2,F3`。这不承诺四项均在一天完成；超时保留checkpoint，完成率不足不形成最终成绩。旧 `run_akr_final_three.sh` 是依赖旧source run的兼容入口，新机器优先使用本页命令。

## 收尾准则

S1完成后先比较简单方案，不把KV结果一定较好作为完成条件。数据/配置/模型身份不全、raw query journals不全时，记录missing或incomplete，不能填0。不要重复看query挑alpha、rank或seed；不要再拓展环境声音、VLM或新的模型家族来掩盖当前任务上的简单替代。
