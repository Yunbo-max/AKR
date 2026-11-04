# AKR ICLR rebuttal：稳定性修复与完整运行说明

本次是基于 `Yunbo-max/AKR` 的 `95bcf7c99cc011c1d0d852abfa640fe2d01690f1` 的新增代码。原论文、历史结果、原 S1/F1/F2/F3 配置和主回归器不改。没有提供具体 reviewer 原文，因此以下是针对现有方法问题的补证据计划，不冒充 reviewer 的逐条要求。

**状态：代码与 CPU 验收；真实音频/Omni GPU 结果待本地主机运行。**

## 1. 哪些是修复，哪些是新变体

### P0：实际更新量记录（测量修复）

保留旧的 planned norm，新增 `effective_relative_norms`、`effective_delta_frobenius`、`unchanged_fraction`。
实际增量定义为 `updated.float() - original.float()`，而不是只量 `applied`。低精度相加后的小增量可能消失或发生量化。
新日志不等于提高识别分数。旧 `relative_norm` 的正常输入数值行为保持不变。

还增加了实际 prefill hook 覆盖检查、非有限数检查、答案拼接后的上下文长度检查。方向或位置不匹配不能悄悄当作成功注入。

### P1：`relative_cap`（新推断变体）

原模式对每个 active K/V block 强制定长：
`Delta = alpha * ||Z|| * direction / ||direction||`。
因此在归一化之前乘一个正标量，不会降低最终更新量。

新增模式：
`Delta = eta * direction * min(1, alpha * ||Z|| / (eta * ||direction||))`。
它不把较小的预测自动放大到固定范数。零方向严格无操作。
**cap 限制的是舍入前的更新；BF16 相加后的真实比例另行记录，不声称最终量化后严格满足同一上界。**
原模式仍是默认；此变体不回填历史表格。

### P2：支持侧答案损失校准（新方法变体）

保留原 support fit 的 centered SVD、ridge 与 normalizer。预测方向改为：
`g_gamma(x) = gamma_0 * mean_g + B diag(gamma_1,...,gamma_r) a_hat(x)`。
默认 r=4，共5个增益，不训练 backbone，也不替换原生 decoder。

使用**实际 BF16 注入路径**下的答案 CE 做有限坐标搜索，而不是对最终舍入假装存在精确梯度。初始增益1，范围[0,2]，步长0.5/0.25；目标为平均每录音答案CE加 `0.01*mean((gamma-1)^2)`。同时保留所有坐标共享同一增益的 scalar control。

增益只能使用 calibration 标签。每个预先声明的校准方法（scalar、coordinate）只把它最终选定的一个候选交给 guard，与 native 比较一次。校准无收益或 guard 恶化就使用零修正；错误/非有限数据不是正常回退。不得依据 query 的准确率再选 gain、alpha、rank、seed。
这个控制器没有全局优化、逐query正确率保证或多轮稳定性定理。

## 2. 实验矩阵：默认10个完整单元

| Suite | 数据任务 | 模型 | 条件 | Support draws | Query | 作用 |
|---|---|---|---|---:|---|---|
| `core` | MA-CT / MarmAudio 六类 call type | Qwen2.5-Omni-7B | Full / LP1 | 3 | 当前 manifest 的97/56/62条，条件间相同 | 方法、稳定性与消融 |
| `scale` | 相同 MA-CT | Qwen2.5-Omni-3B | Full / LP1 | 首个draw | 与对应7B相同97条 | 同族跨规模，不是跨家族 |
| `dogs` | BD-ID / BEANS Dogs 十个个体 | Qwen2.5-Omni-7B | Full / LP1 | 1 | 官方test全139条 | 另一动物识别任务 |
| `watkins`（可选） | BW-SP / BEANS Watkins 31物种 | Qwen2.5-Omni-7B | Full / LP1 | 1 | 官方test全339条 | 类别数量扩展，覆盖全部31类 |

默认 `core,scale,dogs` 共10个模型×条件×support单元。`--tasks all` 加Watkins，共12个。数量来自当前登记manifest；本地输入发生变化必须新run ID，不能为保持某个计数裁剪query。

Full是模型可见的0–8kHz基带，LP1是原采样率上1kHz低通后抗混叠重采样到16kHz。不做每个条件独立峰值归一化，不恢复输入已经丢失的高频。

### 标签预算与隔离

| 数据任务 | Fit | Calibration | Guard | 总标签预算 |
|---|---:|---:|---:|---:|
| MA-CT | 8/类＝48 | 2/类＝12 | 2/类＝12 | **72** |
| BD-ID | 8/类＝80 | 2/类＝20 | 2/类＝20 | **120** |
| BW-SP | 2/类＝62 | 1/类＝31 | 1/类＝31 | **124** |

主回归器只用fit。P2额外利用calibration/guard。因此比较时同时报告fit-only AKR/ridge，以及可使用**全部标注**的 `akr_all` / `ridge_all` / probe-to-text / probe-to-LM。不能把72-label calibrated结果写成48-label方法无代价增益。

MA-CT按真实recording_id把fit/calibration/guard/query分开；无caller标识，不称unseen-caller。
BEANS保留官方test所有query，只从train抽取三种支持角色。以source_filename分组不等于恢复完整录音身份。
**当前manifest发现Dogs 1条、Watkins 8条train候选与test有同名来源；新协议在分配support前排除这些训练候选，保留所有test。**还检查已声明sha256与实际波形字节。排除原因与IDs写入protocol，不修改旧manifest或旧分数，不伪称原官方完整训练协议复现。

这些数据已在项目中检查过；新划分是重新评估，不称untouched test。重叠draw的均值/样本SD仅描述一致性，不当作多个独立新数据集。

## 3. 对比方法和输出

每个单元13个基础方法：

| 名称 | 输出/意义 | 标注 |
|---|---|---|
| `native` | 原模型生成 | 无任务support |
| `zero` | 预测方向，alpha=0；必须与native完全一致 | no-op检查 |
| `zero_field` | 零方向，alpha>0；必须与native完全一致 | 独立no-op检查 |
| `fixed_relative` | fit均值修正，原相对范数 | fit |
| `akr_relative` | 原continuous AKR，原相对范数 | fit |
| `akr_cap` | 原continuous方向，范数上限 | fit |
| `scalar_cap` | 整个预测方向共用一个校准增益 | fit+cal+guard |
| `gain_cap` | 均值和各低秩坐标分别校准 | fit+cal+guard |
| `akr_all` | 原AKR获得全部标注；不使用增益校准 | 全部 |
| `ridge_fit` | fit特征直接分类 | fit |
| `ridge_all` | 全部支持特征直接分类 | 全部 |
| `probe_to_text` | 同一ridge_all预测直接映射文字，准确率应一致 | 全部 |
| `probe_to_lm` | ridge_all预测作为可能错误的外部证据，保留query音频 | 全部 |

首个MA-CT/7B draw在Full和LP1另做：fixed / predicted / matched random / shuffled feature-gradient pairing，各alpha 0.003/0.01/0.03，以及K-only / V-only cap。前者比较相同计划范数下方向与条件化；后者是不同干预位置消融，不声称K-only与K+V总能量相同。
默认共158个方法单元，包含no-op与相同输出的核对项，不是158项独立科学发现；加Watkins为184个。

指标：accuracy、macro-F1、invalid rate、相对native的repaired/harmed、planned/effective更新比例、未改变元素比例、gain接受/回退原因；Probe-to-LM额外报告服从次数、把正确probe改错和把错误probe改对。
所有负面和零收益都保留。完成与否由预测完整性决定，不要求p值或正收益。

记录每次生成/答案loss调用时间与peak memory，特征/梯度/校准共享成本另列。生成表中的时间**不是**包括首遍特征和支持拟合的完整端到端延迟；缓存复用也不代表内在计算免费。Scalar与五参数校准实际候选数量不同，不能声称计算量完全相同。

## 4. 精确模型设置

| | 7B | 3B |
|---|---|---|
| Hub ID | `Qwen/Qwen2.5-Omni-7B` | `Qwen/Qwen2.5-Omni-3B` |
| Revision | `ae9e1690543ffd5c0221dc27f79834d0294cba00` | `f75b40e3da2003cdd6e1829b1f420ca70797c34e` |
| 当前架构对应feature index | 24 | 32 |
| 当前架构对应K/V层 | 7,14,21,24 | 9,18,27,31 |
| 模式 | Thinker-only，Talker关闭 | 同左 |
| Precision / batch | BF16 / 1 | 同左 |
| Decode | greedy，最多12个答案token | 同左 |

代码从已登记layer fractions生成层索引，实际加载的层与feature index记录于BACKEND文件。新数据/模型revision不要覆盖已有run。
Rank4，ridge penalty10，relative alpha.01；cap raw eta300。旧主结果的feature0、20-total错误富集support、raw eta300不是此协议。
不加入没有已实现后端的Qwen3/GLM，不从3B对比宣称跨家族泛化。

## 5. 数据与环境准备

复用原先正常工作的GPU环境与原始音频，不升级正在运行的实验环境。新环境按 `requirements-cpu.txt` 和 `requirements-qwen.txt` 安装；先安装与实际驱动兼容的PyTorch/torchvision。`80gb`是内存分配配置，不是显存资格测试或速度保证。Transformers仍固定于现有后端版本。

MA-CT的canonical manifest应指向真实文件：

```bash
python scripts/prepare_marmaudio_expert_validation.py \
  --archive /path/to/Technical_Validation_Data.zip \
  --annotations /path/to/Annotations.tsv \
  --output-dir data/marmaudio/expert_validation \
  --manifest data/manifests/marmaudio_expert_validation.csv
```

Dogs/Watkins使用已materialize的protocol manifest：
`data/manifests/beans_dogs_protocol.csv`、`beans_watkins_protocol.csv`。
其下载和duration处理入口保持 `REPRODUCE.md`、`DATASETS.md` 中原说明。没有音频时check会报具体缺项；不会填零、跳过数据集或自动把旧的分数拿来补。

## 6. 一次启动、同命令续跑

```bash
# 在应用补丁后的AKR根目录，已有正确GPU环境中执行。
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q

# 只有元数据计划；不读模型、不下载模型、不需要旧RUN/LOCK。
python scripts/run_akr_rebuttal.py plan --tasks core,scale,dogs

# 完整输入预检；不加载模型。
python scripts/run_akr_rebuttal.py check \
  --tasks core,scale,dogs --run-id iclr_rebuttal_v1 --profile 24gb

# 顺序运行核心、规模、Dogs；需要实际GPU、音频和可用的模型权重。
CUDA_VISIBLE_DEVICES=0 python scripts/run_akr_rebuttal.py run \
  --tasks core,scale,dogs --run-id iclr_rebuttal_v1 \
  --profile 24gb --budget-hours 24 --execute

# 整理完成率和表格；不再加载模型。
python scripts/run_akr_rebuttal.py report --run-id iclr_rebuttal_v1
```

H100 80GB可将profile改为80gb，但必须在首次运行前确定，并用对应新run ID。只跑主实验使用 `--tasks core` 和单独run ID；包含Watkins用 `--tasks all` 与单独run ID。固定同一队列、配置、代码、环境、音频身份后，重复run命令从逐条journal恢复。

24小时是**每次调用的墙钟上限**，不是保证全部完成，也不是全研究累计预算。上限到达会发安全停止，最多90秒退出宽限；被硬终止的单条操作下次重做，已完成条目保留。不要删除锁或改参数强行续旧目录。真实模型首次运行可能仍遇到驱动/显存/依赖问题；CPU测试不认证它们。

## 7. 输出如何用于rebuttal

目录：`results/akr_rebuttal/iclr_rebuttal_v1/`。

- `STUDY.json`：模型、角色ID、输入排除记录和所有计划单元。
- `*/STANDALONE_PROTOCOL.json`、`*/WAVEFORM_HASHES.json`：数据与实际波形身份。
- `*/policies/*/*/{scalar,coordinates}/POLICY.json`：全部候选损失、锁定增益、guard和回退记录。
- `*/trials/.../items/*.json`：每个query的原生输出、增量、成本；标签在生成结束后评分时加入另一个文件。
- `RESULTS.csv`：完整单元的指标；部分单元不混入平均。
- `SUMMARY.csv`：只有预定draw全部完成的cohort才有均值/样本SD。
- `REPORT.md`：可直接检查的结果表。
- `SUPPORT_COSTS.json`：共享前向/支持梯度和答案loss校准调用成本。
- `COVERAGE.json`、`FINAL_STATUS.json`：完整率；出现ERROR时保留原因。

优先整理三张rebuttal表：
1. MA-CT三draw：原方法 vs cap vs scalar/gain，与全部标签预算的AKR/ridge并列。
2. 第一draw的方向/强度/K/V消融和BF16实际施加量。
3. 同设置3B与Dogs，报告增益、误伤和回退；无收益也展示。

历史论文分数保留在原表；新版结果明确注明是新增协议/新变体。Native fallback率高只能说明控制器保守，不能单凭回退宣称方法有效或稳定。要同时看准确率、修复、误伤、注入生效率和简单替代结果。
