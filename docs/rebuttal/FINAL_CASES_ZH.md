# AKR 最终 rebuttal 实验：独立 case、运行顺序与证据边界

更新日期：2026-11-07。本文对应 `research/rebuttal-train50-cases-20261107` 分支。

**本轮交付是实验代码与运行方案。新 Train-50 实验尚未进行真实模型/GPU 评测，因此没有新增 accuracy、F1 或加速结论。** CPU 测试、合成音频和注入小模型的验证只用于检验实现契约。最终验证见 [VALIDATION_20261107.json](VALIDATION_20261107.json)。

最新的 Train-50 决定替代此前 Cal-40 扩展草稿。历史结果仍按原协议解释。逐位审稿人的对应关系见 [REVIEWER_MAP.md](REVIEWER_MAP.md)；数据和声学特征细节见 [DATA_AND_FEATURES.md](DATA_AND_FEATURES.md)。

## 1. 主实验协议

| 项目 | 固定方案 |
|---|---|
| 数据 | Dogs、Watkins、Bats、HumBugDB 的 BEANS 划分；使用已固定 revision 的 DBD 镜像 |
| 标注预算 | 每个数据集总计 **50 条** official-train support，按类别分层；不是每类 50 条 |
| 主种子 | `20261107`；选择过程不读取模型对错或置信度 |
| 测试 | 保留完整 official test；没有 `max_queries` 或挑选“可修复”测试样本的开关 |
| 主模型 | Qwen2.5-Omni Thinker 7B、3B；Talker 关闭，冻结权重，greedy decoding |
| 其他架构 | C12 单独运行 GLM-4-Voice 9B，使用其兼容环境 |
| 音频条件 | `full` 与 `lp:1000`；同一比较中的 support/query/特征基线使用同一规定的音频条件 |
| 划分复用 | 同 dataset、seed、预算的 case/model/condition 使用相同 ID；C11 只改变 support 的频率条件 |
| 允许的拟合 | 标准化、ridge、梯度原型仅由 support 拟合；验证集不额外提供训练标签，query 标签仅用于评分 |
| 去重 | 将真实 source filename 和规范化解码 PCM 等价关系做传递闭包；移除与 test 重叠的训练候选，不删除 test |
| 完成标准 | 所有规定方法均覆盖全部登记 query，逐条 journal 与身份/哈希检查通过 |

配置中的完整测试规模如下。这里是上游元数据核对和预检要求，**不是本轮已完成全量数据下载或推理的数量**。

| 数据集 | 训练总量 | 验证总量 | 完整测试量 | 主实验使用训练标签 |
|---|---:|---:|---:|---:|
| Dogs | 415 | 139 | 139 | 50 |
| Watkins | 1,017 | 339 | 339 | 50 |
| Bats | 6,000 | 2,000 | 2,000 | 50 |
| HumBugDB | 5,577 | 1,859 | 1,859 | 50 |

来源、revision、原始 shard 数和验证方式均在 [数据说明](DATA_AND_FEATURES.md) 中。预检若无法选满 50 个不重叠的训练组，或无法覆盖每类，就停止并保存失败原因，不自动降低预算或漏掉类别。

`source_filename` 不是已证实的母录音、地点、日期或个体标识，因此不声称更强的 recording/session/individual-disjoint 泛化。`full` 表示统一 16 kHz 模型输入内不加 LP1 滤波，不代表保留原始超声频带。Bats 的整数类别保持为字符串 ID，不凭空添加语义类名。单一主 support 种子的结果不能用 query bootstrap 冒充跨 support 抽样方差。

## 2. 方法如何比较

### 连续 AKR 与新原型方法分别保留

`akr_regression_original` 原样调用仓库原始连续映射：冻结声学特征 → 中心化 SVD/ridge → 连续 K/V 修正方向。当前主对照固定 rank=4、ridge=10、`relative_norm=0.01`。这里的 original 指**连续映射实现**；这一新 Train-50 对照不是历史主表 raw eta=300 的完全复现。历史证据由 C16 单独处理。

`akr_soft_cap` 使用以下路径：

1. 50 条 support 的冻结声学特征拟合 ridge 类别分数。
2. support 的负 answer-loss 梯度按类别求 K/V 原型 `mu_c`。
3. query 类别分数经温度 softmax 得到 `p_c(x)`。
4. 用 support 类别先验 `pi_c = n_c / 50` 中心化：`d(x) = sum_c (p_c(x) - pi_c) * mu_c`。
5. 对选定层、选定位置的 K/V 做有上限的加性干预，再由冻结 Speech LM 生成答案。

中心化是 support-prior centering；只有 support 各类完全均衡时才等于均匀类别中心化。它使 `p(x)=pi` 时方向为零。模型权重不更新；query 不做带真实标签的梯度计算。

默认固定 `tau=1`、`eta=300`、每个 K/V block 的相对范数上限 `rho=0.3`。若 `D_b` 是在有效位置展开的方向，实际缩放系数为 `min(eta, rho * ||H_b|| / ||D_b||)`。因此小方向不会像固定相对范数那样被强制放大到上限；BF16 加法后是否真的改变激活另行记录。该默认值是登记的实验决策，并非已验证的最优点。

`akr_hard_centered` 是 hard routing 对照。C01 的 hard 与 soft 同时具有缩放差别，因此**不能仅凭 C01 的差距声称 soft routing 是唯一原因**；C05 的 hard/soft × fixed-norm/cap 四格对照负责分离两者。

### 超参数不从完整 test 选择

50 总 support 可能出现每类仅 1 条样本。此前“全类两折 support CV”的提议不适用于这种划分；任一类只有 1 条时无法同时覆盖两个训练折。因此主实验直接锁定上面的默认值，不再声称做了全类两折选择。C10 完整报告预设敏感性网格，不回填 test 上最好的设置到主表。C09 的 100/200 support 是独立预算曲线，必须清楚标记超出主实验 50 标签预算。

## 3. 17 个独立入口

所有入口位于 `experiments/rebuttal_cases/`。C00–C15 默认只打印 `plan`；执行需要明确的 `run --execute`。C16 使用历史 runner 的独立命令格式，见后文。

| Case / Python 文件 | 实验内容 | 主要问题与阅读结果 |
|---|---|---|
| C00 `c00_protocol.py` | Train-50 与完整 test 登记、source/PCM 重叠检查 | support/test 是否混淆，是否依赖选错样本；看 ID、每类支持量、排除记录、完整测试数 |
| C01 `c01_main.py` | Native、原始连续映射、hard-centered、soft-cap | 主结果；看全量 accuracy、macro-F1、invalid、相对 native 的修复和损害 |
| C02 `c02_ridge_decoder.py` | Native、连续/soft AKR、direct ridge、probe-to-text、top-1 probe-to-LM、全概率/先验 probe-to-LM | 同监督 ridge 是否足够、decoder 有何额外作用；看两种文本通道的改错/救回与成本 |
| C03 `c03_acoustic_features.py` | Native、Omni feature ridge、MFCC ridge、eGeMAPS ridge、BEATs ridge | 传统声学与冻结自监督特征基线；全部用相同 50 support 和 ridge 读出 |
| C04 `c04_audio_icl.py` | Native、连续/soft AKR、50-example audio ICL 的 3 个预设顺序 | AKR vs ICL；所有 50 条均进入 ICL，不因上下文限制悄悄缩减样本 |
| C05 `c05_routing_scale.py` | Hard/soft × relative_norm/relative_cap，另含 native/连续对照 | 路由和缩放的贡献能否分开 |
| C06 `c06_centering.py` | Centered 与 uncentered soft-cap | support-prior 中心化是否必要 |
| C07 `c07_direction_controls.py` | Fixed、random、shuffled-pair、shuffled-prototypes；matched continuous-cap | 是否只是随机扰动或平均梯度；打乱配对与未打乱对照使用相同 cap 协议 |
| C08 `c08_kv_layers.py` | K-only、V-only；等数量单层 early/middle/late；audio/text/full-prefill 位置 | 层、K/V 和干预位置的作用；多层默认与单层结果另列 |
| C09 `c09_support_size.py` | 嵌套 50/100/200 support 上的连续、soft、ridge | 支持量/标注成本曲线；原连续对照保持其自身缩放参数 |
| C10 `c10_sensitivity.py` | 连续 rank/ridge；soft tau/eta/cap 的预定网格 | 超参数透明度与稳定性；展示整组，不选择 test 最优 |
| C11 `c11_frequency_transfer.py` | Full/LP1 support × Full/LP1 query，连续与 soft | 收益是否只在 LP1；是真正跨音频条件还是匹配条件效果 |
| C12 `c12_glm_architecture.py` | GLM-4-Voice 9B 上 native/连续/hard/soft | 不同模型家族；需生产权重验收和完整结果 |
| C13 `c13_matched_lora.py` | Same-50 language-decoder LoRA 与 native/连续/soft | 同标注预算的参数更新对照；rank8、lr5e-4、5 epochs，单列训练成本 |
| C14 `c14_cost_noop.py` | Native/连续/soft、alpha=0、zero-field；组件成本 | 无干预精确一致性、实际 BF16 更新、耗时/调用/内存证据 |
| C15 `c15_naturelm.py` | Frozen NatureLM native specialist 与 native generalist | 专家模型适用范围；不是 NatureLM+AKR，也不是同预训练监督比较 |
| C16 `c16_historical_readouts.py` | 从真实原始登记读取 S1 六种 readout 对照 | 原主表缺 ridge/decoder 对照；严格复用历史 support/query，不能用 Train-50 顶替 |

C10 的网格是：连续 rank `{1,4,16}`、ridge `{1,10,100,3000}`；soft temperature `{0.25,1,4}`、eta `{30,300,3000}`、cap `{0.03,0.1,0.3}`。这是逐因素敏感性而非这些轴的全笛卡尔积。

C02 新增 `probe_distribution_to_lm`：把与 soft router 相同的完整类别概率和 50-support 类别先验，按固定标签顺序用完整 float JSON 送入冻结 LM，明确它是未经校准的外部证据。它补齐 top-1 文本对照丢失非最大类别分数的信息差，但不传递梯度原型，也不因此证明 decoder 的一般必要性。

C07 random 是固定种子的随机全局方向，按预测方向的扁平向量范数匹配；不能把它写成每个 K/V block 都严格范数匹配。C08 的早/中/晚比较均为单层；与默认四层的差异同时包含层数量因素。

## 4. 建议固定的执行矩阵

先用完整的主对照回答有效性和公平性，再做机制。以下是本轮建议送审的矩阵；同一 case 的额外模型/数据入口已经支持，但不用默认把所有消融扩成相同规模。

| 顺序 | Cases | 预先规定的矩阵 | 用途 |
|---|---|---|---|
| P0 | C00 | 四个数据集，各登记一次 | 在加载模型前确认数据可用、50 support 和完整 test |
| P1 | C01 | 四数据集 × Qwen 7B/3B × Full/LP1 | 主结果与规模比较，16 个独立任务 |
| P1 | C02、C03、C04 | 各跑四数据集 × Qwen 7B × Full/LP1 | ridge/decoder、声学特征、同 support ICL；每 case 8 个任务 |
| P1 | C14 | Dogs/Watkins × Qwen 7B/3B × Full/LP1 | 8 个完整 no-op/cost 任务；其他 case 也持续记录成本和实际更新 |
| P2 | C05、C06、C07、C08、C10 | 各跑 Dogs/Watkins × Qwen 7B × Full/LP1 | 个体分类与物种分类两种任务上的机制/敏感性；每 case 4 个任务 |
| P2 | C09、C11、C13 | 各跑四数据集 × Qwen 7B × Full/LP1 | 支持量、条件迁移、同监督 LoRA；每 case 8 个任务 |
| P2 | C12 | 四数据集 × GLM 9B × Full/LP1 | 8 个跨家族任务；先解决生产环境可执行性 |
| 条件项 | C15 | 四数据集 × Qwen 7B native / NatureLM native × Full | 4 个任务；需资产、运行环境和标签适用性核验 |
| 条件项 | C16 | 找回的原始登记指定的全部 episodes | 历史对照；数量由真实登记确定，不预造支持/测试 ID |

这对应 C00–C15 共 **108 个独立任务**（包含 4 个无模型的数据审计），再加 C16。它是建议的完整运行清单，**不是已经启动的队列，也不承诺某个时长内完成**。ICL 可能受到上下文/显存限制，NatureLM/GLM 尚未进行生产权重验收；失败必须作为资源/兼容性结果报告，不能改成较小预算的同名结果。机制矩阵预先选择 Dogs 与 Watkins，不按新测试成绩选择最有利的数据集。

## 5. 运行命令

### 准备数据与环境

在已安装 GPU 驱动兼容的 PyTorch/torchvision 环境中安装主环境依赖；不要用 CPU 测试环境覆盖现有 CUDA 环境。

```bash
python -m pip install -r requirements-qwen.txt
python -m pip install -r requirements-train50-features.txt

python scripts/prepare_train50_data.py --dataset dogs
python scripts/prepare_train50_data.py --dataset watkins
python scripts/prepare_train50_data.py --dataset bats
python scripts/prepare_train50_data.py --dataset humbugdb
```

数据准备写入新的 `data/manifests/train50/` 和 `data/beans/train50/`。已有 HF snapshot 可用 `--snapshot-dir` 或 `--local-files-only`；完整规则在数据说明中。它不会准备 BEATs/NatureLM 模型权重。

### 每个 case 单独启动

以下以 Dogs/7B/Full 为例；不同 case 的脚本不互相启动。`plan` 不读取音频、不加载模型。`check` 会读取实际音频并登记/核对身份，但不加载大型模型。

```bash
python experiments/rebuttal_cases/c00_protocol.py check \
  --dataset dogs --model qwen7b --condition full --run-id final_v1

python experiments/rebuttal_cases/c01_main.py plan \
  --dataset dogs --model qwen7b --condition full --run-id final_v1

python experiments/rebuttal_cases/c01_main.py check \
  --dataset dogs --model qwen7b --condition full --run-id final_v1 --profile 48gb

python experiments/rebuttal_cases/c01_main.py run --execute \
  --dataset dogs --model qwen7b --condition full --run-id final_v1 --profile 48gb

python experiments/rebuttal_cases/c02_ridge_decoder.py run --execute \
  --dataset dogs --model qwen7b --condition full --run-id final_v1 --profile 48gb

python experiments/rebuttal_cases/c01_main.py report \
  --dataset dogs --model qwen7b --condition full --run-id final_v1
```

7B→3B 改 `--model qwen3b`；LP1 改 `--condition lp:1000`。其余数据集用 `watkins`、`bats`、`humbugdb`。`--profile` 是模型加载的显存预算，不是已验证能装下所有上下文的承诺。主上下文上限 32768 tokens，输出上限 32 tokens；超限不截掉 support 或 query。

同一 run 的 `check` 和 `run` 须保持 profile、config、assets 等设置一致；改变已登记输入/代码/依赖/参数后换新的 `--run-id`。在同一配置下重发相同 `run` 命令即可从逐条 journal 续跑。`run --cache-only` 只允许复用已有数组与预测，不允许补载模型推理。

### 可选资产与独立环境

C03 要提供真实预训练 BEATs 源码和权重，并安装真实 openSMILE。保存本机路径与哈希为 JSON，例如顶层 `{"beats": {...}}`；字段在 [DATA_AND_FEATURES.md](DATA_AND_FEATURES.md)。然后：

```bash
python experiments/rebuttal_cases/c03_acoustic_features.py check \
  --dataset dogs --model qwen7b --condition full --run-id final_v1 \
  --profile 48gb --assets /absolute/path/to/assets.json
```

`check`/`run` 使用同一 assets 文件。缺 BEATs/openSMILE 会明确失败，不以 MFCC 或 Omni 特征替代。BEATs 的 checkpoint SHA-256 必须填真实值。

C12 使用已经准备好的 GLM 解释器，不能直接在 Qwen 的 Transformers 版本中运行。已有可迁移设置脚本见 `scripts/setup_glm_env.sh`，需按本机位置设置 `AKR_CORE_ENV`/`AKR_GLM_ENV`，详见 [既有多模型环境说明](../COMPREHENSIVE_RUN.md)。命令中的解释器路径由实际环境决定：

```bash
/absolute/path/to/glm-env/bin/python experiments/rebuttal_cases/c12_glm_architecture.py run --execute \
  --dataset watkins --model glm4voice9b --condition full --run-id final_v1 --profile 48gb
```

C15 需要本地 NatureLM 源码、NatureLM/Llama 权重、tokenizer、BERT config、inference YAML 和真实 revision，放在 assets JSON 的 `naturelm` 字段。它的窗口协议、预训练重叠、标签适用性与运行环境限制见 [NATURELM_SCOPE.md](NATURELM_SCOPE.md)。这些资产未在本轮下载。

C16 单独读取真实历史源：

```bash
python experiments/rebuttal_cases/c16_historical_readouts.py check \
  --source-run results/akr_closing/ORIGINAL_RUN \
  --config configs/akr_final_three.yaml --run-id historical_readouts_v1

python experiments/rebuttal_cases/c16_historical_readouts.py run \
  --source-run results/akr_closing/ORIGINAL_RUN \
  --config configs/akr_final_three.yaml --run-id historical_readouts_v1
```

上面的 `ORIGINAL_RUN` 是需要替换的实际登记位置。当前仅支持已登记 `relative_active_state_norm` 的 source schema；历史 raw eta=300 协议仍待找回与核对，不能把此命令成功写成原主表完全重现。详见 [HISTORICAL_READOUTS.md](HISTORICAL_READOUTS.md)。

## 6. 每个 case 保存什么

C00–C15 输出在：

```text
results/akr_train50/<run-id>/<case>/<dataset>__<model>__<condition>__s<seed>/
```

例如 C01 Dogs/7B/LP1 对应 `results/akr_train50/final_v1/c01/dogs__qwen7b__lp_1000__s20261107/`。不同 case/model/dataset/condition/seed 不覆盖彼此，也不依赖私有历史 cache。不同进程不能同时写入同一 case 目录。

| 文件/目录 | 用途 |
|---|---|
| `REGISTRATION.json` | 全部行/音频哈希、support/query ID、每类数量、完整测试计数、代码与来源身份 |
| `CASE.json` | 固定方法、超参数、模型配置、资产哈希、环境、注入测试后端标识 |
| `STATUS.json` | 预检/运行/失败/完成；任何失败先撤销旧的完成状态 |
| `trials/<method>/METHOD.json` | 每种方法的具体参数、拟合/测试身份、映射与训练来源 |
| `trials/<method>/items/` | 每条预测、原始输出、成本与干预信息；用于断点续跑和复查 |
| `worker/cache/` | 本 case 的声学特征、仅 support 梯度及校验记录 |
| `maps/` | query 方向/类别概率（适用时）、拟合及路由成本记录 |
| `training/` | C13 的同监督 LoRA 训练状态、适配器和身份记录 |
| `REPORT.json`、`RESULTS.csv` | 从完整登记与合法 journal 重算的结果；不信任旧汇总自报的样本数 |
| `CACHED_VALIDATION.json` | 写入最终成功状态前的完整性/报告哈希回执 |

C16 单独保存至 `results/akr_historical_readouts/<run-id>/`，不写入历史源目录。

## 7. 最终表格与判断规则

主表至少报告 accuracy、macro-F1、invalid rate、测试 N、support N。Invalid 算错，保留原始输出。对 native 记录 repaired、harmed、净增益；对 probe-to-LM 记录“正确 probe 被改错”和“错误 probe 被救回”。同一 query/group 的配对 bootstrap 与辅助 McNemar 分开解释，不能当作独立数据集的重复实验。

成本表将 support 特征、support backward、映射拟合/路由、query 特征、generation、LoRA 训练和模型加载分开。`query_feature_plus_generation_seconds` 含 query 特征与生成；拟合/全 query 路由成本单列，不冒称纯在线延迟。首个完成计算的成本保留；cache-only 续跑不会被写成“零计算方法”。磁盘 cache I/O 不在模型组件成本中。NatureLM 的调用单位是 **每个音频窗口的一次 generate**，不能与 Qwen decoder forward 数直接作等量比较。

C14 要求 alpha=0 和 zero-field 的规范化预测、raw answer 与生成 token 记录都与 native 一致。所有干预方法记录 planned/effective 相对范数与 unchanged fraction；“方向非零”本身不是 BF16 激活真的发生改变的证据。

若 ridge 持平或更强，就如实回答直接分类的作用，不能预先声称 decoder 必要。若只 LP1 有收益，就把结论限制在该条件。若 GLM 或某数据集不提升，仍保留结果。若 ICL 因全部 50 条超过上下文而失败，报告完整预算的可行性失败，另立小预算 case 才能比较小预算性能。

## 8. 尚不能写成“已回答”的内容

1. **新科学结果尚未产生。** 实现完成只代表可安排执行；GPU 内存、模型兼容性和真实收益仍待验证。
2. **原始审稿全文/顺序未恢复。** 当前按保存文件中的 `tbdR`、`h8E8`、`7PoR` 主题映射；不是捏造的 Reviewer 1/2/3 编号或原文逐句引用。
3. **原主表历史 raw-eta 登记未恢复。** C16 有独立代码和严格校验，但缺真实源不能填历史 ridge 单元格，Train-50 表也不能代替它。
4. **真实场景漂移/BirdSet 未得到实证回答。** C11 只是受控频率条件迁移。当前四数据集主协议不支持把 BirdSet 的其他标签/划分语义直接塞入同名 case。若继续保留真实漂移主张，需要单独固定地点/设备/时间分组、源域 support、完整目标域 test、标签对应与适合任务的指标；只有具备这些来源后才能编写和登记对应实证 case。此次回复应收窄到已测协议，不把这个缺口标成已解决。
5. **NatureLM 原生基线不等于 AKR 迁移。** 官方模型卡列出 Watkins 为预训练来源；确切 clip 重叠未审计。Dogs 的个体 ID 也不能仅靠预训练物种知识解释。若要主张 AKR 跨 NatureLM，需要另外实现并验收该模型的 K/V adapter 与同 support native/+AKR 对照。

这五项在 reviewer map 中保留为待结果、待来源或需收窄主张的条目。它们不会因 CPU 测试通过而自动变成论文证据。
