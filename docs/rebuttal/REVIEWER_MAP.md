# AKR 审稿问题—独立实验—条件式回复映射

清单与条件模板日期：2026-11-07；研究定位更新：2026-11-10。
11 月 7 日准备时，本轮科学实验尚未运行；后续已有 [final_v1 汇总快照](../../results/akr_train50/final_v1)。下文保留按准备日期记录的执行边界，实际完成状态和数值须核对相应报告与原始 journal。代码、单元测试、数据集元数据核对和 synthetic/injected-backend 测试都不是识别结果。
以下英文为需按已核验结果填写的模板，不是已提交的 rebuttal，也不预设收益为正。

**研究目标：改善原 LM decoder 生成的答案。** Native、同支持 Audio ICL / probe-to-LM 和可比强度的内部干预是主要对照；LoRA 是参数适配参考。Direct ridge 保留为冻结表示上的外部监督读出诊断，不作为 LM 生成改善成立的必要门槛，也不称为 LM 主干微调。详见 [LM_OUTPUT_SCOPE_ZH.md](LM_OUTPUT_SCOPE_ZH.md)。

## 1. 评审来源和身份边界

目前恢复的是保存的**次级摘要**，不是三份原评审全文：

- `EXPERIMENT_DESIGN_ZH.md`，保存时间 2026-11-07 13:19:05 UTC。
  文件自述依据用户提供的三份评审与讨论邮件；第 9 节把问题归于
  `tbdR`、`h8E8`、`7PoR`。
- `REBUTTAL_PRIORITY_PLAN_ZH.md`，保存时间 2026-11-07 11:07:11 UTC。
  这是相应的补证据建议，亦非原评审文本。

原始评审措辞、完整问题清单、OpenReview forum URL、三份评审出现顺序及评分尚未找回。
**不得把这三个 ID 擅自对应成原始 Reviewer 1/2/3，也不得把本文件的归纳放进引号冒充审稿人原话。**
后续原文若恢复，应核对每条遗漏、措辞和归属；本文件只覆盖目前次级保存材料明确记录的主题。

次级摘要中的已知归属：

| ID | 保存的关切主题 |
|---|---|
| tbdR | 原主表缺 ridge；20 clips 是否只是特选测试；为什么需要 decoder；层与超参数选择不明 |
| h8E8 | MFCC/GeMAPS/BEATs 不足；收益主要依赖 LP1 和错误筛选；其他任务、模型家族与较大数据 |
| 7PoR | AKR–ICL 比较缺失与协议混乱；NatureLM 适用性；BirdSet/真实 drift；可读性、术语与图注 |

当前执行依据是 `akr_final/train50_cases.py`、`akr_final/train50_runner.py`、
`configs/akr_train50.yaml`，以及
[历史补表边界](HISTORICAL_READOUTS.md)、[数据与特征定义](DATA_AND_FEATURES.md)、
[NatureLM 范围](NATURELM_SCOPE.md)。早期保存设计中的每类 k=4、三套支持种子、
MarmAudio、DogSpeak、两折 OOF 选参等建议，**不作为当前 Train-50 主协议**。

## 2. 当前实验和结果读取入口

主协议只包括 Dogs、Watkins、Bats、HumBugDB：从 official train 选 **50 总计**、
分层、outcome-blind 的支持样本，固定种子 `20261107`，评估完整 official test。
当前登记的 test 数量分别为 **139、339、2,000、1,859**；这属于协议目标与来源元数据，
不是本轮已完成的预测条数。验证集标签不用于主方法拟合；不得静默截断 test、删类或挑选 query。
`source_filename` 与 decoded PCM 的去重检查不能证明已恢复独立 parent recording/session。
固定一次支持抽样不能报告为三种子均值/SD，重复同一 query 也不能算新增独立测试样本。

| Case | 当前实际比较或检查 | 主要可读证据 |
|---|---|---|
| C00 | 完整官方分区、Train-50 支持、类覆盖、重复/重叠与身份审计 | `REGISTRATION.json`、`CASE.json`、`STATUS.json`；支持/测试 IDs、class counts、来源与 waveform hashes |
| C01 | native、`akr_regression_original`、`akr_hard_centered`、`akr_soft_cap` | 每方法完整预测、accuracy/macro-F1/invalid、repair/harm、paired 差异 |
| C02 | native、原连续 AKR、soft、`ridge_direct`、`probe_to_text`、top-1 `probe_to_lm`、完整分布 `probe_distribution_to_lm` | ridge 与文字渲染相等；同 RidgeReadout/tau=1 的完整 class probabilities 与同 50-support priors 文本通道；decoder agreement/repair/harm；成本 |
| C03 | native、Omni/MFCC/eGeMAPS/BEATs + 同一 ridge | 特征真实定义/版本/来源、同支持读出指标与预测 |
| C04 | native、原连续 AKR、soft、全 50 条 audio ICL 的三个预定顺序 | 三个完整 ICL journals；上下文是否容纳全部示例；每顺序结果、invalid、成本 |
| C05 | hard/soft × `relative_norm`/`relative_cap`，外加 native、原 AKR | 指标、routing probabilities、planned/effective state norms、BF16 unchanged fraction |
| C06 | soft centered 与 uncentered，加 native | class counts/priors、prototype/map identity、预测与有效更新 |
| C07 | 原/continuous-cap、soft、fixed mean、random、shuffled feature–gradient pair、shuffled prototypes | 映射元数据、flat-field norms、实际每 block 更新与指标 |
| C08 | K-only、V-only、K+V；early/middle/late 单层；audio/text/full-prefill 位置 | 实际层索引、相同层数的比较、intervention telemetry 与指标 |
| C09 | nested 50/100/200 总支持量的原连续 AKR、soft、ridge，加 native | 各预算 IDs/类分布/嵌套性、完整 test 一致性、成本与结果 |
| C10 | 预定 rank/ridge/temperature/raw gain/cap 单因素诊断 | 每个固定候选完整输出；不得据 test 重新选主方法 |
| C11 | Full/LP1 支持拟合交叉到 Full/LP1 query；原连续 AKR 与 soft，加 native | `fit_condition`/query condition 身份、同 query 配对结果；不是自然环境 drift |
| C12 | GLM-4-Voice 的 native、原连续 AKR、hard、soft | 真后端可行性、固定 model revision、层/mask/no-op 与完整 test 结果 |
| C13 | 相同 50 标签的实际 language-decoder LoRA，加 native、原 AKR、soft | adapter 训练 receipt、epochs/LR/exposures、预测、拟合与 query 成本 |
| C14 | native、原 AKR、soft、zero-alpha、zero-field；分项成本 | exact no-op、实际 hooks/update norms；支持 backward、特征、拟合、两遍 query 和模型加载成本 |
| C15 | frozen native NatureLM specialist 与对应 generalist native | 同 registration/query、specialist asset hashes、所有 window 原文与无效项、已知训练来源范围 |
| C16 | 另一路径：已有历史 source registration 的 native/fixed/continuous/ridge/text/probe-to-LM 补表 | 历史 `RUN/LOCK/EPISODE`、S1 完整 IDs/metrics、`ARTIFACTS.json`、raw 历史 parity=false |

C00–C15 在各独立 case 输出目录读取 `CASE.json`、`REGISTRATION.json`、`STATUS.json`、
`REPORT.json`、`RESULTS.csv`，并核对 `trials/<method>/METHOD.json`、`manifest.json`、
`items/*.json`。完整性由不可变 registration 和 journal 重建，不接受只截一行 CSV。
`REPORT.json.methods` 提供 accuracy、macro-F1、invalid、repaired/harmed、paired 统计和成本；
`maps/*.npz` 与对应 cost 文件提供已保存的 correction/probability/拟合证据。
具体字段以实际文件为准，未保存的诊断不能声称已经输出。

引用为科学证据前，必须确认完整方法/完整 test 的身份校验通过、真实后端运行，且没有
`injected_backend` 或 `contract_test_injected` 标记。C00 的 `preflight_passed` 只说明协议检查，
不是科学结果。其他 case 的 complete 也不代表效果为正或必然显著。成本应加上 query 第一遍
特征提取与第二遍生成，不能只引用 generation timing 或以 cache replay 充当实际推断。

## 3. tbdR：逐问题回应

### T1. 原主表缺 ridge；20 clips 与测试样本规模/筛选混淆

- **对应实验：**C00、C16；C01–C02 是另列的新 Train-50 证据。
- **具体读取：**历史每个 draw 的支持和 query IDs、原始分区/选择说明、condition、注入
  scaling、模型/特征层、逐 query 预测；C16 的 source lock hashes、六方法 S1 metrics、
  `historical_raw_eta_parity`。新表同时读 C00 的完整 official-test 登记。
- **可写逻辑：**先澄清单位：保存摘要称旧实验的 **20 是 support 数**，五个 draw 的
  query 数分别为 **39/64/57/68/82**。这些是待原始登记逐一复核的历史数字，不是新结果。
  修订后的表头应逐 draw 分列 n_support、n_query、类分布、support 筛选规则与测试条件。
  然后承认原表缺直接 ridge 的比较，需要同 IDs/同协议补齐，不能用 Train-50 替换。
- **未解决范围：****C16 仅支持已有登记的 `relative_active_state_norm` 历史 source。原
  主表 raw eta=300 协议尚未复原。**没有相容原始 source 时，C16 必须阻断，不能另抽
  20 条来冒充旧实验。即便 C16 成功，也只是 compatible source 上的新 readout extension，
  不能声称恢复了 raw 主表。20-support 澄清也不消除 error-enriched support 的选择问题。

英文条件模板：

> We will distinguish support size from query size in every table. The archived
> summary records 20 support clips and five query populations of 39/64/57/68/82;
> these counts require confirmation against the original episode registrations.
> Our historical supplement can use recovered relative-active-state-norm locks,
> but the original raw-eta=300 main-table registration remains unresolved. Any
> completed Train-50 table will be labeled as a new protocol, rather than a
> replacement for the missing matched comparison in the original table.

### T2. 直接 ridge 已能分类，为什么还需要 decoder/KV 修正？

- **对应实验：**主 C02；机制 C05–C07；效益/代价 C13–C14；可兼容历史源另读 C16。
- **具体读取：**C02 中 ridge_direct、probe_to_text、probe_to_lm、probe_distribution_to_lm、原连续 AKR、soft 的
  同 query 指标；probe agreement、correct_probe_spoiled、incorrect_probe_rescued；
  `cost_components` 与 query_feature_plus_generation_seconds；逐 query 的 external_prediction。
  完整分布方法为 `kind=probability_lm`：检查其确实复用相同 RidgeReadout 和 tau=1，
  全类别 probabilities 与同 50-support 的 class priors 按固定 label 顺序、完整 float JSON
  序列化进入 prompt；保留数值/prompt 身份，确认没有 query 标签进入信息通道。
  C05–C07 读 class route/gradient value/注入幅度控制，不把分类准确率归因于未经验证的机制。
- **可写逻辑：**把目标说清楚：AKR 研究如何利用支持信息改善原 decoder 生成的答案；
  主要比较 native、同支持的生成条件和内部修正对照，记录识别、无效输出、repair/harm 与成本。
  Direct ridge 另列为外部读出诊断；它训练读出参数而不更新 LM 主干，不作为 LM 生成改善的
  必要门槛。对 hard/soft 路由变体，同时报告路由器决策经 LM 生成后的保留、纠正和损坏。
- **实现与未解决范围：**当前 C02 保留 **top-1 ridge 文本证据**，另已实现完整 class
  probabilities/priors 的 `probe_distribution_to_lm`。11 月 7 日准备时尚未运行，后续 C02 汇总
  已在 `final_v1` 上传，需按报告和原始记录解释。这补上同路由
  信息的文本通道，不能称与 K/V 方法获得相同 gradient/prototype 信息；文本通道没有
  类别梯度修正 values。ridge 与 probe_to_text 是相同 class prediction 的表示检查，不能
  当成两项独立强基线。即使完整分布比较完成，也不能建立 decoder 的一般必要性或排除
  所有语言替代方案。C13 的 LoRA 也不是计算量/搜索历史完全匹配的所有 PEFT。

英文条件模板：

> If the complete matched runs are available, we will report direct ridge,
> rendering the same ridge label as text, passing its top-1 label to the frozen
> LM, and passing its full class-probability vector and support priors through a
> fixed text prompt, alongside K/V correction on identical support and query
> identities. The distribution prompt uses the same RidgeReadout and temperature
> of 1, with fixed label order and full-float JSON serialization. This matches
> routing information, not gradient/prototype information. We will report
> when the decoder preserves, damages, or repairs the readout decision, together
> with end-to-end costs. Our objective is to improve answers generated by the
> existing frozen LM. Direct ridge is an external readout diagnostic that fits
> classifier parameters without fine-tuning the LM backbone. The main claims
> concern the observed improvements and mechanisms of LM-generated answers.

### T3. 层、超参数和方法选择依据不清楚

- **对应实验：**C00、C05–C08、C10、C14；支持量 C09。
- **具体读取：**CASE 中有效 method spec、模型 layer count/实际索引、feature layer、
  rank 与 scaling 类型；C08 的同层数 early/middle/late；C10 每个预定候选；planned 与
  effective norms、unchanged fraction、零更新结果。rank 只解释连续 SVD 回归，不用于
  类别 prototype router。
- **可写逻辑：**当前 Train-50 主 soft 设置固定为 tau=1、raw_eta=300、cap=.3、ridge=10；
  原连续 AKR 控制保留其方法身份，以 rank=4、relative_norm alpha=.01 运行当前新协议。
  当前 layer fractions 为 .25/.5/.75/.875，feature fraction=.875；逐模型报告解析后的
  索引，不能把 fraction 直接称作通用层编号。C10 是预定单因素诊断，不在 test 上择优改主表。
- **未解决范围：**这些固定值不是已验证最优值，C10 也不是两折 OOF 选出的结果。
  Train-50 中某类可能只有一条支持，不能虚构所有类均可行的两折 CV。单个 seed 的敏感性
  不证明稳健于任意支持集；固定相对范数会抹平幅度信息，cap 也可能饱和，均须实际测量。

英文条件模板：

> We will state the fixed settings and the resolved layer indices for each model.
> The prespecified sensitivity cases will be reported in full and will not select
> a new main configuration using test outcomes. Equal-count single-layer controls
> address depth separately from the default multilayer configuration. These
> analyses characterize the selected configuration; they do not establish a
> universal optimum.

## 4. h8E8：逐问题回应

### H1. 缺少 MFCC、GeMAPS、BEATs 等声学基线

- **对应实验：**C03；共享身份 C00；与 decoder 的关系 C02。
- **具体读取：**相同 50 支持/完整 test 的 Omni ridge、MFCC ridge、eGeMAPS ridge、BEATs
  ridge；所有提取器 provenance、真实特征名、维数、scaler/读出只用 support 的身份。
  MFCC 当前为 13 静态 + delta + delta-delta 的 mean/std，共 78 维；当前 openSMILE
  实现是 **eGeMAPSv02 88 functionals**，不是 GeMAPSv01b 62 维。
- **可写逻辑：**比较简单 DSP、标准声学 functionals、预训练音频表示与冻结 Speech-LM
  表示，保持当前下游标签预算/查询一致。明确区分外部预训练监督与本任务 50 标签。
- **未解决范围：**不能把 eGeMAPSv02 简写成已经实现/运行了精确 GeMAPSv01b。
  如果恢复的原文明确要求后者，要标为未补或另加真实实现。BEATs 缺官方本地权重时是
  missing，不允许以 MFCC 近似或第三方不可辨识权重顶替。当前也没有 full-train SOTA 结论。

英文条件模板：

> If C03 completes with the registered assets, we will add MFCC, eGeMAPSv02
> functionals, official pretrained BEATs embeddings, and frozen Speech-LM features
> with the same support labels and ridge readout. We will name the exact feature
> variants and distinguish their external pretraining from the 50-label task
> budget. eGeMAPSv02 is not an exact implementation of GeMAPSv01b; any remaining
> requested variant will be identified explicitly.

### H2. 主要收益依赖 LP1 和按错误筛选的 support

- **对应实验：**C00–C01 的 Full/outcome-blind 主评测；C11 的人工带宽控制；原协议补证据 C16。
- **具体读取：**支持选择时是否不读 Full/LP1 模型成败；四任务 Full 主表全部方法的完整
  output；C11 的 2×2 fit/query condition 与同 query IDs；invalid/repair/harm 和置信区间。
- **可写逻辑：**把普通支持 Full 结果列为新协议，旧 error-enriched LP1 作为有明确范围
  的历史发现保留。若新协议无收益/负收益，照实限制结论；不能只展示 LP1 或挑任务。
- **未解决范围：**新 Train-50 **不是同一历史 query/同一预算下** error-enriched vs
  outcome-blind 的成对选择消融，因此不能单独量化旧选择偏差造成了多少收益。
  C11 只有原连续/soft 迁移，没有在此 case 登记 transferred ridge 诊断；不得捏造该结果。
  LP1 是人工信号操作，不能代替自然生态录音 drift。

英文条件模板：

> We will separate the historical error-enriched low-pass protocol from the new
> outcome-blind Train-50 evaluation on Full inputs. All completed task and
> condition results, including neutral or negative changes, will be reported.
> The crossed Full/LP1 experiment tests a controlled bandwidth change; it does
> not by itself establish robustness to naturally occurring recording shifts or
> isolate the magnitude of the historical support-selection effect.

### H3. 任务、模型规模/家族与较大数据上的覆盖不足

- **对应实验：**C01/C03/C04/C13 的四任务；Qwen 3B/7B；C09 nested 支持量；C12 GLM；
  C15 只作 specialist native 参照。
- **具体读取：**四个完整 test 的独立结果/各任务类别定义；同 budget/同 IDs 的 Qwen
  3B/7B；C12 真 GLM 后端的 hook、finite gradients、no-op 和全部查询输出；C09 的
  50⊂100⊂200，及无法满足预算/类覆盖的明确失败。
- **可写逻辑：**区分同家族模型规模证据、GLM 跨家族证据、更多任务/更多 query、更多
  support labels 这四件事。C09 的 100/200 标签不得混入“same-50”主成绩。GLM
  每个模型重新拟合自身 map，不跨模型复制修正向量。
- **未解决范围：**Qwen 3B→7B 不能单独叫跨家族；当前 C12 只有 native/original/hard/soft，
  没有登记 GLM ridge、ICL 或 LoRA，不能声称完整的跨家族基线矩阵。
  一家 GLM 不证明任意架构通用；C15 也不提供 NatureLM + AKR 证据。
  更多固定测试 query 不等同未见物种、未见环境、全新任务泛化；Bats 数字标签不推断语义。

英文条件模板：

> If completed, the four-task tables, Qwen size comparison, nested-support study,
> and GLM case will be reported as distinct forms of evidence. The 100/200-label
> extensions will be separated from the 50-label main protocol. Cross-family
> claims will be limited to the actually validated GLM conditions; neither two
> Qwen sizes nor a native NatureLM baseline establishes general architectural
> transfer of AKR.

## 5. 7PoR：逐问题回应

### P1. AKR 与 audio ICL 缺直接比较、支持预算/协议混乱

- **对应实验：**C00、C04；native/修正接口完整性 C14；历史补表边界 C16。
- **具体读取：**全 50 条真实多音频 ICL prefix 是否完整送入；三个预定 order_seed
  0/1/2 的 method spec 与同 query 预测；实际 input tokens、context-limit 失败、raw
  answers、invalid 和 cost。三个顺序不是三个独立支持抽样或独立 test。
- **可写逻辑：**以相同支持录音/标签/处理条件比较显式 map fitting 与 implicit ICL，明确
  这仍不是 compute-matched 学习过程。固定顺序全部报告，不从 test 选择最优顺序。
- **未解决范围：**若 50 音频不容纳，当前 case 必须明确 blocked/unsupported，不能
  静默截断示例，也不能临时换成 k=2 后声称仍是 same-50。C16 不包含 audio ICL，
  因此不能称已补齐原主表的历史 ICL 条件。

英文条件模板：

> We will compare AKR with audio ICL using the exact registered 50 support clips
> and the same complete query set, reporting all three predetermined support
> orders. If the model cannot accommodate all demonstrations, that condition
> will be reported as unsupported rather than silently reducing the prefix.
> Matching examples and labels does not make explicit fitting and in-context
> inference compute-equivalent.

### P2. NatureLM 专用模型适用性

- **对应实验：**C15；数据/标签语义 C00；边界详见 `NATURELM_SCOPE.md`。
- **具体读取：**相同 registration 下的 generalist native 和 NatureLM native；specialist
  source/checkpoint/base/config hashes 与 revisions、有效生成配置、每个窗口输出/
  invalid、完整 query 覆盖。实际 production loading 不成功时保留原因，不填假成绩。
- **可写逻辑：**这是外部预训练的 frozen **native specialist baseline**。它不拟合当前
  Train-50 支持，必须明确其 zero task-adaptation 与不等量的外部预训练监督。
  Watkins 被官方 card 列入训练数据；精确 clip 重叠未知，不能说完全未见数据。
- **未解决范围：****C15 不是 AKR 跨 NatureLM 适配。**回答“AKR 能否迁移该架构”还需要
  独立 K/V adapter、音频 mask/support gradient/zero-update 核验及真实 native/+AKR 对照。
  Dogs 的任意个体名/ID、Bats 保留的数字标签可能没有 pretrained 语义映射；低分不能
  自动解释为声学能力弱。NatureLM 10s 窗口/尾部与一致标签聚合是显式 adapter 协议，
  不是原官方 leaderboard 结果，也不同于 generalist 的原生长音频接口。
  当前 C15 runner 是同一进程先 generalist、关闭后再加载 NatureLM，未实现隔离子进程。
  官方依赖的 Transformers>=4.44.2、PEFT>=0.11.1 并不声明式排斥当前 Qwen 固定版本，
  但联合环境/实际权重兼容性仍待生产验收，不能把分环境运行建议当成已完成实现。

英文条件模板：

> C15, if completed, will provide a frozen native NatureLM-audio reference on the
> same registered recordings. It will not demonstrate AKR adaptation of NatureLM:
> no NatureLM K/V correction adapter is implemented. We will disclose that Watkins
> is named among the specialist's training sources, that exact clip overlap is
> unaudited, and that arbitrary individual/numeric labels may not have a learned
> semantic mapping. Its external pretraining and zero task-adaptation setting
> will be distinguished from the Train-50 adaptation budget.

### P3. BirdSet、真实声景/自然 drift

- **对应实验：**C11 只能提供有界的人工频谱干预；C00–C15/C16 **没有 BirdSet 实验**。
- **具体读取：**C11 的真实 Full/LP1 定义、filter/采样率、paired IDs 和完整输出；
  对照拟改正文/摘要是否仍含超出证据的 natural drift/所有 bioacoustic tasks 声明。
- **可写逻辑：**收窄结论为当前 single-label tasks 与 controlled low-pass shift。
  如果未运行 BirdSet，不写“覆盖命名 benchmark”；不把 Dogs/Watkins/Bats/HumBugDB
  全量单标签结果或 LP1 迁移描述成真实多标签声景漂移鲁棒性。
- **未解决范围与后续证据要求：**固定真实 soundscape/domain splits、完整官方数据协议、
  标签与重叠审计；建立多标签 answer/loss/aggregation 与匹配 baselines；使用适用的
  mAP/AUROC 等指标并保留原协议含义；冻结训练/开发选择后再评估真实环境迁移。
  不通过挑单标签子集、换 accuracy 或拼接多个受控频段冒充 BirdSet 协议。

英文条件模板：

> We will narrow the claim to the evaluated single-label recognition tasks and
> controlled bandwidth perturbations. Full/LP1 transfer is not evidence of
> robustness to natural soundscape or recording-domain drift. BirdSet has not
> been evaluated in this suite. Addressing it requires the appropriate official
> soundscape splits, multilabel inference and metrics, and matched baselines;
> this remains an explicit evidence gap.

### P4. 可读性、术语、方法定义与图注

- **对应工作：**所有 case 的证据用于改写；C00、C08、C14 最直接支撑定义和协议说明。
  **没有一个运行实验能自动完成写作修订。**
- **具体读取/改写：**对照实际 method spec 和 effective metadata，逐一定义 support/query、
  标签词表、audio-token K/V、pre-RoPE projection、特征池化、梯度符号、中心化对象、
  continuous SVD-ridge 与 class prototypes 的差异。区分外部路由 keys/values 与模型
  内部 K/V，softmax weights 不叫已校准 posterior。把 frozen backbone、support backward、
  CPU map fit、query first feature pass 和 second generation pass 分开解释。
- **图表要求：**图注写明 dataset/task、模型/revision、Full/LP1、n_support/n_query、
  支持抽样/筛选、metric 单位、旧/新协议和方法身份。主文表格显式标 missing/failed。
  一次固定支持抽样不画“跨支持集 SD”。原连续 AKR 的旧数值不得贴给 soft/hard 新名。
- **未解决范围：**最终稿逐页改写、记号检查、caption 检查与 claim–evidence 核对仍待执行；
  本文件不是已完成论文修订记录。原评审全文未恢复，不能声称每条写作意见都已覆盖。

英文条件模板：

> We will revise the method overview, define the intervention states and each
> symbol before use, and distinguish the original continuous regressor from the
> prototype-routing extensions. Each result caption will state task, condition,
> support/query counts and protocol provenance. The final response will give
> concrete page and figure pointers only after the manuscript changes exist.

## 6. 回复与改稿使用规则

先完成 case 的身份/完整性校验，再从真实结果写“观察到什么、它回答了哪一部分、还缺什么”。
不能在模板里先填 improved/generalizes/efficient/robust 等结论。正、零、负结果全部保留；
未完成单元格写 missing/failed 与原因。paired CI 的分组来源只是可用 source metadata，
不能称已证实 parent-recording-disjoint。

每位 reviewer 的最终答复可引用同一证据表，但不重复堆砌全部实验。改稿中将原主结果、
compatible-source 历史 readout extension、Train-50 新评测和新 soft-router 方法分开。
当前最明确的未闭环项是：原主表 raw-eta=300 登记/配对基线、原始三份评审全文/顺序、
NatureLM + AKR 适配、自然 drift/BirdSet，以及真实模型科学结果。完整 probabilities/priors
文本通道已实现但未运行，仍待实际 matched 比较；它不匹配梯度 prototype 信息，也不证明
decoder 一般必要性。任何已实现脚本或通过的 CPU 测试都不能把这些项自动变成已解决。
