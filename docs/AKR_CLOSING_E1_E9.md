# 未完成实验与收尾计划（与单版本论文同步）

## 当前证据定位

本文是：**少量同监督的识别差异 + acoustic-to-KV 连续修正回归 + 已完成的部分修复与消融**。不以新tokenwise/factorized方法已经成功为前提。纯动物声音，保留MarmAudio / BEANS Dogs / Watkins / BEANS-Zero。以下是待补的结果，不是本次已运行的实验。

## E1 — 完成K=8候选评分与位置控制

沿用同一75条MarmAudio query和对应48条support，保留sequence-sum与mean-token两种评分，核对event ID、label顺序、prompt版本、音频顺序及完整性。不能将52/75等partial结果算作完整准确率。将first-class固定、轮换第一类、循环反转、保持类频次随机排列分开；比较query-dependent正确率和first/last class agreement。20%且first-copy100%支持输出集中，不单独证明“第一位置导致错误”。

产物：全75预测、output分布、mapping/order版本、配对统计。完成前正文保留已有K1/K2和单独的K8 interleaved结果。

## E2 — 连续回归的锁定确认

主方法必须对应`ConditionalGradientRouter`的SVD+ridge梯度系数回归，不可用类别字典的成绩替换。锁定feature layer、support IDs、rank、ridge penalty、raw eta或relative alpha、prompt和输出解析器。与native、fixed mean、same-support ridge在相同query上比。报告accuracy、macro-F1、invalid和额外前向成本。

MarmAudio已有五split 12.35→32.47，但这是历史20-total-support协议；新确认应分别标明复现与新预注册。Dogs已有多次validation和exploratory test，不可重命名为untouched。建立新的清楚记录的confirmation或重复分组评测；不要用声明替代真实划分历史。

## E3 — “为何不用probe”的直接比较

固定support、query、表征层和标签词表，比较：

| 对照 | 输出由谁决定 | 需要记录 |
|---|---|---|
| Ridge/router直接分类 | 外接读出 | 类别预测、置信度/分数、accuracy |
| Probe-to-text | 将同一预测映射成文本 | 与classifier完全一致的基线 |
| Probe-to-LM | 将预测作为文本证据交给原LM | 是否服从、是否改错、token/latency |
| Continuous AKR | 回归得到的KV增量 | native文本输出及intervention norms |
| Class-routed pooled | 先分类再查类质心修正 | router正确/错误分层、注入后变化 |

如KV没有超出简单读出的任务价值，论文仍可报告native decision repair，但不能说它证明了更一般的推理恢复。Cross-prompt一次paraphrase成功不是这个对照的替代品。

## E4 — 固定标签的声学机制检查

同一support标签y分别使用原音频、静音、其他类别音频；保持长度、prompt、目标词token一致。计算梯度相似度、质心能量及held-out预测能力。为避免目标标签天然带来的聚类，重点分析“同目标不同声音”和class-centroid残差。比较这些target训练出的预测器在query上的修复，而不仅看support标签能否从梯度解码。

梯度依赖J^T(e_y−p)，所以same-label聚类本身不等于模型发现了动物语义空间。Dogs每类只有2个gradient时不要估计有意义的per-class rank。

## E5 — 同范数、同query的关键消融

需要完成、不可仅凭已有12条Oracle替代的deployable对照：K-only、V-only、K+V；audio-only、text-only、full-prefill；early/middle/late layer；real predicted direction、zero、fixed mean、matched random、support-label shuffle。各组报告实际施加的范数、invalid率与对应完整样本数。

空/零方向必须是严格no-op。Query labels只在预测结束后评分，不进入推断路径。Raw eta与relative alpha不要混在同一数值轴。

## E6 — 可选扩展，不作为主方法已经成功

Factorized仅在完成selection后选rank；锁定后confirmation；失败就保留失败，不能反复看test调超参。Watkins至少1/2/4-per-class才能检查全部31类的支持覆盖，旧20-total失败不能推导高类数能力边界。已有tokenwise在某一alpha局部优于permutation并不等于稳定优于pooled。

## E7 — 单独确认第一位置效应

保持support音频和标签集合完全不变，只旋转哪一类位于第一位置，并对所有类别做counterbalance。比较每个query的预测是否随首类变化，同时报告原始accuracy与首类一致率。当前first-copy100%也可能对应固定类别偏好；只有跨排列的干预结果才支持因果位置解释。

## E8 — 强化原生适配基线

LoRA当前one epoch、rank8、q/v，仅能支撑该budget下的比较。增加development-only的学习率/步数选择、训练loss和native accuracy曲线，检查mask、labeltokenization和adapter加载。匹配支持样本，分别报告训练成本。不能把oneepoch负结果扩成“LoRA无法解决”。

Audio-silence实现是conceptual baseline，不等于完整复现audio-specialist-head论文。需引用其正确定位，faithful复现完成前不声称击败那个正式方法。

## E9（可选）— 成本、模型族与匿名发布

如资源允许，在3B复现已锁定的核心修复，不展开另一套方法搜索。记录support预处理/反向时间、两次queryforward、peakmemory和完整生成时间。纯动物scope不强制增加非动物任务。BEANS-Zero维持capped12-component口径，未完成原始完整时长版本前不称官方全量成绩。

## 执行顺序

先E1和E3，其次E2/E4/E5，最后E6/E7/E8；E9为可选扩展。每个任务都有完整率和实验身份；缺数据时输出明确缺项而不是填0。所有已观察过的测试集历史都记录在案。

## 本轮写作没有做的操作

没有启动新的GPU实验，没有改动GitHub，没有证明新的rank/gain，没有上传新权重。本文依据已读取的仓库与报告重新写作；未来结果必须通过完整性检查后再替换正文数字。
