# 多次自采样 → MTP OPSD：迁移基线

## 本轮问题与状态

用户选定的是先跑通迁移基线：**同一个已经预训练好 MTP 的模型，自己多采几条，
用自生成参考来做 on-policy self-distillation，再把监督扩展到原生 MTP 头。**
本轮交付状态是 `generated_unexecuted`，没有新的准确率或训练完成记录。

工作分支：`research/mtp-opsd-baseline`。基础模型统一为固定 revision 的
`Qwen/Qwen3.5-4B`。这里的原生 MTP 包含一个附加预测层，即主头 + 下一步条件预测头。
它不是任意长度的并行独立预测器；本轮没有随机添加新的未来预测头。

## 1. 一次参数更新做什么

1. 取一个 GSM8K train 问题。读取后立即移除答案列；训练样本只暴露
   `id / question / source_index`。
2. 当前模型在相同问题上独立采样 G 条 response，G 为 4 或 8。
   一整组生成期间参数保持不变。
3. Math-Verify 解析最后一个完整 boxed/fbox 答案，将双向等价的答案分组。
   长度截断、没有最终答案、未闭合 thinking 均不能成为合格参考或负例。
4. 最大答案组人数 / **原始 G** 不小于 0.5 时，取该组最长 response 为参考。
   并列最大组按固定 RNG 选择。因此 G=4 的 2/2 并列也可能符合阈值；
   这里沿用 0.5 的 modal-consensus 规则，不能把它描述成严格过半。
5. 从有效、答案不同的 response 中选择最长的一条作为学生轨迹。
   默认 `max_targets=1`，使第一次开发运行的每组反传预算有界。
6. 教师与学生使用同一时刻的参数。教师的 system context 额外包含该自生成参考；
   学生只看到题目。之后两者追加**同一条学生 response 的原始 token IDs**。
   教师关闭梯度，学生计算全词表 forward KL，更新 LoRA。
7. 每个合格问题组最多一次 optimizer update。全体有效答案一致、无合格组、
   或上下文长度超限会记录为 skip，不能算成有效训练步。

这是参考上下文提供的条件差异。直接在完全相同参数、相同输入上让模型蒸馏自己，
KL 没有这样的学习信号。多数参考可能错误，教师并不被假定天然更准确。

## 2. NTP 与 MTP 的具体差别

| 项目 | NTP OPSD | MTP OPSD |
|---|---|---|
| 学生组采样 | 每一步使用主头 | 主头采一个 token，原生 MTP 头再采一个 token，循环 |
| 教师主头 | 使用 | 使用 |
| 教师原生 MTP 头 | 不使用 | 使用 |
| 学生原生 MTP 头 | 不使用 | 使用并获得梯度 |
| KL | 主头的全部 response 位置 | 主头 + MTP 头，各自覆盖全部合法 response 位置 |
| 训练后评测 | NTP、MTP 两种解码分别评测 | NTP、MTP 两种解码分别评测 |

MTP 采样**不做 speculative verification**。如果先让主模型校验并纠正每个草稿，
保留下来的输出会重新对应 NTP policy，不能作为这里独立的 MTP rollout 对照。

原生头的位置约定：

- 主头 `head0[t]` 预测输入位置 `t+1`。
- MTP 头 `head1[t]` 使用主干状态 `h[t]` 和 token `x[t+1]` 的 embedding，
  预测输入位置 `t+2`。
- 学生 prompt 长 P、response 长 L 时，主头从 row P−1 监督 L 个位置；
  MTP 头从 row P−1 监督 response[1:] 的 L−1 个位置。
- 教师按其自己的 prompt 长度对齐，避免参考文本长度造成错位。
- KD 使用全部合法前缀，不只使用实际 pair generation 中的奇偶活跃位置。
  这是额外的 all-prefix 辅助监督，不能称为“恰好等于轨迹联合 KL”。

令两头在上述位置得到的平均 forward KL 分别为 K0 与 K1。本轮损失为：

- NTP：L = K0。
- MTP：L = (K0 + K1) / 2；若 response 只有一个 token，则仅 K0。
- 多个 target 启用时再对 target 取平均。默认一个。
- 教师分布 stop-gradient，温度 T=1，完整词表，不使用 top-k 近似目标。

词表 logits 分成 16 个位置一块。冻结共享 LM 输出投影后，先累计每个 hidden
leaf 的梯度，再一次向主干/MTP 传播 VJP，避免为每一词表块重复整个 LM 的反传。
实现仍需真实 GPU 验收；静态公式核对不等于数值运行通过。

## 3. 实际调什么参数

基座和 LM output head 固定；标准 LoRA 公式为 W x + (alpha/r) B A x。
默认 r=16、alpha=32、dropout=0、AdamW learning rate=1e-5。
主干 attention、MLP、线性 attention 投影及 MTP 的对应线性层按
[common.json](../../configs/mtp_opsd/common.json) 的列表挂载 adapter。
保存格式由本基线实现，不能直接当成 PEFT adapter 或 merged HF model。

NTP 控制使用相同的 adapter 配置，其中未参与 forward 的 MTP adapter 没有梯度。
MTP case 必须记录教师与学生分别调用 MTP；不能只给学生添加未来头、
却让教师仍使用主头伪装为 MTP 教师。

首轮最有意义的变量只有 G=4/8 与 NTP/MTP 两种训练方式。
温度、LoRA rank、学习率、阈值、max targets、token budget 先固定。
改变这些值需要新 run ID；改变模型结构/运行配置还需要同 overrides 的新 qualification。

## 4. 与已有工作的关系

本实现参考 [U-OPSD v2](https://arxiv.org/html/2608.06296v2) 的多条自采样、
最大一致答案组、最长一致参考、不同答案轨迹和 forward KL 思路。
截至本轮源码检查，[作者仓库](https://github.com/williamium3000/u-opsd)
没有发布可直接迁移的训练实现，因此这里是**按论文机制实现的开发基线**，
不是作者官方代码的完整复现，也不声称复现其报告分数。

本轮明确的工程/协议选择包括：Qwen3.5-4B 原生条件 MTP、LoRA、
G=4/8、最长一个不同答案 target、512 prompts/128 updates 上限、
non-thinking chat、1536 tokens 和独立全量 native evaluation。
这些选择不能静默与原论文完整训练设置混同。

[OPSD / Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734)
是同模型获得额外条件后指导学生轨迹的相关背景；
本轮使用自生成参考，不将训练集 gold solution 交给教师。

本轮没有实现 block-consensus 新目标、JEPA、显式 future latent、内部更新教师、
外部大模型、搜索 verifier、MTP head 去除后的新蒸馏方法或独立 SFT/LoRA baseline。
LoRA 在这里是优化器作用的参数化方式；NTP/MTP OPSD 都用它，
因此不能宣称本轮已经回答“OPSD 比 LoRA 更好”。

## 5. 什么结果才值得继续

必须先看 Native → NTP OPSD、Native → MTP OPSD，再比较同 G 的两种训练。
对同一个训练 checkpoint，分别看 NTP 与 MTP 解码：如果只在 MTP 解码提高，
与在共同 NTP 解码下也提高，是不同的证据。

同时报告有效 updates、参考/target 合格率、skip 原因、token 数、耗时、
两头 KL 和完整测试分母。G=8 是更多自采样预算，不能仅按准确率宣称更高效。
固定 prompt/update 上限也不保证实际 updates 或 FLOPs 相同；根据真实日志比较成本。

首次运行只采用训练 seed=42、评测 seed=43，每题 n=1。
`avg@1` 与 `pass@1` 数值完全相同，不能当作两项独立收益。
多 seed、n>1、严格等计算预算和第一关键决策机制分析留给有实测后再决定。
首轮正结果也不能直接支持“未来监督必然改善最早决策”或新方法优越性的主张。
