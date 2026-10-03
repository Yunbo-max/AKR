# AKR：research-autopilot 实际导出版本复核

日期：2026-10-03。入口 **M**（已存在方法、代码、论文、结果）；模式为回顾性审计与已授权的代码/材料迁移，不是GPU实验。目标 `Yunbo-max/AKR`，继续现有迁移PR #1，不重复创建另一个迁移分支。

## 使用了什么

实际读取用户2026-10-02导出的 `research-autopilot.zip` 中 `SKILL.md`、intake、state/routing、artifact contracts、host adapters、literature、collision、importance-preserving gate及handoffs参考文件。文件哈希和使用范围在 `SKILL_USE_20261003.json`。这比原 `AUTORESEARCH_CHECK_ZH.md` 所依据的2026-09-27设计提案更完整。它是实际导出内容，不等于证明当前云端安装与该导出逐字相同；未假称调用未暴露的独立写作/绘图子技能。

不倒签Parent Problem、Natural Gate 0、Gate A、独立审稿或host批准记录。原有历史结果先于本次检查。当前只有一个助手的有界分析；不发放正式novelty-clearance、full-validation PASS或投稿保证。

## 核心结论

**保留现有方法与有界主结果；不需要新idea，不扩动物数据集或模型家族。先补“最简单替代”对照S1。**

| 检查项 | 已有证据 | 判定及边界 |
|---|---|---|
| Parent problem | 同一批少量标注使冻结ridge显著强于audio ICL | 自然任务是动物声音识别与标签绑定；不是翻译动物语言 |
| Natural Gate 0 | 有固定数据集自然录音；主要修复针对构造的LP1条件 | 尚无“真实部署失败总体中，强简单替代仍不能解决多少”的合格census；不回顾性宣称Gate0 PASS |
| 方法对象 | support答案loss的负K/V状态梯度，audio pooling，中心化SVD，ridge系数预测，加性state intervention | 回归的是连续修正而非类别；不是把原KV投影掉，也不是重新训练Speech LM |
| 主效果 | MA-CT五录音组draw：native12.35、fixed13.01、AKR32.47%；macro-F1 5.92→24.54 | 支持20-total错误富集support、raw eta300的协议内效果；draws可重叠 |
| 条件化 | Dogs相对范数控制：较大登记alpha下pooled优于fixed/random | 局部方向/条件化证据；保留无增益强度；不可用class-routed成绩替代continuous |
| 可预测性 | 独立分组F1：rank4 NMSE Full .6813、LP1 .7641，比mean1及shuffle低 | 标量导出可读取；本次没有原始大梯度重算；不是下游准确率 |
| 推广 | 固定均衡support的F2/F3仅7B/LP1小幅准确率改善且macro-F1下降，3B无准确率提升 | 不支持稳定跨频谱/跨模型族泛化；完整数值与直接ridge保留 |
| 最强简单替代 | 同support ridge已达到较高分类性能 | 不能把“保留原decoder”本身当作实际用途优势。S1直接测ridge输出文字及probe-to-LM |
| 文献碰撞 | KV steering、专门生物声学训练、频谱编码和audio-ICL后训练均有相邻工作 | 详见LITERATURE_AUDIT.md；部分原文、部分摘要/模型卡，覆盖有限，不作最终novelty判决 |
| 迁移与执行 | 旧公共代码已导入；既有集成分支恢复closing/final；实际CPU基线137通过 | 当前旧yuhanlydia/animal返回404；不能证明其最新HEAD全部等价；不影响已恢复文件的可核验性 |

## 唯一默认待跑：S1

父任务不变：少量标注下的MA-CT识别。输入Full和LP1，冻结Qwen2.5-Omni-7B；固定每类8条support、r4、ridge10、relative alpha .01，不做query调参。支持集选择由固定配置和真实manifest确定；全部confirmation query入评测。

并列 Native、Fixed、Continuous AKR、Ridge direct、Probe-to-text、Probe-to-LM。最后两者分别直接把同一ridge预测输出为文字，或把预测标为可能错误的证据提供给原模型，同时保留query音频。标签只在support拟合与预测结束后评分时使用。

可证伪预期：如果内部修正提供超出简单标签读出的用途，应该在同一任务终点与明确成本下显示收益；若direct/probe-to-text或probe-to-LM匹配或超过AKR，则“KV是必要/更优”的主张不成立，停止以此为目的的调参扩展。不设置“必须找到好结果才完成”。报告accuracy、macro-F1、invalid、repair/harm、服从probe数量、把probe正确改错及错误改对数量和实际调用/用时。

这是本次恢复后明确登记的待跑比较，不是伪造新的未见过数据的确认实验。实现和CPU假后端测试不等于真实S1结果。上限24小时加有界安全终止宽限；超时保留journal，partial不计最终成绩。F1/F2/F3已有完成标量，只按恢复需要显式复评；不自动追加GLM/Qwen3/其他数据集。

## 允许的论文表述

在所报告的错误富集support与录音分离评测下，AKR改善原模型动物声音识别；不同录音的修正可以从冻结声学特征预测；外部readout的频谱迁移与native识别是不同实验终点。

不允许：首创KV steering、已证明动物语义理解、普遍推理恢复、已超过最强简单classifier、各频谱/模型都改善、把原有测试重命名为untouched、把机器单元测试视为科学证据。

## 交付检查

完整代码/资产检查、测试日志及paper编译在 `VERIFICATION_20261003.json`；原始来源与恢复冲突在 `migration/COMPLETION_SOURCE_MAP_20261003.json`。保留旧报告作为历史，不把旧失败/撤回项抹掉。没有运行新模型、重新选择rank/gain、发布模型权重、迁移受限原始录音或公开用户的custom skill实现。
