> 更新：本文件是2026-10-02的设计规范式历史复核。实际2026-10-02导出技能的本轮检查见 [RESEARCH_AUTOPILOT_AUDIT_20261003.md](RESEARCH_AUTOPILOT_AUDIT_20261003.md)。

# AKR：running-autoresearch 规格式复核

检查日期：2026-10-02。入口 M（已有方法、代码、结果和稿件）；本轮为 audit + implement，不是 GPU run。

## 依据与执行边界

本轮读取的是用户保存的 `2026-09-27-autoresearch-skill-design.md`（设计提案、文件库版本1），不是已安装的 `SKILL.md` 引擎。采用其中已有项目资产审计、最简单替代、主张—证据、回顾性验证和负结果保留要求。没有运行其未获得的验证脚本，也没有伪称独立多代理裁决。复核记录格式是本仓库的报告格式，不冒充该设计规范的完整 schema 实现。

目标仓库是 `Yunbo-max/AKR`。公开旧源 `Yunbo-max/animal-omni-kv@9e8c244b797c24cba1933998bbafbf44a36b0d86` 已导入；当前连接读取 `yuhanlydia/animal` 返回404，因此较新代码从本对话交付的 E1–E9、F1–F3、No-Old-Locks 包恢复。后者不等于验证了私有仓库最新 HEAD。原始音频、基础权重和大梯度数组没有被附到本轮，所以不能说这些资产已迁移。

## 检查结果

| 检查 | 结论 | 依据 / 下一步 |
|---|---|---|
| 问题是否清楚 | 有界问题成立 | 动物声音的冻结表示可分类，原生标签生成不一定有效利用同样监督；不能把全监督 probe 与 zero-shot 的差直接称公平 gap |
| 数据是否充分覆盖问题 | 不急着扩充 | MA-CT、BD-ID、BW-SP 分别是 call type、caller、species；BZ-12 保留历史外部诊断，不在默认实验队列 |
| 数学与实现 | 已有实现可对应 | 负答案梯度、音频位置池化、中心化SVD、support标准化、ridge坐标回归、query加性K/V更新；不是对原始KV做子空间投影，也不是教师策略蒸馏 |
| 主效果 | 保留原协议内证据 | MA-CT五组 mean accuracy 12.35→32.47%，fixed13.01%；主support是20条错误富集录音，不是均衡8/class |
| 条件化控制 | 有局部证据 | BD-ID matched active-block norms比较；完整强度都保留。class-routed和continuous不能互换成绩 |
| 修正可预测 | 已完成的标量记录支持 | F1 rank4 NMSE Full .6813 / LP1 .7641，优于mean1和shuffle；本轮未拿到所有原始梯度数组来重算 |
| 跨频谱 / 跨规模 | 不能概括稳定泛化 | 固定均衡设置中仅7B/LP1有小准确率改善且macro-F1下降；3B无准确率增益；off-diagonal未优于shuffle |
| 最简单替代 | 仍缺直接完整比较 | 同support ridge已很强；新S1并列ridge-to-text、probe-to-LM、native/fixed/AKR，不预设KV必要 |
| 文献碰撞 | 范围有限，未作新颖性通过裁决 | 原始来源初查见 LITERATURE_AUDIT.md；KV steering、模态差距、专门生物声学训练、多频带编码均是相关路线 |
| 执行复现 | CPU测试及静态入口可核对 | 完整模型运行依赖真实数据与CUDA；没有旧RUN/LOCK也能独立启动；启动生成新身份，不伪造旧元数据 |

## 允许和不允许的表述

允许：在明确的错误富集支持集和录音分离评测协议中，AKR改善原模型自己的识别；声音特征可预测支持集留出样本的修正目标；频谱读出迁移和原生识别是不同测量。

不允许：首次发明KV steering；已恢复动物交流语义；普遍恢复推理；优于最简单外部分类器；所有模型/频谱都改善；F1的低误差自动证明下游识别有效；已运行GLM/Qwen3；将已观察数据改称untouched。

## 本轮路由决定

**维持已有方法，不另造新idea；补最简单替代检查和复现入口。**原来的完整正面结果保留其实验身份，扩展的混合/负面结果也完整保留。这里没有足够的当前完整原始数据及独立碰撞裁决来签发正式 full-validation PASS；广泛泛化/优于probe主张应 REVISE，精确原协议主张保留。

### 唯一默认待跑任务：S1

MA-CT、Qwen2.5-Omni-7B、Full与LP1，固定48条均衡support，使用配置生成的全部录音组分离query；同一组样本上的 Native / Fixed / Continuous AKR / Ridge / Probe-to-text / Probe-to-LM。方法配置r4、lambda10、alpha.01，不看query结果调参。输出accuracy、macro-F1、invalid、修复/改错、probe服从率及误改计数。S1只是新实现，当前没有真实模型成绩。

F1、F2、F3不是新的待完成承诺。它们已有已记录结果；仅在需要恢复或另行指定复评时运行。数据来源变化、软件版本变化或配置变化必须用新run ID；不把新运行当作旧结果的位级复现。

### 结束规则

每组输出完整才计算最终表；超时保存逐样本日志，不用partial填0或报胜利。若AKR没有胜过简单读出，保留native-state repair的有界定位；不继续为了证明KV必要而扩大参数搜索。24小时是墙钟上限，不是未知机器的完成时长保证。
