# Web → Local：MTP OPSD baseline

## 用户已确认的范围

“自己多采几条，再做 MTP 版 OPSD 的 baseline，先跑了算是迁移过来的；
更新到 AKR，换一个 GitHub 分支。”

本次按该明确选择实现已有思路的迁移基线，不启动新方法候选筛选。
仓库 `Yunbo-max/AKR`；分支 `research/mtp-opsd-baseline`；
创建时 main 基点 `bbe2a445d0b63294cf595c92f8de3d09637f5d7f`。
用户明确要求分支交付，因此使用独立分支，无需合并 main。

## 交付物与证据状态

| 项目 | 内容 / 状态 |
|---|---|
| 原生模型接口 | Qwen3.5-4B 文本主干 + 实际预训练 MTP 参数；已生成，待 GPU 验收 |
| 多条自采样 | G=4/8，majority reference、最长 disagreeing target、全词表 FKL；已生成 |
| 两种训练 | NTP 与主头+MTP 头监督，教师学生均用各自实际头；已生成 |
| 启动与保存 | 六独立配置/脚本、checkpoint、恢复、原始日志、汇总；已生成 |
| 评测 | 固定 harness 原生完整 GSM8K/MATH-500，trained checkpoint 双解码；已生成 |
| 源码审查 | 模块 owners + 交叉阅读，位置对齐/原生 MTP/接口/恢复身份已核对 |
| 静态验证 | 见 STATIC_REVIEW.json；只解析源码/JSON/shell，不导入项目 |
| 安装与下载 | 未执行；Local 显式 download/check |
| CUDA qualification | 未执行；Local qualify 为必要入口 |
| 训练与准确率 | 未执行，无新分数、无完成训练记录 |
| 资源身份 | 未提供/未确认 SSH host、GPU 和缓存；Local 记录实际情况 |

**不要把历史 README 的声学 AKR CPU tests、旧结果或 release checks 当作本分支验收。**

## Local 入口

完整步骤、下载清单、逐 case 命令、恢复和输出路径：
[LOCAL_AGENT_RUNBOOK.md](../../LOCAL_AGENT_RUNBOOK.md)。

计划的 bounded development run：

- 一个固定官方 pretrained MTP checkpoint。
- GSM8K train 问题洗牌后最多 512 个；不暴露 gold answer 给训练方法。
- 四个 trained cases，各最多 128 updates、累计 24 小时训练预算。
- 首轮训练 seed=42，评测 seed=43，评测每题 n=1。
- GSM8K test 1,319 + MATH-500 test 500，全量原生任务。
- 两个 native controls，四个 trained cases 各用 NTP/MTP 两种解码评测。
- 训练预算到达会正常落盘停止，不会自行扩大到完整收敛训练。
- 原始大文件留在执行目录；没有 Hugging Face 上传目的地或对外模型发布动作。

## 已处理的实质性实现问题

标准 Transformers loader 不直接提供本轮所需的原生 MTP 控制流程，
因此明确加载全部 15 个官方 MTP tensors，并实现条件头的真实 KV cache。
不允许遗漏 MTP 后静默跑 NTP，不允许随机初始化未来头。

对齐依据是主头 row t → token t+1，MTP row t 与下一 token embedding → token t+2。
teacher reference 会改变 prompt 长度，教师/学生必须分别对齐 response 位置。
原始学生 token IDs 在两条上下文下直接追加，避免重新分词改变轨迹。

大词表 KL 采用 hidden-state 梯度累计后一次 LM VJP，避免逐块重复主干 backward。
仍要求真实 GPU 检查：源码审查不是数值验证，也未测吞吐。

断点验证覆盖 source/config/prompt order/adapter/optimizer/RNG。
`run --resume` 只在 completed summary、current status、latest checkpoint 三者一致时
跳过训练。预算 ledger 包含失败和重放开销，不通过 resume 重置训练时间。

## 方法与来源边界

[方法与已有工作的关系](METHOD.md)明确说明：这是论文机制启发的迁移实现，
不是 U-OPSD 官方训练代码复现，不声明新方法，也没有未经实验支持的性能结论。

源码契约：

- [Transformers 5.16.1](https://github.com/huggingface/transformers/tree/93c8b7b485963a10800c91f55304db6be211c2bd)
  的 Qwen3.5 decoder/attention 和 checkpoint conversion。
- [VeOmni 固定版本](https://github.com/ByteDance-Seed/VeOmni/tree/95f93e35f6ca5fae0b2994203a86b15b73e6a66b)
  的 Qwen3.5 MTP shift、normalization、native 参数映射。
- [vLLM 固定版本](https://github.com/vllm-project/vllm/tree/92f3ee5ae4abb86f2b175b13e623dab4a21b0986/vllm/model_executor/models)
  的 Qwen3.5 MTP 架构契约。
- [lm-evaluation-harness 固定版本](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520)
  的原生 tasks、filters、scorers、result schemas。

## 完成判定

代码送达 GitHub 不代表实验完成。Local 交付需要保留真实 qualification，
四个训练状态（含零更新或预算停止）、六个 case 的完整评测状态和汇总。
只有真实结果才能回答 MTP OPSD 是否超过同 G 的 NTP OPSD、更多自采样是否值得、
以及是否改善共同 NTP 解码下的模型能力。
