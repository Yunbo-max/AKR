# AKR 绘图 Prompt 定稿包

本包替换先前过长且包含多套备选布局的 prompt 文件。现有 15 张图和 F1/F2/F3 三份模板均已改成单个可直接复制的英文版本，不需要排列组合。

## 使用

直接复制 `prompts/` 或 `new_experiments/` 对应 `.md` / `.txt` 文件的全部文字即可。两种扩展名的内容完全相同。每份的八个小节是给绘图工具的指令，不是要印在图片里的文字。

`FIGURE_PROMPTS_ALL.md` 和 `.txt` 是合并阅读稿；复制其中单张图的代码块内容，不要整份投入一次单图生成。

长度按英文字符计数，包含空格、Markdown 小标题和 LF 换行；不是 4,500–5,000 个英文单词。每一份独立 prompt 实测均在 4,500–5,000 字符内。

## 构图与文字

方法主流程图为 16:9，清楚区分 support 拟合与 query 推断。单幅矩阵、诊断平面和敏感性分析为 1:1；并排曲线与横向对比为 16:9。每张已经确定一个画幅，没有要求你自行选择 A/B/C。

白底、克制配色、矢量线条和统一字体层级；不加总标题或大段图中文字。只保留模块名、坐标轴、图例、必要数学与少量关键数值；长说明置于外部图注。英文 prompt 详细不等于最终图片文字繁多。

## 数据与结果身份

`data/`、`additional_analysis/`、`templates/` 保持旧交付包的文件内容不变。数值图需同时提供对应源数据，使用绘图代码精确排点和标值。不要让图片模型凭想象补点、误差条或热图数值。

F1/F2/F3 仍是等待完整指标的绘图模板；null/空单元不是 0，也不预示性能应当上升。F2 保持可选。本文写作、模型实验、GitHub 代码均未被本次 prompt 改写任务更动。

现有主表继续使用原生 LaTeX，不要绘制成图片。涉及新增模型的空格不属于已有实验结果。

## 清单

| 图 | 内容 | 画幅 | 字符数 | 文件 |
|---|---|---|---:|---|
| 01 | 任务与同监督动机 | 16:9 | 4857 | [prompts/01_motivation.md](prompts/01_motivation.md) |
| 02 | AKR 方法主图 | 16:9 | 4776 | [prompts/02_method_pipeline.md](prompts/02_method_pipeline.md) |
| 03 | 三任务频谱读出分析 | 16:9 | 4852 | [prompts/03_spectral_readout.md](prompts/03_spectral_readout.md) |
| 04 | 连续 AKR 五划分结果 | 16:9 | 4929 | [prompts/04_continuous_AKR_results.md](prompts/04_continuous_AKR_results.md) |
| 05 | 梯度方向与读出漂移 | 1:1 | 4738 | [prompts/05_gradient_rotation.md](prompts/05_gradient_rotation.md) |
| 06 | 双向频谱迁移 | 16:9 | 4845 | [prompts/06_bidirectional_transfer.md](prompts/06_bidirectional_transfer.md) |
| 07 | Dogs 频谱迁移差距矩阵 | 1:1 | 4891 | [prompts/07_dogs_transfer_matrix.md](prompts/07_dogs_transfer_matrix.md) |
| 08 | Watkins 频谱迁移差距矩阵 | 1:1 | 4887 | [prompts/08_watkins_transfer_matrix.md](prompts/08_watkins_transfer_matrix.md) |
| 09 | 读出变化与迁移差距平面 | 1:1 | 4914 | [prompts/09_readout_difference_plane.md](prompts/09_readout_difference_plane.md) |
| 10 | 跨 prompt 修复与改错 | 16:9 | 4948 | [prompts/10_prompt_repairs_and_harms.md](prompts/10_prompt_repairs_and_harms.md) |
| 11 | 完整首划分类别混淆变化 | 1:1 | 4960 | [prompts/11_class_confusion_change.md](prompts/11_class_confusion_change.md) |
| 12 | 输出覆盖与识别质量 | 1:1 | 4925 | [prompts/12_label_coverage.md](prompts/12_label_coverage.md) |
| 13 | 模型规模与频谱交叉 | 1:1 | 4864 | [prompts/13_model_size_bandwidth.md](prompts/13_model_size_bandwidth.md) |
| 14 | 回归秩与特征层交互 | 1:1 | 4898 | [prompts/14_rank_feature_depth.md](prompts/14_rank_feature_depth.md) |
| 15 | 修正强度与输出稳定性 | 1:1 | 4900 | [prompts/15_strength_stability.md](prompts/15_strength_stability.md) |
| F1 | 可压缩性与可预测性 | 16:9 | 4797 | [new_experiments/F1_prompt.md](new_experiments/F1_prompt.md) |
| F2 | 连续 AKR 跨频谱迁移 | 1:1 | 4757 | [new_experiments/F2_prompt.md](new_experiments/F2_prompt.md) |
| F3 | 7B 与 3B 规模比较 | 16:9 | 4826 | [new_experiments/F3_prompt.md](new_experiments/F3_prompt.md) |

## 检查

运行 `python verify_prompts.py` 可重新检查字符数、英文内容、八段结构、文件一致性与所需数据是否存在。`VERIFICATION.json` 保存本次实际校验。
