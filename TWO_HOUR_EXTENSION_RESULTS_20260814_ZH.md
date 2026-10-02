# 两小时高价值实验扩展结果（2026-08-14）

## 结论摘要

本轮最可靠的新证据不是单纯增加一个 benchmark 分数，而是补强了三段论文因果链：

1. **Same-support 下 representation 学得很快，native decision 学得很慢。** MarmAudio 的 ridge probe 从 K=1/class 的 46.67% 上升到 K=8/class 的 84.00%，K=16/class 为 82.67%，表明约 K=8 已饱和；相同 support 的 audio ICL 在 K=1、2 时仅为 8.00%、17.33%。严格的 Support-to-Decision Gap 分别为 38.67pp（95% bootstrap CI 25.33–52.00，McNemar p=1.08e-6）和 41.33pp（28.00–54.67，p=1.23e-7）。
2. **低通 shift 后大量类别信息仍可解码，而 native generation 长期停留在地板。** Dogs 六个频率条件的 condition-specific probe 为 87.77%–94.96%，native 均为 2.88%；Watkins probe 为 74.04%–88.20%，native 仅约 5.60%–6.49%。这排除了“低通后模型完全听不见”这一简单解释。
3. **正确 KV correction 本身具有强类别几何。** Dogs 预设 layer 22 上，类别质心解释 89.19% raw gradient energy；中心化后的 class-specific variance 占 88.48%，between/within trace ratio=7.68，leave-one-out centroid label accuracy=100%。MarmAudio 多个设置同样具有 77.58%–93.29% centroid energy 和 100% LOO accuracy。这给 conditional pooled/factorized repair 提供了机制依据。

本轮没有把未完整的 factorized repair 和 K=8 constrained candidate scoring 当成结果；二者只保留为可续跑 checkpoint。

## 1. Support-to-Decision scaling

固定同一组 75 条 recording-disjoint MarmAudio query，support 使用确定性的 nested prefixes：

| K/class | support 总数 | ridge probe | nearest centroid | audio ICL |
|---:|---:|---:|---:|---:|
| 1 | 6 | 46.67% | 29.33% | 8.00% |
| 2 | 12 | 58.67% | 46.67% | 17.33% |
| 4 | 24 | 73.33% | 50.67% | 已有完整产物，待统一统计 |
| 8 | 48 | 84.00% | 78.67% | 已有 free-generation 产物；candidate scoring 本轮未完成 |
| 16 | 96 | 82.67% | 76.00% | 未运行 ICL |

K=16 相对 K=8 没有显著提升：ridge −1.33pp（CI −9.33–6.67，p=1.0），centroid −2.67pp（CI −8.00–2.67，p=.625）。因此当前数据支持“representation readout 在 K≈8 饱和”，而不支持继续增加 probe support 会持续提高。

权威产物：

- `results/marmaudio_support_scaling_k16_statistics.json`
- `results/marmaudio_equal_support_readouts_k16_7b.csv`
- `results/marmaudio_equal_support_split_k16_seed20260814.json`

## 2. Acoustic shift 下的 latent-to-decision failure

所有 probe 均使用完整官方 train/valid/test split；每个 condition 只在自己的 validation 上选择 layer 和 ridge alpha，再 refit train+valid，test 只评一次。

### BEANS Dogs

| 条件 | native generation | condition-specific probe | gap |
|---|---:|---:|---:|
| full observable baseband | 2.88% | 92.81% | 89.93pp |
| 0–1 kHz | 2.88% | 87.77% | 84.89pp |
| 0–2 kHz | 2.88% | 91.37% | 88.49pp |
| 0–4 kHz | 2.88% | 91.37% | 88.49pp |
| 0–6 kHz | 2.88% | 89.21% | 86.33pp |
| 0–8 kHz | 2.88% | 94.96% | 92.09pp |

各低通 probe 与 full probe 的配对差异均不显著（所有 95% CI 包含 0）。这里最准确的表述是：Dogs 的 native output 已在地板，而线性可解码身份信息对这些低通条件极其稳健。

### BEANS Watkins

| 条件 | native generation | condition-specific probe | gap |
|---|---:|---:|---:|
| full observable baseband | 6.19% | 88.20% | 82.01pp |
| 0–1 kHz | 6.19% | 74.04% | 67.85pp |
| 0–2 kHz | 6.49% | 77.88% | 71.39pp |
| 0–4 kHz | 5.60% | 85.84% | 80.24pp |
| 0–6 kHz | 6.19% | 86.73% | 80.53pp |
| 0–8 kHz | 5.90% | 84.07% | 78.17pp |

Watkins 1 kHz 和 2 kHz 相对 full 分别损失 14.16pp、10.32pp，说明真实 information loss/representation shift 存在；但即使最差的 1 kHz probe 仍达 74.04%，比 native 高 67.85pp，所以 information loss 远不足以解释 generation collapse。

权威产物：

- `results/beans_dogs_frequency_probe_7b_summary.json`
- `results/beans_watkins_frequency_probe_7b_summary.json`
- `results/beans_dogs_probe_transfer_matrix_7b_summary.json`
- `results/beans_watkins_probe_transfer_matrix_7b_summary.json`

## 3. Correction Geometry decomposition

定义每个正确标签 gradient field：

`g_i = mu_{y_i} + epsilon_i`。

重点量是 class centroid raw-energy fraction、中心化 between-class variance fraction、between/within trace ratio 和 leave-one-out nearest-centroid label accuracy。

| 数据/设置 | 预设层 | centroid raw energy | centered class fraction | between/within | LOO label acc. |
|---|---:|---:|---:|---:|---:|
| MarmAudio failure set, n=111 | 20 | 77.58% | 66.29% | 1.97 | 100% |
| MarmAudio K2 full, n=12 | 20 | 93.29% | 92.18% | 11.79 | 100% |
| MarmAudio K2 1 kHz, n=12 | 20 | 86.66% | 85.43% | 5.87 | 100% |
| Dogs K2 1 kHz, n=20 | 22 | 89.19% | 88.48% | 7.68 | 100% |

Dogs layer 22 的 global effective rank 为 8.04，median cosine-to-class-centroid=.966，median relative residual norm=.285。最稳妥的结论是：correction field 在全局上包含多个类别 mode，但单条修正的大部分能量由类别共享方向解释。这解释了 fixed global KV 失败、class-conditional pooled KV 成功，并直接动机化低自由度 factorized correction。

注意：K2/class 的单类样本只有 2 条，单类 effective rank 的估计没有解释价值；论文应报告全局 rank、能量分解和 leave-one-out 结果，不应夸大 per-class rank。

权威产物：`results/correction_geometry_decomposition_7b.json`。

## 4. 跨频率 correction direction

在完全相同 K=2/class support 上，六个频率条件的 same-vs-different-label cosine separation 始终很强（0.775–0.826），但相对 full 的 paired gradient cosine 随截止频率降低而旋转：1/2/4/6/8 kHz 分别约 .515/.625/.723/.825/.979。

因此更准确的机制表述是：**label geometry 在 shift 下保留，但具体 corrective direction 是 condition-dependent 的。** 这支持 query/condition-aware repair，而不是一个固定全局方向。

权威产物：`results/kv_geometry_marmaudio_equal_support_k2_cross_frequency_7b.json`。

## 5. Factorized repair：实现完成，实验未过完整性门槛

已实现 rank-2/4/8 的 token×feature truncated-SVD field、relative intervention norm 和同 query method batching；小矩阵 Gram eigendecomposition 与 full SVD 的数值一致性误差为约 1e-6，相关单元测试通过。

预注册协议为：从官方 Dogs validation 分出 stratified 30-query rank-selection 和 untouched 109-query confirmation；selection 比较 pooled、rank-2/4/8、full tokenwise；只有固定 rank 在 confirmation 至少 2/3 alpha 上同时严格超过 pooled 与 full tokenwise，才允许碰 test。

两小时截止时 selection 仅完成 16/30 query（每个 query 五种方法，共 80 行），所以：

- 不选择 rank；
- 不报告 partial accuracy；
- 不运行 confirmation；
- 不触碰 test。

checkpoint：`results/beans_dogs_AJ_factorized_kv_lp1_rank_selection.csv`。

## 6. 未完成任务与下一步顺序

1. resume factorized selection 剩余 14 query，锁 rank 后跑 109-query untouched confirmation；未过 gate 就停止，不跑 test。
2. resume MarmAudio K=8 constrained-candidate scoring；它能区分 lexical/decoding failure 和 native decision-boundary failure。
3. 只在 factorized gate 通过后跑 cross-prompt transfer；否则优先用已成功的 pooled conditional KV 做 cross-prompt。
4. LoRA data scaling 可用现有 Dogs epoch 内 checkpoints 完成，但优先级低于上述两项，因为本轮 correction geometry 和 shift diagnostics 已经形成更直接的机制证据。

## 7. 当前论文主张的安全边界

本轮结果支持：

> 在 matched support 和多种 acoustic conditions 下，Qwen 的冻结表示可以快速、稳定地编码动物类别，但 native generative decision 无法等比例利用这些信息；正确的 KV correction 具有强 class-conditioned geometry，并随 acoustic condition 系统旋转。

本轮结果尚不支持：

- factorized repair 优于 pooled repair；
- repair 可跨 prompt transfer；
- gap 大小能够定量预测 repair gain；
- 方法已跨到 generic environmental audio。

这些必须等完整、预注册的实验完成后再写进摘要或主 claim。
