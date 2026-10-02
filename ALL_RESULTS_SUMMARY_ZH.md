# Animal-Omni-KV 全部实验结果中文汇总

更新时间：2026-08-14

这份文档汇总最终审计后的全部有效结果，包括主实验、公平监督协议、频率诊断、KV 干预、LoRA、失败边界、无效实验和最终论文判断。

## 一、最终科学结论

最稳健的结论已经从最初的：

> “probe 95.42%，generation 29.49%，所以存在 65.93pp grounding gap”

修正为：

> 在完全相同的少量监督样本下，Qwen 的冻结声学表示能够快速学会动物声音类别，但原生生成、音频 ICL 和 LoRA 仍不能稳定利用这些监督信息控制答案。

这是更公平、更经得起 reviewer 检查的结论。

同时，KV cache 确实具有 label-specific causal repair capacity，support gradients 也有很强的类别和 token 结构；但当前 token-wise 方法没有通过预注册稳定性 gate。因此目前最诚实的论文定位是：

> equal-supervision diagnostic + causal KV geometry + partial pooled repair

还不能写成一个已经成功的 token-wise repair methods paper。

## 二、MarmAudio：546 条四位专家一致标注

Qwen 的 “full” 只表示其 16 kHz 输入能够观察到的 0–8 kHz，不是 MarmAudio 原始的 0–48 kHz。

| 模型/方法 | Accuracy | Macro-F1 |
|---|---:|---:|
| Qwen2.5-Omni-3B generation | 20.33% | 12.73% |
| Qwen2.5-Omni-7B generation | 29.49% | 21.94% |
| Qwen-7B frozen grouped probe | 95.42% | 95.32% |
| AVES-bio grouped probe | 93.41% | 93.09% |
| MarmAudio 官方 ResNet-50，原始 0–48 kHz | 88.64% | 88.68% |
| MarmAudio 官方模型，低通到 0–8 kHz | 18.32% | 5.16% |

Qwen-7B 比 3B 高 9.16 个百分点，paired bootstrap 95% CI 为 `[+4.76, +13.37]`。

Qwen-7B 各类别准确率：

| 类别 | Accuracy |
|---|---:|
| Infant Cry | 0.00% |
| Phee | 63.00% |
| Seep | 18.99% |
| Trill | 0.00% |
| Tsik | 82.35% |
| Twitter | 13.13% |

Frozen probe 各类别准确率：

| 类别 | Accuracy |
|---|---:|
| Infant Cry | 99.00% |
| Phee | 97.00% |
| Seep | 96.20% |
| Trill | 89.16% |
| Tsik | 92.94% |
| Twitter | 96.97% |

原始的 95.42% 对 29.49% 差异是 65.93pp，但这是“全监督 probe 对 zero-shot generation”，只能作为表示能力上界，不再作为公平 grounding-gap 主证据。

## 三、Constrained candidate scoring

对同一批 546 条 MarmAudio：

| Readout | Accuracy | Macro-F1 |
|---|---:|---:|
| Free generation | 29.49% | 21.94% |
| Bare candidates，平均 token log-prob | 30.95% | 20.26% |
| Bare candidates，sequence sum | 33.88% | 25.36% |
| 带定义 candidates，平均 token log-prob | 22.34% | 10.46% |
| 带定义 candidates，sequence sum | 18.86% | 6.58% |

结论：

- constrained scoring 能解释大约 1.46–4.39pp；
- 但远远解释不了 95.42% probe 与原生决策之间的差异；
- 给标签定义反而降低表现；
- bare mean scoring 从不预测 Infant Cry、Trill、Twitter，说明 native candidate boundary 本身也严重偏置。

在固定的 75 条 recording-disjoint 平衡 query 上：

| Readout | Accuracy |
|---|---:|
| Free generation | 41.33% |
| Candidate mean | 42.67% |
| Candidate sequence sum | 49.33% |

K=1 ridge probe 是 46.67%，与 zero-shot sequence-sum 的 49.33% 在这 75 条上没有显著差异。因此只拿极低 K 的 probe 对 candidate scoring，并不能证明强 gap；随着 support 增加后 gap 才变得明显。

## 四、Equal-supervision：完全相同 support

### 4.1 MarmAudio，75 条 recording-disjoint query

| K/class | 总 support | Audio ICL free | Audio ICL candidate | Centroid | Ridge probe |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 8.00% | 4.00% | 29.33% | 46.67% |
| 2 | 12 | 17.33% | 未运行 | 46.67% | 58.67% |
| 4 | 24 | context 未运行 | context 未运行 | 50.67% | 73.33% |
| 8 | 48 | context 未运行 | context 未运行 | 78.67% | 84.00% |

K=1：

- Ridge 比 free ICL 高 38.67pp；
- 95% CI：`[+25.33, +52.00]`；
- p=`1.08e-6`。

Ridge 比 candidate ICL 高 42.67pp：

- CI：`[+30.67, +54.67]`；
- p=`4.07e-9`。

K=2：

- Audio ICL 为 17.33%，但全部 75 条都预测 Twitter；
- 因此它是单类别 collapse，不是有效利用更多 support；
- 相同 12 条 support 的 ridge probe 为 58.67%；
- Ridge − ICL = +41.33pp；
- CI：`[+28.00, +54.67]`；
- p=`1.23e-7`。

这部分是目前最强的公平监督证据之一。

### 4.2 Dogs，1 kHz，139 条官方 validation

| K/class | 总 support | Audio ICL | Centroid | Ridge probe |
|---:|---:|---:|---:|---:|
| 1 | 10 | 7.19% | 24.46% | 35.25% |
| 2 | 20 | 未运行 | 22.30% | 35.25% |

K=1 ridge 比 ICL 高 28.06pp：

- CI：`[+20.14, +36.69]`；
- p=`8.65e-10`。

这里的正确解释是 arbitrary acoustic category support-utilization failure，不能再声称 Qwen zero-shot 本来就应该知道 Rudy、Zoe 等名字对应哪只狗。

### 4.3 Watkins，339 条官方 validation

| K/class | 总 support | Centroid | Ridge probe |
|---:|---:|---:|---:|
| 1 | 31 | 26.84% | 32.15% |
| 2 | 62 | 34.22% | 42.48% |
| 4 | 124 | 48.38% | 57.52% |

随着每类 support 增加，representation readout 稳定提升。

旧 Watkins conditional-KV 使用 K=20 total，但 Watkins 有 31 类，support 没覆盖全部类别。因此这个旧失败不能再被解释成一般方法失败。

## 五、任意标签 A–F：排除语言先验

在 MarmAudio 中，将六类 call type 映射为任意单 token 标签 A–F。

固定映射实验：

| 设置 | Accuracy |
|---|---:|
| K=0 candidate scoring | 9.33% |
| K=1/class audio demonstration | 12.00% |

K=1 − K=0：

- +2.67pp；
- CI：`[-2.67, +8.00]`；
- p=`.625`。

K=0 只预测 A/D/B；加入 demonstration 后只预测 A/E。说明 demonstrations 改变了输出先验，但没有可靠学会声音到符号的对应。

进一步做六种 cyclic counterbalanced mappings，共 450 个 query–mapping 对：

| 设置 | 平均 Accuracy |
|---|---:|
| K=0 | 9.56% |
| K=1/class | 17.33% |

差异为 +7.78pp：

- query-clustered CI：`[+2.67, +12.44]`；
- sign-flip p=`.004`。

但保持每种 mapping 输出频率的 shuffled association null 区间为 15.33%–17.78%，实际 17.33% 并没有超过 shuffled audio-label association，one-sided p=`.152`。

因此 Audio ICL 主要是把坏的字母先验修回接近六分类 chance，没有学会可靠的 query-dependent arbitrary binding。

## 六、MarmAudio 频率结果

### 6.1 Qwen-7B generation

| 输入条件 | Accuracy | Full 相对下降 |
|---|---:|---:|
| Full observable 0–8 kHz | 29.49% | — |
| Low-pass 0–1 kHz | 12.64% | −16.85pp |
| Low-pass 0–2 kHz | 15.20% | −14.29pp |
| Low-pass 0–4 kHz | 19.23% | −10.26pp |
| Low-pass 0–6 kHz | 23.08% | −6.41pp |
| Low-pass 0–8 kHz | 28.39% | −1.10pp |

Full 减各 condition 的 paired 95% CI：

| 条件 | 95% CI |
|---|---:|
| 1 kHz | `[+13.00, +20.70]` |
| 2 kHz | `[+10.81, +17.95]` |
| 4 kHz | `[+7.14, +13.55]` |
| 6 kHz | `[+3.85, +9.16]` |
| 8 kHz | `[-0.37, +2.56]` |

单独移除 0–1、1–2、2–4、4–6 或 6–8 kHz 时，没有任何一个窄频段移除的 paired CI 排除 0。所以结论是累计频谱依赖，不是某一个窄频段绝对必要。

### 6.2 Condition-specific frozen probe

| 条件 | Accuracy |
|---|---:|
| Full | 95.42% |
| 1 kHz | 77.29% |
| 2 kHz | 79.85% |
| 4 kHz | 82.60% |
| 6 kHz | 93.96% |
| 8 kHz | 94.87% |

1 kHz 时：

- Generation：12.64% / 5.88% Macro-F1；
- Condition-specific probe：77.29% / 76.91%。

说明低频确实丢失了一部分声学信息，但 generation 的下降远大于表示中可解码信息的下降。

### 6.3 Full-trained transfer probe

只在 full-band 训练分类器，然后不作修改直接应用到 degraded representation：

| 条件 | Transfer probe |
|---|---:|
| 1 kHz | 34.98% |
| 2 kHz | 54.58% |
| 4 kHz | 59.52% |
| 6 kHz | 88.64% |
| 8 kHz | 95.42% |

因此 degraded representation 既保留大量类别信息，也发生明显 distribution/readout shift。

### 6.4 3B 对 7B 的频率尺度效应

- Full：7B 比 3B 高 9.16pp；
- 1 kHz：3B 17.22%，7B 12.64%，7B 反而低 4.58pp；
- 1 kHz CI：`[-7.51, -1.65]`；
- 2 kHz：7B−3B = +0.55pp，CI 包含 0；
- 4 kHz：+3.48pp；
- 6 kHz：+7.88pp；
- 8 kHz：+8.42pp。

7B 的优势随可用带宽增加，不是“模型越大，在严重频谱退化下也越稳健”。

## 七、Prompt、音频和能量控制

固定平衡的 300 条 MarmAudio core：

| 控制 | Accuracy |
|---|---:|
| Canonical prompt | 30.33% |
| 反转 label 顺序 | 19.00% |
| seeded 非平凡 permutation | 31.33% |

反转 label 顺序：

- −11.33pp；
- CI：`[-16.33, -6.67]`；
- 与 canonical prediction agreement 只有 43.0%。

随机 permutation：

- +1.00pp；
- CI：`[-3.33, +5.33]`；
- prediction agreement 63.67%。

说明 deterministic decoding 下，label 顺序仍是显著测量因素。

### 音频替换控制

把每条音频循环替换为下一类别的音频：

- 按原始 target 评分：16.67%，正好六分类 chance；
- 按替换后音频的真实类别评分：30.33%，与 canonical 完全相同。

说明输出确实跟随 waveform，不是纯文本 prior。

### Silence

- Accuracy：15.00%；
- Macro-F1：5.72%；
- 242/300 预测 Phee；
- 58/300 预测 Seep。

模型使用声学信息，但同时受到很强的输出先验控制。

### RMS matching

将低通音频 RMS 恢复到对应 full 音频的 RMS：

| 条件 | 原低通 | RMS matched | 提升 |
|---|---:|---:|---:|
| 1 kHz | 12.33% | 18.33% | +6.00pp |
| 2 kHz | 14.00% | 16.00% | +2.00pp |
| 4 kHz | 16.67% | 19.33% | +2.67pp |

1 kHz 提升 CI 为 `[+2.67, +9.33]`。但仍远低于 full 的 30.33%，所以频谱实验不是单纯由音量衰减造成。

## 八、Dogs 完整结果

### 8.1 Zero-shot 与 frozen representations

官方 test，139 条：

| 方法 | Accuracy | Macro-F1 |
|---|---:|---:|
| Qwen-7B zero-shot generation | 2.88% | 0.56% |
| Qwen-7B frozen probe | 92.81% | 90.95% |
| AVES-bio fixed-split probe | 83.45% | 80.72% |

Zero-shot 139 条全部预测 Rudy。Qwen probe 最佳层为 layer 10。

Qwen probe 比 AVES-bio 高 9.35pp：

- paired bootstrap CI：约 `[+3.60, +15.11]`；
- exact McNemar p=`.00443`；
- Qwen-only correct：16；
- AVES-only correct：3。

这证明 Qwen representation 对 Dogs individual identity 非常强，但 zero-shot 名字映射本身不能被当作公平 grounding 证据。

### 8.2 Dogs 频率

11 个 full/low-pass/band-removal 条件，共 1,529 个预测，全部为 Rudy。因此全部条件都是 2.88% Accuracy / 0.56% Macro-F1。这是 generation head floor effect，不代表个体信息与频率无关。

### 8.3 K=2/class LoRA

训练：

- 20 条 support；
- 每只狗 2 条；
- one epoch；
- rank 8；
- q/v projection；
- learning rate `2e-4`。

Validation 139 条：

- Accuracy：2.88%；
- Macro-F1：0.56%；
- 全部预测 Rudy。

同 support ridge probe 为 35.25%。Ridge − LoRA：

- +32.37pp；
- CI：`[+23.74, +41.01]`；
- p=`1.97e-11`。

因此少量 support 下，LoRA 也没有解决 output collapse。

### 8.4 全训练集 Dogs LoRA

另一条排队中的扩展流水线最终完成了 Dogs 全训练集 LoRA，并在官方 untouched test 上评测一次。

训练设置：

- 官方 train：415 条；
- official validation loss monitor：64 条；
- one epoch；
- rank 8；
- q/v projection；
- query/test labels 只用于最终 post-hoc scoring。

训练与测试结果：

- Train loss：1.4114；
- Validation loss：1.2159；
- Test n：139；
- Accuracy：25.18%；
- Macro-F1：12.26%；
- Invalid：0；
- 输出分布：Mac 101、Luke 32、Zoe 6。

它明显高于 zero-shot 和 K=2/class LoRA 的 2.88%，但仍远低于相同官方 fixed-split 上 frozen probe 的 92.81%。该模型已经保存在本地，尚未作为第三个 Hugging Face 模型发布。

## 九、Dogs matched-support KV

139 条 validation，relative intervention norm。

### 9.1 连续 conditional router

| 方法 | α=.003 | α=.01 | α=.03 |
|---|---:|---:|---:|
| Fixed mean | 2.88% | 2.88% | 5.04% |
| Conditional pooled | 2.88% | 2.88% | 12.95% |
| Conditional ordered tokenwise | 2.88% | 7.19% | 10.79% |
| Token-permuted | 2.88% | 3.60% | 3.60% |
| Matched random | 2.88% | 2.88% | 2.88% |

α=.03 时：

- pooled 比 fixed 高 7.91pp；
- CI：`[+2.88, +13.67]`；
- p=`.0074`。

Ordered tokenwise 比 token-permuted 高 7.19pp：

- CI：`[+1.44, +12.95]`；
- p=`.0309`。

这说明 token 位置确实存在 causal structure。但是 ordered tokenwise 相对 pooled 为 −2.16pp，CI `[-9.35, +5.04]`，p=`.70`。它没有稳定胜过 pooled，预注册 gate 失败，所以没有运行 test。

### 9.2 Class-dictionary router

Router 使用相同 K=2/class ridge readout，router accuracy 为 35.25%。

| 方法 | α=.003 | α=.01 | α=.03 |
|---|---:|---:|---:|
| Class-routed pooled | 2.88% | 15.83% | 20.86% |
| Class-routed tokenwise | 10.79% | 13.67% | 0.00% |
| Token-permuted | 3.60% | 2.16% | 0.72% |

α=.003 时 tokenwise：

- 比 pooled 高 7.91pp，p=`.0074`；
- 比 token permutation 高 7.19pp，p=`.0063`。

但 α=.01 时 pooled 更高 2.16pp；α=.03 时 tokenwise 产生 99.28% invalid output。Gate 要求 tokenwise 至少在两个 α 上超过 pooled，实际只在 α=.003 超过。因此 gate 失败，没有 test。

20.86% pooled 是有效的 partial repair，但 tokenwise 不能宣称成功。

## 十、Dogs 任意 A–J KV 复制

139 条 validation，所有 support gradients 都针对 A–J 重新计算。

| 方法 | α | Accuracy | Macro-F1 | Invalid |
|---|---:|---:|---:|---:|
| Native | 0 | 8.63% | 2.73% | 0.00% |
| Token-permuted | .003 | 7.91% | 5.64% | 25.90% |
| Token-permuted | .01 | 4.32% | 5.74% | 57.55% |
| Token-permuted | .03 | 0.72% | 1.82% | 95.68% |
| Pooled | .003 | 14.39% | 11.84% | 0.00% |
| Pooled | .01 | 18.71% | 15.26% | 0.72% |
| Pooled | .03 | 20.14% | 18.49% | 6.47% |
| Tokenwise | .003 | 15.11% | 11.22% | 24.46% |
| Tokenwise | .01 | 6.47% | 9.26% | 74.82% |
| Tokenwise | .03 | 0.00% | 0.00% | 99.28% |

α=.003 时 tokenwise 比 permutation 高 7.19pp，CI `[+0.72, +14.39]`，exact p=`.064`，未达到 .05。

Tokenwise 只比 pooled 高 0.72pp，CI `[-5.76, +7.19]`，p=`1.0`。α=.01 后 tokenwise 低于 pooled 12.23pp。

因此 A–J 结果再次证明 class-conditioned pooled cache effect 存在，token order 可能有作用，但 token-wise 方法不稳定；validation gate 失败，没有运行 test。

## 十一、Audio–silence steering baseline

MarmAudio 75 条 query：

- support-only calibration 选择 β=.1；
- candidate readout：41.33% → 42.67%；
- 提升 +1.33pp。

Dogs A–J，139 条：

- 选择 β=.3；
- 结果保持 5.04%，没有提升。

因此一个通用 listening direction 不能自动学会 arbitrary acoustic-category-to-symbol mapping。这只是 matched conceptual baseline，不声称是对 specialist-head 方法的完全 faithful reproduction。

## 十二、Watkins 完整结果

### 12.1 Zero-shot 与 frozen probe

官方 test，339 条：

| 方法 | Accuracy | Macro-F1 |
|---|---:|---:|
| Qwen-7B native full 0–8 kHz | 5.60% | 1.87% |
| Qwen-7B frozen probe | 88.20% | 88.25% |
| AVES-bio fixed-split probe | 85.84% | 85.48% |
| Thinker LoRA | 31.56% | 23.44% |

以前出现的约 6.19% 是 paired 1 kHz KV test baseline；权威 full 0–8 kHz native test 是 5.60%。

Native 输出主要集中在 False Killer Whale（140）、Beluga White Whale（107）和 Bottlenose Dolphin（84）。Qwen probe 最佳层为 layer 8。

Qwen probe 比 AVES-bio 高 2.36pp，paired CI 约 `[-0.88, +5.90]`，exact p=`.243`，差异不显著。

### 12.2 Watkins LoRA

训练设置：

- 1,017 条官方 train；
- one epoch；
- rank 8；
- q/v projection；
- 3.83M trainable parameters；
- 约占模型 0.043%；
- learning rate `2e-4`；
- accumulation 8；
- 64 条 validation 仅用于 loss monitor；
- 最终 validation loss：0.466。

Test：

- Accuracy：31.56%；
- Macro-F1：23.44%；
- 8/339 invalid。

相比 native 5.60% 有明显提升，但仍远低于 frozen probe 88.20%。

### 12.3 Watkins 频率

Low-pass generation 大致保持在 5.60%–6.49%。只有移除 0–1 kHz 显著下降：

- Full：6.19%；
- 移除 0–1 kHz：3.54%；
- 差异：+2.65pp；
- CI：`[+0.29, +5.01]`。

其他频率变化基本是 floor effect。

### 12.4 旧 conditional KV

旧协议 K=20 total，31 个类别没有全部被 support 覆盖。1 kHz paired baseline 为 6.19%，conditional 约 5.60%–5.90%，fixed 约 5.60%–6.49%，没有改善。

这个结果仍保留为历史失败，但不能用来判断完整 K-per-class 协议下的方法能力。

## 十三、AVES-bio 完整 fixed-split

全部 representation 已补齐：Dogs 693/693，Watkins 1,695/1,695。

| 数据集 | Qwen frozen probe | AVES-bio |
|---|---:|---:|
| Dogs | 92.81% / 90.95% F1 | 83.45% / 80.72% F1 |
| Watkins | 88.20% / 88.25% F1 | 85.84% / 85.48% F1 |

Dogs 上 Qwen representation 显著高于 AVES-bio；Watkins 上 Qwen 高 2.36pp，但差异不显著。这说明 Qwen 表示并不缺少主要类别信息，主要问题在监督映射和原生生成接口。

## 十四、Oracle KV capacity

### 14.1 历史 raw-eta 111 条结果

111 条 full-correct、1 kHz-wrong MarmAudio 样本：

| η | Recovery |
|---:|---:|
| 0.01 | 12.61% |
| 0.03 | 23.42% |
| 0.1 | 53.15% |
| 0.3 | 100.00% |
| 1.0 | 93.69% |

111/111 都能在至少一个注册 η 下被恢复。但它使用 ground-truth label gradient，只是 capacity upper bound，不是可部署方法。

### 14.2 最终 matched-norm causal controls

固定的 12 条 full-correct/lp1-wrong 样本：

| 干预 | 结果 |
|---|---:|
| Full-prefill correct-label，α=.0003 | 12/12 恢复正确 |
| Full-prefill wrong-label，α=.001 | 11/12 被推到错误目标；0/12 保持正确 |
| Full-prefill matched random | 所有 α 均为 0/12 |
| Full-prefill，audio token permutation | 12/12 |
| Audio-only correct-label | 5/12 曾恢复 |
| Audio-only token permutation | 1/12 曾恢复 |
| Audio-only matched random | 0/12 |

解释：

- Cache 有很强的 label-specific causal capacity；
- 但 full-prefill token permutation 仍能 12/12，说明 prompt/decision token gradient 可能承担了主要作用；
- Audio-only 只有 5/12，才是更严格的 acoustic-token causal evidence；
- 所以不能把 111/111 写成“audio token mechanism 已被证明”。

Dogs 没有 full-correct/lp1-wrong 样本，因为全部 collapse 到 Rudy，Oracle 不可定义。Watkins 只有 5 条 eligible，η=1 时 5/5 恢复，但样本太少，不能作为 deployable evidence。

## 十五、KV gradient geometry

### 15.1 MarmAudio

- Same-label median cosine：0.734；
- Different-label median cosine：−0.145；
- Median gap：0.867；
- Nearest-cosine-neighbor label accuracy：96.40%–100%；
- Median：100%。

最佳分离附近 layer 20：same 0.838，different −0.190，gap 1.028。但 failure-selected 集合只有 4/6 个标签，且类别不平衡，因此它是 mechanism evidence，不是平衡训练集。

### 15.2 Dogs

K=2/class、每类两条 support。

整体：

- Same-label median cosine：0.776；
- Different-label median cosine：−0.012；
- Gap：0.784；
- 不同 projection 的 nearest-neighbor accuracy：45%–100%。

最佳 combined K/V layer 为 layer 22：

- same-label cosine：0.862；
- different-label cosine：−0.042；
- gap：0.903；
- global effective rank：8.04；
- between/within trace ratio：7.68。

Token energy：250 个 audio tokens 中，有效能量相当于约 15.7 个 token，63.95% 梯度能量位于前 1/3 时间段。这证明 support corrective field 强烈 class-conditioned，且 token energy 很稀疏。

### 15.3 Cross-task low-rank geometry

| 数据集 | Median rank-4 energy | Median off-diagonal cosine |
|---|---:|---:|
| MarmAudio | 85.40% | 0.238 |
| Dogs | 76.99% | 0.026 |
| Watkins | 75.44% | 0.208 |

Rank-4 energy 范围：MarmAudio 73.79%–93.24%，Dogs 69.10%–96.63%，Watkins 50.52%–91.58%。

因此不存在一个跨任务都可靠的 global mean direction，也不能声称 rank 4 普遍足够。Watkins 的旧 20-gradient set 少于 31 类，同类 cosine 无法定义。

## 十六、Query-label-free MarmAudio conditional KV

### 16.1 Recovery-enriched diagnostic

21 条 held-out full-correct/lp1-wrong 样本：

| η | Fixed | Conditional |
|---:|---:|---:|
| 300 | 23.81% | 76.19% |
| 1000 | 52.38% | 76.19% |

原始 recovery-enriched 75 条 query：

- lp1 baseline：14.67% / 6.84% Macro-F1；
- Fixed KV：12.00% / 10.41%；
- Conditional KV：44.00% / 29.98%。

因为 query recording 是从已知包含 eligible failure 的 recordings 中选出的，所以只能作为 enriched diagnostic。

### 16.2 五个 primary recording-group splits

先从全部 93 个 recordings 抽 query groups，再检查 eligibility；support recordings 完全不重叠。

| 方法 | Accuracy，mean ± SD |
|---|---:|
| Degraded baseline | 12.35 ± 4.16% |
| Fixed KV | 13.01 ± 10.20% |
| Conditional KV | 32.47 ± 7.50% |

差异：

- Conditional − baseline：`+20.12 ± 8.02pp`；
- Conditional − fixed：`+19.46 ± 5.15pp`；
- 两个差异在五个 splits 上全部为正。

Macro-F1 从 baseline 5.92% 提升到 conditional 24.54%。这是目前 conditional pooled KV 最有力的 query-label-free 结果。

## 十七、BEANS-Zero

### 17.1 原始 300 条 screening

| Component | Accuracy |
|---|---:|
| Overall | 11.00% |
| call-type | 60.00% |
| zf-indiv | 72.00% |
| Watkins | 0.00% |
| unseen species/genus/family | 全部 0.00% |

### 17.2 完整 2,950 条扫描

| Component | N | Correct | Accuracy |
|---|---:|---:|---:|
| call-type | 283 | 174 | 61.48% |
| zf-indiv | 324 | 217 | 66.98% |
| unseen-family-cmn | 117 | 0 | 0.00% |
| unseen-family-sci | 130 | 2 | 1.54% |
| unseen-family-tax | 123 | 0 | 0.00% |
| unseen-genus-cmn | 265 | 0 | 0.00% |
| unseen-genus-sci | 262 | 0 | 0.00% |
| unseen-genus-tax | 258 | 0 | 0.00% |
| unseen-species-cmn | 373 | 0 | 0.00% |
| unseen-species-sci | 364 | 0 | 0.00% |
| unseen-species-tax | 336 | 0 | 0.00% |
| Watkins | 115 | 0 | 0.00% |
| Overall | 2,950 | 393 | 13.32% |

Whole-phrase target containment 为 13.36%，而 exact match 为 13.32%。Watkins 以及所有 open genus/species/common/taxonomic components 仍为 0%。所以 near-zero 结果不是因为模型答对了名称但外面包了一层自然语言。

结论：Qwen 并不是完全听不到动物声音。它在 closed coarse vocal attribute 上能达到 61%–67%，但对 open-vocabulary taxonomic/species-name binding 几乎完全失败。

## 十八、历史 Dogs conditional-KV 结果

这些保留，但不是最终公平协议主证据。

Dogs 1 kHz test：

| 方法 | K | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Baseline | — | 2.88% | 0.56% |
| Fixed | 5 | 2.88% | 0.56% |
| Conditional | 5 | 17.27% | 9.45% |
| Fixed | 10 | 3.60% | 3.06% |
| Conditional | 10 | 11.51% | 8.49% |
| Conditional | 20 | 7.19% | 6.92% |

K=5 conditional − baseline 为 +14.39pp，CI `[+8.63, +20.14]`。

后续看过初始 test 后再做的 layer/rank follow-up：layer 28、rank 8，test 20.86% / 18.07% Macro-F1，fixed mean 3.60% / 2.57%。因为它发生在初始 test 已被观察之后，只能明确标成 exploratory follow-up，不能冒充预注册主结果。

## 十九、无效或隔离的结果

以下结果仍保留用于溯源，但不能作为论文主证据：

- Zero-shot Dogs 名字对 fully supervised probe：只能作为上界差异；
- Dogs validation 前 64 条 manifest prefix：只覆盖 6/10 类，已经隔离；
- Batch=5 与 batch=1 在 25 条中有 2 条预测不同，所有正式 KV 结果固定 batch=1；
- Raw-eta tokenwise sweeps：只算 engineering pilot；
- Oracle raw-eta 111/111：被 matched-norm 12 条 causal control 取代；
- Watkins K=20 total：没有覆盖 31 类，不能解释方法一般失败；
- 混合 batch-size 的旧频率 CSV：不能引用；
- `Audio_Examples.zip` 的文件名与专家标注只有 17/60 一致，不能计算科学指标；
- 早期 partial AVES extraction 已由完整 CUDA extraction 取代。

## 二十、最终论文判断

目前可以强写的：

1. 在同等 K-shot supervision 下，frozen readout 能快速学会 bioacoustic 类别，而 Audio ICL、生成和少量 LoRA 仍出现 collapse；
2. Candidate scoring 只能解释一小部分 native generation failure；
3. Support KV gradients 具有强 label-conditioned geometry；
4. Token 位置和排列确实有 causal effect；
5. Query-label-free conditional pooled KV 在五个 recording splits 上稳定优于 baseline 和 fixed KV；
6. 任意 A–F/A–J 标签实验排除了单纯语言标签先验解释；
7. BEANS-Zero 证明 Qwen 对 closed vocal attributes 有一定能力，但 open taxonomic binding 几乎为零。

目前不能强写的：

1. 不能把 95.42% probe 对 29.49% zero-shot 直接定义成公平的 65.93pp grounding gap；
2. 不能声称 first KV steering；
3. 不能声称 token-wise repair 已成功；
4. 不能声称 111/111 证明 audio-token causal mechanism；
5. 不能把 Watkins K=20-total 失败当作方法边界；
6. 不能把 Dogs zero-shot 名字失败解释成“模型知道是哪只狗但说不出来”。

最终 framing：

> Generalist Audio-LM 在少量同等监督下存在显著的 support-to-decision failure。冻结声学表示可以利用 support 学出类别边界，但原生生成接口不能稳定形成相同映射。KV cache 具有 label-specific causal repair capacity，conditional pooled repair 可以部分改善决策，而 token-wise repair 的稳定性仍未解决。

## 二十一、模型与运行状态

已经完成并上传的模型：

| 模型 | 状态 | 结果 |
|---|---|---|
| Watkins Qwen-7B Thinker LoRA | 已上传 | 31.56% Accuracy / 23.44% Macro-F1 |
| Dogs K=2/class LoRA | 已上传 | 2.88%，全部 Rudy；仅作为负对照 |

新增完成并已准备发布的模型：

- Dogs 全训练集 LoRA 已完成 one epoch；
- 415 条 official train，139 条 untouched official test；
- Test Accuracy：25.18%；
- Macro-F1：12.26%；
- Invalid：0；
- 本地 adapter 已完整保存；
- Hugging Face 目标仓库：`https://huggingface.co/humanlong/qwen2.5-omni-7b-dogs-full-lora`。

随后自动开始的额外 Dogs frequency representation extraction 已按用户要求停止；已完成正式产物均保留，未删除 checkpoint 或结果。

最终系统状态：

- GPU utilization：0%；
- GPU memory：约 4 MiB；
- 正式实验进程：0；
- BEANS-Zero：2,950/2,950；
- Dogs A–J：1,390/1,390 logical rows；
- 测试：19/19 通过；
- Artifact audit：23/23 通过；
- Fair-protocol audit：27/27 通过；
- 两个 Hugging Face 模型均完成逐文件下载及 SHA-256 一致性验证。

## 二十二、发布地址

- GitHub（私有）：https://github.com/Yunbo-max/animal-omni-kv
- Watkins LoRA（私有）：https://huggingface.co/humanlong/qwen2.5-omni-7b-watkins-lora
- Dogs K=2 LoRA（私有）：https://huggingface.co/humanlong/qwen2.5-omni-7b-dogs-k2-lora
- Dogs full-train LoRA（私有）：https://huggingface.co/humanlong/qwen2.5-omni-7b-dogs-full-lora
- GitHub 发布分支：`main`

## 二十三、2026-08-14 两小时高价值扩展

本轮新增了三组可直接加强论文的完整结果：

- MarmAudio support scaling 扩到 K=16/class：ridge 在 K=1/2/4/8/16 时为 46.67/58.67/73.33/84.00/82.67%，表明约 K=8 饱和；与相同 support audio ICL 的严格 gap 在 K=1、2 时分别为 38.67pp 和 41.33pp，配对检验均显著。
- Dogs 与 Watkins 六频段 condition-specific probe 全部完成。Dogs 在所有条件仍为 87.77%–94.96%，而 native 为 2.88%；Watkins 为 74.04%–88.20%，而 native 约为 5.60%–6.49%。低频确有信息损失，但无法解释 native generation 的地板表现。
- correction geometry 能量分解完成。Dogs layer 22 的 class centroid raw-energy fraction 为 89.19%，centered class-specific variance fraction 为 88.48%，between/within=7.68，LOO centroid label accuracy=100%；MarmAudio 多个设置得到一致结论。

Factorized repair 已实现并通过相关单元测试，但预注册 rank-selection 在两小时截止时仅完成 16/30 query，因此没有选择 rank、没有运行 confirmation/test，也没有把 partial accuracy 当作正式结果。MarmAudio K=8 constrained candidate scoring 同样只完成 50/75，已保存 checkpoint 并停止。

完整协议、数值、限制和续跑顺序见 `TWO_HOUR_EXTENSION_RESULTS_20260814_ZH.md`。

## 二十四、2026-08-14 Order-control 续跑状态

MarmAudio counterbalanced order controls 已完成预定三组中的最后一组 K=8 interleaved。结果为 75 条 query、accuracy 20.00%、macro-F1 19.74%、invalid 0%、last-label copy rate 0%。这个对照说明 K=8 ICL 的错误不是简单复制最后一个 support label。

随后自动启动的跨数据集 order controls 在 RTX 3090 24GB 上显存不足：Dogs K=2 在 audio tower attention 处 OOM，Watkins K=1 在 thinker self-attention 处 OOM。脚本已按协议隔离 incomplete 输出；没有把这两段作为正式结果。
