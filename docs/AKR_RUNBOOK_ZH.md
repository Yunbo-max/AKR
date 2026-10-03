# AKR ICLR 2027：E1–E9 执行手册

## 1. 本次代码的范围

本次只更新执行代码与说明，不启动科学 GPU 实验，不生成新权重或新的准确率。
实验定义以 `docs/AKR_CLOSING_E1_E9.md` 的用户原始计划为准。

主方法严格调用 `src/animal_omni/conditional_kv.py` 中的
`ConditionalGradientRouter`：support 梯度中心化 → SVD → ridge 预测梯度系数。
`akr_closing/repair.py` 中的 `RepositoryGradientRouter` 是适配层，导入失败会停止，
不会静默切换另一算法。包内数值 reference 仅服务 CPU 测试，生产 CLI 不允许选择它。

干预位于 **pre-RoPE k_proj/v_proj outputs**，不是直接编辑后旋转的 cache。
backbone 冻结；support 上拟合辅助回归器不等于 training-free。
continuous、class-soft、class-hard、tokenwise/factorized 均分开命名、分开报告。

## 2. 应用更新并发布

本次连接对 `Yunbo-max/animal-omni-kv` 只有读取权限；实际远端写入返回 403。
因此代码必须先由有写权限的执行 agent 应用、测试、提交。收到本包不代表 GitHub 已更新。

完整 ZIP 包含补丁、源码 overlay、验证日志以及 `apply_update.sh`。
在已有 repo 的干净工作树中执行（不修改/删除历史结果）：

```bash
# 在解压后的交付目录；默认仅应用并测试，不提交或推送
bash apply_update.sh /root/animal-omni-kv

# 有权限的 agent 可用此选项：应用、测试、提交、推送当前 main 分支
bash apply_update.sh /root/animal-omni-kv --publish
```

手动方式：

```bash
cd /root/animal-omni-kv
git pull --ff-only
git apply --check /path/to/Animal_AKR_E1_E9_Update/animal_akr_e1_e9.patch
git apply /path/to/Animal_AKR_E1_E9_Update/animal_akr_e1_e9.patch
.venv/bin/python -m pytest -q tests/test_akr_closing_base.py tests/test_akr_contracts.py tests/test_akr_extended.py
# 按交付包 changed_paths.txt 精确提交新增文件与 README，勿 git add 全部数据/结果。
```

补丁基于远端 main `9e8c244b797c24cba1933998bbafbf44a36b0d86` 所读代码。
README 源 blob 为 `015bad9c1bb4e39f34e023012807d3cdc1baec6b`，本包已核对该文件字节哈希。
对既有文件只插入 README 收尾入口；新代码使用独立 `akr_closing/` 命名空间，
不覆盖上一份 `iclr2027_final/` 包。不得将旧入口与新入口输出合并成同一实验身份。

## 3. 数据与环境前提

沿用原 repo 的可用 `.venv`、模型缓存和 waveform 文件，不自动升级依赖。
CPU 测试使用 numpy/scipy/sklearn/torch/pytest/soundfile/PyYAML。
真实实验额外需要仓库所支持的 transformers、accelerate、qwen-omni-utils、
huggingface_hub；E8 需要 peft。安装问题应参照现有 `REPRODUCE.md`，
不要把本次 CPU 环境版本当作新的 GPU 锁定环境。

主数据 manifest：

| 数据 | manifest | 划分 |
|---|---|---|
| MarmAudio | `data/manifests/marmaudio_expert_validation.csv` | 使用真实 recording_id 分组；已观察数据的新分组重评 |
| Dogs | `data/manifests/beans_dogs.csv` | 保留官方 train/test，原 valid 分 selection/confirmation |
| Watkins | `data/manifests/beans_watkins.csv` | 同 Dogs；标签从 configs/beans_watkins.yaml 读入 |

E1/E7 还需要原 `results/marmaudio_equal_support_split_seed20260814.json`、
`configs/marmaudio_fair_gap.yaml` 和原 candidate CSV（若存在则只复制保留）。
其中必须是原 75 个 query、每类 8 个共 48 个 support，且 recording 不重叠。

`old_root` 默认 `/root/animal-omni-kv`，新主流程和 E1/E7 会将该前缀映射到当前 repo 根。
新主流程先验证所有 waveform 存在、标签覆盖、真实 group 元数据和 train/test 重复。
同一 RUN_ID 内 source 文件或 waveform 内容变化会拒绝恢复。
配置中的全部数据集都会预检；只运行某个数据集时，在开跑前复制配置并删去其他数据集，
使用新的 RUN_ID，不能在一个已经开始的 RUN_ID 中改配置。

## 4. 入口与资源策略

```bash
cd /root/animal-omni-kv

# 仅数据、配置、实验身份检查；无模型加载、无 GPU 实验
.venv/bin/python scripts/run_akr_closing.py --run-id akr_closing_v1

# 标准运行：先测试、预检、真实后端 parity，随后执行默认任务
RUN_ID=akr_closing_v1 PROFILE=24gb bash scripts/run_akr_closing.sh
```

默认任务顺序：`E1,E3,E2,E4,E5,E7,E8`。
配置包含 3 个固定 seeds、7B、full/1 kHz、语义/任意标签条件。
这是一套完整矩阵，不是限时 smoke；不会用 partial accuracy 代替完整结果。
可以选择任务：

```bash
RUN_ID=akr_closing_v1 PROFILE=24gb TASKS=E1,E3 bash scripts/run_akr_closing.sh
# 通过同配置真实后端 parity 后，可直接续跑选定阶段
.venv/bin/python scripts/run_akr_closing.py --run-id akr_closing_v1 \
  --profile 24gb --tasks E2,E4,E5 --execute --resume
```

`--execute` 才加载真实模型；没有该参数默认只是预检。
`PROFILE=16gb/24gb/48gb` 分别限制 GPU 权重放置预算，并允许 CPU offload。
这不是所有长 audio-ICL prefix 都能装入显存的保证。OOM 会停止并保留 checkpoint；
context 超预算会写 BLOCKED，不会删 support 或截断 query 来假装完成。
默认 batch=1，正式结果不得混入历史 batch=5 的 tokenwise 产物。

后端一致性脚本先顺序加载原 `QwenThinkerRunner` 和新 direct Thinker backend，
比较声明的 smoke 声音上的预测。该检查在用户 GPU 上运行；本次未运行。
一致只说明该 smoke 的离散输出匹配，不是全模型数值等价证明。
E1/E7 单独用旧 runner 并固定原模型 revision；新主实验也固定 revision。

## 5. E1：保持实验身份的历史续跑

程序 `akr_closing/legacy.py`：复制历史 candidate CSV 到新输出区，不删除、不移动、不覆盖旧文件。
检查每行 event、K/class=8、总 support=48、目标标签和六个候选的两个 finite log-prob。
记录 config/split/源码/WAV 的 SHA-256，以及继承行的内容哈希；跑后再次检查旧行未被改写。

只补缺失 query；新结果保留原两个 scoring rule。完整 75/75 后才输出各自准确率、
macro-F1、输出分布与配对统计。未完整只输出 completed/expected/missing IDs。
源 partial 的历史运行信息无法凭空还原，所以 provenance 明确标识“继承的历史部分结果”。
这不是新预注册实验。

E7 是 E1 的配套位置实验，另用全 75 queries 的 order×query 网格，避免把单一固定首类的
100% agreement 当成首位置因果结论。

## 6. E2/E3：真实连续方法与 probe 对照

新实验划分是 locked re-evaluation：旧 Dogs/Watkins test 已观察，不能重新叫 untouched。
MarmAudio 专家集的新分组也不能称独立外部固定 test。

E3 先在 development selection 部分，以预先指定 rank=4、alpha=.01 比较直接 ridge、
probe-to-text、probe-to-LM、continuous、class-soft/class-hard、fixed 和 zero；
不读取 confirmation 标签来决定这些设置。

E2 的 rank 1/4/8、ridge penalty 1/10、relative alpha .003/.01/.03 仅在 selection 搜索。
锁定后原样在 confirmation 运行，记录 `LOCK.json`。主 gate 根据 continuous 相对 native 的
配对增益及 invalid 判断，不能改由 class-dictionary 成绩过关。
通过后才允许 `test_locked_reevaluation`，未通过保留负结果并跳过 test。
这不恢复任何旧 tokenwise/factorized gate。

每个 query 的推断对象 `Query` 没有 label 字段；真值只用于完成预测后的评分。
所有支持样本、query 集、特征层、目标词、解析器、prompt、参数、归一化方式及源码均写入身份记录。

E3/E2 的 direct ridge、probe-to-text 输出严格相同；probe-to-LM 则把同一分类预测作为文本证据，
记录 LM 是否服从、是否将正确预测改错。class-routed 和 continuous 都按 router 正误分层报告。
直接 classifier 分数是回归/分类分数，不宣称为已校准概率。

## 7. E4/E5：机制与消融

E4 对每个支持标签保持目标 token 不变，分别使用真实音频、同长度静音、其他类别音频。
其他类别 donor 会 resample 后截取或补零以匹配时长，不 time-stretch、不循环复制、不独立 RMS 归一化。
另加入同声音错误目标、保持类频次的 support-label permutation、类质心目标与类质心残差目标。

关键是：每种目标集都拟合一个**连续回归器**，保持原始 support/query acoustic features 相同，
实际在 confirmation query 生成并统计修复，不仅计算 support cosine。
梯度聚类含目标标签效应；不将 LOO gradient-label accuracy 当成 label-free 动物语义解码结果。

E5 在相同 query 和已锁定设置上运行 K-only/V-only/K+V、四个选层中的 early/middle/late、
audio/text/full-prefill、固定均值、匹配随机、负方向、query-field shuffle，以及 E4 的支持标签打乱。
记录每一 projection 的实际 base/delta Frobenius 范数与比例。
**relative alpha 是每个激活 projection／scope 的相对范数，不等于不同层数或 K-only/KV
之间的绝对总能量相同。**各 scope 由各自 support 梯度重新拟合；论文不得隐去此条件。
主入口不混用 raw eta，历史 raw-eta 表保持单独口径。

`zero_alpha` 和 `zero_direction` 必须与 native 的 raw 文本及最终标签逐项完全一致；
任何 no-op 不一致都会阻止继续报告。

## 8. E7：真正的顺序干预

E7 固定完全相同的 48 条 audio-label support 对和原 75 queries。
生成 historical order、每类作为第一类的 interleaved cycle、逆序 cycle、保持类别频次的随机排列。
**每一种顺序评估全部 query**，而不是每个 query 只分配一个顺序。
保存每条预测的 support 顺序、first/last label、recording、raw 文本、耗时。
完整后报告每个顺序 accuracy/F1/invalid、首尾一致率、query 预测变化率及 paired comparison。
这支持“顺序的干预效应”；因为多处位置一起改变，不等同于完全隔离首个 token 的作用。

## 9. E8：有预算的原生 LoRA 适配

不是再执行历史 one epoch：新 baseline 使用相同 support、固定 optimizer step 网格。
默认 Dogs/MarmAudio、1 kHz、K2/K8、LR 2e-5/2e-4、最大128步、每32步完整 development evaluation、
rank8 decoder q/v LoRA、accumulation8。配置改变须新 RUN_ID。

验证 prefix label mask=-100，target tokens 及 EOS 有监督；确保只有 LoRA 参数训练。
保存 loss/native-accuracy 曲线和 wall time；每个 optimizer step 保存 adapter、optimizer、
Python/NumPy/Torch/CUDA RNG。两个轮换 resume 槽控制磁盘占用，正式 eval checkpoints 保留。
中断在完整 optimizer step 后停止，未提交 microbatch 不会被当成完整 step。

学习率和 checkpoint 仅由 development 选择，之后重新载入 adapter，要求保存的 development
预测与 reload 预测一致，才跑 confirmation。没有 test 调参。
这条真实 GPU 训练链本次未跑；CPU 只验证 scheduling、mask、选择协议及非有限数防护。
既有 one-epoch 低准确率仍只代表其预算，不写成 LoRA 不可能成功。

## 10. E6/E9：可选，不改变主线

```bash
# E6：原 class-routed factorized 的新 batch-one 重评
.venv/bin/python scripts/run_akr_closing.py --run-id akr_closing_v1 --tasks E6 --execute --resume

# E9：需要已有完整7B LOCK；把所选设置迁移到3B，不另搜参
.venv/bin/python scripts/run_akr_closing.py --run-id akr_closing_v1 --tasks E9 --execute --resume
```

E6 依赖原 30/109 selection/confirmation JSON、已提取 token gradients/reps，以及原 factorized
scripts。为了不混合已知不等价的 batch 5/1，旧16/30 batch-five partial 只留历史，不导入新 batch-one
重评。这里**不会自动运行 test**，旧失败 gate 完全不改。
旧脚本/manifest 含机器路径时需在原路径运行或先审计路径；缺依赖会明确报错。

Watkins 新主流程已用每类1/2/4支持，不另使用20-total来宣称边界。
E9 在 `.../<RUN_ID>_3b_locked/` 输出，保存7B锁的哈希、相同支持/确认IDs、相同比例选层。
不在3B重新选择 rank/alpha、不运行新test。成本记录会区分 support forward+backward、feature forward
与生成，缓存命中不冒充一次新的执行耗时。额外前向数量由 hook 实测，不默认全等于2。

BEANS-Zero 的既有 capped12-component 结果保留，不重新扫描或扩展物种；本包没有环境声任务。
Audio-silence specialist-head 的 faithful 复现不在此新增代码内；既有版本继续标为 conceptual baseline。

## 11. 结果目录、停止和完整性

`results/akr_closing/<RUN_ID>/` 下：

| 路径/文件 | 含义 |
|---|---|
| `EXECUTION_IDENTITY.json`, `RUN.json` | 固定代码、环境、配置、模型revision和源文件 |
| `splits/`, `EPISODE.json` | 真实事件/recording划分、每类支持和实验身份 |
| `features/`, `support_gradients/` | 可恢复缓存及成本/target-token旁文件 |
| `items/`, `manifest.json` | 原子逐项预测、expected IDs、内容hash |
| `LOCK.json`, `GATE.json` | 参数锁和continuous确认判断 |
| `E1/`, `E7/`, `E8/`, `E6/` | 各自独立的完整率、结果、状态 |
| `COMPLETE_METRICS.json`, `complete_metrics.csv/.tex` | 只包含完整结果的表格 |
| `COVERAGE_ALL.json`, `TASK_STATUS.json`, `FINAL_STATUS.json` | 缺项、阻塞、停止与任务完成状态 |

程序只标记完整结果；missing 不填0，失败实验完成也可为 completed。
未通过 gate 的 test 无准确率；不要把“完成全部允许阶段”误解成“所有候选方法成功”。

`Ctrl+C`/`SIGTERM` 请求安全停止。原子预测完成或完整优化步后退出；对于其拥有的旧脚本子进程，
wrapper 会转发停止信号、等待退出并回收，只终止自身子进程，不扫杀其他 GPU 工作。
必须等退出完成再启动同一目录的新任务。`kill -9` 只能保留已完成原子checkpoint，未保存工作需重做。

```bash
.venv/bin/python scripts/run_akr_closing.py --run-id akr_closing_v1 --report-only
```

该命令不依赖模型或数据存在，可在另一台机器汇总已拷贝产物。
如代码、配置、profile、环境、源数据哈希改变，同一 RUN_ID 拒绝续跑；创建新的名字并写明变动，
不要删 identity 文件来绕过保护。

## 12. 本次验证的准确边界

已做 addon CPU 单元/合成集成测试、语法/CLI检查，以及在独立临时 Git 工作树中应用补丁并重跑测试。
合成 backend 只用于软件正确性，不是科学 benchmark。
没有获得这次远端写权限，没有执行模型GPU路径，没有新训练权重，没有新实验指标。
所有已有19项仓库测试的最终兼容性，应由执行 agent 在完整repo/原环境中再次运行；
本地只有新增测试与其前置测试，没有把未执行的原仓库全测试算作通过。
