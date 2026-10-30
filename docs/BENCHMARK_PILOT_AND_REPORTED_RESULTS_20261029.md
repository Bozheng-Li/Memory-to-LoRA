# 早期测试选型与论文实际指标

> 2026-10-30：本文中的管理方案和实验结果为当时记录；当前实现已改为 [直接 D2L](DIRECT_D2L_FLOW.md)，旧成绩不代表新链路结果。

核查日期：2026-10-29（Asia/Shanghai）。本轮读取现有论文正文、原始 HTML 表格及服务器已有数据/结果审计；**没有启动新模型推理或训练**。以下论文数字是作者报告，不是我们的复现成绩，也不构成完整、实时的 SOTA 排名。

本轮网络搜索连续返回 HTTP 504，因此主要依据已于 2026-10-28 保存的原文。表格逐行提取保存在 [benchmark_tables_20261029.json](../research/method_motivation_20261028/sources/benchmark_tables_20261029.json)；原 HTML 路径和 SHA256 随文件记录。DPPM 引用固定为已保存的 [v1 正文](../research/method_motivation_20261028/sources/dppm.md)。

## 1. 当前建议

**基座固定 Qwen3-4B-Instruct-2507；首轮先做 MemoryAgentBench 的 FactConsolidation-SH 6K/32K，随后做 PersonaMem-v2 32K-Text；PrefEval 70/300 turns 作为针对长期保留的补充。**

理由是：MAB 可以直接观察更新与旧事实干扰，短版本成本可控；PersonaMem-v2/PrefEval 有 DPPM 的同基座、同冻结 D2L 接口结果，能对齐最近的参数维护近邻。

自然对话外部验证可加 LongMemEval cleaned；完整 Agent 的执行收益再用 MemoryArena 的可运行环境验证。早期不把全部 benchmark 同时铺开。

## 2. 基座为什么选这个具体版本

| 角色 | 首版配置 | 原因 |
| --- | --- | --- |
| 任务 reader/执行 Agent | Qwen3-4B-Instruct-2507 | 兼容服务器现有 D2L checkpoint；与 DPPM 主表一致 |
| 记忆管理调用 | 同一基座，结构化提示，无额外梯度训练 | 满足 Agent 自己决定接纳、巩固和替换的当前方案 |
| D2L | 当前 qwen_4b_d2l/checkpoint-20000，冻结 | 避免先引入编译器训练与后端变更 |
| 精度/解码 | 先固定 bf16 和确定性解码，再核对短答/动作格式 | 属于建议配置，运行前核对 checkpoint/源码支持，不把配置当作已验证结果 |

不能仅因为 8B/新一代模型更强，就把当前编译产物加载到另一个 backbone。它依赖目标模块形状和激活分布。Gemma-2-2B-IT、Mistral-7B-Instruct-v0.2 在 DPPM 中也有匹配 compiler 的实验，可用于以后跨基座验证；本项目尚未因此准备好这些本地配置。

确定性解码用于首轮机制 pilot。MAB 现有部分官方配置采用 temperature=0.7；正式比较应固定所选 revision 的配置，并对所有方法统一。采用不同解码的 pilot 不能直接声称严格复现论文表格。

硬件补充：只读查询显示服务器有一张约 48 GiB GPU 和多张约 24 GiB GPU，但查询时空闲显存仅约 3–6 GiB。机器总显存不等于当前可用额度，本轮没有启动任务或停止其他进程。现有模型是否能运行所需组合，要以正确配置的实际峰值为准。

## 3. Benchmark 的先后顺序

| 阶段 | 数据 | 主要测什么 | 首轮规模/限制 |
| --- | --- | --- | --- |
| 接口与能力检查 | 来源明确的简单检索/短更新样本、官方数据的 evidence reference | 纯文本与单块参数是否能用，短答/评分是否正确 | 属于机制诊断，不称正式 benchmark 全量成绩 |
| 第一项主诊断 | MAB FactConsolidation-SH 6K → 32K | 新事实生效、旧值冲突、流式维护、预算满后的保留 | 每个设置 100 题；独立源历史非常少 |
| 第一项扩展 | MAB FactConsolidation-MH 6K → 32K | 多跳关系在替换/巩固后是否保留 | 先确认全文/evidence reader 具备能力再扩展 |
| 第二项自然任务 | PersonaMem-v2 32K-Text，沿用 DPPM processed split/评测条件 | 用户事实、归属、当前状态、适当遗忘与个性化 | DPPM 使用 5,000 test examples；早期从开发/验证侧固定小样本，不能调测试后再称独立验证 |
| 长期保留补充 | PrefEval 10/70/300 turns | 早期偏好能否跨干扰和局部淘汰继续影响回答 | 10 turns 可作短参考，70/300 更有机会触发维护；DPPM processed test 为 1,620 instances |
| 自然对话外部验证 | LongMemEval cleaned 的 update/temporal 等能力 | 自然历史中的知识更新、时间与弃答 | 原数据 500 题；版本、judge 与历史分组固定，不能与 MAB 的派生 LME 子集混称 |
| Agent 执行验证 | MemoryArena 一个可执行环境起步 | 记忆影响后续工具动作和任务成功率 | 需环境 rollout 与主体能力检查，不能以 QA 分数替代 |

LoCoMo 可以作为方法交集补充，但只有少量公开独立对话，不作为唯一结论来源。LongBench v2/HELMET 更适合静态长上下文，暂不成为这个局部/全局更新框架的首轮主线。

### 服务器现有 MAB 数据的行号

依据 2026-10-29 的 [已有文件审计](../research/method_motivation_20261028/sources/existing_artifact_audit_20261029.json)，当前位于 [resources/data/mab/Conflict_Resolution.parquet](../resources/data/mab/Conflict_Resolution.parquet) 的数据：

| 0-based 行号 | 设置 | 题数 |
| --- | --- | ---: |
| 4 | FactConsolidation-SH 6K | 100 |
| 5 | FactConsolidation-SH 32K | 100 |
| 0 | FactConsolidation-MH 6K | 100 |
| 1 | FactConsolidation-MH 32K | 100 |

文件有 8 行，但只有 4 个完全不同的 context；SH/MH 对应长度共享 context，长度变体也不能自动算成独立新历史。因此它适合机制评测，不适合直接用题级随机划分建立大量独立训练样本。

## 4. 最近的同基座论文：DPPM 的具体值

来源：[DPPM v1，Table 1/2、§4.1、附录 D](https://arxiv.org/html/2610.11776v1)。下表为作者在 **Qwen3-4B-Instruct-2507** 设置中的 Overall accuracy，单位 %。

| 方法 | PersonaMem-v2 Overall | PrefEval Overall |
| --- | ---: | ---: |
| No Context | 28.54 | 37.72 |
| Rolling Summary | 29.16 | 40.49 |
| Mem0 | 30.46 | 75.49 |
| RAG，BGE-M3 | 32.50 | 57.96 |
| Full Context | 32.96 | 45.31 |
| Direct D2L | 24.86 | 30.06 |
| RPMem，DPPM 中同冻结 D2L encoder 的实现 | 45.73 | 82.47 |
| DPPM | **54.22** | **86.79** |

相关细分结果：

| 方法 | Persona Self | Persona Current | PrefEval 10 turns | 70 turns | 300 turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct D2L | 25.15 | 24.92 | 40.19 | 25.56 | 24.44 |
| RPMem | 49.46 | 53.38 | 79.56 | 83.81 | 84.04 |
| DPPM | **54.69** | **58.38** | **84.52** | **88.15** | **87.70** |

解释与限制：

- PersonaMem-v2 Overall 平均其 processed test 的 5,000 个问题；Self/Current 按论文给定 annotation 筛选。原文将对应过滤写为 `who=self` 和 `updated=False`，复现时以作者实际评测代码为准，不能自行按名称重构子集。
- PrefEval Overall 平均其 1,620 个测试实例，各间隔覆盖三种 preference forms；作者按主题划分 16 个训练主题与 4 个 held-out 主题。
- RPMem/DPPM 主比较使用 seeds 42–46；Table 2 中 DPPM 对应 `54.22±0.48 / 86.79±0.68`，RPMem 对应 `45.73±0.24 / 82.47±0.58`。不将这些 ± 擅自解释为跨用户置信区间。
- 作者对每个 dataset 分别训练 5 epochs；我们的首版是提示决策、无梯度训练。它们是强参考结果，不是零训练方案天然应达到的通过线。
- 这里 RPMem 的 45.73/82.47 是 **DPPM 论文的匹配实现**，不是 RPMem 原论文 Qwen3-8B 主表；两者不可混用。
- 同样 backbone 不等于同样物理存储、source budget、写入成本或训练数据。正式复现需要进一步对齐这些条件。
- Direct D2L 较低说明其该配置存在改进空间，不证明任何局部/全局机制都会有效。

### DPPM 的成本数字，不能当成本项目端到端延迟

作者报告 DPPM 的 answer-forward 为 **46.14 ms**，比 RPMem 多 2.43 ms；测试为一张 A800、batch=1，包含预热与少量计时 forward。它不是完整管理→编译→安装→解码的总延迟。

在作者选定的一道最长历史问题、较长历史配置上，Full Context 输入为 139,319 tokens，峰值为 18,245.97 MiB；DPPM 为 7,762.32 MiB。它是特定样本/读取配置的比较，不是全数据平均，也不包含我们的管理决策与新旧参数暂存开销。

## 5. MAB 的具体值：分清短版本和主表

来源：[MemoryAgentBench 论文](https://openreview.net/forum?id=DT7JyQC3MR)，保存 HTML 的 Table 5（`S4.T5`）。该表在 FactConsolidation 上报告 accuracy %：

| 模型 | SH 6K | SH 32K | MH 6K | MH 32K |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o | 92.0 | 88.0 | 28.0 | 10.0 |
| o4-mini（原表 O4-mini） | 100.0 | 61.0 | 80.0 | 14.0 |

这正是先测 SH 的理由：多跳和长度增加可以显著改变任务难度，不能只因分数低就归因于记忆替换。该表不是 Qwen3-4B 的成绩，也不是我们的目标分数线。

主 Table 3 是另一组标准主表设置，不能与上述 6K/32K 表直接拼接：

| 方法/reader | AR Avg. | FC-SH | FC-MH | Overall |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o-mini，长上下文 | 49.2 | 45.0 | 5.0 | 42.2 |
| BM25＋GPT-4o-mini | 60.5 | 48.0 | 3.0 | 41.5 |
| HippoRAG-v2＋GPT-4o-mini | 65.1 | 54.0 | 5.0 | 41.6 |
| Mem0＋GPT-4o-mini | 32.6 | 18.0 | 2.0 | 21.1 |
| GPT-5-mini，长上下文 | 74.4 | 78.0 | 28.0 | 60.6 |

主表未特别指定的 RAG/memory 方法使用 GPT-4o-mini。表中 `Qwen3-Embedding-4B` 是 embedding retriever 名称，不代表 reader 是 Qwen3-4B。当前论文版本称 SF/Selective Forgetting 的地方，与项目代码的 CR 命名也需按 revision 对齐。

### MAB 官方指标

需要按子任务使用官方 scorer：相关文档 QA/FactConsolidation 使用 Substring EM 类评分；分类/部分推理任务用 exact match，推荐用 Recall@5，其他自然任务可能依赖 judge。

我们的首轮重点是保留原生质量分，同时额外检查“新值命中但仍把旧值当当前事实”的错误。不能仅凭回答包含正确字符串就断言冲突已解决。旧 pilot 的自写 `MAB-style subem` 也不能未经核对就称为官方评分器。

## 6. LongMemEval 的具体值与口径

来源：[LongMemEval 原论文](https://arxiv.org/abs/2410.10813)，Table 3。以下为 **LongMemEval-M、value=round、Top-10**，QA 结果为论文 judge 下的正确率，转换为 %；retrieval 指标与 QA 指标分开：

| 检索 key | Recall@10 | GPT-4o QA | Llama-3.1-70B QA | Llama-3.1-8B QA |
| --- | ---: | ---: | ---: | ---: |
| K=V，原始记录 | 69.2 | 67.0 | 62.4 | 53.4 |
| K=V+fact，事实扩展 | 78.4 | 72.0 | 68.2 | 57.2 |

这些是原论文版本，不是 cleaned 数据上的 Qwen3-4B 成绩。原论文 Table 8 中 Qwen2.5-7B 在 LongMemEval-S 的 LC Direct 为 12.8%，round RAG 的 K=V 为 45.2%，K=V+fact 为 46.2%；也说明主体能力与读取方法会共同改变分数。

早期若用 oracle/evidence sessions 做能力诊断，必须标注 gold evidence 只给参考条件，不作为正式记忆 writer 输入。正式自然记忆评测固定 cleaned revision、judge 模型/prompt、问题类别和独立历史分组。

## 7. 真正 Agent 环境：MemoryArena 的具体值

来源：[MemoryArena 论文](https://arxiv.org/abs/2602.16313)，Table 3。以下取原表 `All Task Avg SR`，将 0–1 转为 %。memory/RAG 行的任务执行模型为 **GPT-5.1-mini**：

| 配置 | 平均任务成功率 SR |
| --- | ---: |
| GPT-5.1-mini＋Long Context | 16 |
| GPT-5.1-mini＋Letta | 14 |
| GPT-5.1-mini＋Mem0 | 13 |
| GPT-5.1-mini＋Reasoning Bank | 12 |
| GPT-5.1-mini＋BM25 | 17 |
| GPT-5.1-mini＋Text-Embedding-3-Small | 20 |

它还报告 PS（进度）和 sPS。SR 不是 QA accuracy；低 SR 中可能有环境执行、规划、工具和记忆共同造成的困难。尤其不能据此期待当前 4B/D2L 直接达到这些闭源模型成绩，也不能把其 adapter 直接挂到 GPT-5.1-mini。

## 8. 服务器现有 pilot：只读审查，不是新框架的结果

读取 `/home/libaizheng/Lora-Grpo-memory/research/method_motivation_20261028/ARTIFACT_AUDIT_20261029.json` 和关联脚本/结果文件得到：

| 已有 CR 条件 | n | 自写 SubEM % |
| --- | ---: | ---: |
| no_memory | 25 | 0 |
| text_last2k | 25 | 12 |
| text_bm25_top4 | 25 | 8 |
| full_context | 25 | 0 |
| d2l_first2k / last2k / bm25_2k，各条件 | 各 25 | 各 0 |

这些记录对应旧脚本的 row 0，即 **MH 6K**。新局部池/全局联合维护尚未在这些记录中运行；完整文本参考也很低，不能拿它们直接证明新框架无效。

另一个人工短码检索 pilot 的既有记录为：text_full 20/20；d2l_full 2/20；d2l_needle 20/20。短码稀释记录中，D2L 256-token 条件 9/12，512-token 为 2/12，1024/2048-token 各 1/12；各条件用的是不同生成样本，不能把它们当成严格配对因果曲线。这些只能提示需要检查任务适配、输入构成与工程设置。

已有结果缺少完整、不可变的运行 manifest；当前脚本/提示与旧输出可能已不同，源码还可见 v2 结果保存失败日志。正式实验需固定原生 scorer、实际 checkpoint/module/scale/reset、解码长度与提示、输入 token 和来源版本。我们本轮没有修改旧脚本或重新运行推理。

## 9. 首轮实际应该报告什么

主质量：MAB 官方分项分数，PersonaMem-v2/PrefEval 的原生 accuracy；后续自然对话 judge accuracy 和 Agent SR 分表。

针对本方法另报：

1. **退出块后保留率**：在前后都应成立的事实中，退出前可答的题退出后还能答多少，同时报告全部有效事实的无条件正确率和样本数，避免条件筛选掩盖总体质量。
2. **过期值误用率**：仍将过期值主张为当前值的比例；合理提及历史旧值不算错误。
3. **新信息接纳与多跳保留**：新观察是否可用，跨块关键关系是否丢失。
4. **端到端成本**：管理调用、source/metadata token、D2L 编译、安装、KV 重建、解码；写入/读取延迟、完整任务时间、持久字节及峰值显存分别报告。

首轮预算可沿用四个局部 rank-8＋rank-32 全局内容，共 64 内容宽度；与八个局部 rank-8 比较，共享项各只加入一次。再比较全局/局部分别更新和联合更新。相同 rank 还需要源记录和管理调用的完整成本对照。

最先做的检查是 SH 6K 的 no-memory、全文/证据参考、合理文本记忆/检索和冻结 D2L。确认 reader/评分流程能区分这些条件后，再将局部替换与全局巩固接入正式比较。不在本轮提前给新方法承诺分数，也不以某篇不同基座论文的值作为硬性通过线。
