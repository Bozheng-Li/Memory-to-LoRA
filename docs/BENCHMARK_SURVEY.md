# 长上下文与 Agent 记忆：Benchmark 和 Baseline 调研

> 日期：2026-10-28 · 第一轮公开资料核查  
> 目的：先根据已有顶会工作确定实验对象，再决定方法；D2L 是候选写入后端，不预设为优化对象。  
> 核查依据：论文正文、作者官方仓库、ACL Anthology 元数据；本地材料保存在 [research/benchmark_sources](../research/benchmark_sources)。  
> 范围：这是有来源的候选与使用关系梳理，不是全领域论文的频次统计；未运行任何新 benchmark。

## 1. 三类实验回答不同问题

| 类别 | 基本流程 | 能支持的结论 | 不能单独支持的结论 |
| --- | --- | --- | --- |
| 静态长上下文 | 给长文本和问题，输出答案 | 长文本检索、理解、推理或压缩质量 | 持续写入、事实维护、跨任务经验复用 |
| 增量/跨会话记忆 | 分批读历史，随后回答问题 | 历史保留、时间关系、知识更新、未知信息处理 | Agent 真正依靠记忆执行工具任务 |
| 交互 Agent 记忆 | 执行一系列相互依赖的任务，记忆影响后续动作 | 记忆对任务成功、后续决策和成本的作用 | 不应将其成功率与静态 QA accuracy 混为一谈 |

关键是先选择要验证的记忆能力。顶会标签有助于选择可比较的协议，但不是选择科学问题的充分依据。

## 2. 已核查的 Benchmark

### 2.1 记忆方向：优先候选

| Benchmark | 会议/来源 | 数据与能力 | 对本项目的价值 | 主要限制 |
| --- | --- | --- | --- | --- |
| **MemoryAgentBench** | **ICLR 2026**；[论文/OpenReview](https://openreview.net/forum?id=DT7JyQC3MR)、[官方仓库](https://github.com/HUST-AI-HYZ/MemoryAgentBench) | 逐块写入历史、同一长历史多次查询；AR、TTL、LRU、CR 四类能力 | 最直接检验增量写入、固定预算、多问题复用与更新冲突 | 多数 AR/CR 任务独立长序列很少，适合作评测，不宜直接当作大量独立训练历史；多评分器 |
| **LongMemEval** | **ICLR 2025**；[论文](https://arxiv.org/abs/2410.10813)、[官方仓库](https://github.com/xiaowu0162/LongMemEval) | 500 个评测问题；信息提取、多会话推理、知识更新、时间推理、弃答 | 适合自然对话记忆、更新与旧信息处理；已有强检索设计对照 | 原始协议按问题构建历史，不能假设所有题共享同一用户历史；数据版本须固定 |
| **LoCoMo** | **ACL 2024**；[官方仓库](https://github.com/snap-research/locomo) | 当前公开版 10 段长期多会话对话，每段多个 QA；单跳、多跳、时间、开放域、对抗题 | 许多记忆方法共同使用，容易形成方法对比；天然支持同历史多问 | 独立对话只有 10 段；题级数量大不意味着独立历史数量大 |
| **LongMemEval-V2** | 2026 公开工作；会议归属本次未独立核实；[论文](https://arxiv.org/abs/2605.12493)、[官方仓库](https://github.com/xiaowu0162/LongMemEval-V2) | 451 个问题；web/enterprise agent 轨迹；状态、工作流、环境陷阱、前提意识 | 更接近从历史经验中形成可复用记忆 | 主协议仍是轨迹后的 QA；有多模态输入和极长历史，现有文本 D2L 不直接覆盖 |

MemoryAgentBench 中：

- AR = Accurate Retrieval，包含单/多跳文档 QA、LongMemEval 派生任务、EventQA。
- TTL = Test-Time Learning，包含示例驱动的分类和推荐。
- LRU = Long-Range Understanding，包含长摘要和 DetectiveQA；此处不是缓存淘汰算法 Least Recently Used。
- CR = Conflict Resolution，主要是 FactConsolidation 单跳与多跳任务。部分论文版本使用 Selective Forgetting（SF）名称，必须固定论文、代码和数据版本，不能混用命名后假定协议完全相同。

它的“inject once, query multiple times”设计已经直接覆盖写一次、多次查询。因此该设置是我们可以采用的评测约束，不可独立当作新贡献。

**独立序列数是选择该 benchmark 的关键限制。**本次读取论文 Appendix B 的 Table 6，得到以下作者统计：

| 子任务 | 长序列数 | QA 数 | 平均长度（作者 tokenizer） |
| --- | ---: | ---: | ---: |
| SH-Doc QA | 1 | 100 | 197K |
| MH-Doc QA | 1 | 100 | 421K |
| LongMemEval (S*) | 5 | 300 | 355K |
| EventQA | 5 | 500 | 534K |
| FactConsolidation-SH | 1 | 100 | 262K |
| FactConsolidation-MH | 1 | 100 | 262K |

这些是论文对应版本的统计，不是本次对 HF 数据实例的重新计数。100 题不等于 100 个独立写入样本，6K/32K/64K 长度变体也不自动增加独立历史。它可以检验固定历史上的多问题表现；训练写入器、测试未见历史和估计跨历史不确定性，需要另有独立语料与划分。

LongMemEval 官方仓库在 2025 年发布了 cleaned 历史；本轮应优先使用 [cleaned 数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)，并记录具体 revision。原论文的 S 规模约为 115k token/问题，M 规模更大；这些是作者统计，实际实验长度需要用本项目 tokenizer 重算。

### 2.2 静态长上下文：适合主线候选或辅助验证

| Benchmark | 已核会议 | 检验能力 | 在本项目中的建议角色 |
| --- | --- | --- | --- |
| **LongBench** | **ACL 2024**；[论文](https://aclanthology.org/2024.acl-long.172/)、[官方代码](https://github.com/THUDM/LongBench/tree/main/LongBench) | 21 个任务，包含 QA、摘要、few-shot、合成任务与代码补全 | 低成本对照与旧工作兼容；单个 2Wiki 任务不宜代表长期记忆 |
| **LongBench v2** | **ACL 2025**；[Anthology](https://aclanthology.org/2025.acl-long.183/)、[官方仓库](https://github.com/THUDM/LongBench) | 503 个选择题；六类现实任务；8k–2M **words** | 若选长上下文理解路线，可作为自然任务主候选；原生单题协议不证明多问题复用 |
| **RULER** | **COLM 2024**；[论文](https://openreview.net/forum?id=kIoBbc76Sy)、[官方仓库](https://github.com/NVIDIA/RULER) | 13 个任务，覆盖检索、多跳追踪、聚合、QA；长度/难度可控 | 便宜的机制诊断与长度曲线；原始 NIAH 单项不够 |
| **NoLiMa** | **ICML 2025**；[论文](https://arxiv.org/abs/2502.05167)、[官方仓库](https://github.com/adobe-research/NoLiMa) | 降低问题与证据间的字面匹配，检验关联检索 | 防止方法只学字面筛选；不能证明更新与长期维护 |
| **HELMET** | **ICLR 2025**；[论文](https://arxiv.org/abs/2410.02694)、[官方仓库](https://github.com/princeton-nlp/HELMET) | recall、RAG、rerank、citation、long QA、summarization、ICL 七类 | 长上下文能力覆盖更全面；首轮应选择与主问题相关的任务 |
| **∞Bench** | **ACL 2024**；[官方仓库](https://github.com/OpenBMB/InfiniteBench) | 超 100k token 的长文本任务，包含自然与合成任务 | 超长外推验证；不会仅因文本长就成为 Agent 记忆测试 |

LongBench v2 的长度单位为 words，不能写成同数值的 tokens。其会议归属由 ACL 官方 2025 元数据确认；仅看仓库旧 BibTeX 会误以为仍是 2024 预印本。

HELMET 全套预处理数据在官方仓库说明中约 34 GB，不宜为了第一轮判定就全部下载。RULER 当前仓库还包含更新分支，首轮复现要指定原始版本，不把新分支分数与 2024 表格直接拼接。

### 2.3 真正的交互任务：保留一条明确候选

**MemoryArena（ICML 2026）**：

- [论文](https://arxiv.org/abs/2602.16313)、[项目页](https://memoryarena.github.io/)、[官方代码](https://github.com/ZexueHe/MemoryArena)。
- 包含 bundled web shopping、progressive web search、group travel、math & physics 等相互依赖的多会话任务。
- 记忆写入过去交互，后续子任务必须使用过去状态或结果；Agent 动作会影响未来环境。
- 主要指标是任务成功率 SR、进度 PS，以及部分场景的软进度 sPS，同时评测延迟。
- 会议接收信息由 MemoryAgentBench 作者官方仓库明确给出；MemoryArena 当前代码仍注明 preview。

它比对话 QA 更接近完整 Agent 闭环，但环境搭建、rollout 和模型能力要求更高。不能在 4B 主体基本不能执行任务时，把低成功率归因于记忆；应先做无记忆、完整/证据历史参考的主体能力 pilot。

SWE-bench 可以评测补丁正确性，但现成“仓库 + issue → patch”不自动测跨会话记忆。本阶段依照用户的新要求，将 coding 自建场景降为后续候选。

## 3. 别人的方法实际用了哪些数据和对照

| 工作 | 核查到的出处 | 使用的主要实验 | 论文中的主要对照 | 对我们设计的影响 |
| --- | --- | --- | --- | --- |
| **A-MEM** | **NeurIPS 2025**；[论文](https://arxiv.org/abs/2502.12110)、[论文复现代码](https://github.com/WujiangXu/AgenticMemory) | LoCoMo，多种基础模型 | 全历史、ReadAgent、MemoryBank、MemGPT 等 | LoCoMo 是方法对比的重要交集；调用生产库不一定等于复现论文 |
| **MemoryOS** | **EMNLP 2025 主会**；[官方仓库](https://github.com/BAI-LAB/MemoryOS) | LoCoMo，层次化记忆维护 | 作为其他近期记忆论文的强对照 | 简单缓存/摘要不是唯一有竞争力的 baseline |
| **Memory-R1** | [论文](https://arxiv.org/abs/2508.19828)；会议归属本次未核实 | LoCoMo 训练；MSC、LongMemEval 迁移评测 | LoCoMo/RAG、A-MEM、Mem0、MemoryOS、Memory-SFT | PPO/GRPO 学记忆操作已存在；训练预算与回答模型是否更新必须核对 |
| **Mem0** | [论文](https://arxiv.org/abs/2504.19413)、[代码](https://github.com/mem0ai/mem0)；会议归属本次未核实 | LoCoMo；含质量、token 和延迟分析 | full context、不同 chunk/k 的 RAG、A-MEM、MemoryBank、MemGPT、ReadAgent 等 | 要同时比较普通检索、结构化记忆和完整历史；不要只挑单一弱 RAG 配置 |
| **MemoryAgentBench** | **ICLR 2026** | 增量的 AR/TTL/LRU/CR | 长上下文、简单 RAG、embedding RAG、结构化 RAG、Mem0/Cognee/Zep/MIRIX 等 | 可沿用成熟协议；原论文的模型和 chunk 条件并非完全一致 |
| **MEM1** | 官方代码确认 **COLM/NeurIPS 2025 workshop**，本次未确认主会；[论文](https://arxiv.org/abs/2506.15841)、[代码](https://github.com/MIT-MI/MEM1) | 多目标 HotpotQA/NQ 问答、WebShop、在线 Web QA 迁移 | Search-R1、DeepResearcher、A-MEM、同模型压缩 prompt、SFT 等 | 长程 Agent 的 memory/reasoning 联合 RL 已有工作；不能混为文本历史召回 |
| **MemAgent** | [论文](https://arxiv.org/abs/2507.02259)、[代码](https://github.com/BytedTsinghua-SIA/MemAgent)；会议归属本次未核实 | 长 QA、RULER 和长度外推 | 原生长上下文模型、训练后的 memory agent | 已有固定窗口的 RL 流式读写；训练/测试答案 verifier 不同，要核对 |
| **MemGen** | **ICLR 2026**；[论文](https://arxiv.org/abs/2509.24704)、[代码](https://github.com/KANABOON1/MemGen) | ALFWorld、TriviaQA/PopQA、数学、GPQA、KodCode/BigCodeBench | SFT、GRPO、REINFORCE、MemoryBank、ExpeL、AWM、latent computation 等 | latent/parametric 的经验记忆不等于对话事实存储；不能因名称含 memory 就纳入同一主表 |
| **MemoryArena** | **ICML 2026**；[代码](https://github.com/ZexueHe/MemoryArena) | 相互依赖的多会话任务 | long context、Letta、Mem0/Mem0-g、MIRIX、ReasoningBank、BM25、dense RAG、MemoRAG、GraphRAG | 如果声称改善 Agent 后续执行，需要这类任务结果，QA 不足以替代 |

该矩阵反映本轮核查的实验与主要对照，不表示已经成功安装、适配或复现这些方法。

## 4. 必须注意的协议差异

### 4.1 “相同 benchmark”不等于相同评测

1. **LoCoMo 对抗题是否排除。** Memory-R1 正文排除 adversarial 子集；我们若保留，应同时报告全类别结果和可比较子集。
2. **LoCoMo 的公开版与早期版。** 当前仓库为 10 段对话；初始版本曾有 50 段，不能混用独立历史数量。
3. **按问题划分还是按用户/历史划分。** 同历史未见问题衡量 query generalization；未见用户/历史衡量更强的写入泛化。前者不是自动无效，但不能声称后者。
4. **Memory-R1 的比例与数字。** 正文称 1:1:8，但同时列出 152/81/1307；这些数并不严格符合该比例。复现需要核查实际 split manifest，不照抄一句比例。
5. **LongMemEval 数据清洗版本。** cleaned 历史与早期结果不能直接视为同版本比较。
6. **LongMemEval 与 V2。** V2 是不同数据和能力，不是把原 500 题换成更长的原样评测。
7. **MemoryAgentBench 的派生数据。** 它的 LongMemEval/∞Bench 等子集经过重构，不能直接称为原 benchmark 全量结果。

### 4.2 评分器本身会改变结论

- MemoryAgentBench 的部分 AR/CR 用 Substring Exact Match；TTL/DetectiveQA 用 exact match；推荐用 Recall@5；自然记忆与摘要包含 LLM judge。
- SubEM 可能把同时列出旧值和新值的回答算对；若研究冲突消解，要额外检查是否仍主张旧事实，而不只报命中正确实体。
- strict exact match 对格式敏感。官方分数保持原评分器；若提供更宽松解析，只作为额外结果，所有方法同样使用。
- LoCoMo 的 token F1、BLEU-1 与 semantic judge 不等价；LongMemEval 的官方主质量评价依赖 judge。
- 固定 judge 模型版本、prompt、解码和错误重试；不同方法的答案盲评，并抽样检查人工一致性。不能把更换 judge 后的数字直接对照论文旧表。

### 4.3 输入边界与信息泄漏

原始历史和时间戳可以进入写入器。gold answer、evidence、`has_answer`、`answer_session_ids`、LoCoMo QA evidence 或 gold event summary 都不能作为可见历史。

主“写入后查询”协议先生成记忆，随后揭示问题；问答时不将测试问题、答案或 judge 反馈写回同一记忆。若研究在线对话，则按原生协议明确哪些新观察可以触发更新，不同时声称只写一次。

普通 RAG 通常在查询出现后检索，这是一种合法且重要的强 baseline。应标注写入阶段的问题可见性，不因其可见查询就排除对比。

### 4.4 模型、容量和时间账

- D2L 的现有 checkpoint 与 Qwen3-4B 的结构绑定，不能直接加载到任意闭源 reader 或别的基础模型。
- MemoryAgentBench 原论文大量 memory/RAG 方法使用 GPT-4o-mini，同时比较更强长上下文模型；Mem0 等的 chunk 设置也与部分简单 RAG 不同。它们是系统对比，不能直接归因为单一记忆机制。
- 应提供同 reader 的机制对照，以及原生系统配置的外部对比；异构系统的结果分表标注。
- 同最终 rank 不等于同 token 预算；同 token 预算不等于同磁盘字节。记录存储、读入 token、KV、参数驻留、写入调用和每题成本分别报告。
- MemoryAgentBench 数据本身包含 RULER/LongMemEval 派生来源，额外跑这些 benchmark 不自动形成完全独立的泛化证据。

### 4.5 已读到的当前仓库配置与入口

通过 GitHub contents API 取得的配置与入口已保存在 [mab_code](../research/benchmark_sources/mab_code)。它们显示：

- FactConsolidation 的 6K 单跳/多跳配置有 `max_test_samples: 1`、`context_max_length: 6000`、`chunk_size: 4096`；与“100 个独立历史”的假设不符。
- BM25 和 Contriever 配置均为 `retrieve_num: 10`、`temperature: 0.7`。不能用某篇论文的 top-k=1/2 或默认贪心解码来描述所有当前 baseline。
- `main.py` 先为一个 context 初始化并写入，再遍历该 context 的题组，调用 `send_message(..., memorizing=False)`。这与写入后多问的设计一致；各 agent 内部是否修改查询状态仍需单独核查。
- 当前 6K 配置的 chunk 值与论文部分任务的 512 设置不同。实际运行需要固定 commit、配置和命令行覆盖，不能把 README、论文和配置当成同一不可变协议。

数据卡和数据实例的 HF 直连失败，镜像 API 返回 403；没有将未取得的数据写成已经逐条核验。论文数据表、已取得的源码配置与真实下载后的实例计数必须区分。

## 5. 本项目应该准备的 Baseline 层级

### 5.1 必做的基础对照

| 对照 | 解决的问题 | 实现注意 |
| --- | --- | --- |
| No-history / recent window | 模型本来会多少，记忆是否有用 | 同 reader、同当前问题与模板；recent window 的容量固定 |
| Full history | 可读取全部历史的参考质量 | 报告上下文上限与 OOM；它不是数学性能上界 |
| Fixed/rolling summary | 普通内容压缩已经能做到多少 | summary 的调用、更新成本与来源明确 |
| BM25 | 词面检索是否已足够 | 同允许的检索内容与返回 token 预算 |
| Dense RAG | 语义检索强基线 | embedding、chunk、top-k 在开发集校准；按返回 token 预算比较 |
| Timestamp/fact-aware retrieval | 更新与时间题是否只是索引问题 | LongMemEval 已有事实扩展、时间过滤/查询扩展思路 |
| Frozen D2L | 即时参数记忆本身是否提供收益 | 原始写入/分块规则固定；共享项和有效 scale 单独核对 |

reranker 可作为加强版 Dense RAG，调参和额外推理成本必须计入。全文能放下时，使用可复用 prefix cache 的成本对照。

### 5.2 按主场景选择强方法，避免一次堆满

- 对话/增量记忆：优先 **Mem0、A-MEM、MemoryOS**；若方法包含学习写入，则 **Memory-R1/Memory-SFT** 必须重点对比。
- 长上下文读写：优先固定摘要、加强 RAG、**ReadAgent/MemAgent** 类读写系统；其训练模型与本地 reader 的差别需说明。
- 交互经验记忆：优先 **ReasoningBank、Letta/MemGPT、Mem0**，按 MemoryArena 的统一任务框架比较。
- 含 latent/parametric 经验学习时，再考虑 **MemGen**。它不是所有对话事实记忆场景都必须原样复现的 baseline。

这是目标列表；公开 checkpoint、代码能否使用同基座、训练费用与完整协议还需逐项确认。不能提前写成“所有 baseline 已可复现”。

### 5.3 最关键的两因素对照

将“记忆形成/维护策略”与“存储/读取表示”分开：

| 写入策略 | 文本表示 | D2L 参数表示 |
| --- | --- | --- |
| 固定摘要/记录策略 | 固定记录作为 prompt memory | 同一份固定记录经冻结 D2L 编译 |
| 学习的记忆策略 | 学习记录作为 prompt memory | 同一份学习记录经冻结 D2L 编译 |

这样才能判断收益来自写入策略、参数表示，还是两者配合。相同记录控制信息来源，但两种表示的物理成本并不相等，要单独报告质量与成本。

再补充 D2L 的“仅共享项”控制、纯正文写入控制，以及原始基座/含共享项 reader 的结果，排除全局共享适配被误计为内容收益。

如果训练写入策略，主机制对照冻结 answer reader；Memory-R1 完整系统训练多个组件时，另设原生系统表或明确重新训练的组件，不能把 answer reader 的改进归到 memory manager。

## 6. 当前建议：先选能力，再锁正式实验

以下是调研后的候选组合，不是已经决定的主实验。

| 候选研究对象 | 推荐主评测 | 外部验证 | 机制诊断 | 相对代价 |
| --- | --- | --- | --- | --- |
| **增量记忆的保留与冲突/更新** | MemoryAgentBench 的 AR + CR；之后扩展其他能力 | LongMemEval cleaned；LoCoMo 辅助方法对比 | NoLiMa 或 RULER | 中等；最适合现有文本 D2L |
| **静态长上下文的有限预算表示** | LongBench v2 或 HELMET 相关任务 | 另一套自然长上下文任务 | RULER + NoLiMa | 中等到高；需排除 reader 本身推理能力不足 |
| **跨会话 Agent 的经验复用** | MemoryArena 的一个可执行环境起步 | LongMemEval-V2 的轨迹 QA | 受控历史/更新探针 | 高；环境与长 rollout 成本更大 |

当前更倾向第一类，理由是它有顶会原生增量协议、相关强 baseline、多问题与冲突任务，也与本地文本编译器更兼容。MemoryAgentBench 作为正式评测候选，训练历史需另行选择；自然对话泛化用独立数据检验。最终选择前，仍需核查实际发布版的数据实例、标签和训练划分，以及原生 D2L 和强 RAG 的小规模表现。

第一轮不应同时做三类完整主实验。LoCoMo 适合作方法交集，但只有 10 段独立历史，不建议作为唯一训练与唯一最终证据。QuALITY 与本地 2Wiki 保留为表示/API 诊断，不继续充当记忆主场景。

## 7. 选择场景前还缺哪些证据

1. 从候选数据卡及少量实例核对 schema、历史 ID、问题分组、更新时间与可用训练划分；不先下载全套超大数据。
2. 在训练/开发侧做 No-history、full/evidence reference、Dense RAG、frozen D2L pilot，检查现有基座能否回答这些题。
3. 查看失败是否集中于检索、跨会话关系、状态纠正、旧事实干扰、精确值或编译损失，再确定具体方法。
4. 明确方法与 Memory-R1、A-MEM、MemoryOS、MemAgent 等的真实差别；接入 D2L 或增加 GRPO 不是充分贡献。
5. 固定主 benchmark、外部评测、baseline、同模型条件、split、评分器与成本预算，之后再进入实现和训练。

这些核查是下一阶段的任务；本次已完成的是公开证据调研与方案重定向，尚未证明某种 agent 场景或方法有效。
