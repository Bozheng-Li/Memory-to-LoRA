# 已有现象、解决办法与本项目动机边界

> 2026-10-29 整理说明：本文保留历史文献证据及旧方案勘误，不是当前方法方案。当前唯一正文为 [Agent 自主管理局部与全局记忆](AGENT_MANAGED_LOCAL_GLOBAL_FLOW_20261029.md)。下文对学习分组、评分器和 PACT 的建议属于当时的候选讨论。

日期：2026-10-28。所有实验数字均来自所引论文，不是本地复现。来源版本、阅读深度及正文捕获见 [SOURCE_INDEX.md](../research/method_motivation_20261028/SOURCE_INDEX.md)。

## 1. 核心判断

**现象有依据，但最初的大部分方法元素已有直接近邻。** 现有证据足以支持研究增量记忆中的保留、更新和干扰；不足以支持“我们首次发现参数记忆干扰”“参数写入不可逆”“混合通道＋GRPO 即创新”。

值得优先验证的缺口是一个具体机制问题：在冻结的文档到参数编译器下，把哪些事实共同编码为一个可替换单元，会不会改变后续更新时的有效信息损失、过期信息影响和重编译成本？这里的“会不会”尚无当前 checkpoint 的实测答案。参数分区和局部编辑本身也已有研究，不能只靠换研究对象宣称创新。

## 2. 现象一：写入无关参数记忆，可能降低原有问答表现

**直接证据。** [D2L，附录 C.4/Table 8](https://arxiv.org/html/2602.15902v1) 在 Gemma-2-2B 设置中，将 SQuAD 配套文档替换为简单 assistant prompt 或约 4K token 的无关书籍章节，再测试原问题。

| 作者报告的方法 | assistant | distracting |
| --- | ---: | ---: |
| Base model w/ replaced context | 0.201 | 0.175 |
| D2L | 0.096 | 0.126 |
| CD，10 generated queries | 0.211 | 0.203 |

**观察与解释要分开。** 退化是观测；“训练问题总与文档相关，造成相关性先验”是作者的解释假设。论文建议加入无关问题训练或知识保留约束，并未在该表证明这些办法已经解决问题。

**已有办法及对项目启示。** 保留约束、replay、按作用域激活记忆都有先例，见后文 WISE/MEMOIR/Cortex。我们的 pilot 必须测无关问题，并加入 base/shared-only 控制。该实验不是 Qwen3-4B 本地结果，也不是新旧事实更新实验。

## 3. 现象二：容量、模块选择和组合存在不同瓶颈

[Understanding LoRA as Knowledge Memory](https://arxiv.org/html/2603.01097v3)，§4、§6 Q8–Q11 给出三类不同结果：

1. 固定 rank 的单模块随知识量增加出现容量限制；提高 rank 有益，但参数效率并非始终更好。
2. 同参数预算、oracle routing 下，多个小模块能优于单个大模块；实际 embedding routing 有明显差距，甚至不如单模块。
3. CAT 的严重退化很大程度来自 **scale mismatch**；正确缩放能恢复大部分差距。TIES 是重要合并对照。即便使用 oracle 来源和 TIES，组合 1–5 个模块时仍观察到随模块数增加的下降。

**已有办法。** 模块路由、scale 校准、TIES、参数＋文本混合读取已经存在。

**不能推出的结论。** 不可把任意下降都归因于“语义串扰”；不可宣称此前没人测多模块干扰；也没有证据给当前 D2L checkpoint 预设 `accuracy = a − b log K`。这篇论文主要研究训练型 LoRA，机制能否迁移需要本地对照。

**对本项目的约束。** 分清单块编译失败、错误路由、合并 scale、跨块推理和真正组合干扰。若修正 scale、共享项和缓存后问题消失，就不能把工程错误包装成科学现象。

## 4. 现象三：长期递归更新可能衰减早期信息，已有专门的参数维护方法

### RPMem：任务训练的循环门控

[RPMem §2.3、§3.3/Table 2](https://arxiv.org/html/2609.23466v2) 把会话编码到统一 latent，通过 coordinate-wise recurrent gate 整合，再解码为 backbone 对应的 LoRA。其 Qwen3-8B/PERMA consolidation 消融为：

| 规则 | 作者报告 Avg. |
| --- | ---: |
| Rank concatenation | 13.84 |
| Latent averaging | 45.52 |
| Latest Session | 53.56 |
| Factor averaging | 54.21 |
| Learned gate | 85.52 |

这些规则的存储宽度、历史处理方式与 scale 并不构成全部严格匹配的单变量实验，不能直接用它证明“同 rank 学习机制必然占优”。它已经覆盖“跨会话参数记忆＋学习如何整合”。

### DPPM：冻结 D2L 上联合保留与更新，是最直接的近邻

[DPPM §3、§4.4](https://arxiv.org/html/2610.11776v1) 冻结 D2L compiler/decoder/backbone，仅训练 Evidence pooling、ordered Delta update 与 fusion/calibration。它已经把“直接保留历史证据”和“顺序修订”分开处理。

在 32 个干扰段之后，作者报告归一化 latent sensitivity：PersonaMem-v2 为 DPPM 0.970、RPMem 0.268；PrefEval 为 0.667、0.116。**这是 latent sensitivity，不是事实召回率或 QA 准确率。**

Table 4 的两数据集平均：CE 70.51、CE＋GRPO 69.93、CE＋OPD(64) 70.80；OPD(64) 同时改了 teacher prompt，不能当作纯粹单因素增益。GRPO 的 correctness reward 用于采样答案动作，也不能等同于我们的分组动作策略。

**启示。** “冻结 D2L”“保留＋更新”“加入 GRPO”都已有先例。若研究只是再学一个整合门控，差异很弱。新的切口需要落到不同的决策单位与可验证机制，例如事实到可替换编译块的组织，以及由组织引起的更新损失。

## 5. 现象四：文本记忆也会遗漏、损坏和难以读出，已有验证与自修复

### TrustMem：不能把写入正确性当作默认成立

[TrustMem §3、§4.4–4.5](https://arxiv.org/html/2606.25161v1) 用 coverage、preservation、faithfulness verifier 检查同一状态下的候选转换，再用 Transition-Ranked GRPO 训练。

其 463-instance 验证集、GPT-4o-mini 审查下，TrustMem 自身 omission/corruption/hallucination 错误率分别为 8.08%/0.14%/0.01%。这些不是 baseline 的错误率。其综合分数 66.3，去 verification reward 为 58.7，PRM-style GRPO 为 60.3。

这说明记忆转换的忠实性是需要测量的现实问题，但**加 verifier 或局部转换奖励已不是新机制**。作者训练成本约 32 H100、两天，也不能直接视为本地轻量可用方案。

### MemDream：自生成问题再修复记忆也已有近邻

[MemDream §3、§4.3](https://arxiv.org/html/2609.34545v1) 从 memory graph 生成 grounded probes，诊断 buried/scattered/redundant/noisy，执行 extract/synthesize/merge/soft decay，并以 GRPO 学修复。

LoCoMo、GPT-4.1 reader 下，作者报告 F1 为 55.14；无 Dreamer 为 52.09；无 Consolidator GRPO 为 52.98；w/o Decay 的硬剪除配置为 41.04。完整离线维护周期约 22 分钟。

**启示。** “自测→定位失败→修复”“为未来查询做维护”都不能单独主张新颖。仅从当前 memory 生成的问题无法覆盖所有已经遗漏的源事实；若保留源记录来审查，源记录也属于记忆预算。

## 6. 现象五：有记忆或分数提高，不等于实际利用了记忆语义

[Large Language Model Agents Are Not Always Faithful Self-Evolvers §3–5](https://arxiv.org/html/2601.22436v3) 对 raw/condensed experience 进行 Empty、Irrelevant、Corrupt、Filler 等干预。其设置中，raw 的语义扰动往往影响较大，condensed 的语义扰动影响常较小；ReasoningBank 的 condensed-only 对照也有此现象。

**已有办法/评测启示。** 对语义做干预，比只做“开/关 memory”更能检查机制。我们需要内容正确、内容换错、shared-only、无记忆的对照。

**适用边界。** 结果只支持论文所测框架与任务，不证明文本记忆普遍无效，也不证明参数记忆更好。不能把它当成使用 D2L 的充分理由。

## 7. 现象六：混合介质和反事实路由已经很接近初始方案

| 工作 | 已有机制 | 对初始设想的影响 |
| --- | --- | --- |
| [Memory-R1](https://arxiv.org/html/2508.19828v5) | PPO/GRPO 学 ADD/UPDATE/DELETE/NOOP 等文本记忆动作 | 动作式记忆管理＋RL 已有 |
| [UniMem，2607.26017](https://arxiv.org/html/2607.26017v1) | 已知任务走 procedural/参数记忆，未知/稀疏经验进 episodic buffer，聚类后巩固，可扩容 | 参数与情景记忆混合、后续巩固已有；不等于固定预算 D2L 逐事实维护 |
| [MemoryAthena](https://arxiv.org/html/2609.25853v3) | 冻结 direct/generated memory 路径，用相对 direct 路径的未来 token likelihood 反事实收益训练小 router，条件接纳并 bounded interpolation | 冻结后端＋预测边际收益路由已存在；对象是推理 residual 路径，不是跨会话源记录的更新 |
| [Towards Unified Memory Adaptation…](https://neurips.cc/virtual/2026/loc/sydney/poster/155155) | 官方摘要明确 textual/latent/parametric 三路径 | “首次统一三通路”不可主张；全文算法未核验，不能推测其没有路由 |
| [Cortex](https://arxiv.org/pdf/2610.20399) | prospective queries、context teacher→无上下文 student 蒸馏、旧 query replay、non-target retention | 为未来用途写入参数并保护非目标能力已有；本轮仅审摘要/导论 |

[Harness the Memory §5.2–5.3](https://arxiv.org/html/2608.15008v1) 还表明，介质和 retrieval k 的收益依任务变化。例如其 ALFWorld M7 从 k=1 的 32.1% TSR 降到 k=5 的约 25%，no-memory 为 22.4%；LoCoMo 上增加检索则可能有益。这不是 D2L 实验，也不是普适定律。

**启示。** 混合通路是合理基线，不能自动升格为创新。静态“数字进表、语义进参数”的规则也必须先测：当前没有本地证据证明 D2L 对某类值必然失败。

## 8. 现象七：更新后旧知识的影响确有依据，但不能声称整个领域没有办法

### 对 selective consolidation 的失败诊断

[Memory Depth, Not Memory Access §7](https://arxiv.org/html/2606.26806v1) 在 Memora stale-memory 诊断中报告：frozen 的 forgetting absence 为 91/222，EVAF 为 95/222，McNemar p=0.57；high-mutation slice 为 49/126 对 52/126，p=0.36。作者明确这不构成外部验证胜利，现有 EVAF 没有解决 delete/update validity。

这是一个具体方法的边界证据。其主要实验是小模型和合成环境，不能据此断言当前 4B 自然对话必然出现同样失败，更不能宣称“参数更新从未解决”。

### 为什么文本补丁未必足够

[JuICE/Taming Knowledge Conflicts §3](https://arxiv.org/html/2503.10996v2) 研究 pretrained parametric knowledge 与 contextual knowledge 冲突，发现干预效果依冲突上下文变化，单独有效的干预不一定能简单叠加。它不能证明我们的 LoRA 会压过正确文本，但足以提醒：**不能把“新值已放进 prompt”当作行为已更新的证明。**

LoRA 审查的混合读取表也有这种谨慎需求：Qwen3-8B/QuALITY 上，RAG 为 64.53，SingleLoRA＋RAG 为 44.10，MultiTop3＋RAG 为 64.31；ICL 为 72.89，MultiTop3＋ICL 为 76.26（作者 Table 2）。这是其配置间比较，不是严格隔离其他变量后的普适因果效应。

### 更新局部性已有成熟研究

- [SERAC](https://proceedings.mlr.press/v162/mitchell22a/mitchell22a.pdf) 存 explicit edit memory，用 scope classifier 决定是否由 counterfactual model 覆盖 base 输出。作用域、局部更新、无关能力保留早有研究。
- [WISE §2.3](https://arxiv.org/html/2405.14768v3) 采用 main/side memory、参数随机子空间 sharding、merging 和作用域路由。
- [MEMOIR §3](https://arxiv.org/html/2506.07899v1) 用样本依赖 sparse activation mask 限制写入的参数列，并在推理时激活相关编辑、抑制无关 memory。

**对候选框架的限制。** “拆成模块就能少干扰”“隔离更新范围”“只对相关查询开记忆”都不是我们的新颖点。需要证明的更窄差异是：在冻结编译器的事实分组与整块替换接口下，是否存在可学习的当前读取/未来重写权衡，而且简单变化率规则、普通分块和全量重编译不能同样解决。

## 9. 对 refine-logs/PACT 草稿的必要修正

以下针对旧文档中的主张，不改写其原文，也不把 CPU 模拟结果当作模型实测。

| 旧表述/推断 | 本轮核查后的修正 |
| --- | --- |
| D2L 只处理单文档，没有多块组合 | §3.2 明确分块并沿 rank 拼接，训练也考虑 compositionality |
| 2048 token 是结构上限 | 是相关训练/实验配置，不是由架构证明的硬上限；实际可用长度需测 |
| 参数写入不可逆，文本写入可逆 | 有损不等于不可替换；保留预算内源记录时可重编译。源信息已丢失时，文本摘要同样无法恢复 |
| 没有人测多 LoRA 并存干扰 | S02 Q10/Q11 已有直接研究；冻结 D2L 的特定机制仍可另测 |
| 串扰只由内容重叠决定，不取决于槽数 | scale、路由、模块数、信息量、查询类型都可能影响；应做受控因素比较 |
| 必有 `a − b log K` 容量律 | 目前是猜测，没有本地拟合、置信区间或理论支撑；不预设曲线 |
| ID/数字几乎必然无法保真 | 本地没有分类实测；一个单跳例子也不足以证明普遍可靠或普遍失败 |
| 参数/字面/丢弃路由＋GRPO 是主要新颖点 | 混合路径、记忆 RL、反事实路径选择都有近邻；仅换动作空间的差异较弱 |
| 学习策略优于所有固定策略 | 只能比较预先明确、认真调优的实际基线；“所有固定策略”包含不合理的全称主张 |
| CPU 假设模拟器证明真实信道与 RL 必要性 | 其字面保真与串扰由人为参数规定；规则和 GRPO 均达 1.0，不能支持真实模型现象或 RL 独有贡献 |
| 探针势函数加到每步 advantage 就自动保持最优策略 | 标准 potential-based shaping 结论要求正确的转移奖励、折扣、终止边界和回报计算；不能直接替代对具体更新式的证明 |
| GRPO 天然最适合延迟记忆奖励 | 是优化选择而非任务定理；DPPM 的结果已说明奖励训练并非自动提高平均成绩 |

## 10. 动机应如何落脚

可被文献支持的逻辑链是：

1. 参数记忆能减少读取时的历史文本输入，但有容量、组合和无关影响问题。
2. 跨会话维护已有有效门控和双路径方法，不能再把“学更新”本身当新贡献。
3. 在可替换的编译块接口下，一条事实的改变需要重写包含它的整块；分组可能同时影响读取质量和更新时有效旧信息的保留。
4. 第 3 点中的实际附带损失及其可预测性，仍需在本地冻结 compiler/reader 上验证，不能由前两点直接推出。
5. 若它成立，候选研究对象是**面向更新的事实分组与重编译联合决策**；若简单规则或全量重编译已足够，就收敛到简单方案，不强行训练 RL。

历史候选框架见 [旧 METHOD_FRAMEWORK.md](../research/archive/plan-history/agent-managed-focus-20261029/research/method_motivation_20261028/METHOD_FRAMEWORK.md)，验证顺序和否定条件见 [旧 PILOT_PROTOCOL.md](../research/archive/plan-history/agent-managed-focus-20261029/research/method_motivation_20261028/PILOT_PROTOCOL.md)。
