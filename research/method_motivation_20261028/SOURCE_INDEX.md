# 来源与阅读范围

本轮截止日：2026-10-28。下表中的正文捕获不等于逐页精读；阅读范围列说明实际用来核查论点的部分。抓取 UTC 时间保存在各份来源文件头部。论文数值均为作者报告，本轮未复现。

| ID | 原始来源 | 本地捕获 | 本轮核查范围 |
| --- | --- | --- | --- |
| S01 | [Doc-to-LoRA: Learning to Instantly Internalize Contexts](https://arxiv.org/html/2602.15902v1) | [d2l.md](sources/d2l.md) | §3.2 块组合、训练设置、附录 C.4/Table 8 干扰；不把 Gemma 实验迁移成 Qwen 实测 |
| S02 | [Understanding LoRA as Knowledge Memory: An Empirical Analysis](https://arxiv.org/html/2603.01097v3) | [lora_audit.md](sources/lora_audit.md) | 容量、§6 Q8–Q11、§7 混合读取、相关附录；是训练型 LoRA，不是冻结 D2L |
| S03 | [TrustMem: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory](https://arxiv.org/html/2606.25161v1) | [trustmem.md](sources/trustmem.md) | 转换验证、TR-GRPO、§4 消融与错误审查、成本 |
| S04 | [Remember Before You’re Asked: MemDream for Self-Probing Memory Evolution](https://arxiv.org/html/2609.34545v1) | [memdream.md](sources/memdream.md) | §3 自探针与图修复、§4.3 消融、离线维护成本 |
| S05 | [Large Language Model Agents Are Not Always Faithful Self-Evolvers](https://arxiv.org/html/2601.22436v3) | [faithfulness.md](sources/faithfulness.md) | §3–5 对经验语义的干预、condensed-only 对照及解释边界 |
| S06 | [RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents](https://arxiv.org/html/2609.23466v2) | [rpmem.md](sources/rpmem.md) | §2.3 循环门控、§3.3/Table 2 合并对照、成本与 backbone 解耦 |
| S07 | [DPPM: Dual-Path Parametric Memory for Personalized Language Models](https://arxiv.org/html/2610.11776v1) | [dppm.md](sources/dppm.md) | 冻结 D2L 上的 Evidence/Delta/fusion、§4.4 latent retention、Table 4 目标函数 |
| S08 | [UniMem: Complementary Episodic-to-Parametric Memory for Boundary-Agnostic Task Streams](https://arxiv.org/html/2607.26017v1) | [unimem_stream.md](sources/unimem_stream.md) | §3.2–3.3 任务路由、episodic buffer、聚类与参数巩固；与其他同名工作区分 |
| S09 | [Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](https://arxiv.org/html/2508.19828v5) | [mem_r1.md](sources/mem_r1.md) | ADD/UPDATE/DELETE/NOOP 与训练/答题协议相关段落 |
| S10 | [Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents](https://arxiv.org/html/2608.15008v1) | [harness.md](sources/harness.md) | §3 harness、§5.2–5.3 介质/检索规模差异；不视为冻结 D2L 评测 |
| S11 | [Adaptive Routing over Latent and Generated Memories](https://arxiv.org/html/2609.25853v3)（MemoryAthena） | [adaptive_routing.md](sources/adaptive_routing.md) | §3.2–3.3 与附录 B/C：反事实 likelihood 路由、冻结路径、bounded interpolation |
| S12 | [Memory Depth, Not Memory Access: Selective Parametric Consolidation for Long-Running Language Agents](https://arxiv.org/html/2606.26806v1) | [depth.md](sources/depth.md) | selective consolidation、§7 Memora 失效诊断、§9 限制 |
| S13 | [Taming Knowledge Conflicts in Language Models](https://arxiv.org/html/2503.10996v2)（JuICE） | [juice.md](sources/juice.md) | §3.1–3.2 参数/上下文冲突与干预不可简单叠加；不是新写入 LoRA 的实验 |
| S14 | [Cortex: Consolidating Context into Parametric Memory](https://arxiv.org/pdf/2610.20399) | [cortex_pdf.md](sources/cortex_pdf.md) | 摘要、导论：prospective queries、蒸馏、replay/retention；未完成全文机制审查 |
| S15 | [Towards Unified Memory Adaptation for LLM Agents: Textual, Latent, and Parametric Pathways](https://neurips.cc/virtual/2026/loc/sydney/poster/155155) | [unified_abstract.md](sources/unified_abstract.md) | **仅官方摘要**，确认三路径主张；未核全文路由算法 |
| S16 | [WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://arxiv.org/html/2405.14768v3) | [wise.md](sources/wise.md) | 摘要、导论、§2 sharding/merging 与 memory routing、§3 指标；不把“impossible triangle”表述当普适定理 |
| S17 | [MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](https://arxiv.org/html/2506.07899v1) | [memoir.md](sources/memoir.md) | 摘要、导论、§3 样本依赖稀疏掩码与相关编辑激活；未复现其长期编辑结果 |
| S18 | [Memory-Based Model Editing at Scale](https://proceedings.mlr.press/v162/mitchell22a/mitchell22a.pdf)（SERAC） | [serac.md](sources/serac.md) | 摘要、导论、Figure 1、scope classifier/explicit edit memory 机制；未做完整数值审计 |

## 补充线索与限制

- [How LoRA Remembers? A Parametric Memory Law for LLM Finetuning](https://arxiv.org/html/2605.30260v1) 已在上一轮捕获为 `../2026-10-26-lora-memory/novelty/root/memft_2605.30260v1.md`。本轮用它排除“首次研究容量”的泛化主张，不借其容量关系外推当前 checkpoint。
- 针对 grouping/recompilation 的补充检索找到 [ChunkWise LoRA](https://arxiv.org/html/2601.21109v1)，已读摘要和方法：对象是推理序列中的动态 rank/scale 与 KV 调度，本轮不引用其性能数值，也不作为跨会话事实维护的直接证据。
- SplitLoRA 的 OpenReview 提取失败；检索到 [ICLR 2026 官方 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5035a409f5798e188079e236f437e522-Paper-Conference.pdf) 和 [arXiv](https://arxiv.org/html/2505.22370v3)。摘要/检索片段显示是持续学习梯度空间划分；未完成正文阅读，不据此作具体方法差异判断。
- Cortex HTML 返回 404，改读 PDF；arXiv 版本日期与正文日期分别为 2026-10-14 / 2026-10-15，不混作同一字段。
- S08、S15，以及旧方案提及的 Uni-Mem 2610.29671 不是可以混用的一个引用。本轮没有对 2610.29671 的机制追加核验。
- 本包是围绕现象和动机的定向审查，不是系统综述或完整新颖性证明。尤其不能从“检索结果未出现相同表述”推出“此前无人研究”。
