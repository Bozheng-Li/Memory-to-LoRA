# 项目文档

- [当前方法方案](DIRECT_D2L_FLOW.md)：按 chunk 直接生成 rank-8 LoRA，累积完整历史因子。
- [参数记忆代码高层设计](HIGH_LEVEL_DESIGN.md)：目录、代码及注释规范、独立 YAML、运行流程和核心函数契约。
- [当前实现与验证](IMPLEMENTATION.md)：实现范围、检查结果、运行方法和真实模型验证边界。
- [MAB 数据说明](MAB_DATA_GUIDE.md)：四类数据的分布、一行的结构、SH/MH 和真实更新例子。
- [早期测评选型与论文指标](BENCHMARK_PILOT_AND_REPORTED_RESULTS_20261029.md)：先测什么，以及相关论文成绩的适用范围。
- [Benchmark 与 baseline 调研](BENCHMARK_SURVEY.md)：候选数据与方法的来源和比较。
- [文献证据综述](EVIDENCE_REVIEW.md)：已有现象和历史主张勘误；其中旧候选建议不代表当前方法。

原始论文、检索日志、实验记录和旧方案存放在 [research/](../research/README.md)，实验资源统一存放在 [resources/](../resources/README.md)，数据位于 [resources/data/](../resources/data/)。

[2026-10-29 局部/全局管理方案](AGENT_MANAGED_LOCAL_GLOBAL_FLOW_20261029.md) 仅保留为历史记录，已被直接 D2L 主线取代。
