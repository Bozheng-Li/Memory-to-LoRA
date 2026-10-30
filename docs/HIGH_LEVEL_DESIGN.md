# 直接 D2L 链路设计

更新：2026-10-30。当前实现以 [直接 D2L 方法](DIRECT_D2L_FLOW.md) 为唯一主线，配置协议为 `schema_version: 2`。
代码继续遵守项目 [AGENTS.md](../AGENTS.md)：复用原生后端，删除失效机制，严格检查输入，不增加兼容分支。

## 模块职责

| 模块 | 当前职责 |
| --- | --- |
| [config.py](../Main-Code/src/param_memory/config.py) | 严格 YAML、路径解析、数值与本地资源校验 |
| [schema.py](../Main-Code/src/param_memory/schema.py) | 不可变事件、chunk、状态、因子引用与结果契约 |
| [events.py](../Main-Code/src/param_memory/events.py) | 原文无损分片、连续前缀与游标推进 |
| [state.py](../Main-Code/src/param_memory/state.py) | 空历史、来源完整性和单个 chunk 的编译输入 |
| [budget.py](../Main-Code/src/param_memory/budget.py) | 真实 token 边界、文本与张量占用计账 |
| [d2l.py](../Main-Code/src/param_memory/d2l.py) | 冻结模型加载、原生编译、统一组合、安装及推理 |
| [runtime.py](../Main-Code/src/param_memory/runtime.py) | chunk 触发、因子追加、原子发布、失败恢复与回答 |
| [evaluation.py](../Main-Code/src/param_memory/evaluation.py) | MAB 历史/问题/gold 隔离、官方评分和产物保存 |
| [run_logging.py](../Main-Code/src/param_memory/run_logging.py) | 独立日志生命周期、阶段信息与即时 JSONL |

`manager.py`、`planning.py`、管理提示及对应计划协议已删除。当前运行路径不存在全局事实表、
固定槽位、接纳/跳过动作、替换队列、来源补查或摘要生成。

## 配置

```yaml
schema_version: 2
memory:
  chunk_tokens: 2048
```

`memory` 只有 `chunk_tokens`，必须为正整数。样例与主配置为 2048，
`mab_sh6k_budget512_baseline.yaml` 现为 chunk=512 的对照配置。
编译前检查含模板的真实 token IDs，后端加载后检查上下文编码器实际容量。

`generation.task` 保留原来的显式输入/输出、解码和单次调用缓存配置，`prompts.task_path`
指定任务模板。任务输入上限加输出上限不能超过基座容量。
模型、数据、评分、输出、日志和种子继续只从 YAML 读取；相对路径以 YAML 所在目录为基准。
rank、层数、模块维度和原生 scale 从实际 checkpoint 读取，当前 checkpoint 为 rank=8。

旧的 `local_slots`、`global_blocks`、局部/全局块预算、持久文本淘汰预算、工作元数据预算、
管理调用预算、`generation.manager` 和 `prompts.manager_path` 均已删除，旧字段会被严格拒绝。
仓库内三个 YAML 已迁移。历史运行产物保留为旧协议记录，不作为当前配置使用。

## 状态和不变量

`Snapshot` 只包含 `scope_id`、单调递增的 `version`、`work`、全部 `factor_refs` 和最后一次提交回执。
`WorkState` 包含任务规则、尚未编译的 `recent_spans`、唯一 `pending_candidate` 和已处理游标。
事件只保留身份、顺序、原文及可用时间信息，写入触发仅取决于 chunk 边界和历史收尾。

每个 `CompileSpec` 保存一个完整 chunk 的原文、真实 token IDs、来源引用及输入指纹。
原文字段允许仅含空白的非空片段；身份与任务规则仍要求含非空白字符。
原生 D2L 编码会对每个 chunk 做首尾 `strip`，原文记录保持不变，输入指纹同时绑定二者。
`CompiledMemory` 保存历史顺序下的全部编译规格、原始因子和当前组合产物。
原始因子不可原地修改；新状态只能追加一个 chunk，旧输入和旧因子必须保持一致。
原文副本用于来源审计和复现，任务输入不读取这些编译记录。

对 N 个 chunk，内容宽度为 `N × rank`；非空状态的组合宽度为 `(N + 1) × rank`。
最后一个 rank 区域是原生共享项。空状态没有内容或共享项，直接使用纯基座。
宽度是 A/B 的分配维度，不等于数学矩阵秩，也不是可无损保存的事实数量。

## 核心接口

| 接口 | 输入、结果与约束 |
| --- | --- |
| `empty_snapshot(scope_id, goal)` | 创建版本 0 的独立历史，不分配因子。 |
| `iter_spans(event, max_tokens, encode_context)` | 按真实模板长度输出连续来源片段，连接后恢复整个事件原文。 |
| `MemoryRuntime(initial, backend, config)` | 只接受空历史，检查模型容量并清除之前的 LoRA 绑定。 |
| `observe(span)` | 接收下一片原文并返回本次维护结果；不足一块时继续累积，暂停后禁止交付新片段。 |
| `build_candidate(work, trigger, limits, encode_context)` | 从尚未处理原文中取有界连续前缀；无内容时返回 `None`。 |
| `maintain(candidate)` | 编译、组合并提交一个 chunk；失败保留上一版因子和待处理原文并返回 `paused`。 |
| `answer(question)` | 在 `ready` 或 `frozen` 状态下使用当前因子和未编译上下文回答；结果不自动回写历史。 |
| `freeze()` | 所有历史和尾段提交后固定版本；存在待处理原文、pending 或故障时拒绝冻结。 |

在线调用在 `observe` 成功后可继续 `answer`。历史收尾时用 `HISTORY_END` 构造候选并调用
`maintain`，直到没有待处理原文，再调用 `freeze`；MAB 外壳已实现此过程。

## 执行与失败边界

1. `iter_spans` 按真实 tokenizer 上限切分当前事件，所有片段连接后恢复原文。
2. `MemoryRuntime.observe` 检查作用域、原文 hash 和接续位置，将片段加入工作上下文。
3. 达到上限，或下一完整片段使前缀超限时，构造一个有界 chunk；跨事件原样拼接，不插入新文本。
4. `build_compile_spec` 验证来源、游标、候选身份、原文和含模板的 token 长度。
5. `_compile_chunk` 只编译当前 chunk，复用全部历史原始因子并统一调用原生组合函数。
6. `_commit` 检查追加关系与因子身份，安装完整组合并执行真实前向，然后一起发布因子与游标。
7. 仅成功提交的原文离开工作上下文，后续推理继续使用累积 LoRA。

出现预期编译、组合、安装或前向错误时，保留 pending 和已提交版本，尝试恢复当前提交版，
随后进入 `paused`。恢复本身失败时清除绑定，禁止回答或继续接收事件。未知异常清理后重抛。
日志 I/O 位于提交恢复边界之外，记录失败不会把已经提交的 chunk 重新消费。

`answer` 在 `ready` 或 `frozen` 状态下读取当前 LoRA；任务 prompt 只包含任务规则、未编译上下文
和当前问题。每次调用重新前向，不跨顶层调用保留 KV，不把生成答案自动回写历史。
MAB 入口按原顺序交付 context，收尾编译全部剩余 chunk 后调用 `freeze`，所有题读取同一版本；
完整问题集和 gold 只存在于评测外壳，gold 只交给官方评分函数。

## 验证要求

行为测试覆盖 chunk 触发、超出原槽位数量后全部保留、Unicode 与来源顺序、尾段、空历史、
失败事务与问题隔离。原生 CPU 张量测试检查 rank=8、A/B 方向、共享项一次、原生 scale 和重新安装。
真实 checkpoint 测试另需 CUDA 和显式配置，运行命令及本轮执行状态见 [实现说明](IMPLEMENTATION.md)。
