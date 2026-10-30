# 首版实现与验证

更新：2026-10-29。对应 [HIGH_LEVEL_DESIGN.md](HIGH_LEVEL_DESIGN.md) v0.2。

[Main-Code/](../Main-Code/README.md) 已实现设计规定的 12 个 Python 模块、52 个核心函数、
两个提示模板、严格 YAML 配置、单元测试和独立模型集成测试。
本文补充实际实现约定、43 个辅助函数契约及本次验证边界。
事实选择质量、参数可读性和 SH6K 模型分数仍需 GPU 实验验证。

## 实现约定

- 所有模型调用集中在 `d2l.py`；任务和管理共享同一冻结基座，另有实际上下文编码器。
  纯状态模块不运行模型或写文件，`runtime.py` 是维护、事务、恢复和发布的唯一控制点。
- CLI 只接受动作和 `--config`。参数从一份严格 YAML 解析，路径以该文件所在目录为准。
  不通过环境变量覆盖业务参数，不下载替代模型，不提供假编译器或文本记忆降级。
- 来源位置是原事件的 Unicode 码点区间，hash 对应完整 `SourceSpan` 原文。
  候选 ID 由有序 `(span_id, source_hash)` 列表稳定序列化后计算。
  创建与校验共用候选构造、来源完整性和接续规则；游标保存 `order/event_id/end`。
- 局部槽是固定 tuple，空槽为 `None`。已处理候选原文立即离开 W，未处理后缀继续保留。
  pending 的来源和编译文本副本也计入窗口预算，不能用窗口中的旧原文冒充长期参数读取。
- `scope` 是当前历史/任务身份；具体服务、环境、条件和关系须写进 `claim`。
  全局记录按稳定 ID 排序，用真实 D2L 模板长度做 next-fit 装箱。
  局部退出前复制仍需保留的证据子区间并改写引用，不用 Agent 改写替代原始证据。
- 合法失效通知先发布安全参数子集，再要求基于新版本重提。
  后续失败只能恢复最新安全状态；安全安装或恢复失败则暂停。
  普通更新在编译、完整预算检查、安装和真实前向全部成功后才发布。
- skip 也恢复被管理模式清除的任务绑定；不跨顶层调用保留 KV。
  因子复用比较输入与编译器指纹，并重新绑定当前来源/记录版本。
  全局有效编译文本变化时重编译全部全局块，仅来源版本变化时可复用相同输入的张量。
- 全部局部和全局原始因子一起调用原生 `combine_lora`，共享项只加入一次。
  有 N 个内容块时分配宽度为 `(N + 1) × checkpoint.rank`，空内容走纯基座。
  安装后直接调用 `base_model.generate`，避免上游包装入口再次组合。
- 首版使用 manifest 的单历史选择项，多历史选择明确拒绝。
  writer 只获得历史，reader 在冻结后逐题获得问题，gold 只交给官方评分器。

加载器使用一次原生构造和严格 hypernetwork 权重加载，避免 `from_state_dict` 重建模型。
上游内部加载编码器未传递完整设备/离线参数，因此仅在构造作用域内限定其加载函数，
使实际资源、精度和注意力实现遵循 YAML，退出作用域后还原。没有修改上游资源源码。

## 用量和失败口径

持久文本包括来源、编译文本副本、卡片、F、派生依赖索引、回执、因子引用，
以及 `CompiledMemory` 持有的编译文本和版本绑定。
输入 IDs 是数值载荷，其长度单列，不把十进制展示文本重复计作自然语言。
原始/组合张量与模型常驻字节分开统计。

`peak_staged_factor_bytes` 统计当前与单次暂存的已知原始/组合产物，复用对象不重复计算。
编译或组合失败仍记录此前成功分配的占用，不包含后端内部尚未返回的临时分配。
CUDA 实测峰值另报。维护账本记录失败操作耗时，无返回 token 用量的调用尝试单列。
`elapsed_seconds` 截止推理/评分阶段结束，不包含最终 manifest 指纹汇集和文件落盘。

`official_mean_scored` 只对实际完成官方评分的题求均值，完全未评分时为 `null`。
完整题目分母、未尝试数和失败数保留；`failure_inclusive_score` 使用完整分母。
失败题没有伪造的逐题分数。`manifest.artifacts_complete=true` 只表示文件已写完，
运行状态看 `status`，是否冻结可读看 `task_readable`。

## 本次验证

环境：Python 3.10.22、PyTorch 2.6.0、Transformers 4.51.3、PEFT 0.17.1、Pydantic 2.14.1。
根目录 [requirements-d2l.txt](../requirements-d2l.txt) 管理 D2L 依赖，
[pyproject.toml](../Main-Code/pyproject.toml) 管理项目新增依赖与开发工具。
本地验证环境只读复用了既有环境的大型依赖，PEFT 与项目依赖单独安装；
常规安装按 [Main-Code README](../Main-Code/README.md) 中的两份依赖清单执行。

| 检查 | 实际结果 |
| --- | --- |
| `timeout 60s python -m pytest tests/unit -q` | **49 passed in 30.46s** |
| `python -m ruff check src tests` | **All checks passed** |
| `python -m build` | 在独立构建环境中成功生成 wheel 与 sdist |
| `python -m param_memory validate-config --config configs/mab_sh6k.yaml` | 配置与实际本地资源校验通过，不加载模型 |
| HLD 函数/注释核对 | 52/52 核心函数已实现；辅助函数均有参数/返回契约 |
| 无 GPU 的 `run-mab` CLI | 按预期非零退出，生成全部 7 类产物；100 题未尝试，0 题评分，均分 `null` |
| 独立 GPU 集成入口 | **1 skipped**，当前进程没有 CUDA；不算模型测试通过 |

单元测试覆盖 YAML、来源和游标、无损分片、模板预算、装箱、证据存活、统一调用额度、
纠正优先、安全恢复、因子复用、冻结隔离、官方评分、损坏数据和失败分母。
原生后端契约测试使用小型真实 CPU Qwen3/PEFT 基座和原生 LoRA 运算，
检查 A/B 方向、scale、共享项、重复安装、reset 和直接基座生成。
其中内容因子由固定测试张量提供，不能据此宣称真实 checkpoint 编译通过。

无 GPU 的 CLI 产物可查看
[manifest](../research/runs/mab_sh6k_lg_joint-20261029T101141610036Z-b8be5f016049427e962e0f171a817acd/manifest.json)
和 [metrics](../research/runs/mab_sh6k_lg_joint-20261029T101141610036Z-b8be5f016049427e962e0f171a817acd/metrics.json)。
这是一份失败处理验证记录，不是模型评测结果。

已尝试申请 `gpuA800`，作业 1870192 返回 `Requested nodes are busy`。
真实 checkpoint 集成测试和 SH6K 模型推理均未完成，不能给出模型成绩或参数可读性结论。
获得 GPU 后，从 Main-Code 执行：

```bash
timeout 1800s python -m pytest tests/integration -q -rs \
  --d2l-config configs/mab_sh6k.yaml
python -m param_memory run-mab --config configs/mab_sh6k.yaml
```

## 代码规模与长文件职责

相对原先只有 YAML 样例的 Main-Code，新增 **4,479 行 Python 源码、1,333 行 Python 测试**，
包含注释和空行，不含模型资源、缓存、构建产物及历史脚本。
根目录没有 Git 元信息，因此按新增文件统计，不声称是 Git diff 的计数。

| 超过 80 行的新增业务模块 | 行数 | 必需职责 |
| --- | ---: | --- |
| schema.py | 562 | 严格状态/JSON 模型、枚举、异常与共享序列化 |
| config.py | 260 | 唯一键 YAML、严格配置、静态及资源校验 |
| budget.py | 171 | 真实 token、完整副本计账和统一预算 |
| events.py | 241 | 原文分片、唯一来源/接续规则、候选和游标 |
| state.py | 603 | 纯状态推导、依赖、证据保留、失效闭包及装箱 |
| planning.py | 370 | 响应、版本、来源、互斥动作及最终状态校验 |
| manager.py | 137 | 管理可见材料、生成 schema 和一次共享基座调用 |
| d2l.py | 708 | 原生加载、张量契约、编译、组合、绑定与模型调用 |
| runtime.py | 801 | 有限维护、事务发布、安全恢复、冻结及成本账本 |
| evaluation.py | 556 | 数据/问题/gold 隔离、官方评分、失败口径及产物 |

较长的装配、状态转移和事务函数保留集中步骤，以保护完整输入、来源依赖及唯一发布点，
docstring 中已说明原因。测试按这些跨模块不变量组织；本说明的篇幅用于补齐辅助函数契约。

## 补充函数契约

以下是 HLD 的 52 个核心函数之外新增的 43 个辅助函数和属性，签名及说明从实际源码提取。
异常列表列出项目边界或明确文件错误，未知异常不被静默转换为成功。
模块内辅助函数由该模块核心流程调用；纯状态和计账辅助函数可跨层共享。


### budget.py

- `prefix_end(text: str, max_tokens: int, encode: EncodeText) -> int`  
  找一个真实编码可装入的原文前缀，返回 Unicode 结束位置。  
  参数：text: 待保留原文，不通过 token 解码重建。 max_tokens: 包含调用方模板的正整数额度。 encode: 实际原始或模板编码函数。  
  返回：可装入的非空原文前缀结束位置，单位为 Unicode 码点；不保证最长。  
  异常：BudgetExceeded: 额度非正，或搜索无法形成可装入的非空前缀。  
  代码：[budget.prefix_end](../Main-Code/src/param_memory/budget.py)。

- `measure_work(work: WorkState, encode_text: EncodeText) -> int`  
  按完整 W 序列化计数，包含目标、元数据和 pending 中的副本。  
  参数：work: 完整工作状态，包含 pending 的来源与文本副本。 encode_text: 实际原始 tokenizer 编码。  
  返回：稳定序列化后的完整 W token 数，不修改输入。  
  代码：[budget.measure_work](../Main-Code/src/param_memory/budget.py)。


### config.py

- `fingerprint_file(path: Path) -> str`  
  流式计算本地资源 SHA-256；只读文件，不把大权重整体复制进内存。  
  参数：path: 本地资源文件。  
  返回：流式计算的 SHA-256 十六进制摘要。  
  异常：OSError: 文件无法读取。  
  代码：[config.fingerprint_file](../Main-Code/src/param_memory/config.py)。


### d2l.py

- `CompiledMemory.size_bytes() -> int`  
  返回这份状态实际持有的原始及组合张量字节。  
  返回：此状态引用的原始及组合张量总字节数，不含模型常驻内存。  
  代码：[d2l.CompiledMemory.size_bytes](../Main-Code/src/param_memory/d2l.py)。

- `empty_compiled() -> CompiledMemory`  
  创建纯基座状态，不编译空字符串、不分配共享项。  
  返回：无原始或组合因子的 CompiledMemory，不分配共享项。  
  代码：[d2l.empty_compiled](../Main-Code/src/param_memory/d2l.py)。

- `_readonly(tensors: dict) -> Mapping[str, Mapping[str, Tensor]]`  
  冻结两层模块映射；张量本身仍依赖调用方遵守只读契约。  
  参数：tensors: 原生生成的两层模块与 A/B 张量字典。  
  返回：只读映射；张量不复制，调用方仍必须遵守不原地修改的契约。  
  代码：[d2l._readonly](../Main-Code/src/param_memory/d2l.py)。

- `D2LBackend._load_native(config: ModelConfig, native: object) -> None`  
  复用原生构造器和严格权重加载，避免 from_state_dict 重建两次编码器。  
  参数：config: 唯一模型配置。 native: 已核对源码路径的原生 hypernet 模块。  
  返回：None；构造一次原生模型、严格装入权重，并记录实际资源指纹。  
  异常：BackendError: checkpoint 契约或声明的模型路径不匹配。  
  代码：[d2l.D2LBackend._load_native](../Main-Code/src/param_memory/d2l.py)。

- `D2LBackend._load_native.load_local(model_name_or_path: str, **kwargs: object) -> object`  
  限定上游内部资源路径，并传入明确设备、精度、注意力实现。  
  参数：model_name_or_path: 上游请求的模型路径，必须等于配置中的本地基座。 **kwargs: 上游模型构造参数；设备、精度、注意力与离线限制在此明确传入。  
  返回：原生 get_model 创建的真实模型。  
  异常：BackendError: 上游尝试加载未声明的模型。  
  代码：[d2l.D2LBackend._load_native.load_local](../Main-Code/src/param_memory/d2l.py)。

- `D2LBackend._initialize_layout(native: object) -> None`  
  从实际模型核对层、模块、A/B 方向、rank、scale、dtype 和设备。  
  参数：native: 提供原生组合与安装接口的模块。  
  返回：None；建立真实 FactorLayout，并统计基座、编码器及 hypernetwork 常驻字节。  
  异常：BackendError: 层索引、模块维度或冻结状态不成立。  
  代码：[d2l.D2LBackend._initialize_layout](../Main-Code/src/param_memory/d2l.py)。

- `D2LBackend._synchronize() -> None`  
  等待本设备实际工作完成，使耗时和失败归属于当前操作。  
  返回：None；CUDA 操作已经完成，CPU 无需额外等待。  
  代码：[d2l.D2LBackend._synchronize](../Main-Code/src/param_memory/d2l.py)。


### evaluation.py

- `load_official_scorer.score(prediction: str, answers: tuple[str, ...]) -> float`  
  只适配原生调用参数，不另写文本清洗或匹配规则。  
  参数：prediction: 未经自行清洗的原始回答。 answers: 本题的可接受答案集合。  
  返回：官方多答案最大匹配分数，不重写原生归一化。  
  代码：[evaluation.load_official_scorer.score](../Main-Code/src/param_memory/evaluation.py)。

- `_consume_history(case: EvalCase, runtime: MemoryRuntime, records: list[MaintenanceResult]) -> None`  
  逐片交付历史，循环收尾有限剩余前缀；暂停后停止交付新事件。  
  参数：case: 外壳持有的数据；只取 context 与合法来源身份。 runtime: 当前历史的运行时。 records: 外壳维护列表，按发生顺序追加结果。  
  返回：None；完整消费历史后冻结，或在暂停时停止交付。  
  异常：StateError: 来源顺序或冻结条件不成立。 BudgetExceeded: 单个片段无法放入声明的容量。  
  代码：[evaluation._consume_history](../Main-Code/src/param_memory/evaluation.py)。

- `_answer_questions(case: EvalCase, runtime: MemoryRuntime, scorer: ScoreFn) -> tuple[Prediction, ...]`  
  逐题读取同一冻结快照，失败题保留 None 分数，不把任何回答回写历史。  
  参数：case: 与冻结历史对应的题组。 runtime: 已冻结的运行时。 scorer: 固定官方评分函数。  
  返回：逐题 Prediction；单题失败保留 None 分数及原始错误。  
  异常：StateError: 调用期间快照或冻结状态发生变化。  
  代码：[evaluation._answer_questions](../Main-Code/src/param_memory/evaluation.py)。

- `_run_metrics(case: EvalCase, records: tuple[MaintenanceResult, ...], predictions: tuple[Prediction, ...], elapsed_seconds: float) -> dict[str, object]`  
  以全部题目为运行分母，单列已评分均值、未尝试和失败数。  
  参数：case: 提供完整题目分母。 records: 本次全部维护结果。 predictions: 实际尝试过的题目结果。 elapsed_seconds: 完整运行的已耗秒数。  
  返回：完整题目分母、成功/失败/未尝试数量、已知成本及进程内存的指标字典。  
  代码：[evaluation._run_metrics](../Main-Code/src/param_memory/evaluation.py)。


### events.py

- `validate_span(span: SourceSpan) -> None`  
  核对原文长度、Unicode 区间和完整片段的 SHA-256。  
  参数：span: 已交付的原始片段或独立保留的证据。  
  返回：None 表示来源元数据与原文一致；不证明原文语义真实。  
  异常：StateError: 区间长度或 hash 不符。  
  代码：[events.validate_span](../Main-Code/src/param_memory/events.py)。

- `check_continuation(span: SourceSpan, previous: SourceSpan | Cursor | None) -> None`  
  核对新片段接续已处理游标或当前窗口的最后片段。  
  参数：span: 新交付的片段；原文完整性另由 validate_span 检查。 previous: 上一片段、最后处理游标，或表示历史起点的 None。  
  返回：None 表示顺序与事件身份连续，不允许重放或换用同一事件身份。  
  异常：StateError: 作用域、事件身份、位置或顺序冲突。  
  代码：[events.check_continuation](../Main-Code/src/param_memory/events.py)。

- `candidate_from_spans(spans: tuple[SourceSpan, ...], trigger: Trigger) -> Candidate`  
  从同一有序来源集合生成规范候选文本与稳定身份。  
  参数：spans: 当前未处理连续前缀；不能是空集合。 trigger: 调用方已知的维护边界，不参与内容身份计算。  
  返回：同事件原样拼接、跨事件加换行的候选；不登记 pending 或分配槽位。  
  异常：StateError: 来源集合为空。  
  代码：[events.candidate_from_spans](../Main-Code/src/param_memory/events.py)。


### planning.py

- `_unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]`  
  在 JSON 边界拒绝重复键，避免解析器悄悄覆盖先前动作。  
  参数：pairs: JSON 解码器交付的原始键值对。  
  返回：未丢弃重复信息的唯一键映射。  
  异常：PlanValidationError: 发现重复 JSON 键。  
  代码：[planning._unique_object](../Main-Code/src/param_memory/planning.py)。

- `validate_header(snapshot: Snapshot, candidate: Candidate, reply: ResponseHeader) -> None`  
  统一核对响应版本、作用域来源和候选身份；所有响应都必须先通过。  
  参数：snapshot: 当前快照。 candidate: 本次维护的候选。 reply: 已严格解析的公共响应头。  
  返回：None 表示 schema、状态版本、候选身份和来源作用域一致。  
  异常：PlanValidationError: 版本、身份或候选来源不合法。  
  代码：[planning.validate_header](../Main-Code/src/param_memory/planning.py)。

- `_expected_version(objects: dict, key: str, version: int) -> None`  
  校验一个现存对象的乐观版本，不接受缺失对象或陈旧动作。  
  参数：objects: 当前对象 ID 到有版本记录的映射。 key: 动作目标 ID。 version: 计划声明的预期版本。  
  返回：None 表示目标存在且版本一致。  
  异常：PlanValidationError: 对象不存在或版本陈旧。  
  代码：[planning._expected_version](../Main-Code/src/param_memory/planning.py)。

- `_validate_actions(snapshot: Snapshot, candidate: Candidate, plan: MemoryPlan) -> None`  
  校验槽位选择、动作身份、互斥集合及所有外部来源引用。  
  参数：snapshot: 当前安全状态。 candidate: 本次可引用的新来源。 plan: 身份已核对的计划。  
  返回：None 表示普通动作的对象、互斥集合与修复要求成立。  
  异常：PlanValidationError: 动作、别名、修复覆盖或佐证声明不合法。 StateError: 在线来源或来源引用不合法。  
  代码：[planning._validate_actions](../Main-Code/src/param_memory/planning.py)。


### runtime.py

- `MemoryRuntime._add_cost(**values: int | float) -> None`  
  累加当前唯一维护账本，不在快照中保存完整成本历史。  
  参数：**values: 本次操作新增的账本字段用量，单位由字段后缀指定。  
  返回：None；只更新当前维护成本，不向 Agent 暴露完整历史。  
  代码：[runtime.MemoryRuntime._add_cost](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._feedback(error: Exception) -> ValidationFeedback`  
  按管理输出 token 上限收紧反馈；完整错误仅由评测外壳保存。  
  参数：error: 本次计划解析或预算错误。  
  返回：由真实 tokenizer 限长的修复反馈；完整错误只进入外壳日志。  
  异常：BudgetExceeded: 反馈无法装入管理输出额度。  
  代码：[runtime.MemoryRuntime._feedback](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._needs_repair(candidate: Candidate) -> bool`  
  识别不能按普通失败丢弃的纠正和已停用内容。  
  参数：candidate: 仍在维护的当前候选。  
  返回：是否存在纠正触发或已停用内容；为真时不能按普通跳过确认候选。  
  代码：[runtime.MemoryRuntime._needs_repair](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._finished(snapshot: Snapshot, candidate: Candidate, status: MaintenanceStatus, plan_id: str | None) -> Snapshot`  
  仅为可终结候选计算新 W 和有界回执；此函数不发布状态。  
  参数：snapshot: 当前或拟提交的完整状态。 candidate: 可合法终结的 pending。 status: 提交、跳过或普通放弃结果。 plan_id: 最后计划身份，尚无合法计划时为 None。  
  返回：已推进游标并保留最后一份回执的新快照，不发布状态。  
  异常：StateError: 候选与工作窗口身份不符。 BudgetExceeded: 终结后的 W 超限。  
  代码：[runtime.MemoryRuntime._finished](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._settle_without_commit(candidate: Candidate, outcome: MaintenanceStatus, plan_id: str | None) -> MaintenanceStatus`  
  skip/普通放弃也恢复任务绑定；未完成纠正只恢复最新安全子集并暂停。  
  参数：candidate: 当前待终结候选。 outcome: 未提交时的跳过或普通放弃结果。 plan_id: 最后合法计划身份或 None。  
  返回：恢复任务绑定后的结果；未完成纠正返回 paused 并保留 pending。  
  异常：StateError: 最新安全状态无法恢复。 BudgetExceeded: 候选终结后的完整状态超限。  
  代码：[runtime.MemoryRuntime._settle_without_commit](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._recover_failure(candidate: Candidate, plan_id: str | None, errors: list[str]) -> MaintenanceStatus`  
  只恢复最近安全前态；安全失效安装本身失败时禁止重装已知错误因子。  
  参数：candidate: 失败事务的 pending。 plan_id: 最后合法计划身份或 None。 errors: 本次维护的错误列表；恢复错误会追加到其中。  
  返回：普通失败的 aborted，或不能安全继续时的 paused；不尝试更旧版本。  
  代码：[runtime.MemoryRuntime._recover_failure](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._record_factor_peak(blocks: tuple[RawFactors, ...], combined: CombinedFactors | None) -> None`  
  统计当前与暂存产物共存的真实字节，失败路径也保留已知分配峰值。  
  参数：blocks: 本次已成功生成或复用的原始内容因子。 combined: 已成功生成的组合产物，尚未组合时为 None。  
  返回：None；只更新维护成本，不发布状态或持有新的因子引用。  
  代码：[runtime.MemoryRuntime._record_factor_peak](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._compose(blocks: tuple[RawFactors, ...]) -> CombinedFactors | None`  
  记录一次统一组合的时间，空内容仍按后端契约返回 None。  
  参数：blocks: 本次全部活跃原始因子。  
  返回：后端生成的完整组合产物或 None，同时记录耗时和已知暂存峰值。  
  异常：BackendError: 原生组合失败；已知成本仍记录。  
  代码：[runtime.MemoryRuntime._compose](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._reset() -> None`  
  记录模式切换成本；顶层调用从不保留外部 KV。  
  返回：None；清除模型绑定并累计模式切换时间。  
  异常：BackendError: 原生 reset 失败。  
  代码：[runtime.MemoryRuntime._reset](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._install_and_prefill(snapshot: Snapshot, compiled: CompiledMemory) -> None`  
  安装一份完整状态并以待恢复 W 做真实前向，记录失败操作的成本。  
  参数：snapshot: 待安装的完整工作状态。 compiled: 与该状态对应的完整因子。  
  返回：None；绑定与真实前向完成，尚不发布快照。  
  异常：BudgetExceeded: 工作输入超出任务模式额度。 BackendError: 安装或前向失败；已发生成本仍计入。  
  代码：[runtime.MemoryRuntime._install_and_prefill](../Main-Code/src/param_memory/runtime.py)。

- `MemoryRuntime._task_input(snapshot: Snapshot, question: str) -> TokenIds`  
  只编码合法目标、工作观察、摘要和当前问题，不暴露管理事实表或卡片。  
  参数：snapshot: 只向任务模式展示其中的目标、W 和摘要。 question: 当前交付的问题。  
  返回：带任务模板的完整 IDs，不包含 F、局部卡片、管理日志或其它题答案。  
  异常：BudgetExceeded: 完整输入超过任务模式额度。 BackendError: 模板编码失败。  
  代码：[runtime.MemoryRuntime._task_input](../Main-Code/src/param_memory/runtime.py)。


### schema.py

- `stable_json(value: object) -> str`  
  稳定序列化 JSON 数据，供计账和指纹共用，不删除任何字段。  
  参数：value: 已由 model_dump(mode="json") 或边界解析得到的 JSON 数据。  
  返回：保留 Unicode、键排序且无多余空白的完整文本。  
  代码：[schema.stable_json](../Main-Code/src/param_memory/schema.py)。

- `text_hash(text: str) -> str`  
  返回原文 UTF-8 的 SHA-256；不正规化空白或 Unicode。  
  参数：text: 待计算指纹的完整文本，保留 Unicode 和空白。  
  返回：原文 UTF-8 字节的 SHA-256 十六进制摘要。  
  代码：[schema.text_hash](../Main-Code/src/param_memory/schema.py)。

- `compile_fingerprint(text: str, token_ids: TokenIds) -> str`  
  绑定真实文本和完整输入 IDs；来源版本另外保存在 CompileSpec 中。  
  参数：text: 实际编译视图。 token_ids: 对应的完整模板化输入 IDs。  
  返回：绑定文本及输入 IDs 的指纹；来源与记录版本在 CompileSpec 中另行绑定。  
  代码：[schema.compile_fingerprint](../Main-Code/src/param_memory/schema.py)。


### state.py

- `online_sources(snapshot: Snapshot, candidate: Candidate | None=None) -> dict[str, SourceSpan]`  
  索引本作用域在线来源，拒绝同 ID 不同原文；不查询离线历史。  
  参数：snapshot: 当前作用域的有界快照。 candidate: 可选的当前候选，只读取其已交付来源。  
  返回：从来源 ID 到原始片段的临时映射，不保留或补查离线历史。  
  异常：StateError: 作用域、原文完整性或同 ID 来源内容冲突。  
  代码：[state.online_sources](../Main-Code/src/param_memory/state.py)。

- `span_ref(span: SourceSpan) -> SourceRef`  
  引用一个已交付的完整片段，位置仍相对原事件。  
  参数：span: 已经过完整性检查的原始片段。  
  返回：引用整个片段的 SourceRef，区间仍相对原事件。  
  代码：[state.span_ref](../Main-Code/src/param_memory/state.py)。

- `validate_source_refs(refs: tuple[SourceRef, ...], sources: dict[str, SourceSpan]) -> None`  
  在唯一来源边界核对身份、hash 和原事件区间；失败抛 StateError。  
  参数：refs: 记录或计划携带的来源引用。 sources: 已核对完整性的在线来源映射。  
  返回：None 表示所有引用的身份、hash 和半开区间有效。  
  异常：StateError: 来源不存在、hash 不符或区间越界。  
  代码：[state.validate_source_refs](../Main-Code/src/param_memory/state.py)。

- `check_candidate(snapshot: Snapshot, candidate: Candidate) -> None`  
  核对候选是未处理前缀且编译文本来自该前缀，不能覆盖另一个 pending。  
  参数：snapshot: 候选对应的当前快照。 candidate: 待登记或待校验的候选。  
  返回：None 表示候选是游标之后的未处理连续前缀，文本及 ID 与原文一致。  
  异常：StateError: 来源、游标、pending 身份或规范候选内容冲突。  
  代码：[state.check_candidate](../Main-Code/src/param_memory/state.py)。

- `plan_aliases(snapshot: Snapshot, plan: MemoryPlan) -> dict[str, str]`  
  按作用域、计划身份和别名生成稳定全局 ID；不分配随机业务 ID。  
  参数：snapshot: 提供当前作用域。 plan: 新增别名已完成唯一性校验的计划。  
  返回：计划内别名到稳定全局记录 ID 的映射，不修改状态。  
  代码：[state.plan_aliases](../Main-Code/src/param_memory/state.py)。

- `_retain_evidence(records: tuple[GlobalRecord, ...], blocks: tuple[LocalBlock | None, ...], available: dict[str, SourceSpan]) -> tuple[tuple[GlobalRecord, ...], tuple[SourceSpan, ...]]`  
  在局部退出前复制仍需保留的证据子区间，并改写全部新状态引用。  
  参数：records: 拟保留的完整全局记录。 blocks: 拟提交的局部槽位。 available: 计划可以合法引用的在线来源。  
  返回：改写来源引用后的全局记录及仍被引用的来源；旧状态不变。  
  代码：[state._retain_evidence](../Main-Code/src/param_memory/state.py)。

- `_compile_spec(key: str, kind: str, text: str, refs: tuple[SourceRef, ...], versions: tuple[ObjectVersion, ...], encode_context: EncodeText) -> CompileSpec`  
  绑定一次真实编码和当前来源，供局部与全局编译统一使用。  
  参数：key: 内容块的稳定标识。 kind: local 或 global。 text: 实际编译文本。 refs: 当前来源绑定。 versions: 当前记录或局部对象版本。 encode_context: 实际 D2L 模板编码。  
  返回：包含真实输入 IDs、文本指纹和当前来源版本的 CompileSpec。  
  代码：[state._compile_spec](../Main-Code/src/param_memory/state.py)。

- `_global_text(records: tuple[GlobalRecord, ...]) -> str`  
  序列化有效事实及范围/时间/状态，不把内部 hash、理由或用途编入事实。  
  参数：records: 按稳定顺序排列的有效记录。  
  返回：保留事实、种类、范围、时间与状态的编译文本；不写入管理理由或内部 hash。  
  代码：[state._global_text](../Main-Code/src/param_memory/state.py)。

