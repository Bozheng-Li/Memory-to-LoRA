# 有界局部与全局参数记忆

按 [HIGH_LEVEL_DESIGN.md](../docs/HIGH_LEVEL_DESIGN.md) 实现同一冻结基座的任务/管理模式、
有界工作窗口、局部事件记忆、全局事实记录和原生 D2L 参数读取。首版不训练模型。
管理器只输出计划；来源、版本、依赖和最终预算通过校验，且安装与真实前向成功后，
Runtime 才发布新状态。纠正失败只能恢复不包含已知错误的安全子集。

## 安装和运行

使用 Python 3.10。先在项目根目录按 [requirements-d2l.txt](../requirements-d2l.txt)
安装 D2L 依赖，再安装本包的新增依赖和开发工具：

```bash
cd Main-Code
python -m pip install -e '.[dev]'
cp configs/mab_sh6k.example.yaml configs/mab_sh6k.yaml
python -m param_memory validate-config --config configs/mab_sh6k.yaml
python -m param_memory run-mab --config configs/mab_sh6k.yaml
```

已存在实际配置时直接修改它，不覆盖自己的参数。所有可编辑运行参数只在 YAML 中，
相对路径以 YAML 所在目录为基准。`validate-config` 检查结构和本地资源，不加载模型。
当前 checkpoint 带量化上下文编码器，完整运行需要已分配的 CUDA 设备；
登录节点可执行纯状态测试和数据检查。代码不会回退到另一模型、注意力实现或文本记忆。

`model.d2l_root`、checkpoint、基座、tokenizer、聊天模板和官方评分源码都从既有资源读取，
不复制到 Python 包。D2L 张量 rank、目标模块、层数、精度和 scale 从实际 checkpoint 派生。
通过构造期限定的原生加载边界，基座和上下文编码器都使用配置中的设备和注意力实现；
原生模型只构造一次，再严格加载 hypernetwork，避免上游 `from_state_dict` 的二次重建。

## 实现约定

- 工作窗口包含候选的元数据、来源副本和编译文本；接收下一片前检查候选容量。
  候选终结后已处理原文立即离窗，剩余未处理前缀继续保留。
- `scope` 是当前历史/任务 ID；具体服务、环境和成立条件必须写在 `claim` 中。
  `source_refs.start/end` 是原事件的 Unicode 码点位置，hash 对应完整来源片段。
- `local_blocks` 是固定槽位 tuple，空槽为 `None`；禁用块仍占槽位和文本预算。
  全局记录按稳定 ID 排序，完整记录按含模板的长度做 next-fit 装箱。
- 一个维护事务的首次生成、补查、JSON 修复、预算修复、纠正后重提共享调用上限。
  有效失效通知先提交安全状态，再要求新的版本号；旧响应的普通动作不会被自动沿用。
- 原始内容因子和组合因子分开存储。共享项仅在统一组合时加入一次；
  安装后直接调用 `base_model.generate`，顶层调用间不保留 KV。
- MAB 入口按 `metadata.source` 筛选，并把固定版本行号用作校验。
  当前 manifest 的单历史选择项可用；多历史选择会明确拒绝。
  writer 只接收 context，冻结后逐题读取相同快照，gold 仅用于官方评分。

## 检查

```bash
timeout 60s python -m pytest tests/unit -q
python -m ruff check src tests
python -m build
```

测试替身只存在于 `tests/`。真实模型测试单独执行，并设置外部硬期限：

```bash
timeout 1800s python -m pytest tests/integration -q \
  --d2l-config configs/mab_sh6k.yaml
```

未提供模型配置时集成测试明确跳过；没有 GPU 不算模型验证通过。
集成测试检查真实因子方向/scale、共享项次数、reset、重复安装和重建一致性。
测试是否完成及实测结果见 [IMPLEMENTATION.md](../docs/IMPLEMENTATION.md)，
配置样例不是已经验证过的事实记忆质量结论。

## 运行产物

每次运行在 `output.directory/<run_id>/` 创建新目录，拒绝覆盖：

| 文件 | 内容 |
| --- | --- |
| `resolved_config.yaml` | 解析后的绝对路径配置，仅用于复现 |
| `manifest.json` | 代码/数据/模板/模型指纹、固定版本声明、执行状态和编译输入绑定 |
| `maintenance.jsonl` | 所有维护的身份、原始管理响应、失败、调用数和实际可观测成本 |
| `predictions.jsonl` | 逐题原始回答、官方分数、快照版本和错误 |
| `metrics.json` | 全部题目分母、成功/失败/未尝试数、成本、稳态和暂存占用 |
| `final_snapshot.json` | 最后状态、在线来源和因子引用 |
| `memory.safetensors` | `raw.<block_key>.<module>.A/B` 与 `combined.<module>.A/B` |

`official_mean_scored` 只对实际完成评分的题求均值；完全未运行时为 `null`。
`failure_inclusive_score` 使用全部题目的分母，未完成题没有伪造的逐题分数。
`manager_calls_without_returned_usage` 标出没有返回 token 用量的失败调用，
此时成本表仅包含已知用量。`peak_staged_factor_bytes` 是共存原始/组合张量的占用，
另列 CUDA 实测峰值；模型驻留项报告参数/缓冲区字节，CUDA 峰值还包含其它运行分配。
`manifest.artifacts_complete=true` 表示文件已写完；执行是否成功看 `status`，
暂停/失败的状态不能当作可回答问题的完整记忆。

## 模块职责与代码规模

这是从单个 YAML 样例建立完整实现。超过 80 行的新增业务文件分别承载：
`schema.py` 的严格数据契约，`config.py` 的配置/资源边界，`budget.py` 的统一计账，
`events.py` 的原文与游标，`state.py` 的来源依赖/状态推导，`planning.py` 的计划校验，
`manager.py` 的管理提示，`d2l.py` 的实际模型/张量契约，`runtime.py` 的事务与恢复，
`evaluation.py` 的协议隔离、官方评分和结果落盘。测试文件按这些可观察契约组织。
较长的事务/装配函数在 docstring 中说明保持步骤集中以保证哪些不变量。
