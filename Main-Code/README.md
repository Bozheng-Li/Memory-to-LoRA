# 直接 D2L 参数记忆

执行过程中的连续上下文达到 chunk 边界后，直接生成当前 checkpoint 的 rank=8 LoRA，
将新因子追加到全部历史因子并安装，用于后续推理。历史尾段也会编译。
方法见 [直接 D2L 说明](../docs/DIRECT_D2L_FLOW.md)，模块与接口见 [高层设计](../docs/HIGH_LEVEL_DESIGN.md)。

## 安装与运行

使用 Python 3.10 及项目 [requirements-d2l.txt](../requirements-d2l.txt) 对应环境：

```bash
cd Main-Code
python -m pip install -e '.[dev]'
python -m param_memory validate-config --config configs/mab_sh6k.yaml
python -m param_memory run-mab --config configs/mab_sh6k.yaml
```

新环境可从 `configs/mab_sh6k.example.yaml` 创建配置；已有配置直接修改，不覆盖自己的参数。
当前 checkpoint 带量化上下文编码器，真实编译与推理需要已分配的 CUDA 设备。
`validate-config` 只校验 YAML 和本地资源，不加载模型或下载替代资源。

## 唯一写入规则

```yaml
schema_version: 2
memory:
  chunk_tokens: 2048
```

上限包含真实 D2L 模板和特殊 token。跨事件保持原文顺序，不摘要、不改写；
当下一完整片段无法放入时，先编译已积累前缀，因此部分 chunk 会略小于上限。
上下文编码器的实际容量在模型加载后检查，超大事件由 `iter_spans` 无损分片。
原文及真实 token IDs 分别记录；原生 D2L 编码按其模板规则去掉首尾空白。
仅含空白的非空尾段仍可编译，完整来源原文保留在记录中。

每个 chunk 只生成一组原始 rank-8 因子。8 个 chunk 的内容宽度是 64，加一次共享项后为 72；
第 9 个 chunk 后分别为 72 和 80。因子随历史增加，未设置固定数量或替换策略。
编译、安装与前向均成功后，才发布新版本并移走已编译的工作上下文；失败保留原文并暂停。

全局槽位、局部队列、管理器、计划协议及其提示和预算已从主链路删除。
`generation` 只保留 `task`，`prompts` 只保留 `task_path`。旧 YAML 字段会被拒绝，
仓库三个配置已迁移，其中 `mab_sh6k_budget512_baseline.yaml` 使用 chunk=512。
所有运行参数在 YAML 中显式指定，相对路径按 YAML 所在目录解析。

任务解码仍由 `generation.task` 明确控制，不继承模型目录中会覆盖本次配置的生成预设。
缓存配置只影响单次生成，顶层调用间不保留 KV。rank、层、模块维度和 scale 来自实际 checkpoint。

## 任务与评测

`MemoryRuntime(initial, backend, config)` 接收空历史；通过 `observe` 写入已发生上下文，
通过 `answer` 携带当前累积 LoRA 推理。`empty_snapshot(scope_id, goal)` 不再接收槽位数量。

MAB 入口按 `metadata.source` 和固定 revision 选择单份历史，保持事实序号及原文顺序；
历史结束时编译全部尾段并冻结，所有问题读取同一版本。gold 只用于官方评分。
编译后的原文仍保存在编译记录中用于复现，不再拼回任务 prompt，也不读取其它题的答案。

## 日志和产物

日志写入 YAML 的 `logging.directory/<run_id>/`，与最终结果目录分开：

| 日志文件 | 内容 |
| --- | --- |
| `run.log` | 编译、组合、安装、前向、答题、评分等阶段及耗时和异常 |
| `resolved_config.yaml` | 启动时保存的完整配置 |
| `maintenance.jsonl` | 每个 chunk 的提交版本、结果、错误和真实成本 |
| `predictions.jsonl` | 逐题回答、分数、版本和错误 |

每个 chunk 或问题完成后立即追加记录。日志级别不影响结果 JSONL，已有目录不会被覆盖。
管理回复、计划和管理调用计数不再出现在新日志协议中。

最终产物写入 `output.directory/<run_id>/`：

| 文件 | 内容 |
| --- | --- |
| `resolved_config.yaml` | 解析后的运行配置 |
| `manifest.json` | 代码、模型、数据、任务提示指纹和完整编译输入 |
| `metrics.json` | 官方分数、失败分母、成本、chunk 数、内容宽度和占用 |
| `final_snapshot.json` | 最后版本、剩余工作上下文、全部因子引用 |
| `memory.safetensors` | 全部原始 A/B 和加入一次共享项的组合 A/B |

`manifest.artifacts_complete=true` 仅表示文件已写完，执行结果仍以 `status` 为准。
`official_mean_scored` 只统计完成评分的题，`failure_inclusive_score` 使用全部题目作分母。
`memory_usage.compiled_chunks` 和 `allocated_width` 分别报告 chunk 数和含共享项的宽度，
同时报告实际因子字节、暂存峰值及可用时的 CUDA 峰值。

## 检查

```bash
timeout 60s python -m pytest tests/unit -q
python -m ruff check src tests
python -m build
```

断网节点已安装满足 `pyproject.toml` 要求的 `setuptools` 和 `wheel` 时，使用
`python -m build --no-isolation`，直接复用当前环境完成构建。

真实 checkpoint 集成测试需要显式配置和 CUDA：

```bash
timeout 1800s python -m pytest tests/integration -q \
  --d2l-config configs/mab_sh6k.yaml
```

缺少配置或 CUDA 时测试明确跳过，不能算作模型验证通过。本轮验证记录和限制见
[实现说明](../docs/IMPLEMENTATION.md)。本阶段只建立最小直接 D2L 链路，记忆质量留待真实评测。
