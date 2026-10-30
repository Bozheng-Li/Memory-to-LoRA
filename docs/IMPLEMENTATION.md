# 直接 D2L 链路实现与验证

更新：2026-10-30。当前实现对应 [直接 D2L 方法](DIRECT_D2L_FLOW.md) 与 [高层设计](HIGH_LEVEL_DESIGN.md)。

## 本轮实现

主链路为：连续上下文 → chunk → 冻结 D2L 生成 rank-8 因子 → 追加全部历史因子 →
统一组合并安装 → 携带 LoRA 继续推理。全局记录、固定槽位、记忆队列、替换、管理模型、
计划解析、来源补查和摘要字段已删除。当前只保留这条直接写入路径。

`memory.chunk_tokens` 是唯一记忆写入配置，主配置为 2048。模型 rank 来自所选 checkpoint，
当前为 8。原始输入按 Unicode 边界分片，真实 tokenizer 检查包含模板的长度；
成功编译的工作原文离窗，尚未处理的后缀原样保留。历史尾段在冻结前完成编译。
非空的空白片段也保留并编译；原生编码的首尾 `strip` 只影响模型输入，不修改来源记录。

每次只编译一个新 chunk，旧原始因子按对象引用复用。全部原始块统一组合，共享项一次，
使用原生 scale；8 个 chunk 的内容宽度为 64、实际分配宽度为 72。新增 chunk 后宽度继续增长。

状态、日志和评测入口已使用 `schema_version: 2` 对应字段。三个 YAML、CLI、空记忆基线调用
以及实验结果打印脚本已同步。旧配置中移除的字段会直接报错，不自动退回旧机制。

## 事务与复现

来源校验、编译规格和追加约束分别由 `events.py`、`state.py`、`runtime.py` 负责。
编译、组合、安装和真实前向都成功后才发布因子、游标和版本。失败保留 pending 原文与上一版
因子并暂停；恢复绑定失败时清除绑定，禁止继续回答。不会因失败丢弃 chunk 或跳过历史。

每个完成的 chunk 与问题立即写入独立 JSONL。最终报告保留输入指纹、原始和组合张量、
官方分数及真实成本；任何问题或 gold 都不送入 D2L 编译输入，所有评测问题读取同一冻结版本。

## 检查命令

在 `Main-Code/` 下，使用项目 Python 3.10 环境：

```bash
timeout 60s ../.venv-d2l/bin/python -m pytest tests/unit -q
../.venv-d2l/bin/python -m ruff check src tests
../.venv-d2l/bin/python -m build --no-isolation
../.venv-d2l/bin/python -m param_memory validate-config --config configs/mab_sh6k.yaml
```

离线构建复用当前环境中满足 `pyproject.toml` 要求的 `setuptools` 和 `wheel`，不从 PyPI 安装
隔离环境依赖；本轮产物放在临时目录中检查。

单元测试覆盖：超过原槽位数量后全部因子保留、每块只编译一次、chunk 触发、Unicode 连续原文、
空白事件与空白尾段、纯基座空历史、来源完整性、失败恢复、冻结快照、问题/gold 隔离及逐条日志落盘。
原生 CPU 张量测试使用小型 Qwen/PEFT 基座和上游 D2L 组合、安装函数，验证 rank=8 的 A/B 方向、
8 块组合为 64 内容宽度、共享项一次、第 9 块不覆盖旧内容、原生 scale 及重复安装。
这部分测试的内容因子使用测试张量，不是生产 checkpoint 编译质量实验。

真实 checkpoint 测试命令：

```bash
timeout 1800s ../.venv-d2l/bin/python -m pytest tests/integration -q \
  --d2l-config configs/mab_sh6k.yaml
```

## 本轮检查结果

2026-10-30（Asia/Shanghai）续接检查：

| 检查 | 结果 |
| --- | --- |
| 单元测试，外部硬超时 60 秒 | **78 passed**，包含 rank=8 的原生 CPU 张量契约测试 |
| `ruff check src tests` | 通过 |
| `git diff --check` 与实验 shell 语法 | 通过 |
| 三个 YAML 的 `validate-config` | 配置与本地资源均通过 |
| `build --no-isolation` | 源码包和 wheel 均成功生成 |
| 构建产物检查 | wheel 的 11 个模块与源码逐字节一致，旧管理模块和提示已移除；包内 CLI 校验通过 |
| 当前入口与方法文档的本地链接 | 41 个链接通过 |
| 显式配置的真实 checkpoint 集成测试 | **1 skipped**：当前进程没有 CUDA |

当前节点未执行真实 checkpoint 的编译或 SH6K 问答评测。CPU 契约测试与集成测试跳过
不能作为模型验证通过，也不能据此声称问答效果已经改善。

## 实现取舍

历史 LoRA 与复现记录都随 chunk 数增长；当前没有固定容量限制。组合时旧/新组合产物会暂时共存，
峰值与稳态占用分别记录。相关内存增长是本轮完整保留历史的直接代价。
状态校验、失败恢复和即时日志沿用既有实现边界；本轮没有增加新的记忆选择或融合策略。
