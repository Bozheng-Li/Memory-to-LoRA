# 参数记忆代码高层设计

日期：2026-10-29。版本：v0.2。

前置：沿用[当前方法方案](AGENT_MANAGED_LOCAL_GLOBAL_FLOW_20261029.md)确定的**同一冻结 Agent + 局部/全局事实记忆 + D2L**。全局范围暂按同一任务或项目内跨阶段。本文规定**目录和文件内容、代码与注释规范、独立 YAML、启动校验、运行步骤，以及每个函数的签名和契约**。

本文规定的 Python 文件和提示模板已在 [Main-Code/](../Main-Code/README.md) 实现；实现补充契约及验证边界见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。[独立 YAML 样例](../Main-Code/configs/mab_sh6k.example.yaml)中的数值仍是首轮起点，本文流程例子不代表已有模型实验结果。

## 目录

Markdown 预览中点击链接跳转；源码视图中可用 Ctrl+单击。

- [1. 总览](#1-总览)
  - [1.1 谁决定，谁执行](#11-谁决定谁执行)
  - [1.2 参数怎样合在一起](#12-参数怎样合在一起)
- [2. 目录设计与各文件内容](#2-目录设计与各文件内容)
- [3. 代码编写规范](#3-代码编写规范)
  - [3.1 参数与返回值约定](#31-参数与返回值约定)
- [4. 注释与 docstring 规范](#4-注释与-docstring-规范)
  - [4.1 文件头注释](#41-文件头注释)
  - [4.2 函数注释](#42-函数注释)
  - [4.3 行内与分支注释](#43-行内与分支注释)
  - [4.4 YAML 注释](#44-yaml-注释)
- [5. YAML 配置单列](#5-yaml-配置单列)
  - [5.1 配置分组与使用位置](#51-配置分组与使用位置)
  - [5.2 读取和校验规则](#52-读取和校验规则)
- [6. 启动时先查什么](#6-启动时先查什么)
  - [6.1 只看配置就能报的错](#61-只看配置就能报的错)
  - [6.2 加载模型后才查的内容](#62-加载模型后才查的内容)
- [7. 共享类型与命名](#7-共享类型与命名)
  - [7.1 数据类型](#71-数据类型)
  - [7.2 函数签名中的简写](#72-函数签名中的简写)
- [8. 一次记忆更新具体怎么走](#8-一次记忆更新具体怎么走)
  - [8.1 输入：先拿到历史，再拿到问题](#81-输入先拿到历史再拿到问题)
  - [8.2 先形成候选，不先占一个新槽](#82-先形成候选不先占一个新槽)
  - [8.3 管理输出只表示一份计划](#83-管理输出只表示一份计划)
  - [8.4 先算新全局能否成立，再接受局部退出](#84-先算新全局能否成立再接受局部退出)
  - [8.5 编译、组合、安装分别干什么](#85-编译组合安装分别干什么)
  - [8.6 安装成功前，任务看不到新状态](#86-安装成功前任务看不到新状态)
  - [8.7 走一遍局部替换的例子](#87-走一遍局部替换的例子)
- [9. 两份提示文件具体写什么](#9-两份提示文件具体写什么)
- [10. 文件与函数详细规格](#10-文件与函数详细规格)
  - [10.1 `__main__.py`：程序入口](#101-__main__py程序入口)
  - [10.2 config.py：YAML 与配置类型](#102-configpyyaml-与配置类型)
  - [10.3 budget.py：统一用量统计](#103-budgetpy统一用量统计)
  - [10.4 events.py：事件与工作窗口](#104-eventspy事件与工作窗口)
  - [10.5 state.py：纯状态变换](#105-statepy纯状态变换)
  - [10.6 planning.py：输出、来源与计划校验](#106-planningpy输出来源与计划校验)
  - [10.7 manager.py：管理提示与一次生成](#107-managerpy管理提示与一次生成)
  - [10.8 d2l.py：原生模型与参数操作](#108-d2lpy原生模型与参数操作)
  - [10.9 runtime.py：维护、提交与任务调用](#109-runtimepy维护提交与任务调用)
  - [10.10 evaluation.py：数据、评分与运行产物](#1010-evaluationpy数据评分与运行产物)
- [11. 测试文件与函数对应关系](#11-测试文件与函数对应关系)
- [12. 编写顺序与提交要求](#12-编写顺序与提交要求)

---

## 1. 总览

底座是同一个冻结基座，两种调用模式。D2L 把选中的文字编译成 LoRA 因子；Agent 决定保存、巩固、修订和替换什么。

~~~text
任务模式：目标 + 当前问题 + 有界 W + 已安装的局部/全局参数
    ↓ 产生新观察
工作窗口 W → 候选 c
    ↓
管理模式：关闭任务参数绑定，读取候选、局部卡片、全局事实和必要来源
    ↓ 输出 MemoryPlan
校验 → 暂存新状态 → 编译变化内容 → 组合 → 安装和 prefill → 发布新版本
    ↓
恢复任务模式
~~~

先分清四个名字：

| 名字 | 放什么 | 谁能直接读 |
| --- | --- | --- |
| W | 当前目标、近期观察、执行摘要、唯一待处理候选 | 任务与管理模式 |
| L | 局部事件来源、编译文本、卡片和内容因子 | 管理模式读文字，任务模式读已安装因子 |
| F | 有限的全局事实、条件、时间和证据引用 | 管理模式 |
| G | 从 F 的有效视图生成的全局内容因子 | 任务模式通过参数读取 |

F 是管理记录，G 是参数产物。把一条事实放进 F，不代表 D2L 已经编译成功，也不代表任务模型一定读得出来。这三个状态分别记。

### 1.1 谁决定，谁执行

~~~text
AgentManager   决定保存哪些内容，输出计划
planning.py    检查计划的身份、引用、依赖和预算
state.py       计算计划完成后的新状态，不改活跃状态
D2LBackend     编译、组合、安装、生成，不决定淘汰谁
MemoryRuntime  串行组织以上步骤，负责发布和失败恢复
~~~

例如，新候选 c5 到来、四个局部槽已经满了。管理器不能只说“删掉最老的 c2”。它需要说清 c2 中哪些事实进全局、哪些细节放弃、c5 放哪个槽。只有新全局确实装得下并编译成功，c2 才能退出。

这也是文件拆分的理由：Agent 输出的是意图，纯函数负责推导和检查，Runtime 才能让新参数对任务生效。不能在 manager.py 中直接改槽位，或在 d2l.py 中偷偷删除超预算文本。

### 1.2 参数怎样合在一起

局部和全局都由同一个 checkpoint 生成内容因子。读取时收集所有活跃内容块，统一组合，共享项只加一次：

~~~text
内容块数 N = 活跃局部块数 + 活跃全局块数
内容宽度   = N × checkpoint.rank
原生分配宽度 = (N + 1) × checkpoint.rank
~~~

按当前 YAML 的 4 个局部槽、最多 4 个全局块，以及现有 checkpoint 的 rank 8，最大内容宽度为 64，加入一次共享项后最大分配宽度为 72。空槽不占内容块；没有内容时走纯基座路径，不编译空字符串凑槽。

运行时从 checkpoint 读 rank，不把 8 或 72 写成业务代码常量。这里说的是因子宽度，不是矩阵实际秩或可无损记住多少事实。

---

## 2. 目录设计与各文件内容

~~~text
Main-Code/
├── README.md                         # 环境安装、配置填写、运行命令、输出说明
├── pyproject.toml                    # Python 包元数据、新增依赖、开发检查配置
├── configs/
│   ├── mab_sh6k.example.yaml          # 已提供：带注释的配置样例
│   └── mab_sh6k.yaml                  # 运行者根据样例创建的实际实验配置
├── prompts/
│   ├── manager.md                    # 管理指令：接纳、巩固、修订、退出和来源补查
│   └── task.md                       # 任务指令：使用当前目标、W 和参数记忆回答
├── src/param_memory/
│   ├── __init__.py                   # 包声明；不加载模型，不执行实验
│   ├── __main__.py                   # 命令行解析与程序入口
│   ├── config.py                     # YAML 读取、配置类型、配置与资源校验
│   ├── schema.py                     # 共享数据类型、枚举、项目异常
│   ├── events.py                     # 原文分片、工作窗口、候选形成与确认
│   ├── budget.py                     # token 计数、状态用量统计、统一预算校验
│   ├── state.py                      # 纯状态变换、依赖索引、编译输入构建
│   ├── planning.py                   # Agent 输出解析、来源请求与计划校验
│   ├── manager.py                    # 管理提示构建与一次管理生成
│   ├── d2l.py                        # 原生 D2L 加载、编码、编译、组合、安装、生成
│   ├── runtime.py                    # 维护流程、调用额度、提交、恢复与任务回答
│   └── evaluation.py                 # MAB 加载、事件流、官方评分和运行产物
└── tests/
    ├── unit/
    │   ├── test_config.py            # YAML 格式、重复键、字段、路径和预算配置
    │   ├── test_events.py            # 分片无损、有界候选、已处理内容离窗
    │   ├── test_budget.py            # 模板开销、完整状态用量、边界值
    │   ├── test_state.py             # 接纳、巩固、替换、纠正、来源存活
    │   ├── test_planning.py          # 输出解析、版本、来源、冲突动作与可装入性
    │   ├── test_runtime.py           # 有界重试、跳过后恢复、编译/安装失败
    │   └── test_evaluation.py        # 问题隔离、逐题同快照、原生评分
    ├── integration/
    │   └── test_d2l.py               # 真实模型、因子、共享项、模式切换与重装
    └── fixtures/
        ├── events.jsonl             # 测试事件；不是运行配置
        └── plans.jsonl              # 人工合法/非法计划；不是生产降级路径
~~~

目录按实际实现顺序建立，不先创建空业务文件。根目录 [requirements-d2l.txt](../requirements-d2l.txt)维护 D2L 依赖，pyproject.toml 只声明本项目新增依赖和开发工具，避免两个文件维护同一包的不同版本。

资源继续使用 resources/ 中已有模型、checkpoint、数据和上游代码；不复制进 src/。运行产物放在 YAML 指定的输出目录：

~~~text
<output.directory>/<run_id>/
├── resolved_config.yaml              # 本次实际解析后的配置，只供复现
├── manifest.json                     # 代码、模型、模板、数据版本和运行状态
├── maintenance.jsonl                 # 每次维护的输入身份、计划、失败和成本
├── predictions.jsonl                 # 逐题回答与官方分数
├── metrics.json                      # 汇总指标，明确分母与失败数量
├── final_snapshot.json               # 最终有界状态及来源/因子引用
└── memory.safetensors                # 最终参数记忆产物
~~~

这里的 JSON/JSONL 是数据与结果格式。**用户可编辑的运行配置统一使用 configs/ 下的 YAML，不再设计实验配置 JSON。**

---

## 3. 代码编写规范

| 项目 | 必须遵守的规则 |
| --- | --- |
| 语言与类型 | Python 3.10；公开函数、方法、关键内部函数都写参数和返回类型 |
| 命名 | 文件/函数/变量用 snake_case，类用 PascalCase；名称描述具体动作或实体 |
| 单位 | 长度、字节、时间参数使用 _tokens、_bytes、_seconds 等后缀；不同单位不能共用 size |
| 函数职责 | 一个函数完成一个可明确描述的步骤；复杂函数先拆解析、判断、执行 |
| 函数长度 | 超过 40 行先检查职责混合和分支；确有必要时解释原因，不为了行数机械拆函数 |
| 参数 | 传当前函数需要的配置子对象；完整 RunConfig 只进入入口、运行装配和评测编排 |
| 默认值 | 业务默认值写在 YAML；函数不出现模型路径、槽数、生成长度等默认常量 |
| 可变性 | 快照、记录、因子按不可变对象使用；状态函数返回新对象，不原地改调用方数据 |
| 配置访问 | config.py 读取 YAML，其他模块接收类型化配置；不能自行再次读取 YAML 或拼默认配置 |
| 边界校验 | 外部输入用严格 schema；来源、版本、预算规则各有一个实现位置 |
| 异常 | 使用明确的项目异常；只有运行编排层决定重试、拒绝或暂停，不在底层吞异常 |
| 依赖 | 优先复用 D2L 和官方评分函数；不建立通用插件、策略注册器或无调用方的接口 |
| 导入 | 标准库、第三方、本项目依次分组；模型依赖在后端启动时加载，纯状态测试不加载 GPU |
| 日志 | 使用 logging，记录事件 ID、候选 ID、版本、阶段和结果；业务函数不散落 print |

模型加载、编译、安装的调用集中在 d2l.py；控制这些动作何时发生的权力集中在 runtime.py。state.py、budget.py、planning.py 不运行模型、不写文件、不修改当前活跃快照。

保留现有可用逻辑，只增加当前功能需要的代码。不得把缺配置解释成默认值，不以空列表吞掉来源错误，不在参数后端失败时自动改走文本读取。

### 3.1 参数与返回值约定

- 必需参数没有隐式业务默认值；None 只表示契约明确允许的“没有值”。
- 带单位的数值说明单位、取值范围、来源；例如 max_tokens 为正整数、来自对应 YAML 字段。
- 不用含义不清的布尔参数控制多条业务流程；框架 API 自带的 deep=False 等参数按原生契约保留。
- 多项有业务意义的返回值使用本文定义的结果类型，不返回位置含义不明的长 tuple。
- Tensor 返回值必须说明维度、dtype/device 来源、是否允许原地修改。
- 纯函数不得修改输入；有副作用的函数在 docstring 中明确修改哪个对象、写哪些文件。

---

## 4. 注释与 docstring 规范

### 4.1 文件头注释

每个业务模块使用中文模块 docstring，写清模块责任、主要输入输出和有必要说明的边界。不要复制整份方法方案、添加逐日修改日志或作者履历。

~~~python
"""校验记忆管理计划的来源、版本与最终状态预算。

输入为结构化计划、冻结快照和当前候选，输出为待编译的合法更新。
本模块执行确定性检查；事实语义忠实性和参数可读性由独立评测检验。
"""
~~~

### 4.2 函数注释

公开函数与非显然的内部函数使用 Google 风格中文 docstring。第一句说明动作；随后按需要写 Args、Returns、Raises、Notes。下面是 check_token_limit 的注释模板，函数签名在第 10 节统一列出。

~~~python
"""校验实际送入模型的 token 序列是否超出上限。

Args:
    token_ids: 已完成对应模板处理的 token IDs，不能用原文字符数代替。
    limit: YAML 配置提供的 token 上限，必须为正整数。
    name: 被检查对象的名称，用于定位错误，例如管理输入或局部块。

Returns:
    None。通过表示序列长度不超过指定上限。

Raises:
    BudgetExceeded: token 数超过 limit。

Notes:
    不修改或截短 token_ids；超限由调用方决定重提计划或终止操作。
"""
~~~

参数已经在类型注解中表示类型时，docstring 重点说明含义、单位、范围和所有权。None 返回也要说明成功意味着什么。类的说明写用途和所持资源；构造函数说明依赖与加载副作用。

### 4.3 行内与分支注释

这些位置必须解释原因：

- 纠正后先停用旧因子，以及为什么失败时不能恢复错误块。
- 共享项只组合一次，以及为什么安装后调用 base_model.generate。
- 模式切换为什么丢弃 KV，为什么跳过计划也要恢复任务绑定。
- 来源退出前为什么复制证据，编译文本与原文为什么允许不同。
- 预算、张量维度或外部接口存在不直观约束的地方。

~~~python
# 共享项由这次组合统一加入；输入必须是尚未组合的原始内容因子。
combined = backend.compose(raw_blocks)

# 安装已经完成组合，直接使用基座生成，避免上游包装入口再次组合。
result = backend.generate(input_ids, task_generation)
~~~

禁止“循环遍历列表”“设置变量”等重复语法的注释，禁止用注释声称数据已校验而实际没有执行校验。TODO 必须说明缺什么和何时处理，不用 TODO 代替当前范围内必需的实现。

### 4.4 YAML 注释

YAML 在字段上方注明用途、单位、路径基准和影响；注释解释约束，不只是复述键名。数字配置不同时出现在 Python 默认参数、prompt 文本和另一份配置中。实际值以独立 YAML 为准。

---

## 5. YAML 配置单列

配置样例：[Main-Code/configs/mab_sh6k.example.yaml](../Main-Code/configs/mab_sh6k.example.yaml)。

运行时复制为同目录的 mab_sh6k.yaml 后调整。样例中的数值沿用当前方法的首轮建议；它不是已经验证过的模型运行配置。manager.md、task.md 和 Python 实现完成并安装本包后，从 Main-Code/ 使用：

~~~bash
python -m param_memory validate-config --config configs/mab_sh6k.yaml
python -m param_memory run-mab --config configs/mab_sh6k.yaml
~~~

命令行只选择动作与 YAML 文件；首版不增加另一套字段覆盖规则。validate-config 校验结构、数值和资源，不加载模型；模型契约在 run-mab 装配阶段校验。

### 5.1 配置分组与使用位置

| YAML 字段 | 类型与含义 | 使用位置 |
| --- | --- | --- |
| schema_version | 正整数，配置结构版本 | load_config |
| experiment.name | 非空字符串，运行名称 | run_mab、write_run_report |
| experiment.seed | 非负整数，随机种子 | run_mab，在加载模型和生成前统一设置 |
| model.d2l_root | Path，上游仓库目录 | validate_resources、D2LBackend.__init__ |
| model.checkpoint_path | Path，D2L 权重文件 | 同上 |
| model.base_model_path | Path，本地基座与 tokenizer 目录 | 同上 |
| model.chat_template_path | Path，明确选用的聊天模板 | D2LBackend.__init__ |
| model.device | 字符串，实际推理设备 | D2LBackend.__init__ |
| model.dtype | 字符串，基座加载精度 | D2LBackend.__init__ |
| model.attention_implementation | 字符串，受当前后端支持的注意力实现 | D2LBackend.__init__，不自动回退 |
| memory.local_slots | 非负整数，局部槽位总数 | empty_snapshot、check_budget |
| memory.global_blocks | 非负整数，全局编译块上限 | build_compile_specs、check_budget |
| memory.local_block_tokens | 正整数，单个局部编译输入上限 | iter_spans、build_candidate、build_compile_specs |
| memory.global_block_tokens | 正整数，单个全局编译输入上限 | build_compile_specs |
| memory.local_card_tokens | 正整数，局部卡片长度上限 | validate_plan、check_budget |
| memory.persistent_text_tokens | 正整数，全部在线持久文本 token 上限 | measure_snapshot、check_budget |
| memory.working_tokens | 正整数，工作窗口与候选总上限 | append_span、finish_candidate、check_budget |
| memory.source_read_tokens | 非负整数，同次维护累计来源补查上限 | resolve_source_request、MemoryRuntime.maintain |
| memory.max_manager_calls | 正整数，同次维护的管理生成次数上限 | MemoryRuntime.maintain |
| generation.task.max_input_tokens | 正整数，任务完整输入长度上限 | MemoryRuntime.answer |
| generation.task.max_new_tokens | 正整数，任务输出长度上限 | D2LBackend.generate |
| generation.task.do_sample | bool，任务是否采样 | D2LBackend.generate |
| generation.manager.max_input_tokens | 正整数，管理完整输入长度上限 | AgentManager.generate_reply |
| generation.manager.max_new_tokens | 正整数，管理输出长度上限 | D2LBackend.generate |
| generation.manager.do_sample | bool，管理是否采样 | D2LBackend.generate |
| prompts.manager_path | Path，管理指令文件 | AgentManager.__init__ |
| prompts.task_path | Path，任务指令文件 | MemoryRuntime.__init__ |
| data.manifest_path | Path，既有版本与测评入口 | load_eval_case |
| data.selection | 字符串，manifest 中选择的测评项 | load_eval_case |
| data.official_scorer_path | Path，固定版本官方评分代码 | load_official_scorer |
| data.protocol_instruction | 非空字符串，不含问题答案的任务规则 | run_mab，作为双方共同的当前目标 |
| output.directory | Path，运行产物父目录 | write_run_report |

### 5.2 读取和校验规则

1. 使用 PyYAML 的 SafeLoader 派生读取器，禁止重复键和非字符串键；不使用不安全的 yaml.load。
2. 严格拒绝未知字段、缺失字段、错误类型和非法枚举，不把字符串数字或字符串布尔值悄悄转换。
3. 相对路径统一相对配置文件目录解析；不依赖当前工作目录，不隐式展开环境变量或引入 include/继承配置。
4. 配置不接受手填 rank、总参数宽度或另一份模型 ID。rank 来自实际 checkpoint，总宽度从块数派生。
5. load_config 只解析与构造；validate_config 检查数值关系；validate_resources 检查文件和目录，三者调用链固定。
6. 业务模块只接收配置对象。首次运行保存 resolved_config.yaml，作为运行记录而非第二个可编辑配置入口。

config.py 定义 ExperimentConfig、ModelConfig、MemoryConfig、GenerationConfig、GenerationProfiles、PromptConfig、DataConfig、OutputConfig、RunConfig。GenerationProfiles 只有 task 和 manager 两项，两者都使用 GenerationConfig，字段不重复定义两遍。配置模型启用严格类型和 extra="forbid"；只有路径字段在 load_config 中显式从 YAML 字符串转换为 Path。

---

## 6. 启动时先查什么

启动分三步：先读 YAML，再查文件，最后加载模型核对张量。尚未加载模型时，不假装已经知道实际因子维度。

~~~text
YAML
 → load_config：严格结构、明确路径
 → validate_config：数字、枚举和字段关系
 → validate_resources：本地资源是否齐全
 → D2LBackend：加载并核对实际 checkpoint / 基座 / tokenizer / 模板
 → 开始接收事件
~~~

### 6.1 只看配置就能报的错

1. 重复键、未知字段、缺字段、错误类型立即报错。max_manager_calls 写成字符串 "2" 不能悄悄当整数。
2. 槽数和块数为非负整数，局部与全局不能同时为零；token 上限为正整数，来源补查额度可以为零。
3. 管理调用上限必须为正整数；每个生成模式的输入、输出预算分别检查。
4. 模型加载精度、设备和注意力实现必须是当前后端支持的值。
5. 路径相对 YAML 文件目录解析。无论从项目根目录还是 Main-Code/ 启动，同一配置必须指向同一文件。

错误要指出字段、实际值和约束，例如：

~~~text
memory.local_slots=-1，要求非负整数。
memory.local_slots=0 且 memory.global_blocks=0，至少需要一类记忆容量。
model.checkpoint_path 指向的文件不存在：<解析后的路径>。
~~~

这些是错误信息的格式示例，实际值由异常上下文提供。不要只写 assert failed，也不要给出一个临时默认值让程序继续。

### 6.2 加载模型后才查的内容

实际目标模块、层数、rank、A/B 维度和原生 scale 从加载对象取。base_model_path 指向的模型必须与 checkpoint 匹配；设置了另一个模型路径不代表两者兼容。

当前上游存在一份 Qwen 定制聊天模板，与基座 tokenizer 自带模板不同，所以 chat_template_path 要明确。沿用上游 get_tokenizer 的模板规范化，并记录实际生效模板的指纹。编译、预算统计和任务输入必须使用各自约定的同一处理路径；不能“计数时一种模板，调用时另一种模板”。

模型和 D2L 冻结。上下文编码器是额外的实际模型实例，其驻留内存要计入成本；任务模式和管理模式则复用同一个基座。

AgentManager 与 MemoryRuntime 装配时，将各自输入加输出的上限与基座实际上下文容量核对；编译输入上限还要与上下文编码器的容量核对。完整管理输入、实际全局装箱和暂存峰值则只有运行到相应状态才能检查，放在 generate_reply、build_compile_specs 和提交检查中。启动校验通过不代表以后所有候选都会被接纳。

---

## 7. 共享类型与命名

schema.py 使用严格的 Pydantic v2 模型定义 JSON 边界与普通状态；张量容器只在 d2l.py 使用，按只读产物处理。Pydantic 原生的 model_validate、model_dump、model_json_schema 直接复用，不再编写同名包装函数。

### 7.1 数据类型

| 类型 | 必需内容与含义 |
| --- | --- |
| Event | event_id、scope_id、text、order、event_time、observed_at、boundary；一次已发生输入 |
| SourceSpan | span_id、event_id、scope_id、order、start、end、text、source_hash、event_time、observed_at、boundary；原文半开区间，按 Unicode 码点定位 |
| SourceRef | span_id、start、end、source_hash；指向在线来源内的确定子区间，hash 核对被引用的原始 SourceSpan |
| SourceExcerpt | ref: SourceRef、text；补查返回的原文子串，位置和来源身份不因展示而改写 |
| WorkState | goal、recent_spans、summary、pending_candidate、processed_through；游标保存最后处理的事件 order、event_id 与结束位置，初始为 None；全部受工作窗口预算约束 |
| Candidate | candidate_id、source_spans、compile_text、trigger；至多一个待处理候选 |
| LocalBlock | block_id、slot_id、version、source_refs、compile_text、memory_card、claim_refs、status、factor_key |
| GlobalRecord | record_id、version、kind、claim、scope、valid_from、valid_until、observed_at、status、source_refs、evidence_group_ids、support_level、related_record_ids |
| Snapshot | scope_id、version、work、local_blocks、global_records、global_status、sources、factor_refs、last_maintenance；完整只读状态，回执受文本预算约束 |
| DependencyIndex | 来源/记录到局部块和全局记录的反向引用；由 Snapshot 派生 |
| Invalidation | 目标类型/ID/版本、来源引用、原因；已明确的纠正或失效通知 |
| SourceRequest | 公共响应头与 source_requests；每项为 span_id、max_tokens |
| MemoryPlan | 公共响应头、admission、目标槽、candidate_card、全局增删改、局部修订/退出、退出依赖、invalidations、简短理由 |
| ManagerReply | SourceRequest 与 MemoryPlan 的带标签联合类型 |
| ValidationFeedback | code、field、message；可交给下一次管理调用的有界错误信息 |
| SourceReadResult | excerpts: tuple[SourceExcerpt, ...]、tokens_used；实际交付的来源片段和本次读取 token 数 |
| StagedUpdate | base_version、候选新快照、变化局部块、全局视图变化标记；尚未提交 |
| ValidatedPlan | 通过校验的计划、StagedUpdate、CompileSpec 列表和 BudgetUsage |
| CompileSpec | block_key、local/global 类型、编译文本、token_ids、来源/记录版本、输入指纹 |
| FactorRef | block_key、输入指纹、模块/shape/dtype、内容宽度和真实字节；不包含可执行代码 |
| CompiledMemory | specs、按 block_key 保存的原始因子、完整组合因子和 FactorRef；只覆盖当前/单次暂存需要的对象 |
| BudgetUsage | 槽位、块长度、卡片、持久文本、W、因子宽度/字节等实际用量 |
| GenerationResult | text、生成 token IDs、输入/输出 token 数、耗时；保留原始文本，不替 Agent 修补 JSON |
| MaintenanceResult | status、候选/计划 ID、前后版本、错误/遗忘原因、调用次数与成本 |
| EvalCase | 历史 ID、context、questions、answers、必要 QA ID；只由评测外壳持有 |
| Prediction | 历史/问题 ID、快照版本、原始回答、原生分数和独立诊断 |
| RunReport | 已解析配置、版本信息、维护记录、逐题结果、成本、最终快照与因子、完成状态 |

公共响应头包含 response_type、schema_version、plan_id、base_state_version、candidate_id。SourceRequest 的 response_type 为 source_request，MemoryPlan 为 plan。新增记录使用计划内别名，由 state.py 生成稳定 ID；已有对象更新必须带预期版本。

MemoryPlan.admission 为 local、global_only 或 skip；local 时必须同时提供目标槽和 candidate_card，卡片来自同次管理输出，不能暗中增加一次摘要调用。局部初次编译使用 Candidate.compile_text；修订则提供有来源的新编译文本。

枚举定义在 schema.py：Trigger 为 event_end/window_full/correction/history_end；记录种类为 fact/constraint/observation；记录状态为 active/historical/unresolved/invalid；局部执行状态为 active/disabled；维护结果为 committed/skipped/aborted/paused。

boundary 的类型为 Trigger | None，表示输入外壳已知的维护边界；事件分片仅最后一片继承该边界。普通 MAB 编号事实使用 None，候选长度与历史结束仍会触发维护；不能根据未来测试问题生成标签。event_time 未知时用 None，不用 observed_at 冒充事件发生时间。

SourceRef 的 start/end 使用原事件的 Unicode 码点位置，并且必须落在对应 SourceSpan 的区间内。source_hash 使用该原文片段 UTF-8 字节的 SHA-256；展示补查片段时仍引用原 SourceSpan，独立保留证据时才生成新 SourceSpan 并改写依赖。

Snapshot.version 在发布时单调增加；StagedUpdate 保留 base_version，候选快照拟用 base_version + 1。global_status 为 active 或 disabled：纠正涉及全局时先将整个 G 标为 disabled；合法计划完成全局重建后才能恢复 active。disabled 局部块的来源和记录仍计入文本预算，但不进入活跃编译规格。

全局有效视图保留 active 事实，以及带历史限定的 historical 记录、明确陈述冲突未决的 unresolved 记录；invalid 只留管理侧核对，不参与编译。序列化必须保留相应状态、范围和时间，不能将历史值或未决命题写成当前已确认事实。

WorkState.pending_candidate 初始为 None。build_candidate 只返回候选；maintain 核对其来源与工作游标后登记为唯一 pending。一个候选终结后推进游标，剩余未处理片段仍留在 W；不因为形成过候选就把整个窗口当作已处理。

Runtime 只持有当前 Snapshot、对应 CompiledMemory、运行状态（ready/maintaining/paused/frozen）、配置依赖和当前维护账本。维护结束后只在 Snapshot 留最后一份有界回执；完整 MaintenanceResult 交给评测外壳收集，Agent 不能读完整运行日志。

MemoryPlan 的业务字段使用固定名称，避免实现时各模块自行约定：

| 字段 | 内容与约束 |
| --- | --- |
| admission / target_slot / candidate_card | local、global_only 或 skip；只有 local 可带目标槽和必需卡片 |
| global_upserts | 新增或修订的完整全局记录；新增带计划内别名，修订带 record_id 和 expected_version |
| global_deletes | 待删除记录的 record_id、expected_version 与明确遗忘原因 |
| local_revisions | block_id、expected_version、新 compile_text、新 memory_card 与 source_refs |
| local_exits | block_id、expected_version 与退出原因；不把目标槽里的旧块默认为可删 |
| exit_dependencies | 每个退出块所需保留的全局记录 ID/计划内别名，以及明确放弃的内容 |
| invalidations | 已识别失效目标、预期版本、纠正来源和原因；优先于普通动作处理 |
| reason | 同次输出中的简短选择理由，受管理输出总上限限制，不进入事实编译文本 |

skip 的普通更新列表必须为空；它不能撤销有效 invalidations。SourceRequest 与 MemoryPlan 都必须通过公共响应头的版本、作用域和候选身份检查。

### 7.2 函数签名中的简写

~~~python
TokenIds = tuple[int, ...]
EncodeText = Callable[[str], TokenIds]
ChatMessages = tuple[ChatMessage, ...]
ScoreFn = Callable[[str, tuple[str, ...]], float]
~~~

ChatMessage 只有 role 和 content，role 使用模型支持的固定枚举。参数 encode_text 表示不加聊天模板的编码；encode_context 表示 D2L 原生上下文模板化后的编码。两者类型相同、用途不同，不能混用。

RawFactors 和 CombinedFactors 是不同语义的只读张量容器，均按模块保存 A/B：

~~~text
RawFactors.A/B：     [1, n_layers, r, d_in/d_out]，单个内容块
CombinedFactors.A/B：[1, n_layers, allocated_width, d_in/d_out]，完整读取产物
~~~

FactorLayout 保存从实际模型读取的模块、层、rank、维度、dtype/device 和预期宽度。所有 Tensor 都禁止原地修改。schema 的类型注解需要 Tensor 时使用延迟注解/TYPE_CHECKING，不能因此在 CPU 状态测试导入时加载模型。

项目异常集中在 schema.py：ConfigError、BudgetExceeded、PlanValidationError、BackendError、StateError、EvaluationError。它们继承标准异常，不定制无必要的包装层；保留原异常链与出错字段。

---

## 8. 一次记忆更新具体怎么走

### 8.1 输入：先拿到历史，再拿到问题

首轮 MAB 的外壳持有完整数据行，但只按顺序把 context 事件交给 Runtime。questions 在历史处理完后才交给 answer；answers 只交给 score_prediction。

~~~text
load_eval_case
  → context → iter_history_events → iter_spans → runtime.observe
  → 历史结束，逐个处理剩余候选
  → runtime.freeze
  → questions → runtime.answer
  → answers + prediction → score_prediction
~~~

入口调用关系为 main → load_config → validate_config → validate_resources → run_mab → write_run_report。run_mab 统一设种子、装配后端与 Runtime，再执行下面的历史/问题协议。

每题都从相同冻结版本和相同 W 开始。第 1 题的回答不能写入第 2 题上下文，否则测到的就包含题间信息积累。编号越大事实越新的规则从 YAML 的 protocol_instruction 给双方，不拿未来问题帮助选择记忆。

### 8.2 先形成候选，不先占一个新槽

observe 接收当前 SourceSpan，append_span 更新有界工作窗口。到候选长度、事件边界或历史结束时，build_candidate 选出一个带来源的候选。

~~~text
已提交局部池：s1=c1, s2=c2, s3=c3, s4=c4
新输入：c5，只存在于有界 W / Candidate 中

此时没有 s5，也没有 c5 的已安装因子。
~~~

候选可能被拒绝，所以此时不编译、不分配第 K+1 个活跃块。W 放不下新片段时先维护已有待处理内容，再继续交付；不能把尚未评估的事件直接挤掉。

### 8.3 管理输出只表示一份计划

Runtime 先 reset 任务绑定，管理器再读取显式材料，输出 SourceRequest 或 MemoryPlan。

~~~text
source_request：
    需要哪些已有 span_id，每项最多读多少 token

plan：
    基于哪个快照版本、处理哪个候选
    候选进入哪个槽，或仅更新全局，或跳过
    全局新增 / 修订 / 删除哪些记录
    哪些局部块修订或退出
    退出前必须保留哪些全局记录，以及明确放弃什么
~~~

补查、JSON 修复、预算修复与纠正后的重提共用 YAML 中的 max_manager_calls。用掉一次补查后只剩剩余额度，不再额外获得一轮“格式修复重试”。

若响应明确指出已知错误，先校验失效通知并发布安全状态，再基于新版本提修复计划。不能用“这份普通计划还没通过预算”为理由继续读取已知错误。

### 8.4 先算新全局能否成立，再接受局部退出

validate_plan 先核对来源、版本和动作，再让 stage_plan 算出新状态：

~~~text
旧 Snapshot + Candidate + MemoryPlan
    → 暂存新局部 / 新 F / 独立保留证据
    → build_compile_specs
    → 检查槽位、单块长度、固定装箱、全部在线文本、W 和依赖
~~~

全局记录稳定排序后做 next-fit 装箱：当前块放不下，才开下一块。每次判断都包含实际编译模板开销；单条记录太长或 J 块用完就拒绝，不静默删条件、否定或时间。

例如，全局总 token 数低于总上限，也可能因为几条不可拆记录放不进指定数量的块而失败。此时 c2 不能先退出，再假定总会有办法把它的重要事实塞进 G。

引用存在只能证明证据对象存在，不能证明 Agent 的文字完全忠实；这个边界由独立事实/任务测评检查，代码不伪装成语义判别器。

### 8.5 编译、组合、安装分别干什么

~~~text
backend.compile(spec)
    一份文本 → 一个原始内容块；不加共享项

backend.compose(raw_blocks)
    所有活跃局部 + 全局 → 一份组合因子；共享项只加一次

backend.install(combined)
    reset → patch_lora_forward → apply_lora_to_layers

backend.generate(input_ids, generation)
    调 base_model.generate，使用已经安装的绑定
~~~

这里不能把已经 combine 的产物再次传给 ModulatedPretrainedModel.generate，因为该入口还会组合。也不能每个块先加共享项，再把成品相加。

当前原生 A 是 r×d_in，B 是 r×d_out，单层增量是有效 scale 乘 B 的转置再乘 A。forward 沿用原生 lora_alpha，不按拼接后的总宽度再除一次。

局部只重编译变化块；全局视图未变就复用，变化时首版重编译全部全局块。复用的是张量，不是旧来源版本：新快照仍要绑定当前记录/来源与正确输入指纹。

### 8.6 安装成功前，任务看不到新状态

~~~text
旧有效 Snapshot / CompiledMemory
    ↓ 保留作本次安全前态
暂存新状态 → 编译 → 计算终结 W / 回执 → 安装 → 对待发布 W 做 prefill
    ↓ 全部成功
一次性更新 Runtime 当前快照和因子指针
    ↓
回收无引用旧来源和旧产物，恢复任务
~~~

失败发生在发布前，普通更新回到旧有效状态；已经确认纠正时，只能回到不含旧错误的安全状态。恢复失败就暂停，不能继续输出一段实际上没有安装正确记忆的回答。

首版不跨顶层调用复用 KV。reset 处理适配器绑定，KV 由 Runtime/调用生命周期主动释放；管理模式临时上下文不能成为任务模式的隐性记忆。

skip 也要经过恢复路径。原因很具体：管理生成前已经清除了任务绑定，即使记录没变，模型也未必仍挂着原来的参数。恢复成功后才确认普通候选的游标与有界回执；后续任务重新编码这个已确认的 W，不复用恢复检查的 KV。

已提交状态受稳态预算约束；一次提交暂存旧状态与新状态及临时编译结果，峰值必须单独报告，不能声称始终只分配稳态宽度。暂存失败释放本次新对象并恢复安全前态，不建立无限的待编译队列或历史因子缓存。

### 8.7 走一遍局部替换的例子

下面是设计中的预期流程，不是已经测出的模型成绩。使用样例 YAML 的四个局部槽，全局起初为空：

~~~text
s1=c1：项目持续约束
s2=c2：service_A 测试环境一次 429，以及当时的等待和重试日志
s3=c3：尚未完成的故障线索
s4=c4：当前阶段的中间结果

c5：后续步骤需要的新信息
~~~

管理器决定从 c2 保留一条具体观察：

~~~text
g_retry：
  本次测试环境 429 携带 Retry-After: 3；
  当时等待 3 秒后重试成功。
  source_refs = [c2 中实际存在的那段来源]
~~~

执行不是“先把 s2 清空”，而是：

~~~text
1. stage_plan：把这段来源复制为新全局记录可独立引用的证据。
2. build_compile_specs：检查这条记录可装入全局块，以及新状态的全部预算。
3. compile：编译 c5 和新的全局块；c1/c3/c4 的原始因子复用。
4. compose：4 个局部 + 1 个全局，一共 5 个内容块。
5. install + prefill 成功，才发布 s2=c5 和 g_retry 对应的新 G。
6. 回收 c2 的其余原文；被保留的独立证据继续存活。
~~~

按当前 rank 8，这次实际内容宽度为 40，加一次共享项后分配宽度为 48；没有因为配置允许 4 个全局块就填满所有空块。

若第 3 步编译失败，s2 仍是 c2，新全局没有对任务生效。若 c5 只是重复噪声，管理器可以直接 skip，不因有新输入就强制替换。

再假设后续来源明确纠正：“实际等待是 5 秒，3 只是响应字段。”需要改的是同一次事件的等待值，Retry-After: 3 保留：

~~~text
先停用含错误等待值的 G，并清理 W 中把旧值当当前事实的内容
    → 基于新安全版本修订 g_retry，引用纠正来源
    → 重编译 G
    → 安装成功后恢复参数读取
~~~

纠正编译失败时也不能恢复“实际等了 3 秒”的旧 G。来源只说明后来另一请求等待 5 秒时，则是新增观察，不能覆盖原来那一次事件。

---

## 9. 两份提示文件具体写什么

| 文件 | 固定内容 | 每次调用动态加入 |
| --- | --- | --- |
| prompts/manager.md | 保留事实与条件、先巩固后比较剩余价值、允许拒绝候选、来源引用、只输出 JSON、禁止把推测写成事实 | 当前合法目标、候选、局部卡片、全局核心记录、预算、生成的 schema、补查片段/错误反馈 |
| prompts/task.md | 使用当前合法上下文和已安装参数回答、遵守任务输出格式、不把未知事实编造为确定结果 | 当前目标、预算内 W、此时交付的问题 |

提示文件不保存模型路径、槽数、token 上限或数据集题数；这些从 YAML 和运行状态提供。管理输出结构只从 schema.py 生成，不在 prompt 中维护第二份手写 JSON Schema。

来源中的指令作为数据呈现，不进入系统指令层。管理器只输出计划与来源请求，不能因来源文本要求而执行工具操作。预计用途、管理理由和内部 hash 不进入事实编译文本。

---

## 10. 文件与函数详细规格

以下覆盖首版设计的 52 个自定义函数和方法。每个条目给出签名、功能、参数、返回、异常/副作用和调用方；函数体中的省略号是接口规格，不是生产代码。实现新增的辅助函数契约列于 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

### 10.1 `__main__.py`：程序入口

#### parse_args

~~~python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace: ...
~~~

- 功能：解析 validate-config 或 run-mab 子命令，以及必需的 --config。
- 参数：argv 为显式参数序列；None 使用当前进程参数。
- 返回：只包含命令名称和配置路径的 Namespace，不夹带另一套业务默认值。
- 异常/副作用：非法 CLI 参数遵循 argparse 的退出约定；不读数据或模型。
- 调用方：main。

#### main

~~~python
def main(argv: Sequence[str] | None = None) -> int: ...
~~~

- 功能：按固定顺序加载、校验配置和资源；执行配置检查或 run_mab，然后写出结果。
- 参数：argv 传给 parse_args。
- 返回：进程退出码；成功为 0，失败为非零，并说明失败阶段。
- 异常/副作用：配置日志并调用下层；只在入口转换预期项目异常为失败退出，未知错误保留堆栈。
- 调用方：python -m param_memory；入口末尾使用 raise SystemExit(main())。

### 10.2 config.py：YAML 与配置类型

#### UniqueKeySafeLoader.construct_mapping

~~~python
def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[str, object]: ...
~~~

- 功能：在 PyYAML SafeLoader 的映射构建边界拒绝重复键、非字符串键和合并键。
- 参数：node 为当前 YAML mapping 节点；deep 沿用 PyYAML 原生构造参数；self 为当前读取器。
- 返回：键唯一的字符串字典，供严格配置模型校验。
- 异常/副作用：非法节点抛 ConfigError，包含文件位置；不修改全局 SafeLoader 注册表。
- 调用方：load_config 使用的局部 SafeLoader 子类；这是唯一的自定义 YAML 读取扩展。

#### load_config

~~~python
def load_config(config_path: Path) -> RunConfig: ...
~~~

- 功能：读取一份 YAML，严格构造配置类型，将所有路径相对配置文件目录解析为绝对 Path。
- 参数：config_path 为现有 YAML 文件路径；不接受目录或隐式默认路径。
- 返回：RunConfig；数值不补默认，不合并环境变量或其它配置文件。
- 异常/副作用：文件、编码、YAML 或字段错误抛 ConfigError；只读取配置文件。
- 调用方：main。

#### validate_config

~~~python
def validate_config(config: RunConfig) -> None: ...
~~~

- 功能：检查取值范围、局部/全局总容量非零、调用额度、支持的设备/精度/解码参数和数值关系。
- 参数：config 为 load_config 产生的类型化配置。
- 返回：None 表示静态配置约束成立，不代表模型已可运行。
- 异常/副作用：冲突或不支持的配置抛 ConfigError；不修改输入，不加载模型。
- 调用方：main，在资源校验前执行。

#### validate_resources

~~~python
def validate_resources(config: RunConfig) -> None: ...
~~~

- 功能：检查本地模型、tokenizer、checkpoint、模板、数据入口和评分源码存在且类型正确。
- 参数：config 中路径已经解析；输出父目录可以尚未创建，但不能指向已有文件。
- 返回：None 表示必需文件可访问；checkpoint 张量兼容性留给后端加载。
- 异常/副作用：缺失或错误路径抛 ConfigError；不下载、不切换工作目录、不写文件。
- 调用方：main；设计阶段尚未创建的 prompts 文件会在真实运行时被明确拒绝。

### 10.3 budget.py：统一用量统计

#### count_text_tokens

~~~python
def count_text_tokens(text: str, encode_text: EncodeText) -> int: ...
~~~

- 功能：计算不加聊天模板的真实 token 数，供记录、卡片和稳定序列化文本计账。
- 参数：text 为完整待计数文本；encode_text 为当前模型 tokenizer 的原始编码函数。
- 返回：非负 token 数，不使用字符数、词数或近似换算。
- 异常/副作用：编码错误向上传递；不截断或修改文本。
- 调用方：measure_snapshot、来源读取和卡片校验。

#### check_token_limit

~~~python
def check_token_limit(token_ids: TokenIds, limit: int, name: str) -> None: ...
~~~

- 功能：检查已经完成对应模板处理的序列长度。
- 参数：token_ids 为实际输入 IDs；limit 为配置中的正整数上限；name 标识具体对象。
- 返回：None 表示长度小于或等于上限。
- 异常/副作用：超限抛 BudgetExceeded，带 name、实际值和上限；不裁剪。
- 调用方：事件分片、编译规格、管理生成和任务回答。

#### measure_snapshot

~~~python
def measure_snapshot(snapshot: Snapshot, specs: tuple[CompileSpec, ...], encode_text: EncodeText) -> BudgetUsage: ...
~~~

- 功能：按稳定序列化统计来源、编译文本副本、卡片、记录、索引、回执、W 和因子引用的用量。
- 参数：snapshot 可以是提交态或暂存态；specs 为同一状态的完整活跃编译输入；encode_text 必须与运行 tokenizer 一致。
- 返回：BudgetUsage；块输入长度使用 len(spec.token_ids)，文本/W 使用稳定序列化后的实际编码数，不能用裸文本 token 数代替模板后的编译长度。
- 异常/副作用：结构损坏抛 StateError；只读统计，不通过删除字段使预算通过。
- 调用方：validate_plan、运行时提交检查和输出统计；实际张量字节由后端补充。

#### check_budget

~~~python
def check_budget(usage: BudgetUsage, limits: MemoryConfig) -> None: ...
~~~

- 功能：统一检查槽位、块数、块长度、卡片、在线文本和工作窗口上限。
- 参数：usage 为实际统计；limits 为唯一的 memory 配置。
- 返回：None 表示所有适用上限同时满足；全局可装入性还由 build_compile_specs 检查。
- 异常/副作用：任一超限抛 BudgetExceeded；不擅自淘汰记录、增大配置或忽略 disabled 块。
- 调用方：validate_plan、MemoryRuntime 的提交与失效处理。

### 10.4 events.py：事件与工作窗口

#### iter_spans

~~~python
def iter_spans(event: Event, max_tokens: int, encode_context: EncodeText) -> Iterator[SourceSpan]: ...
~~~

- 功能：优先按事件/句子边界拆分原文，每次交付一个能装入局部编译上限的片段。
- 参数：event 由输入外壳持有；max_tokens 来自 local_block_tokens；encode_context 包含真实 D2L 模板。
- 返回：按原文顺序的 SourceSpan 迭代器，记录原始字符位置和 hash；连接片段应恢复原始文本。
- 异常/副作用：连必要最小片段都装不下时抛 BudgetExceeded；不先创建全部分片列表，不用 token 解码结果冒充原文。
- 调用方：run_mab 的输入外壳；Runtime 只接收迭代器当前交付的片段。

#### append_span

~~~python
def append_span(work: WorkState, span: SourceSpan, limits: MemoryConfig, encode_text: EncodeText) -> WorkState: ...
~~~

- 功能：把一个新片段加入工作状态，保留顺序和来源；可释放已经处理且允许离窗的近期内容。
- 参数：work 为旧工作状态；span 为当前片段；limits.working_tokens 为总上限；encode_text 用于真实计账。
- 返回：新的 WorkState，不改变传入 work；已有未完成候选时不得追加第二个候选。
- 异常/副作用：需要先维护才能接纳时抛 BudgetExceeded，由 observe 先处理已有工作；不丢弃未处理事件。
- 调用方：MemoryRuntime.observe。

#### build_candidate

~~~python
def build_candidate(work: WorkState, trigger: Trigger, limits: MemoryConfig, encode_context: EncodeText) -> Candidate | None: ...
~~~

- 功能：按原顺序从未处理片段中取一个能装入上限的连续前缀，形成候选并绑定来源、编译文本和触发原因。
- 参数：work 为冻结工作状态；trigger 为事件闭合、窗口满、纠正或历史结束；limits 提供长度上限；encode_context 核对候选整体的实际编译长度。
- 返回：Candidate；没有待处理内容时返回 None，不用空字符串编译占槽。
- 异常/副作用：已有候选身份冲突抛 StateError，不能形成合法分片抛 BudgetExceeded；不安装参数。
- 调用方：MemoryRuntime.observe，以及 run_mab 结束历史时的最终维护。

#### finish_candidate

~~~python
def finish_candidate(work: WorkState, candidate_id: str, limits: MemoryConfig, encode_text: EncodeText) -> WorkState: ...
~~~

- 功能：在候选已接纳、明确跳过或普通失败被放弃后，清除 pending 并推进处理游标。
- 参数：work 为含当前 pending 的工作状态；candidate_id 必须匹配该候选；limits 提供窗口上限；encode_text 用于真实计账。
- 返回：新的有界 WorkState；仍可按正常近期窗口保留刚观察到的事件。
- 异常/副作用：身份不匹配抛 StateError；不允许用此函数确认尚未处理完的纠正或 paused 候选。
- 调用方：MemoryRuntime.maintain 的终结路径。

### 10.5 state.py：纯状态变换

#### empty_snapshot

~~~python
def empty_snapshot(scope_id: str, local_slots: int, goal: str) -> Snapshot: ...
~~~

- 功能：创建指定作用域的空快照、固定局部槽位和初始工作目标。
- 参数：scope_id 来自当前任务/源历史身份；local_slots 来自 YAML；goal 为已经交付的任务规则。
- 返回：没有局部/全局内容因子的 Snapshot，空槽不产生 D2L 输出。
- 异常/副作用：非法身份或容量抛 StateError；不加载模型、不生成随机业务 ID。
- 调用方：run_mab 为每份源历史创建新运行状态。

#### build_dependency_index

~~~python
def build_dependency_index(snapshot: Snapshot) -> DependencyIndex: ...
~~~

- 功能：从记录中的真实引用派生来源、事实、局部块和全局块之间的反向索引。
- 参数：snapshot 为当前完整状态；引用只能指向本作用域在线可访问的对象。
- 返回：DependencyIndex；不建立第二份可独立编辑的引用计数。
- 异常/副作用：悬空引用抛 StateError；不从日志或历史文件补找来源。
- 调用方：计划校验、失效定位、来源回收与完整计账。

#### stage_plan

~~~python
def stage_plan(snapshot: Snapshot, candidate: Candidate, plan: MemoryPlan) -> StagedUpdate: ...
~~~

- 功能：在隔离副本上应用接纳、全局修订、局部修订/退出；为需保留的事实复制独立来源片段。
- 参数：snapshot 为计划基于的旧版本；candidate 为当前候选；plan 已完成结构、身份和引用检查。
- 返回：StagedUpdate，含新状态与变更范围；版本尚未发布，退出前态仍保留。
- 异常/副作用：动作冲突或无法解析的依赖抛 PlanValidationError；不编译、不改活跃状态、不释放旧来源。
- 调用方：validate_plan；Runtime 不绕过校验直接使用此结果。

#### build_compile_specs

~~~python
def build_compile_specs(update: StagedUpdate, limits: MemoryConfig, encode_context: EncodeText) -> tuple[CompileSpec, ...]: ...
~~~

- 功能：构建局部编译输入及全局有效记录的稳定序列化，确定性装入最多 J 个全局块。
- 参数：update 提供候选新状态；limits 提供两层长度/块数；encode_context 计算含模板的准确输入。
- 返回：全部活跃块的 CompileSpec，绑定实际 IDs、文本、来源/记录版本和输入指纹。
- 异常/副作用：单条记录或固定装箱超限抛 BudgetExceeded；不截断、不按未来问题筛选。
- 调用方：validate_plan；全局按 record_id 稳定顺序 next-fit，紧密关系由 Agent 在记录中完整表达。

#### disable_invalidated

~~~python
def disable_invalidated(snapshot: Snapshot, notices: tuple[Invalidation, ...]) -> Snapshot: ...
~~~

- 功能：生成排除已知失效断言的安全状态，定位并停用相关局部块；全局受影响时停用 G。
- 参数：snapshot 为失效前状态；notices 已通过来源、作用域和版本校验。
- 返回：新安全 Snapshot，保留预算内修复来源；W 中已知错误的片段/摘要同步移除或明确标失效。
- 异常/副作用：无法构造合法安全上下文时抛 StateError；不把未知新值编成事实，不修改原快照。
- 调用方：MemoryRuntime._apply_invalidations；参数安装和版本发布由 Runtime 完成。

### 10.6 planning.py：输出、来源与计划校验

#### parse_manager_reply

~~~python
def parse_manager_reply(text: str) -> ManagerReply: ...
~~~

- 功能：将一次完整管理输出解析成 SourceRequest 或 MemoryPlan，并执行严格结构校验。
- 参数：text 为 GenerationResult.text 的原始 JSON 文本；不接受代码围栏或混杂说明文字。
- 返回：带明确 response_type 的 ManagerReply，不能用空计划代替解析失败。
- 异常/副作用：非法 JSON、截断、未知字段、错误类型抛 PlanValidationError；不运行模型、不修补输出。
- 调用方：MemoryRuntime.maintain；生成成本在解析前已经记账。

#### validate_invalidations

~~~python
def validate_invalidations(snapshot: Snapshot, candidate: Candidate, notices: tuple[Invalidation, ...]) -> tuple[Invalidation, ...]: ...
~~~

- 功能：检查失效通知的目标身份、内容版本、作用域和真实纠正来源。
- 参数：snapshot 为当前安全前态；candidate 提供本次新来源；notices 来自已通过结构校验的 MemoryPlan.invalidations。
- 返回：通过确定性校验的通知序列；校验通过不等于语义判别器证明其真伪。
- 异常/副作用：引用/版本错误抛 PlanValidationError；不自行识别所有自然语言冲突，不修改状态。
- 调用方：MemoryRuntime.maintain，在普通计划预算/动作校验之前执行。

#### resolve_source_request

~~~python
def resolve_source_request(snapshot: Snapshot, candidate: Candidate, request: SourceRequest, remaining_tokens: int, encode_text: EncodeText) -> SourceReadResult: ...
~~~

- 功能：按允许的片段 ID 交付真实来源，检查每项请求及本次维护剩余总额度。
- 参数：snapshot/candidate 是唯一可读取来源；request 提供片段 ID 和长度；remaining_tokens 为调用账本剩余额度；encode_text 用于计账。
- 返回：SourceReadResult；按请求顺序从对应原文片段开头取不超过每项额度的前缀，保持字符边界和 SourceRef，累计真实 token 消耗；不解码 token 拼造原文。
- 异常/副作用：不存在、跨 scope、越界引用抛 PlanValidationError，超限抛 BudgetExceeded；不读任意文件。
- 调用方：MemoryRuntime.maintain；调用方将实际消耗扣入同一维护账本。

#### validate_plan

~~~python
def validate_plan(snapshot: Snapshot, candidate: Candidate, plan: MemoryPlan, limits: MemoryConfig, encode_text: EncodeText, encode_context: EncodeText) -> ValidatedPlan: ...
~~~

- 功能：校验版本、来源、互斥动作、退出依赖；调用 stage_plan、build_compile_specs 和统一预算函数验证最终新状态。
- 参数：snapshot/candidate 必须与计划身份一致；plan 为已解析计划；limits 为 memory 配置；encode_text 用于文本计账，encode_context 用于真实编译输入。
- 返回：ValidatedPlan，包含实际暂存状态、编译规格和用量；不能只有一个 passed=true 标志。
- 异常/副作用：非法动作/引用抛 PlanValidationError，容量或装箱失败抛 BudgetExceeded；不编译或发布。
- 调用方：MemoryRuntime.maintain；同一对象不能同时删除与修订，退出依赖必须在新状态真实存在。

### 10.7 manager.py：管理提示与一次生成

#### AgentManager.__init__

~~~python
def __init__(self, backend: D2LBackend, prompt_path: Path, generation: GenerationConfig) -> None: ...
~~~

- 功能：绑定共享基座后端，读取管理提示，保存本模式生成配置，并核对模型上下文容量。
- 参数：backend 为任务与管理共同使用的同一实例；prompt_path 来自 YAML；generation 为 generation.manager。
- 返回：None，构造一个不持有独立模型、不持有全历史的管理器。
- 异常/副作用：模板文件不可读抛 ConfigError；只读取模板，不启动生成、不自行安装适配器。
- 调用方：run_mab。

#### AgentManager.build_messages

~~~python
def build_messages(self, snapshot: Snapshot, candidate: Candidate, limits: MemoryConfig, source_spans: tuple[SourceExcerpt, ...] = (), feedback: ValidationFeedback | None = None) -> ChatMessages: ...
~~~

- 功能：组织目标、候选、局部卡片、全局核心记录、预算、输出 schema，以及本轮补查/修复信息。
- 参数：snapshot 为管理可见状态；candidate 为新输入；limits 提供预算；source_spans 只能来自合法补查；feedback 为上轮有界错误。
- 返回：ChatMessages；输出 schema 由共享类型生成，未来问题、答案和完整 metadata 不得进入。
- 异常/副作用：必需输入无法组织时抛 PlanValidationError；不访问模型 KV 或离线日志，不修改快照。
- 调用方：MemoryRuntime.maintain。

#### AgentManager.generate_reply

~~~python
def generate_reply(self, messages: ChatMessages) -> GenerationResult: ...
~~~

- 功能：编码管理输入、检查本模式输入上限，调用共享后端完成一次生成。
- 参数：messages 来自 build_messages；前置条件是 Runtime 已清除任务参数绑定。
- 返回：原始 GenerationResult；解析、来源补查和重提由 Runtime 决定。
- 异常/副作用：超长抛 BudgetExceeded，生成失败抛 BackendError；消耗一次真实模型调用，不自行重试。
- 调用方：MemoryRuntime.maintain，每次调用前先确认统一调用账本仍有额度。

### 10.8 d2l.py：原生模型与参数操作

#### D2LBackend.__init__

~~~python
def __init__(self, config: ModelConfig) -> None: ...
~~~

- 功能：加载本地基座、上下文编码器与 hypernetwork，冻结权重，设置实际聊天模板，建立只读模型布局信息。
- 参数：config 为 model 配置；所有路径已解析，注意力实现和基座 dtype 已显式给出。
- 返回：None；实例持有 model、tokenizer、实际 FactorLayout/编译器指纹，不另外创建管理模型。
- 异常/副作用：文件/模型不匹配抛 BackendError；分配 CPU/GPU 内存，不修改原 checkpoint，不下载替代模型。
- 调用方：run_mab；上下文编码器的独立驻留内存必须计入实际成本。

#### D2LBackend.encode_text

~~~python
def encode_text(self, text: str) -> TokenIds: ...
~~~

- 功能：使用当前 tokenizer 对原始文本编码，不添加聊天模板。
- 参数：text 为预算统计对象，不能传入已拼接的其它模式缓存。
- 返回：TokenIds，供 count_text_tokens 和状态序列化计账使用。
- 异常/副作用：编码失败向上传递；不运行模型、不截断、不保留历史。
- 调用方：预算、事件窗口、来源校验；以 EncodeText 回调形式传入纯函数。

#### D2LBackend.encode_context

~~~python
def encode_context(self, text: str) -> TokenIds: ...
~~~

- 功能：沿用上游 tokenize_ctx_text 的上下文处理，包含空 system、user 内容和 assistant 生成前缀。
- 参数：text 是实际编译文本；上游允许的 strip 只作用于编译视图，不能改写原始证据。
- 返回：完整真实编译 IDs，后续 compile 必须使用同一序列。
- 异常/副作用：模板不支持或编码失败抛 BackendError；不静默截断、不编译权重。
- 调用方：iter_spans、build_candidate、build_compile_specs。

#### D2LBackend.encode_messages

~~~python
def encode_messages(self, messages: ChatMessages) -> TokenIds: ...
~~~

- 功能：用当前明确选定的聊天模板编码任务或管理消息。
- 参数：messages 为结构化消息序列；schema 已约束角色与内容类型。
- 返回：包含模板和生成前缀的完整输入 IDs，用同一结果校验长度并生成。
- 异常/副作用：非法消息或模板失败抛 BackendError；不复用前次 KV，不运行模型。
- 调用方：AgentManager.generate_reply、MemoryRuntime.answer 和提交前 prefill。

#### D2LBackend.compile

~~~python
def compile(self, spec: CompileSpec) -> RawFactors: ...
~~~

- 功能：把一份明确有界的编译输入交给 generate_weights，验证实际输出后返回原始内容因子。
- 参数：spec 来自 ValidatedPlan，携带 token_ids、文本/来源版本和输入指纹。
- 返回：单块 RawFactors；rank、层和模块维度取自实际 checkpoint，不添加共享项。
- 异常/副作用：输入不一致、额外不支持产物、非法张量或 OOM 抛 BackendError；占用编译资源，不安装参数。
- 调用方：MemoryRuntime._compile_plan；不得通过反复 internalize 覆盖可变全局缓存。

#### D2LBackend.compose

~~~python
def compose(self, blocks: tuple[RawFactors, ...]) -> CombinedFactors | None: ...
~~~

- 功能：将全部活跃局部/全局原始内容因子交给原生 combine_lora，统一加入一次共享项。
- 参数：blocks 是尚未组合的内容块，顺序来自已校验编译规格。
- 返回：完整 CombinedFactors；空序列返回 None，明确表示纯基座路径。
- 异常/副作用：形状或 dtype/device 不一致抛 BackendError；分配组合张量，不叠加到已有绑定。
- 调用方：MemoryRuntime._compile_plan 和失效后的安全子集重组。

#### D2LBackend.validate_factors

~~~python
def validate_factors(self, factors: RawFactors | CombinedFactors, expected: FactorLayout) -> None: ...
~~~

- 功能：校验模块集合、层、A/B 方向、rank/分配宽度、dtype/device、有限值和实际字节。
- 参数：factors 为实际产物；expected 从加载后的模型布局和此次真实块数派生。
- 返回：None 表示张量契约成立；不代表任务能读出事实。
- 异常/副作用：不一致抛 BackendError；只读检查，不转置、缩放或填零掩盖错误。
- 调用方：compile、compose，以及 Runtime 发布前的完整产物检查。

#### D2LBackend.reset

~~~python
def reset(self) -> None: ...
~~~

- 功能：调用原生 reset，清理参数 forward 绑定与 generated_loras。
- 参数：仅 self，对当前共享模型实例操作。
- 返回：None；模型恢复到无任务记忆绑定的状态。
- 异常/副作用：修改模型绑定，失败抛 BackendError；不删除 Runtime 保存的只读因子，也不负责调用方外部 KV。
- 调用方：install、管理模式进入和错误清理；重复调用必须可验证为安全。

#### D2LBackend.install

~~~python
def install(self, factors: CombinedFactors | None) -> None: ...
~~~

- 功能：先 reset，再用原生 patch_lora_forward/apply_lora_to_layers 安装一份完整组合产物。
- 参数：factors 是已组合且已校验的读取因子；None 保持纯基座。
- 返回：None；只完成绑定，Runtime 尚须 prefill 和发布快照。
- 异常/副作用：修改 forward 绑定，失败抛 BackendError；不写基座权重，不重复组合，不把新因子累加到旧因子。
- 调用方：MemoryRuntime._commit 与 _restore。

#### D2LBackend.prefill

~~~python
def prefill(self, input_ids: TokenIds) -> None: ...
~~~

- 功能：对已安装因子执行一次真实前向检查，确认恢复工作上下文时没有形状或执行错误。
- 参数：input_ids 为 Runtime 准备的合法工作输入，已经过对应输入预算检查。
- 返回：None；首版丢弃此次临时输出/KV，不向任务交付答案。
- 异常/副作用：消耗前向时间和显存，失败抛 BackendError；不推进任务事件。
- 调用方：MemoryRuntime._commit、_restore；这次检查的额外前向成本单独记录。

#### D2LBackend.generate

~~~python
def generate(self, input_ids: TokenIds, generation: GenerationConfig) -> GenerationResult: ...
~~~

- 功能：用当前绑定调用 base_model.generate，并仅解码新生成部分。
- 参数：input_ids 为已校验完整输入；generation 指定本模式 max_new_tokens 与 do_sample。
- 返回：GenerationResult，含原始文本、实际 token 消耗和耗时。
- 异常/副作用：模型生成失败抛 BackendError；单次调用内部可用原生 KV，结束后不跨顶层调用保留。
- 调用方：AgentManager.generate_reply、MemoryRuntime.answer；不调用会再次组合因子的 ModulatedPretrainedModel.generate。

### 10.9 runtime.py：维护、提交与任务调用

#### MemoryRuntime.__init__

~~~python
def __init__(self, initial: Snapshot, backend: D2LBackend, manager: AgentManager, config: RunConfig) -> None: ...
~~~

- 功能：建立一个任务/历史的运行时，绑定当前状态、同一后端、管理器、任务模板和所需配置，核对任务及编译输入容量。
- 参数：initial 是此历史的新空快照；backend 与 manager.backend 必须相同；config 在此装配后只保存需要的配置子对象。
- 返回：None；初始只读因子集合为空，管理调用账本和当前候选为空。
- 异常/副作用：身份或依赖不一致抛 StateError，模板缺失抛 ConfigError；清除前一历史的模型绑定。
- 调用方：run_mab，每份源历史新建 Runtime，模型权重实例可复用。

#### MemoryRuntime.observe

~~~python
def observe(self, span: SourceSpan) -> tuple[MaintenanceResult, ...]: ...
~~~

- 功能：接收一个已发生片段，更新 W，检查边界/容量触发，必要时构造候选并执行 maintain。
- 参数：span 必须属于本作用域且顺序合法；外壳不能一次塞入未来历史或问答标签。
- 返回：本次发生的全部 MaintenanceResult，未触发时为空 tuple；先腾出窗口再处理新边界时可能不止一次，不能漏记成本。
- 异常/副作用：改变本 Runtime 的工作状态，可能消耗管理/编译调用；已暂停、已冻结或顺序错误抛 StateError。
- 调用方：run_mab；若新片段放不下，先维护已有待处理内容再接纳。任何维护进入 paused 即停止交付，不静默丢弃输入。

#### MemoryRuntime.maintain

~~~python
def maintain(self, candidate: Candidate) -> MaintenanceResult: ...
~~~

- 功能：暂停任务，管理生成、解析、来源补查、优先失效、计划校验、编译、提交；统一控制有限重提。
- 参数：candidate 必须来自当前未处理前缀；pending 为空时由本函数登记，已有 pending 时必须完全匹配，不能覆盖另一个候选。
- 返回：MaintenanceResult；committed、skipped、aborted、paused 分别表示提交、主动跳过、普通更新放弃和需停止接收新事件。额度耗尽仍有未完成纠正时必须 paused，不能当普通候选放弃。
- 异常/副作用：可能切换绑定和更新快照；预期失败转换成明确结果，未知异常清理后继续抛出。
- 调用方：observe，以及 run_mab 的历史结束收尾；调用额度、来源额度均按该候选的一次维护累计。

#### MemoryRuntime._apply_invalidations

~~~python
def _apply_invalidations(self, notices: tuple[Invalidation, ...]) -> None: ...
~~~

- 功能：调用 disable_invalidated，过滤受影响原始因子，重组并提交安全状态。
- 参数：notices 已由 validate_invalidations 检查；不能从任意模型自由文本抽取后直接执行。
- 返回：None；成功后安全前态与当前版本均已更新，后续普通计划必须基于新版本。
- 异常/副作用：修改绑定与当前快照；安全状态无法安装时抛 StateError 并停止任务，不能恢复已知错误前态。
- 调用方：maintain；安全提交不提前写本次维护的终结回执，也不重置调用次数。

#### MemoryRuntime._compile_plan

~~~python
def _compile_plan(self, checked: ValidatedPlan) -> CompiledMemory: ...
~~~

- 功能：为变化局部块编译，复用输入未变的局部原始因子；全局视图变化时重编译所有全局块，然后组合一次。
- 参数：checked 必须包含真实待提交状态和完整 CompileSpec，不能直接传未校验的 MemoryPlan。
- 返回：CompiledMemory，包含新状态全部活跃因子、完整组合产物和来源/输入映射。
- 异常/副作用：编译/组合失败抛 BackendError；分配单次暂存，不修改当前已提交因子，不保留无限缓存。
- 调用方：maintain；复用同时比较输入指纹与编译器指纹，不能只比较 block_id。

#### MemoryRuntime._commit

~~~python
def _commit(self, staged: Snapshot, compiled: CompiledMemory) -> None: ...
~~~

- 功能：复核新版本等于当前版本加一、全部引用和预算，安装新因子，prefill 待发布 W，成功后一次性发布新快照和因子引用。
- 参数：staged 为待发布状态；普通成功路径已计算 finish_candidate 后的 W 和有界回执，安全失效路径则保留 pending；compiled 与全部活跃编译输入匹配。
- 返回：None；发布后版本推进，当前状态与实际绑定一致，再释放无引用旧来源和因子。
- 异常/副作用：修改模型绑定和 Runtime 当前指针；失败抛 BackendError/StateError，由 maintain 选择安全恢复。
- 调用方：maintain、_apply_invalidations；发布点在安装与 prefill 成功之后，不能提前交付任务输出。

#### MemoryRuntime._restore

~~~python
def _restore(self, snapshot: Snapshot, compiled: CompiledMemory) -> None: ...
~~~

- 功能：重置残留绑定，完整恢复指定安全快照对应的因子和工作上下文。
- 参数：snapshot/compiled 必须是最近合法安全前态，二者对应；不能传已被纠正作废的版本。
- 返回：None 表示可恢复任务调用；不声称失败的候选已接纳。
- 异常/副作用：重新安装和 prefill；恢复失败抛 StateError，停止任务，不逐个尝试更老版本。
- 调用方：maintain 的跳过、普通失败和异常清理路径；记录未变也需要恢复被管理模式清除的绑定。

#### MemoryRuntime.answer

~~~python
def answer(self, question: str) -> GenerationResult: ...
~~~

- 功能：用任务模板、合法目标、W 和当前参数绑定回答一个问题。
- 参数：question 是此时已交付的问题；不得包含 gold answer 或其它题的答案。
- 返回：原始 GenerationResult；函数不把预测写回长期记忆。
- 异常/副作用：生成会消耗模型资源；暂停/维护中的调用抛 StateError，超输入预算抛 BudgetExceeded。
- 调用方：run_mab；使用 generation.task，首版每次重新编码和 prefill，不跨问题复用 KV。

#### MemoryRuntime.freeze

~~~python
def freeze(self) -> Snapshot: ...
~~~

- 功能：结束历史写入，固定当前版本与 W，使后续问题都读取同一状态。
- 参数：仅 self；调用时不得有未解决的候选、维护事务或失效修复。
- 返回：不可变 Snapshot，用于逐题版本核对和运行产物记录。
- 异常/副作用：将 Runtime 置为只读，之后 observe/maintain 被拒绝；状态未完成时抛 StateError。
- 调用方：run_mab，在向 reader 交付任何测试问题之前；history_end 收尾要循环处理有限剩余前缀，直到 build_candidate 返回 None。

### 10.10 evaluation.py：数据、评分与运行产物

#### load_eval_case

~~~python
def load_eval_case(manifest_path: Path, selection: str) -> EvalCase: ...
~~~

- 功能：读取固定测评 manifest，选择对应 Parquet 和 metadata.source，校验历史/问题/答案对齐。
- 参数：manifest_path 来自 data.manifest_path；selection 来自 data.selection，必须是 manifest 中真实存在的选择项。
- 返回：EvalCase；包含答案的对象只由评测外壳持有，不传给 Runtime 或 AgentManager。
- 异常/副作用：读取本地 JSON/Parquet；版本、选择或结构错误抛 EvaluationError，不按猜测行号替代筛选。
- 调用方：run_mab。

#### iter_history_events

~~~python
def iter_history_events(case: EvalCase) -> Iterator[Event]: ...
~~~

- 功能：将当前 MAB context 按原始编号/事件顺序转成事件流，保留事实原文。
- 参数：case 来自 load_eval_case；函数只能使用 context 和不含答案的来源身份。
- 返回：Event 迭代器；事件时间未知时不伪造时间，编号更新规则来自 YAML 中的共同任务指令。
- 异常/副作用：格式与选定子任务协议不符抛 EvaluationError；不根据测试问题删选事实。
- 调用方：run_mab，再交给 iter_spans 逐片进入 Runtime。

#### load_official_scorer

~~~python
def load_official_scorer(source_path: Path) -> ScoreFn: ...
~~~

- 功能：从固定本地源码载入官方 substring_exact_match 及多答案最大匹配调用路径。
- 参数：source_path 来自 data.official_scorer_path，需记录实际文件指纹与 manifest 指定版本。
- 返回：ScoreFn；只衔接原生函数参数，不重写归一化或匹配算法。
- 异常/副作用：加载评分模块及其必需依赖；缺失或接口不匹配抛 EvaluationError，不使用旧 pilot 的近似实现。
- 调用方：run_mab，一次加载后复用。

#### score_prediction

~~~python
def score_prediction(prediction: str, answers: tuple[str, ...], scorer: ScoreFn) -> float: ...
~~~

- 功能：调用固定官方评分函数，为一条回答计算原生分数。
- 参数：prediction 为未经自行清洗的原始回答；answers 为本题可接受答案；scorer 为已加载的官方调用。
- 返回：原生分数；额外的旧值误用等诊断另列，不改写该分数。
- 异常/副作用：空答案集合或评分错误抛 EvaluationError；不调用 Agent，不回写记忆。
- 调用方：run_mab，问题回答完成之后。

#### run_mab

~~~python
def run_mab(config: RunConfig) -> RunReport: ...
~~~

- 功能：统一设种子并装配后端、管理器和空 Runtime；流式写入、最终候选维护、冻结、逐题回答和官方评分。
- 参数：config 已通过配置和资源校验；实验名称不用于暗中切换算法分支。
- 返回：RunReport，含已发生的完整维护结果、逐题预测、最终状态和失败数量。
- 异常/副作用：读取数据、加载模型并执行推理；paused 提前结束时保留已完成记录和明确失败状态，不能只报告成功题。
- 调用方：main；SH 6K 的题数/行号从数据入口读取，不能硬编码进循环。

#### write_run_report

~~~python
def write_run_report(report: RunReport, output_dir: Path) -> None: ...
~~~

- 功能：在独立运行子目录写解析配置、版本、维护记录、逐题结果、汇总、最终快照和 safetensors 因子。
- 参数：report 只包含本次真实执行结果；output_dir 来自 output.directory。
- 返回：None 表示所需产物写完；运行本身是否成功以 report.status 为准。
- 异常/副作用：写本地文件，写入失败向上传递；不覆盖已有运行，不补写虚构分数，不提供日志读取给 Agent。
- 调用方：main，包括能产生 RunReport 的明确失败运行。

---

## 11. 测试文件与函数对应关系

| 测试文件 | 覆盖对象 | 核心断言 |
| --- | --- | --- |
| test_config.py | construct_mapping、load_config、validate_config、validate_resources | 重复键/错类型/缺字段拒绝；切换 cwd 不改变路径解析 |
| test_budget.py | count_text_tokens、check_token_limit、measure_snapshot、check_budget | 等于上限通过，超过失败；副本/元数据/W 不漏计 |
| test_events.py | iter_spans、append_span、build_candidate、finish_candidate | 原文片段可重建、模板计入、一个 pending、未处理内容不丢失 |
| test_state.py | empty_snapshot、build_dependency_index、stage_plan、build_compile_specs、disable_invalidated | 来源存活；非法悬空；装箱失败；纠正后错误退出有效读取 |
| test_planning.py | parse_manager_reply、validate_invalidations、resolve_source_request、validate_plan | 结构、版本、作用域、来源、互斥动作及全局满时拒绝 |
| test_runtime.py | observe、maintain、_apply_invalidations、_compile_plan、_commit、_restore、answer、freeze | 调用总上限、失效优先、失败不恢复错误、skip 后绑定恢复、冻结后不写入 |
| test_evaluation.py | load_eval_case、iter_history_events、load_official_scorer、score_prediction、run_mab、write_run_report | writer/reader/gold 隔离、逐题同快照、官方多答案评分、失败分母与产物 |
| integration/test_d2l.py | D2LBackend 全部原生接入方法 | 真实 shape/方向/scale、共享项一次、reset、无双重组合、重复安装与重建一致 |

纯状态测试使用测试目录中的模型替身；生产入口不提供假编译器降级。真实 GPU 测试独立运行，不能让配置或来源单元测试加载整个模型。

代码实现后，从 Main-Code/ 运行：

~~~bash
timeout 60s python -m pytest tests/unit -q
python -m ruff check src tests
python -m build
~~~

后端单元测试必须有 60 秒外部硬超时。模型集成实验另设明确进程期限并记录真实成本，不混入这个单元测试命令。文档阶段只校验目录/符号/参数/YAML/引用，不能将其写成实现测试通过。

---

## 12. 编写顺序与提交要求

1. 先建立 pyproject.toml、README、config.py、schema.py；YAML 读取和错误报告可独立验证。
2. 完成 budget.py、events.py、state.py、planning.py；用人工事件与计划测试状态转移。
3. 接入 d2l.py，核对本地实际依赖、checkpoint、模板、因子和安装路径。
4. 完成两个 prompt、manager.py、runtime.py；验证调用额度、跳过和纠正失败。
5. 完成 evaluation.py 与入口，跑 SH 6K 的流式写入、冻结快照和官方评分。

每次提交说明本次实现了哪些函数、运行了哪些检查和净代码行数；新增函数同步参数/返回/异常契约。文件增长超过 80 行说明对应实现职责；不顺带清理无关旧代码。

当前真实接口依据：[hypernet.py](../resources/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py)、[lora_merger.py](../resources/doc-to-lora/src/ctx_to_lora/modeling/lora_merger.py)、[lora_layer.py](../resources/doc-to-lora/src/ctx_to_lora/modeling/lora_layer.py)、[tokenization](../resources/doc-to-lora/src/ctx_to_lora/data/processing.py)、[MAB 入口](../resources/data/mab/first_eval.json)。这些上游函数直接复用，本文件只逐项规定本项目要新增的函数。
