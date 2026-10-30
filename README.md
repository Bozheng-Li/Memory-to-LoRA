# Agent 自主管理的局部与全局参数记忆

当前方案以有助于解决后续问题的事实为记忆内容，由同一 Agent 管理局部池与全局记忆，冻结 D2L 负责参数编译。首版代码、测试和运行入口已在 [Main-Code/](Main-Code/README.md) 实现；真实模型验证状态见 [实现说明](docs/IMPLEMENTATION.md)。

主要入口：[方法方案](docs/AGENT_MANAGED_LOCAL_GLOBAL_FLOW_20261029.md) · [代码高层设计](docs/HIGH_LEVEL_DESIGN.md) · [MAB 数据说明](docs/MAB_DATA_GUIDE.md) · [测评数据](resources/data/mab/README.md)

| 目录 | 用途 |
| --- | --- |
| [docs/](docs/README.md) | 重要方案、测评选型与数据说明 |
| [research/](research/README.md) | 调研来源、中间记录、旧实验结果与历史方案 |
| [Main-Code/](Main-Code/README.md) | 参数记忆实现、独立 YAML、提示模板、测试和 CLI |
| scripts/ | 环境准备和已有实验脚本 |
| [resources/](resources/README.md) | doc-to-lora、基座模型、数据集与编译器 checkpoint |
| wheels/、.venv-d2l/ | 本地依赖与运行环境 |

根目录的 AGENTS.md 保留项目工作规则，requirements-d2l.txt 保留依赖清单。
