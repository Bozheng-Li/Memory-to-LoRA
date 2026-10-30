# 实验资源

实验依赖的外部代码、基座模型、数据集和编译器权重统一放在这里。

| 目录 | 内容 |
| --- | --- |
| [doc-to-lora/](doc-to-lora/) | D2L 上游代码，保留独立 Git 仓库 |
| [models/Qwen3-4B-Instruct-2507/](models/Qwen3-4B-Instruct-2507/) | 本地 Qwen 基座模型与 tokenizer |
| [data/](data/) | 测评数据；首轮入口为 [MemoryAgentBench](data/mab/README.md) |
| [checkpoints/qwen_4b_d2l/](checkpoints/qwen_4b_d2l/) | D2L 编译器配置与 checkpoint-20000 权重 |

项目运行脚本位于 [scripts/](../scripts/)，重要方案位于 [docs/](../docs/README.md)，调研和实验结果位于 [research/](../research/README.md)。本地环境 .venv-d2l/ 和安装包 wheels/ 仍在项目根目录。

doc-to-lora/Qwen/Qwen3-4B-Instruct-2507 通过相对软链接指向 ../../models/Qwen3-4B-Instruct-2507，使 checkpoint 中的模型 ID 在 D2L 仓库内可离线解析。项目脚本和环境准备脚本已使用此布局。

本次通过同一文件系统内重命名迁移资源。历史材料中的旧路径可按 [迁移清单](../research/resource_organization_20261029.json) 查找。
