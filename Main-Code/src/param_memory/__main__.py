"""解析动作与唯一 YAML 配置入口，按固定顺序校验、执行和写出报告。"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .config import load_config, validate_config, validate_resources
from .schema import BackendError, BudgetExceeded, ConfigError, EvaluationError, StateError

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 validate-config/run-mab 及必需 --config；不接受另一套字段覆盖。

    Args:
        argv: 显式参数序列。

    Returns:
        只包含命令名称和配置路径的 Namespace，不夹带另一套业务默认值。
    """
    parser = argparse.ArgumentParser(description="按 chunk 直接编译并累积的 D2L 参数记忆")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-config", "run-mab"):
        action = commands.add_parser(command)
        action.add_argument("--config", type=Path, required=True, help="唯一运行 YAML 文件")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """加载→数值校验→资源校验→运行→写报告；预期失败返回非零，未知错误保留堆栈。

    Args:
        argv: 传给 parse_args。

    Returns:
        进程退出码；成功为 0，失败为非零，并说明失败阶段。
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    phase = "configuration"
    try:
        config = load_config(args.config)
        validate_config(config)
        if args.command == "validate-config":
            validate_resources(config)
            logger.info("配置和本地资源校验通过：%s", args.config.resolve())
            return 0
        # 配置校验命令不导入模型/评测依赖。
        from .evaluation import run_mab

        phase = "run-mab"
        report = run_mab(config)
        return 0 if report.status == "completed" else 1
    except (ConfigError, BackendError, BudgetExceeded, EvaluationError, StateError, OSError) as exc:
        logger.exception("阶段 %s 失败：%s", phase, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
