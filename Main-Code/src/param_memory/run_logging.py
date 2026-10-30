"""管理单次运行的文件/控制台日志、嵌套关联字段及即时结果落盘。

日志上下文只用于诊断，不进入模型输入或 Runtime 的记忆状态。
"""

import logging
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType

from .config import LoggingConfig
from .schema import FrozenModel

_context: ContextVar[Mapping[str, object]] = ContextVar(
    "param_memory_log_context", default=MappingProxyType({})
)


class _ContextFilter(logging.Filter):
    """在输出时绑定当前运行、历史、候选、问题和阶段。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.context = " ".join(f"{key}={value}" for key, value in _context.get().items())
        return True


@contextmanager
def log_context(**fields: object) -> Iterator[None]:
    """临时补充关联字段；异常和正常退出都恢复外层上下文。"""
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def log_phase(logger: logging.Logger, phase: str, **fields: object) -> Iterator[None]:
    """记录阶段开始、耗时和完整异常链；包括中断在内的异常原样上抛。"""
    with log_context(phase=phase, **fields):
        started = time.perf_counter()
        logger.info("started")
        try:
            yield
        except BaseException:
            logger.exception("failed elapsed_seconds=%.6f", time.perf_counter() - started)
            raise
        else:
            logger.info("finished elapsed_seconds=%.6f", time.perf_counter() - started)


@contextmanager
def run_logging(config: LoggingConfig, run_id: str) -> Iterator[Path]:
    """为内部生成的运行 ID 创建独立目录；退出时关闭处理器并恢复日志配置。

    只管理本包的 logging 输出；FileHandler 每条输出都会 flush。
    目录必须全新，拒绝覆盖已有日志。
    """
    directory = config.directory / run_id
    directory.mkdir(parents=True, exist_ok=False)
    handlers = (
        logging.FileHandler(directory / "run.log", mode="x", encoding="utf-8"),
        logging.StreamHandler(),
    )
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s pid=%(process)d %(context)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    logger = logging.getLogger("param_memory")
    level, propagate = logger.level, logger.propagate
    try:
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.addFilter(_ContextFilter())
            logger.addHandler(handler)
        logger.setLevel(config.level)
        logger.propagate = False
        with log_context(run_id=run_id):
            yield directory
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(level)
        logger.propagate = propagate


def append_jsonl(path: Path, record: FrozenModel) -> None:
    """追加一条完整结果并关闭文件刷新缓冲；写入失败直接上抛。"""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(record.model_dump_json() + "\n")
