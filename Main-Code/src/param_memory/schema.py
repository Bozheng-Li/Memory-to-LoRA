"""定义严格 JSON 边界、不可变状态和项目异常。

记录使用冻结 Pydantic 模型和 tuple；张量只经延迟注解出现在运行报告中。
上下文只按 chunk 编译，提交状态不包含槽位、事实表或淘汰计划。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

if TYPE_CHECKING:
    from .config import RunConfig
    from .d2l import CompiledMemory

SCHEMA_VERSION = 2
TokenIds = tuple[int, ...]
EncodeText = Callable[[str], TokenIds]
ScoreFn = Callable[[str, tuple[str, ...]], float]
Nonempty = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Natural = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(gt=0)]


class ConfigError(ValueError):
    """配置结构、数值或资源不合法。"""


class BudgetExceeded(ValueError):
    """实际用量超过声明预算。"""


class BackendError(RuntimeError):
    """原生 D2L 的加载、张量或执行契约失败。"""


class StateError(RuntimeError):
    """运行状态或安全恢复不成立。"""


class EvaluationError(ValueError):
    """数据协议、评分接口或结果结构不合法。"""


class FrozenModel(BaseModel):
    """拒绝隐式类型转换和额外字段，禁止修改已构造记录。"""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class Trigger(str, Enum):
    """输入外壳或工作容量已知的维护边界。"""

    WINDOW_FULL = "window_full"
    HISTORY_END = "history_end"


class MaintenanceStatus(str, Enum):
    """维护终结结果；paused 不确认当前候选。"""

    COMMITTED = "committed"
    PAUSED = "paused"


class Event(FrozenModel):
    """一次已发生的非空输入，空白原文也保留；未知时间保持 None。"""

    event_id: Nonempty
    scope_id: Nonempty
    text: Annotated[str, Field(min_length=1)]
    order: Natural
    event_time: str | None
    observed_at: str | None


class SourceSpan(Event):
    """原事件中按 Unicode 码点定位的半开区间。"""

    span_id: Nonempty
    start: Natural
    end: Positive
    source_hash: Digest


class SourceRef(FrozenModel):
    """引用在线片段的子区间，hash 对应整个被引用片段。"""

    span_id: Nonempty
    start: Natural
    end: Positive
    source_hash: Digest


class Cursor(FrozenModel):
    """最后处理片段的事件序号及结束字符位置。"""

    order: Natural
    end: Positive
    event_id: Nonempty


class Candidate(FrozenModel):
    """有界未处理前缀；形成候选不分配因子。"""

    candidate_id: Nonempty
    source_spans: Annotated[tuple[SourceSpan, ...], Field(min_length=1)]
    compile_text: Annotated[str, Field(min_length=1)]
    trigger: Trigger


class WorkState(FrozenModel):
    """任务目标、尚未编译的上下文和唯一待处理 chunk。"""

    goal: Nonempty
    recent_spans: tuple[SourceSpan, ...]
    pending_candidate: Candidate | None
    processed_through: Cursor | None


class TensorMetadata(FrozenModel):
    """不持有张量的模块、方向、shape 和精度说明。"""

    module: Nonempty
    matrix: Literal["A", "B"]
    shape: tuple[int, int, int, int]
    dtype: Nonempty
    device: Nonempty


class FactorRef(FrozenModel):
    """原始内容因子的输入/编译器身份与实际占用。"""

    block_key: Nonempty
    input_fingerprint: Digest
    compiler_fingerprint: Digest
    tensors: tuple[TensorMetadata, ...]
    content_width: Natural
    size_bytes: Natural


class MaintenanceReceipt(FrozenModel):
    """仅保留最后一次维护的有界身份和结果，不存完整日志。"""

    candidate_id: Nonempty
    status: MaintenanceStatus


class Snapshot(FrozenModel):
    """已提交因子引用及当前工作上下文；历史因子只追加。"""

    scope_id: Nonempty
    version: Natural
    work: WorkState
    factor_refs: tuple[FactorRef, ...]
    last_maintenance: MaintenanceReceipt | None


class CompileSpec(FrozenModel):
    """一段原始上下文的精确文本、token IDs、来源及指纹。"""

    block_key: Nonempty
    text: Annotated[str, Field(min_length=1)]
    token_ids: TokenIds
    source_refs: tuple[SourceRef, ...]
    input_fingerprint: Digest


class BudgetUsage(FrozenModel):
    """完整稳态用量；没有张量产物时参数计数为零。"""

    compiled_chunks: Natural
    chunk_input_tokens: tuple[int, ...]
    persistent_text_tokens: Natural
    persistent_text_bytes: Natural
    working_tokens: Natural
    working_bytes: Natural
    content_width: Natural
    factor_bytes: Natural


class ChatMessage(FrozenModel):
    """模型模板支持的固定角色和纯文本内容。"""

    role: Literal["system", "user", "assistant"]
    content: str


ChatMessages = tuple[ChatMessage, ...]


class GenerationResult(FrozenModel):
    """只含新生成部分，保留原始解码文本及真实成本。"""

    text: str
    token_ids: TokenIds
    input_tokens: Natural
    output_tokens: Natural
    elapsed_seconds: float
    prefill_input_tokens: Natural = 0
    prefill_seconds: float = 0.0


class MaintenanceCost(FrozenModel):
    """一次维护的累计计数；零是未发生操作而非业务默认值。"""

    compile_calls: int = 0
    compile_input_tokens: int = 0
    compile_seconds: float = 0.0
    compose_calls: int = 0
    compose_seconds: float = 0.0
    install_seconds: float = 0.0
    prefill_calls: int = 0
    prefill_input_tokens: int = 0
    prefill_seconds: float = 0.0
    reset_seconds: float = 0.0
    peak_staged_factor_bytes: int = 0


class MaintenanceResult(FrozenModel):
    """交给评测外壳的 chunk 编译结果，永不回灌到上下文。"""

    status: MaintenanceStatus
    candidate_id: Nonempty
    base_version: Natural
    final_version: Natural
    reason: str
    errors: tuple[str, ...]
    cost: MaintenanceCost
    elapsed_seconds: float


class EvalCase(FrozenModel):
    """仅评测外壳持有的完整历史及对齐题组。"""

    history_id: Nonempty
    source: Nonempty
    context: Nonempty
    questions: tuple[str, ...]
    answers: tuple[tuple[str, ...], ...]
    qa_ids: tuple[str, ...]
    dataset_revision: Nonempty
    official_code_revision: Nonempty
    parquet_path: str


class Prediction(FrozenModel):
    """一题的原始预测、官方分数、版本和独立错误诊断。"""

    history_id: Nonempty
    question_id: Nonempty
    snapshot_version: Natural
    question: str
    answers: tuple[str, ...]
    generation: GenerationResult | None
    score: float | None
    error: str | None


@dataclass(frozen=True)
class RunReport:
    """只含真实运行产物；compiled 中的张量不进入 JSON 编码。"""

    run_id: str
    config: RunConfig
    manifest: dict[str, object]
    maintenance: tuple[MaintenanceResult, ...]
    predictions: tuple[Prediction, ...]
    metrics: dict[str, object]
    final_snapshot: Snapshot
    compiled: CompiledMemory
    status: str


def stable_json(value: object) -> str:
    """稳定序列化 JSON 数据，供计账和指纹共用，不删除任何字段。

    Args:
        value: 已由 model_dump(mode="json") 或边界解析得到的 JSON 数据。

    Returns:
        保留 Unicode、键排序且无多余空白的完整文本。
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_hash(text: str) -> str:
    """返回原文 UTF-8 的 SHA-256；不正规化空白或 Unicode。

    Args:
        text: 待计算指纹的完整文本，保留 Unicode 和空白。

    Returns:
        原文 UTF-8 字节的 SHA-256 十六进制摘要。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compile_fingerprint(text: str, token_ids: TokenIds) -> str:
    """绑定真实文本和完整输入 IDs；来源引用另外保存在 CompileSpec 中。

    Args:
        text: 实际编译视图。
        token_ids: 对应的完整模板化输入 IDs。

    Returns:
        绑定文本及输入 IDs 的指纹；来源引用在 CompileSpec 中另行绑定。
    """
    return text_hash(stable_json({"text": text, "token_ids": token_ids}))
