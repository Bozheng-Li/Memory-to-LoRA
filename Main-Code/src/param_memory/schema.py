"""定义严格 JSON 边界、不可变状态和项目异常。

记录使用冻结 Pydantic 模型和 tuple；张量只经延迟注解出现在运行报告中。
公共管理响应的结构是提示生成、解析和校验共用的唯一来源。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

if TYPE_CHECKING:
    from .config import RunConfig
    from .d2l import CompiledMemory

SCHEMA_VERSION = 1
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


class PlanValidationError(ValueError):
    """管理响应的结构、身份、来源或动作不合法。"""


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

    EVENT_END = "event_end"
    WINDOW_FULL = "window_full"
    CORRECTION = "correction"
    HISTORY_END = "history_end"


class RecordKind(str, Enum):
    """全局记录的事实类别，不表示可信度。"""

    FACT = "fact"
    CONSTRAINT = "constraint"
    OBSERVATION = "observation"


class RecordStatus(str, Enum):
    """显式区分现行、历史、未决和无效命题。"""

    ACTIVE = "active"
    HISTORICAL = "historical"
    UNRESOLVED = "unresolved"
    INVALID = "invalid"


class LocalStatus(str, Enum):
    """局部块或全局参数绑定是否允许参与读取。"""

    ACTIVE = "active"
    DISABLED = "disabled"


class MaintenanceStatus(str, Enum):
    """维护终结结果；paused 不确认当前候选。"""

    COMMITTED = "committed"
    SKIPPED = "skipped"
    ABORTED = "aborted"
    PAUSED = "paused"


class Event(FrozenModel):
    """一次已发生的输入；未知时间保持 None。"""

    event_id: Nonempty
    scope_id: Nonempty
    text: Nonempty
    order: Natural
    event_time: str | None
    observed_at: str | None
    boundary: Trigger | None


class SourceSpan(Event):
    """原事件中按 Unicode 码点定位的半开区间。"""

    text: Annotated[str, Field(min_length=1)]
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


class SourceExcerpt(FrozenModel):
    """补查展示的原文，不改变源片段身份。"""

    ref: SourceRef
    text: Annotated[str, Field(min_length=1)]


class Cursor(FrozenModel):
    """最后处理片段的事件序号及结束字符位置。"""

    order: Natural
    end: Positive
    event_id: Nonempty


class Candidate(FrozenModel):
    """有界未处理前缀；形成候选不分配因子。"""

    candidate_id: Nonempty
    source_spans: Annotated[tuple[SourceSpan, ...], Field(min_length=1)]
    compile_text: Nonempty
    trigger: Trigger


class WorkState(FrozenModel):
    """任务目标、近期观察、摘要和唯一待处理候选。"""

    goal: Nonempty
    recent_spans: tuple[SourceSpan, ...]
    summary: str
    pending_candidate: Candidate | None
    processed_through: Cursor | None


class LocalBlock(FrozenModel):
    """固定槽位中的事件来源、编译视图、管理卡和依赖。"""

    block_id: Nonempty
    slot_id: Natural
    version: Positive
    source_refs: Annotated[tuple[SourceRef, ...], Field(min_length=1)]
    compile_text: Nonempty
    memory_card: Nonempty
    claim_refs: tuple[str, ...]
    status: LocalStatus
    factor_key: Nonempty


class RecordContent(FrozenModel):
    """完整事实及证据；scope 为当前任务身份，具体条件写入 claim。"""

    kind: RecordKind
    claim: Nonempty
    scope: Nonempty
    valid_from: str | None
    valid_until: str | None
    observed_at: str | None
    status: RecordStatus
    source_refs: Annotated[tuple[SourceRef, ...], Field(min_length=1)]
    evidence_group_ids: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    support_level: Literal["single_observation", "corroborated", "unresolved"]
    related_record_ids: tuple[str, ...]


class GlobalRecord(RecordContent):
    """有稳定 ID 和乐观版本的全局管理记录。"""

    record_id: Nonempty
    version: Positive


class ObjectVersion(FrozenModel):
    """编译输入绑定的记录或局部对象版本。"""

    object_id: Nonempty
    version: Positive


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
    plan_id: str | None
    status: MaintenanceStatus


class Snapshot(FrozenModel):
    """完整有界提交状态；None 槽不占内容因子。"""

    scope_id: Nonempty
    version: Natural
    work: WorkState
    local_blocks: tuple[LocalBlock | None, ...]
    global_records: tuple[GlobalRecord, ...]
    global_status: LocalStatus
    sources: tuple[SourceSpan, ...]
    factor_refs: tuple[FactorRef, ...]
    last_maintenance: MaintenanceReceipt | None


class Dependency(FrozenModel):
    """一个来源或记录的真实反向依赖。"""

    key: Nonempty
    local_block_ids: tuple[str, ...]
    global_record_ids: tuple[str, ...]


class DependencyIndex(FrozenModel):
    """从快照派生，不能独立编辑的引用索引。"""

    sources: tuple[Dependency, ...]
    records: tuple[Dependency, ...]


class Invalidation(FrozenModel):
    """优先处理的失效通知，必须引用本次交付的纠正来源。"""

    target_type: Literal["local", "global"]
    target_id: Nonempty
    expected_version: Positive
    source_refs: Annotated[tuple[SourceRef, ...], Field(min_length=1)]
    reason: Nonempty


class ResponseHeader(FrozenModel):
    """所有管理响应共用的版本和候选身份。"""

    schema_version: int
    plan_id: Nonempty
    base_state_version: Natural
    candidate_id: Nonempty


class SourceRead(FrozenModel):
    """一次有明确 token 上限的在线来源请求。"""

    span_id: Nonempty
    max_tokens: Positive


class SourceRequest(ResponseHeader):
    """管理器请求补查来源，不授权读取任何文件。"""

    response_type: Literal["source_request"]
    source_requests: Annotated[tuple[SourceRead, ...], Field(min_length=1)]


class GlobalUpsert(RecordContent):
    """新增用 alias；修订用 record_id 和 expected_version。"""

    alias: Nonempty | None
    record_id: Nonempty | None
    expected_version: Positive | None


class GlobalDelete(FrozenModel):
    """带版本和明确遗忘原因的全局删除。"""

    record_id: Nonempty
    expected_version: Positive
    reason: Nonempty


class LocalRevision(FrozenModel):
    """修订真实编译文本及来源，不能只修改管理卡。"""

    block_id: Nonempty
    expected_version: Positive
    compile_text: Nonempty
    memory_card: Nonempty
    source_refs: Annotated[tuple[SourceRef, ...], Field(min_length=1)]


class LocalExit(FrozenModel):
    """明确退出的局部块，不因目标槽被选中而隐式删除。"""

    block_id: Nonempty
    expected_version: Positive
    reason: Nonempty


class ExitDependency(FrozenModel):
    """退出前需成立的全局记录及明确放弃的细节。"""

    block_id: Nonempty
    required_global_ids: tuple[str, ...]
    discarded_content: str


class MemoryPlan(ResponseHeader):
    """联合接纳、巩固、修订、退出计划，仅表示管理意图。"""

    response_type: Literal["plan"]
    admission: Literal["local", "global_only", "skip"]
    target_slot: Natural | None
    candidate_card: str | None
    global_upserts: tuple[GlobalUpsert, ...]
    global_deletes: tuple[GlobalDelete, ...]
    local_revisions: tuple[LocalRevision, ...]
    local_exits: tuple[LocalExit, ...]
    exit_dependencies: tuple[ExitDependency, ...]
    invalidations: tuple[Invalidation, ...]
    reason: Nonempty


ManagerReply = Annotated[SourceRequest | MemoryPlan, Field(discriminator="response_type")]
MANAGER_REPLY_ADAPTER = TypeAdapter(ManagerReply)


class ValidationFeedback(FrozenModel):
    """交给下次管理调用的错误，不包含离线日志。"""

    code: Nonempty
    field: Nonempty
    message: Nonempty


class SourceReadResult(FrozenModel):
    """按请求顺序交付的原文片段及实际 token 消耗。"""

    excerpts: tuple[SourceExcerpt, ...]
    tokens_used: Natural


class StagedUpdate(FrozenModel):
    """纯函数得到的候选状态，发布前不影响任务。"""

    base_version: Natural
    snapshot: Snapshot
    changed_local_ids: tuple[str, ...]
    global_changed: bool


class CompileSpec(FrozenModel):
    """精确编译文本、真实 token IDs 和当前来源/记录绑定。"""

    block_key: Nonempty
    kind: Literal["local", "global"]
    text: Nonempty
    token_ids: TokenIds
    source_refs: tuple[SourceRef, ...]
    record_versions: tuple[ObjectVersion, ...]
    input_fingerprint: Digest


class BudgetUsage(FrozenModel):
    """完整稳态用量；没有张量产物时参数计数为零。"""

    local_slots: Natural
    slot_capacity: Natural
    global_blocks: Natural
    local_input_tokens: tuple[int, ...]
    global_input_tokens: tuple[int, ...]
    local_card_tokens: tuple[int, ...]
    persistent_text_tokens: Natural
    persistent_text_bytes: Natural
    working_tokens: Natural
    working_bytes: Natural
    content_width: Natural
    factor_bytes: Natural


class ValidatedPlan(FrozenModel):
    """经过所有纯状态检查的实际更新和编译输入。"""

    plan: MemoryPlan
    update: StagedUpdate
    specs: tuple[CompileSpec, ...]
    usage: BudgetUsage


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

    manager_input_tokens: int = 0
    manager_output_tokens: int = 0
    manager_seconds: float = 0.0
    source_read_tokens: int = 0
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
    """交给评测外壳的完整维护结果，永不回灌给 Agent。"""

    status: MaintenanceStatus
    candidate_id: Nonempty
    plan_id: str | None
    base_version: Natural
    final_version: Natural
    reason: str
    errors: tuple[str, ...]
    manager_calls: Natural
    cost: MaintenanceCost
    elapsed_seconds: float
    replies: tuple[str, ...]


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
    """绑定真实文本和完整输入 IDs；来源版本另外保存在 CompileSpec 中。

    Args:
        text: 实际编译视图。
        token_ids: 对应的完整模板化输入 IDs。

    Returns:
        绑定文本及输入 IDs 的指纹；来源与记录版本在 CompileSpec 中另行绑定。
    """
    return text_hash(stable_json({"text": text, "token_ids": token_ids}))
