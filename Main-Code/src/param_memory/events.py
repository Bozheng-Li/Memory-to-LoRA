"""无损分片已发生事件，并维护唯一候选和有界工作窗口。

处理游标只有在维护终结后推进；未处理片段永不因窗口压力静默丢弃。
"""

import re
from collections.abc import Iterator

from .budget import check_token_limit, prefix_end
from .config import MemoryConfig
from .schema import (
    BudgetExceeded,
    Candidate,
    Cursor,
    EncodeText,
    Event,
    SourceSpan,
    StateError,
    Trigger,
    WorkState,
    stable_json,
    text_hash,
)


def validate_span(span: SourceSpan) -> None:
    """核对原文长度、Unicode 区间和完整片段的 SHA-256。

    Args:
        span: 已交付的原始片段或独立保留的证据。

    Returns:
        None 表示来源元数据与原文一致；不证明原文语义真实。

    Raises:
        StateError: 区间长度或 hash 不符。
    """
    if span.end - span.start != len(span.text) or text_hash(span.text) != span.source_hash:
        raise StateError(f"片段 {span.span_id} 的位置或 hash 不一致")


def check_continuation(span: SourceSpan, previous: SourceSpan | Cursor | None) -> None:
    """核对新片段接续已处理游标或当前窗口的最后片段。

    Args:
        span: 新交付的片段；原文完整性另由 validate_span 检查。
        previous: 上一片段、最后处理游标，或表示历史起点的 None。

    Returns:
        None 表示顺序与事件身份连续，不允许重放或换用同一事件身份。

    Raises:
        StateError: 作用域、事件身份、位置或顺序冲突。
    """
    if previous is None:
        if span.start != 0:
            raise StateError("首个事件必须从原文位置 0 开始")
        return
    if isinstance(previous, SourceSpan) and span.scope_id != previous.scope_id:
        raise StateError("工作窗口不能跨作用域")
    if span.order < previous.order:
        raise StateError("事件顺序不能倒退")
    if span.order == previous.order:
        if span.event_id != previous.event_id or span.start != previous.end:
            raise StateError("同一事件的片段必须连续且身份一致")
    elif span.start != 0 or span.event_id == previous.event_id:
        raise StateError("新事件必须从位置 0 开始，并使用新的 event_id")


def candidate_from_spans(spans: tuple[SourceSpan, ...], trigger: Trigger) -> Candidate:
    """从同一有序来源集合生成规范候选文本与稳定身份。

    Args:
        spans: 当前未处理连续前缀；不能是空集合。
        trigger: 调用方已知的维护边界，不参与内容身份计算。

    Returns:
        原样拼接的连续上下文；不插入分隔符，不登记 pending。

    Raises:
        StateError: 来源集合为空。
    """
    if not spans:
        raise StateError("空来源不能形成候选")
    text = "".join(span.text for span in spans)
    identity = stable_json([(span.span_id, span.source_hash) for span in spans])
    return Candidate(
        candidate_id=f"c:{text_hash(identity)}",
        source_spans=spans,
        compile_text=text,
        trigger=trigger,
    )


def iter_spans(event: Event, max_tokens: int, encode_context: EncodeText) -> Iterator[SourceSpan]:
    """逐片交付含模板后可装入的原文，连接结果严格恢复 event.text。

    Args:
        event: 已发生的事件；顺序、时间与作用域原样传递。
        max_tokens: YAML 的 chunk 输入上限。
        encode_context: D2L 完整上下文模板编码。

    Returns:
        按原文顺序的 SourceSpan 迭代器，记录原始字符位置和 hash；连接片段应恢复原始文本。

    Raises:
        BudgetExceeded: 必要最小片段也不能装入。
    """
    start = 0
    while start < len(event.text):
        remaining = event.text[start:]
        length = prefix_end(remaining, max_tokens, encode_context)
        if length < len(remaining):
            boundaries = re.finditer(r"(?<=[.!?。！？])\s+|\n+", remaining[:length])
            boundary = next((m.end() for m in reversed(tuple(boundaries))), None)
            if (
                boundary is not None
                and remaining[:boundary].strip()
                and len(encode_context(remaining[:boundary])) <= max_tokens
            ):
                length = boundary
        end = start + length
        text = event.text[start:end]
        check_token_limit(encode_context(text), max_tokens, f"事件 {event.event_id}")
        yield SourceSpan(
            **event.model_dump(exclude={"text"}),
            text=text,
            span_id=f"{event.event_id}:{start}:{end}",
            start=start,
            end=end,
            source_hash=text_hash(text),
        )
        start = end


def append_span(work: WorkState, span: SourceSpan) -> WorkState:
    """接纳连续片段；未完成 chunk、原文损坏或顺序错误时拒绝且不修改旧状态。"""
    if work.pending_candidate is not None:
        raise StateError("pending_candidate 未终结，不能接收新事件")
    validate_span(span)
    previous = work.recent_spans[-1] if work.recent_spans else work.processed_through
    check_continuation(span, previous)
    result = work.model_copy(update={"recent_spans": (*work.recent_spans, span)})
    return result


def build_candidate(
    work: WorkState, trigger: Trigger, limits: MemoryConfig, encode_context: EncodeText
) -> Candidate | None:
    """取未处理来源的有界连续前缀，只形成候选，不登记或编译。

    Args:
        work: 冻结工作状态。
        trigger: chunk 满或历史结束。
        limits: 提供长度上限。
        encode_context: 核对候选整体的实际编译长度。

    Returns:
        Candidate；没有待处理内容时返回 None，不编译空上下文。

    Raises:
        StateError: 已有 pending，不能形成第二个候选。
        BudgetExceeded: 首个未处理片段超过 chunk 编译上限。
    """
    if work.pending_candidate is not None:
        raise StateError("已有 pending_candidate，不能形成第二个候选")
    candidate = None
    for index in range(len(work.recent_spans)):
        proposed = candidate_from_spans(work.recent_spans[: index + 1], trigger)
        if len(encode_context(proposed.compile_text)) > limits.chunk_tokens:
            break
        candidate = proposed
    if candidate is None and work.recent_spans:
        raise BudgetExceeded("首个未处理片段无法装入候选")
    return candidate


def finish_candidate(work: WorkState, candidate_id: str) -> WorkState:
    """确认已终结候选，推进游标并释放已处理前缀；其余片段原样保留。

    只有编译、安装和前向均成功后才发布此状态。已处理原文立即离窗，
    因而任务端不会借近期原句冒充长期参数读取。

    Args:
        work: 含当前 pending 的工作状态。
        candidate_id: 必须匹配该候选。

    Returns:
        推进游标、移除已处理来源的新 W；未处理后缀原样保留。

    Raises:
        StateError: 候选身份或未处理前缀不一致。
    """
    candidate = work.pending_candidate
    if candidate is None or candidate.candidate_id != candidate_id:
        raise StateError("待确认候选身份不匹配")
    count = len(candidate.source_spans)
    if work.recent_spans[:count] != candidate.source_spans:
        raise StateError("候选必须是当前未处理来源的完整连续前缀")
    last = candidate.source_spans[-1]
    result = work.model_copy(
        update={
            "recent_spans": work.recent_spans[count:],
            "pending_candidate": None,
            "processed_through": Cursor(order=last.order, end=last.end, event_id=last.event_id),
        }
    )
    return result
