"""校验原始上下文与来源，为每个 chunk 生成唯一的直接编译输入。"""

from .budget import check_token_limit
from .config import MemoryConfig
from .events import candidate_from_spans, check_continuation, validate_span
from .schema import (
    Candidate,
    CompileSpec,
    EncodeText,
    Snapshot,
    SourceRef,
    SourceSpan,
    StateError,
    WorkState,
    compile_fingerprint,
)


def empty_snapshot(scope_id: str, goal: str) -> Snapshot:
    """创建独立历史的空状态；不分配槽位或空 LoRA。"""
    if not scope_id.strip() or not goal.strip():
        raise StateError("作用域和目标必须非空")
    return Snapshot(
        scope_id=scope_id,
        version=0,
        work=WorkState(goal=goal, recent_spans=(), pending_candidate=None, processed_through=None),
        factor_refs=(),
        last_maintenance=None,
    )


def span_ref(span: SourceSpan) -> SourceRef:
    """引用一个完整原文片段，字符位置相对原事件。"""
    return SourceRef(
        span_id=span.span_id, start=span.start, end=span.end, source_hash=span.source_hash
    )


def check_candidate(snapshot: Snapshot, candidate: Candidate) -> None:
    """候选必须是游标之后的原始连续前缀，不能改写文本或覆盖另一个 pending。"""
    pending = snapshot.work.pending_candidate
    if pending is not None and pending != candidate:
        raise StateError("不能覆盖不同的 pending_candidate")
    if snapshot.work.recent_spans[: len(candidate.source_spans)] != candidate.source_spans:
        raise StateError("候选不是当前未处理来源前缀")
    previous = snapshot.work.processed_through
    for span in candidate.source_spans:
        if span.scope_id != snapshot.scope_id:
            raise StateError(f"来源 {span.span_id} 不属于作用域 {snapshot.scope_id}")
        validate_span(span)
        check_continuation(span, previous)
        previous = span
    if candidate != candidate_from_spans(candidate.source_spans, candidate.trigger):
        raise StateError("候选文本或身份与原始来源不一致")


def build_compile_spec(
    snapshot: Snapshot, candidate: Candidate, limits: MemoryConfig, encode_context: EncodeText
) -> CompileSpec:
    """直接编译完整 chunk，绑定原文与来源；长度包含真实 D2L 模板。"""
    check_candidate(snapshot, candidate)
    token_ids = encode_context(candidate.compile_text)
    check_token_limit(token_ids, limits.chunk_tokens, "chunk 编译输入")
    return CompileSpec(
        block_key=candidate.candidate_id,
        text=candidate.compile_text,
        token_ids=token_ids,
        source_refs=tuple(span_ref(span) for span in candidate.source_spans),
        input_fingerprint=compile_fingerprint(candidate.compile_text, token_ids),
    )
