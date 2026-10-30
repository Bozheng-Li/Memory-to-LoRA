"""解析管理响应并校验身份、来源、互斥动作和最终状态预算。

本模块只执行确定性检查，不能证明 Agent 改写的语义忠实性或参数可读性。
"""

import json

from pydantic import ValidationError

from .budget import check_budget, measure_snapshot, prefix_end
from .config import MemoryConfig
from .schema import (
    MANAGER_REPLY_ADAPTER,
    SCHEMA_VERSION,
    BudgetExceeded,
    Candidate,
    EncodeText,
    Invalidation,
    LocalStatus,
    ManagerReply,
    MemoryPlan,
    PlanValidationError,
    RecordStatus,
    ResponseHeader,
    Snapshot,
    SourceExcerpt,
    SourceReadResult,
    SourceRef,
    SourceRequest,
    StateError,
    ValidatedPlan,
)
from .state import (
    build_compile_specs,
    check_candidate,
    online_sources,
    plan_aliases,
    stage_plan,
    validate_source_refs,
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """在 JSON 边界拒绝重复键，避免解析器悄悄覆盖先前动作。

    Args:
        pairs: JSON 解码器交付的原始键值对。

    Returns:
        未丢弃重复信息的唯一键映射。

    Raises:
        PlanValidationError: 发现重复 JSON 键。
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise PlanValidationError(f"JSON 重复键：{key}")
        result[key] = value
    return result


def parse_manager_reply(text: str) -> ManagerReply:
    """严格解析一次完整 JSON；不去代码围栏、不补括号、不生成替代计划。

    Args:
        text:  GenerationResult.text 的原始 JSON 文本。

    Returns:
        带明确 response_type 的 ManagerReply，不能用空计划代替解析失败。

    Raises:
        PlanValidationError: 非法 JSON、截断、未知字段、错误类型抛 PlanValidationError；不运行模型、
            不修补输出。
    """
    try:
        json.loads(text, object_pairs_hook=_unique_object)
        return MANAGER_REPLY_ADAPTER.validate_json(text, strict=True)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise PlanValidationError(f"管理 JSON 不合法：{exc}") from exc


def validate_header(snapshot: Snapshot, candidate: Candidate, reply: ResponseHeader) -> None:
    """统一核对响应版本、作用域来源和候选身份；所有响应都必须先通过。

    Args:
        snapshot: 当前快照。
        candidate: 本次维护的候选。
        reply: 已严格解析的公共响应头。

    Returns:
        None 表示 schema、状态版本、候选身份和来源作用域一致。

    Raises:
        PlanValidationError: 版本、身份或候选来源不合法。
    """
    if reply.schema_version != SCHEMA_VERSION:
        raise PlanValidationError(f"schema_version 必须为 {SCHEMA_VERSION}")
    if reply.base_state_version != snapshot.version:
        raise PlanValidationError(f"base_state_version 应为 {snapshot.version}")
    if reply.candidate_id != candidate.candidate_id:
        raise PlanValidationError("candidate_id 不匹配当前候选")
    try:
        check_candidate(snapshot, candidate)
    except StateError as exc:
        raise PlanValidationError(str(exc)) from exc


def _expected_version(objects: dict, key: str, version: int) -> None:
    """校验一个现存对象的乐观版本，不接受缺失对象或陈旧动作。

    Args:
        objects: 当前对象 ID 到有版本记录的映射。
        key: 动作目标 ID。
        version: 计划声明的预期版本。

    Returns:
        None 表示目标存在且版本一致。

    Raises:
        PlanValidationError: 对象不存在或版本陈旧。
    """
    if key not in objects or objects[key].version != version:
        raise PlanValidationError(f"目标 {key} 不存在或 expected_version={version} 已过期")


def validate_invalidations(
    snapshot: Snapshot, candidate: Candidate, notices: tuple[Invalidation, ...]
) -> tuple[Invalidation, ...]:
    """核对失效目标版本及新纠正来源；语义真伪仍由独立评测判断。

    Args:
        snapshot: 当前安全前态。
        candidate: 提供本次新来源。
        notices: 来自已通过结构校验的 MemoryPlan.invalidations。

    Returns:
        通过确定性校验的通知序列；校验通过不等于语义判别器证明其真伪。

    Raises:
        PlanValidationError: 引用/版本错误抛 PlanValidationError；不自行识别所有自然语言冲突，不修改
            状态。
    """
    try:
        check_candidate(snapshot, candidate)
        sources = online_sources(snapshot, candidate)
        candidate_ids = {span.span_id for span in candidate.source_spans}
        targets = set()
        locals_by_id = {b.block_id: b for b in snapshot.local_blocks if b is not None}
        globals_by_id = {r.record_id: r for r in snapshot.global_records}
        for notice in notices:
            key = (notice.target_type, notice.target_id)
            if key in targets:
                raise PlanValidationError(f"重复失效通知：{key}")
            targets.add(key)
            objects = locals_by_id if notice.target_type == "local" else globals_by_id
            _expected_version(objects, notice.target_id, notice.expected_version)
            validate_source_refs(notice.source_refs, sources)
            if not any(ref.span_id in candidate_ids for ref in notice.source_refs):
                raise PlanValidationError("失效通知必须引用本次候选交付的纠正来源")
        return notices
    except StateError as exc:
        raise PlanValidationError(str(exc)) from exc


def resolve_source_request(
    snapshot: Snapshot,
    candidate: Candidate,
    request: SourceRequest,
    remaining_tokens: int,
    encode_text: EncodeText,
) -> SourceReadResult:
    """仅交付在线来源的原文前缀，累计真实 token 用量，不解码重造证据。

    Args:
        snapshot: snapshot/candidate 是唯一可读取来源。
        candidate: snapshot/candidate 是唯一可读取来源。
        request: 提供片段 ID 和长度。
        remaining_tokens: 调用账本剩余额度。
        encode_text: 用于计账。

    Returns:
        按请求顺序交付的原文前缀、原始引用和实际 token 用量。

    Raises:
        PlanValidationError: 响应身份错误，或来源不存在、重复及跨作用域。
        BudgetExceeded: 单项额度不能交付原文，或累计用量超过剩余额度。

    Notes:
        整个请求通过预算检查才交付，失败时不把部分来源片段或用量当作已读。
    """
    validate_header(snapshot, candidate, request)
    try:
        sources = online_sources(snapshot, candidate)
    except StateError as exc:
        raise PlanValidationError(str(exc)) from exc
    excerpts = []
    used = 0
    seen = set()
    for item in request.source_requests:
        if item.span_id not in sources or item.span_id in seen:
            raise PlanValidationError(f"来源不存在或重复请求：{item.span_id}")
        seen.add(item.span_id)
        span = sources[item.span_id]
        end = prefix_end(span.text, item.max_tokens, encode_text)
        text = span.text[:end]
        used += len(encode_text(text))
        if used > remaining_tokens:
            raise BudgetExceeded(f"来源补查实际用量 {used} 超过剩余额度 {remaining_tokens}")
        excerpts.append(
            SourceExcerpt(
                ref=SourceRef(
                    span_id=span.span_id,
                    start=span.start,
                    end=span.start + end,
                    source_hash=span.source_hash,
                ),
                text=text,
            )
        )
    return SourceReadResult(excerpts=tuple(excerpts), tokens_used=used)


def _validate_actions(snapshot: Snapshot, candidate: Candidate, plan: MemoryPlan) -> None:
    """校验槽位选择、动作身份、互斥集合及所有外部来源引用。

    各动作共享版本/来源校验；集中列出集合是为了让冲突规则只有一个实现位置。

    Args:
        snapshot: 当前安全状态。
        candidate: 本次可引用的新来源。
        plan: 身份已核对的计划。

    Returns:
        None 表示普通动作的对象、互斥集合与修复要求成立。

    Raises:
        PlanValidationError: 动作、别名、修复覆盖或佐证声明不合法。
        StateError: 在线来源或来源引用不合法。
    """
    ordinary = (
        plan.global_upserts,
        plan.global_deletes,
        plan.local_revisions,
        plan.local_exits,
        plan.exit_dependencies,
    )
    if plan.invalidations:
        raise PlanValidationError("先提交有效失效通知，再基于新版本重提普通计划")
    if plan.admission == "skip" and any(ordinary):
        raise PlanValidationError("skip 的普通更新列表必须为空")
    if plan.admission == "local":
        if plan.target_slot is None or not plan.candidate_card or not plan.candidate_card.strip():
            raise PlanValidationError("local 必须提供 target_slot 和同次生成的 candidate_card")
        if plan.target_slot >= len(snapshot.local_blocks):
            raise PlanValidationError("target_slot 超过固定槽位范围")
    elif plan.target_slot is not None or plan.candidate_card is not None:
        raise PlanValidationError("只有 local 接纳可带 target_slot/candidate_card")
    locals_by_id = {b.block_id: b for b in snapshot.local_blocks if b is not None}
    globals_by_id = {r.record_id: r for r in snapshot.global_records}
    local_actions: set[str] = set()
    for action in (*plan.local_revisions, *plan.local_exits):
        _expected_version(locals_by_id, action.block_id, action.expected_version)
        if action.block_id in local_actions:
            raise PlanValidationError(f"重复或冲突的局部动作：{action.block_id}")
        local_actions.add(action.block_id)
    exit_ids = {action.block_id for action in plan.local_exits}
    if plan.admission == "local":
        old = snapshot.local_blocks[plan.target_slot]
        if old is not None and old.block_id not in exit_ids:
            raise PlanValidationError("目标槽已占用，必须显式声明原块退出")
    global_actions: set[str] = set()
    aliases: set[str] = set()
    for upsert in plan.global_upserts:
        if upsert.record_id is None:
            if not upsert.alias or upsert.expected_version is not None:
                raise PlanValidationError("新增全局记录只能带非空 alias，版本为 null")
            if upsert.alias in aliases or upsert.alias in globals_by_id:
                raise PlanValidationError("全局新增 alias 重复或与现存 ID 冲突")
            aliases.add(upsert.alias)
        else:
            if upsert.alias is not None or upsert.expected_version is None:
                raise PlanValidationError("全局修订必须提供 ID/版本，alias 为 null")
            _expected_version(globals_by_id, upsert.record_id, upsert.expected_version)
            if upsert.record_id in global_actions:
                raise PlanValidationError("同一全局记录重复修订")
            global_actions.add(upsert.record_id)
    for delete in plan.global_deletes:
        _expected_version(globals_by_id, delete.record_id, delete.expected_version)
        if delete.record_id in global_actions:
            raise PlanValidationError("同一全局记录不能同时删除与修订")
        global_actions.add(delete.record_id)
    sources = online_sources(snapshot, candidate)
    for upsert in plan.global_upserts:
        if upsert.scope != snapshot.scope_id:
            raise PlanValidationError("global_upserts.scope 必须属于当前作用域")
        validate_source_refs(upsert.source_refs, sources)
        groups = upsert.evidence_group_ids
        if len(set(groups)) != len(groups):
            raise PlanValidationError("evidence_group_ids 不能重复")
        origins = {sources[ref.span_id].event_id for ref in upsert.source_refs}
        if upsert.support_level == "corroborated" and (len(origins) < 2 or len(groups) < 2):
            raise PlanValidationError("同一原事件的重复片段不能声称为独立佐证")
    for revision in plan.local_revisions:
        validate_source_refs(revision.source_refs, sources)
    if plan.admission != "skip":
        disabled = {b.block_id for b in locals_by_id.values() if b.status == LocalStatus.DISABLED}
        if disabled - local_actions:
            raise PlanValidationError("普通提交必须修订或明确退出所有待修复局部块")
        if snapshot.global_status == LocalStatus.DISABLED:
            invalid = {
                r.record_id for r in globals_by_id.values() if r.status == RecordStatus.INVALID
            }
            if invalid - global_actions:
                raise PlanValidationError("须修订、明确维持无效或删除失效全局记录后再重建 G")


def validate_plan(
    snapshot: Snapshot,
    candidate: Candidate,
    plan: MemoryPlan,
    limits: MemoryConfig,
    encode_text: EncodeText,
    encode_context: EncodeText,
) -> ValidatedPlan:
    """校验并推导最终状态，返回实际编译输入和预算，不运行模型或提交。

    Args:
        snapshot: snapshot/candidate 必须与计划身份一致。
        candidate: snapshot/candidate 必须与计划身份一致。
        plan: 已解析计划。
        limits:  memory 配置。
        encode_text: 用于文本计账，encode_context 用于真实编译输入。
        encode_context: encode_text 用于文本计账，encode_context 用于真实编译输入。

    Returns:
        ValidatedPlan，包含实际暂存状态、编译规格和用量；不能只有一个 passed=true 标志。

    Raises:
        PlanValidationError: 版本、来源、冲突动作或退出依赖不成立。
        BudgetExceeded: 最终文本、槽位或 next-fit 编译装箱超限。

    Notes:
        暂存推导、退出依赖与最终装箱在一次验证中完成；通过前不分配任何模型产物。
    """
    validate_header(snapshot, candidate, plan)
    try:
        _validate_actions(snapshot, candidate, plan)
        update = stage_plan(snapshot, candidate, plan)
    except StateError as exc:
        raise PlanValidationError(str(exc)) from exc
    aliases = plan_aliases(snapshot, plan)
    if set(aliases.values()) & {r.record_id for r in snapshot.global_records}:
        raise PlanValidationError("新增记录的稳定 ID 与已有对象冲突")
    records = {r.record_id: r for r in update.snapshot.global_records}
    dependencies = {d.block_id: d for d in plan.exit_dependencies}
    exit_ids = {action.block_id for action in plan.local_exits}
    if len(dependencies) != len(plan.exit_dependencies) or set(dependencies) != exit_ids:
        raise PlanValidationError("每个退出块必须且只能有一份 exit_dependencies")
    for dependency in dependencies.values():
        if not dependency.required_global_ids and not dependency.discarded_content.strip():
            raise PlanValidationError("没有巩固依赖的退出必须明确写出放弃的内容")
        for reference in dependency.required_global_ids:
            record = records.get(aliases.get(reference, reference))
            if record is None or record.status == RecordStatus.INVALID:
                raise PlanValidationError(f"退出依赖 {reference} 未在新全局有效状态成立")
    specs = build_compile_specs(update, limits, encode_context)
    usage = measure_snapshot(update.snapshot, specs, encode_text)
    check_budget(usage, limits)
    return ValidatedPlan(plan=plan, update=update, specs=specs, usage=usage)
