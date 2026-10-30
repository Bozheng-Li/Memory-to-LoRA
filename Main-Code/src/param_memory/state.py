"""对不可变快照执行纯状态推导、来源依赖校验和确定性装箱。

本模块不运行模型、不写文件。暂存态的退出与证据回收只有安装成功才生效。
"""

from .budget import check_token_limit
from .config import MemoryConfig
from .events import candidate_from_spans, check_continuation, validate_span
from .schema import (
    BudgetExceeded,
    Candidate,
    CompileSpec,
    Dependency,
    DependencyIndex,
    EncodeText,
    GlobalRecord,
    Invalidation,
    LocalBlock,
    LocalStatus,
    MemoryPlan,
    ObjectVersion,
    PlanValidationError,
    RecordStatus,
    Snapshot,
    SourceRef,
    SourceSpan,
    StagedUpdate,
    StateError,
    WorkState,
    compile_fingerprint,
    stable_json,
    text_hash,
)


def empty_snapshot(scope_id: str, local_slots: int, goal: str) -> Snapshot:
    """建立指定作用域的空槽和工作目标；非法身份/容量抛 StateError。

    Args:
        scope_id: 来自当前任务/源历史身份。
        local_slots: 来自 YAML。
        goal: 已经交付的任务规则。

    Returns:
        没有局部/全局内容因子的 Snapshot，空槽不产生 D2L 输出。

    Raises:
        StateError: 非法身份或容量抛 StateError；不加载模型、不生成随机业务 ID。
    """
    if not scope_id.strip() or not goal.strip() or type(local_slots) is not int or local_slots < 0:
        raise StateError("作用域和目标必须非空，local_slots 必须为非负整数")
    return Snapshot(
        scope_id=scope_id,
        version=0,
        work=WorkState(
            goal=goal, recent_spans=(), summary="", pending_candidate=None, processed_through=None
        ),
        local_blocks=(None,) * local_slots,
        global_records=(),
        global_status=LocalStatus.ACTIVE,
        sources=(),
        factor_refs=(),
        last_maintenance=None,
    )


def online_sources(snapshot: Snapshot, candidate: Candidate | None = None) -> dict[str, SourceSpan]:
    """索引本作用域在线来源，拒绝同 ID 不同原文；不查询离线历史。

    Args:
        snapshot: 当前作用域的有界快照。
        candidate: 可选的当前候选，只读取其已交付来源。

    Returns:
        从来源 ID 到原始片段的临时映射，不保留或补查离线历史。

    Raises:
        StateError: 作用域、原文完整性或同 ID 来源内容冲突。
    """
    spans = (*snapshot.sources, *snapshot.work.recent_spans)
    if candidate is not None:
        spans += candidate.source_spans
    result: dict[str, SourceSpan] = {}
    for span in spans:
        if span.scope_id != snapshot.scope_id:
            raise StateError(f"来源 {span.span_id} 不属于作用域 {snapshot.scope_id}")
        validate_span(span)
        if span.span_id in result and result[span.span_id] != span:
            raise StateError(f"来源 ID 冲突：{span.span_id}")
        result[span.span_id] = span
    return result


def span_ref(span: SourceSpan) -> SourceRef:
    """引用一个已交付的完整片段，位置仍相对原事件。

    Args:
        span: 已经过完整性检查的原始片段。

    Returns:
        引用整个片段的 SourceRef，区间仍相对原事件。
    """
    return SourceRef(
        span_id=span.span_id, start=span.start, end=span.end, source_hash=span.source_hash
    )


def validate_source_refs(refs: tuple[SourceRef, ...], sources: dict[str, SourceSpan]) -> None:
    """在唯一来源边界核对身份、hash 和原事件区间；失败抛 StateError。

    Args:
        refs: 记录或计划携带的来源引用。
        sources: 已核对完整性的在线来源映射。

    Returns:
        None 表示所有引用的身份、hash 和半开区间有效。

    Raises:
        StateError: 来源不存在、hash 不符或区间越界。
    """
    for ref in refs:
        span = sources.get(ref.span_id)
        if span is None:
            raise StateError(f"source_refs 指向不存在的在线来源：{ref.span_id}")
        if ref.source_hash != span.source_hash:
            raise StateError(f"source_refs 的 hash 不一致：{ref.span_id}")
        if not span.start <= ref.start < ref.end <= span.end:
            raise StateError(f"source_refs 越界：{ref.span_id}[{ref.start}:{ref.end}]")


def check_candidate(snapshot: Snapshot, candidate: Candidate) -> None:
    """核对候选是未处理前缀且编译文本来自该前缀，不能覆盖另一个 pending。

    Args:
        snapshot: 候选对应的当前快照。
        candidate: 待登记或待校验的候选。

    Returns:
        None 表示候选是游标之后的未处理连续前缀，文本及 ID 与原文一致。

    Raises:
        StateError: 来源、游标、pending 身份或规范候选内容冲突。
    """
    pending = snapshot.work.pending_candidate
    if pending is not None and pending != candidate:
        raise StateError("不能覆盖不同的 pending_candidate")
    if snapshot.work.recent_spans[: len(candidate.source_spans)] != candidate.source_spans:
        raise StateError("候选不是当前未处理来源前缀")
    online_sources(snapshot, candidate)
    previous = snapshot.work.processed_through
    for span in candidate.source_spans:
        check_continuation(span, previous)
        previous = span
    expected = candidate_from_spans(candidate.source_spans, candidate.trigger)
    if candidate != expected:
        raise StateError("候选文本或身份与原始来源不一致")


def build_dependency_index(snapshot: Snapshot) -> DependencyIndex:
    """从真实来源和记录引用派生反向依赖，悬空或重复身份抛 StateError。

    Args:
        snapshot: 当前完整状态。

    Returns:
        DependencyIndex；不建立第二份可独立编辑的引用计数。

    Raises:
        StateError: 悬空引用抛 StateError；不从日志或历史文件补找来源。

    Notes:
        来源和记录的反向边一起从快照派生，避免第二份可独立编辑的引用真相。
    """
    all_sources = online_sources(snapshot)
    sources = {s.span_id: all_sources[s.span_id] for s in snapshot.sources}
    records = {r.record_id: r for r in snapshot.global_records}
    blocks = {b.block_id: b for b in snapshot.local_blocks if b is not None}
    if len(sources) != len(snapshot.sources) or len(records) != len(snapshot.global_records):
        raise StateError("快照包含重复来源或全局记录 ID")
    if len(blocks) != sum(b is not None for b in snapshot.local_blocks):
        raise StateError("快照包含重复局部块 ID")
    source_edges = {key: [set(), set()] for key in sources}
    record_edges = {key: [set(), set()] for key in records}
    objects = [(b.block_id, b.source_refs, b.claim_refs, 0) for b in blocks.values()]
    objects += [(r.record_id, r.source_refs, r.related_record_ids, 1) for r in records.values()]
    for object_id, refs, dependencies, kind in objects:
        validate_source_refs(refs, sources)
        for ref in refs:
            source_edges[ref.span_id][kind].add(object_id)
        for record_id in dependencies:
            if record_id not in records:
                raise StateError(f"{object_id} 存在悬空记录依赖 {record_id}")
            record_edges[record_id][kind].add(object_id)
    for slot, block in enumerate(snapshot.local_blocks):
        if block is not None and block.slot_id != slot:
            raise StateError(f"槽位 {slot} 与 block.slot_id={block.slot_id} 不一致")
    if any(r.scope != snapshot.scope_id for r in records.values()):
        raise StateError("全局记录跨作用域")
    return DependencyIndex(
        sources=tuple(
            Dependency(
                key=k, local_block_ids=tuple(sorted(v[0])), global_record_ids=tuple(sorted(v[1]))
            )
            for k, v in sorted(source_edges.items())
        ),
        records=tuple(
            Dependency(
                key=k, local_block_ids=tuple(sorted(v[0])), global_record_ids=tuple(sorted(v[1]))
            )
            for k, v in sorted(record_edges.items())
        ),
    )


def plan_aliases(snapshot: Snapshot, plan: MemoryPlan) -> dict[str, str]:
    """按作用域、计划身份和别名生成稳定全局 ID；不分配随机业务 ID。

    Args:
        snapshot: 提供当前作用域。
        plan: 新增别名已完成唯一性校验的计划。

    Returns:
        计划内别名到稳定全局记录 ID 的映射，不修改状态。
    """
    return {
        upsert.alias: "g:" + text_hash(stable_json([snapshot.scope_id, plan.plan_id, upsert.alias]))
        for upsert in plan.global_upserts
        if upsert.alias is not None
    }


def _retain_evidence(
    records: tuple[GlobalRecord, ...],
    blocks: tuple[LocalBlock | None, ...],
    available: dict[str, SourceSpan],
) -> tuple[tuple[GlobalRecord, ...], tuple[SourceSpan, ...]]:
    """在局部退出前复制仍需保留的证据子区间，并改写全部新状态引用。

    Args:
        records: 拟保留的完整全局记录。
        blocks: 拟提交的局部槽位。
        available: 计划可以合法引用的在线来源。

    Returns:
        改写来源引用后的全局记录及仍被引用的来源；旧状态不变。

    Notes:
        复制证据与改写依赖一起完成，退出前态仍保持原样直到运行时成功发布。
    """
    local_sources = {ref.span_id for b in blocks if b is not None for ref in b.source_refs}
    retained = {key: available[key] for key in local_sources}
    updated = []
    for record in records:
        refs = []
        for ref in record.source_refs:
            source = available[ref.span_id]
            if ref.span_id not in local_sources and not ref.span_id.startswith("evidence:"):
                # 复制原文区间，不能用 Agent 改写的 claim 充当原始证据。
                text = source.text[ref.start - source.start : ref.end - source.start]
                key = "evidence:" + text_hash(
                    stable_json(
                        [
                            source.scope_id,
                            source.event_id,
                            ref.start,
                            ref.end,
                            text_hash(text),
                        ]
                    )
                )
                source = source.model_copy(
                    update={
                        "span_id": key,
                        "start": ref.start,
                        "end": ref.end,
                        "text": text,
                        "source_hash": text_hash(text),
                        "boundary": None,
                    }
                )
                ref = span_ref(source)
            retained[source.span_id] = source
            refs.append(ref)
        updated.append(record.model_copy(update={"source_refs": tuple(refs)}))
    return tuple(updated), tuple(retained[key] for key in sorted(retained))


def stage_plan(snapshot: Snapshot, candidate: Candidate, plan: MemoryPlan) -> StagedUpdate:
    """在隔离副本上应用已校验计划，返回尚未发布的完整新状态。

    此函数按记录、槽位、证据三阶段推导，步骤保持在一起以保证引用一致。
    参数必须先经 planning 校验；不编译、不修改旧来源或旧槽位。

    Args:
        snapshot: 计划基于的旧版本。
        candidate: 当前候选。
        plan: 已完成结构、身份和引用检查。

    Returns:
        StagedUpdate，含新状态与变更范围；版本尚未发布，退出前态仍保留。

    Raises:
        PlanValidationError: 动作冲突或无法解析的依赖抛 PlanValidationError；不编译、不改活跃状态、
            不释放旧来源。
    """
    aliases = plan_aliases(snapshot, plan)
    records = {r.record_id: r for r in snapshot.global_records}
    for delete in plan.global_deletes:
        del records[delete.record_id]
    for upsert in plan.global_upserts:
        key = upsert.record_id if upsert.record_id is not None else aliases[upsert.alias]
        fields = upsert.model_dump(exclude={"alias", "record_id", "expected_version"})
        fields["related_record_ids"] = tuple(
            aliases.get(ref, ref) for ref in upsert.related_record_ids
        )
        records[key] = GlobalRecord(
            **fields,
            record_id=key,
            version=records[key].version + 1 if key in records else 1,
        )
    blocks = list(snapshot.local_blocks)
    old_blocks = {b.block_id: b for b in blocks if b is not None}
    for exit_action in plan.local_exits:
        blocks[old_blocks[exit_action.block_id].slot_id] = None
    changed = []
    for revision in plan.local_revisions:
        old = old_blocks[revision.block_id]
        blocks[old.slot_id] = old.model_copy(
            update={
                "compile_text": revision.compile_text,
                "memory_card": revision.memory_card,
                "source_refs": revision.source_refs,
                "version": old.version + 1,
                "status": LocalStatus.ACTIVE,
            }
        )
        changed.append(old.block_id)
    if plan.admission == "local":
        key = "l:" + text_hash(stable_json([snapshot.scope_id, candidate.candidate_id]))
        blocks[plan.target_slot] = LocalBlock(
            block_id=key,
            slot_id=plan.target_slot,
            version=1,
            source_refs=tuple(span_ref(s) for s in candidate.source_spans),
            compile_text=candidate.compile_text,
            memory_card=plan.candidate_card,
            claim_refs=(),
            status=LocalStatus.ACTIVE,
            factor_key=key,
        )
        changed.append(key)
    # 事实 ID 的依赖随真实记录推导，不能保留已经删除的记录 ID。
    for slot, block in enumerate(blocks):
        if block is None:
            continue
        source_ids = {ref.span_id for ref in block.source_refs}
        claims = set(block.claim_refs) & records.keys()
        claims.update(
            r.record_id
            for r in records.values()
            if any(ref.span_id in source_ids for ref in r.source_refs)
        )
        blocks[slot] = block.model_copy(update={"claim_refs": tuple(sorted(claims))})
    records_tuple, sources = _retain_evidence(
        tuple(records[k] for k in sorted(records)),
        tuple(blocks),
        online_sources(snapshot, candidate),
    )
    staged = snapshot.model_copy(
        update={
            "version": snapshot.version + 1,
            "local_blocks": tuple(blocks),
            "global_records": records_tuple,
            "sources": sources,
            "factor_refs": (),
            "global_status": snapshot.global_status
            if plan.admission == "skip"
            else LocalStatus.ACTIVE,
        }
    )
    try:
        build_dependency_index(staged)
    except StateError as exc:
        raise PlanValidationError(str(exc)) from exc
    return StagedUpdate(
        base_version=snapshot.version,
        snapshot=staged,
        changed_local_ids=tuple(changed),
        global_changed=(
            _global_text(tuple(r for r in records_tuple if r.status != RecordStatus.INVALID))
            != _global_text(
                tuple(
                    sorted(
                        (r for r in snapshot.global_records if r.status != RecordStatus.INVALID),
                        key=lambda r: r.record_id,
                    )
                )
            )
            or snapshot.global_status != LocalStatus.ACTIVE
        ),
    )


def _compile_spec(
    key: str,
    kind: str,
    text: str,
    refs: tuple[SourceRef, ...],
    versions: tuple[ObjectVersion, ...],
    encode_context: EncodeText,
) -> CompileSpec:
    """绑定一次真实编码和当前来源，供局部与全局编译统一使用。

    Args:
        key: 内容块的稳定标识。
        kind: local 或 global。
        text: 实际编译文本。
        refs: 当前来源绑定。
        versions: 当前记录或局部对象版本。
        encode_context: 实际 D2L 模板编码。

    Returns:
        包含真实输入 IDs、文本指纹和当前来源版本的 CompileSpec。
    """
    ids = encode_context(text)
    return CompileSpec(
        block_key=key,
        kind=kind,
        text=text,
        token_ids=ids,
        source_refs=refs,
        record_versions=versions,
        input_fingerprint=compile_fingerprint(text, ids),
    )


def _global_text(records: tuple[GlobalRecord, ...]) -> str:
    """序列化有效事实及范围/时间/状态，不把内部 hash、理由或用途编入事实。

    Args:
        records: 按稳定顺序排列的有效记录。

    Returns:
        保留事实、种类、范围、时间与状态的编译文本；不写入管理理由或内部 hash。
    """
    fields = {"kind", "claim", "scope", "valid_from", "valid_until", "observed_at", "status"}
    return "\n".join(stable_json(r.model_dump(mode="json", include=fields)) for r in records)


def build_compile_specs(
    update: StagedUpdate, limits: MemoryConfig, encode_context: EncodeText
) -> tuple[CompileSpec, ...]:
    """稳定排序并 next-fit 装箱全部活跃输入；不拆记录、不静默删限定条件。

    Args:
        update: 提供候选新状态。
        limits: 提供两层长度/块数。
        encode_context: 计算含模板的准确输入。

    Returns:
        全部活跃块的 CompileSpec，绑定实际 IDs、文本、来源/记录版本和输入指纹。

    Raises:
        BudgetExceeded: 单条记录或固定装箱超限抛 BudgetExceeded；不截断、不按未来问题筛选。

    Notes:
        局部选择与全局装箱共同定义完整活跃输入，检查实际模板长度后才返回。
    """
    specs: list[CompileSpec] = []
    for block in update.snapshot.local_blocks:
        if block is None or block.status == LocalStatus.DISABLED:
            continue
        spec = _compile_spec(
            block.factor_key,
            "local",
            block.compile_text,
            block.source_refs,
            (ObjectVersion(object_id=block.block_id, version=block.version),),
            encode_context,
        )
        check_token_limit(spec.token_ids, limits.local_block_tokens, block.block_id)
        specs.append(spec)
    if update.snapshot.global_status == LocalStatus.DISABLED:
        return tuple(specs)
    groups: list[tuple[GlobalRecord, ...]] = []
    current: tuple[GlobalRecord, ...] = ()
    for record in sorted(update.snapshot.global_records, key=lambda r: r.record_id):
        if record.status == RecordStatus.INVALID:
            continue
        proposed = (*current, record)
        if len(encode_context(_global_text(proposed))) > limits.global_block_tokens:
            check_token_limit(
                encode_context(_global_text((record,))),
                limits.global_block_tokens,
                record.record_id,
            )
            if current:
                groups.append(current)
            current = (record,)
        else:
            current = proposed
    if current:
        groups.append(current)
    if len(groups) > limits.global_blocks:
        raise BudgetExceeded(f"全局 next-fit 需要 {len(groups)} 块，上限 {limits.global_blocks}")
    for index, group in enumerate(groups):
        specs.append(
            _compile_spec(
                f"global:{index}",
                "global",
                _global_text(group),
                tuple(dict.fromkeys(ref for record in group for ref in record.source_refs)),
                tuple(ObjectVersion(object_id=r.record_id, version=r.version) for r in group),
                encode_context,
            )
        )
    return tuple(specs)


def disable_invalidated(snapshot: Snapshot, notices: tuple[Invalidation, ...]) -> Snapshot:
    """构造停用错误因子的安全快照，保留原始证据供修复，不恢复已知错误。

    失效沿来源重叠和记录依赖传递；全局任何有效记录受影响都停用整个 G。
    W 的摘要缺乏细粒度来源，因此保守清空，纠正候选继续保持 pending。

    Args:
        snapshot: 失效前状态。
        notices: 已通过来源、作用域和版本校验。

    Returns:
        新安全 Snapshot，保留预算内修复来源；W 中已知错误的片段/摘要同步移除或明确标失效。

    Raises:
        StateError: 无法构造合法安全上下文时抛 StateError；不把未知新值编成事实，不修改原快照。

    Notes:
        按来源重叠和记录依赖求闭包，再同时停用参数与清理 W，避免遗漏错误的间接依赖。
    """
    index = build_dependency_index(snapshot)
    sources = online_sources(snapshot)
    local_ids = {n.target_id for n in notices if n.target_type == "local"}
    record_ids = {n.target_id for n in notices if n.target_type == "global"}
    blocks = tuple(b for b in snapshot.local_blocks if b is not None)
    changed = True
    while changed:
        before = (len(local_ids), len(record_ids))
        refs = [ref for b in blocks if b.block_id in local_ids for ref in b.source_refs]
        refs += [
            ref
            for r in snapshot.global_records
            if r.record_id in record_ids
            for ref in r.source_refs
        ]
        origins = [(sources[r.span_id].event_id, r.start, r.end) for r in refs]
        affected_spans = {
            s.span_id
            for s in sources.values()
            if any(
                s.event_id == event and s.start < end and start < s.end
                for event, start, end in origins
            )
        }
        for edge in index.sources:
            if edge.key in affected_spans:
                local_ids.update(edge.local_block_ids)
                record_ids.update(edge.global_record_ids)
        for edge in index.records:
            if edge.key in record_ids:
                local_ids.update(edge.local_block_ids)
                record_ids.update(edge.global_record_ids)
        changed = before != (len(local_ids), len(record_ids))
    pending = snapshot.work.pending_candidate
    pending_ids = {s.span_id for s in pending.source_spans} if pending is not None else set()
    if affected_spans & pending_ids:
        raise StateError("纠正候选同时含已知错误来源，无法构造合法安全工作上下文")
    work = snapshot.work.model_copy(
        update={
            "summary": "",
            "recent_spans": tuple(
                s for s in snapshot.work.recent_spans if s.span_id not in affected_spans
            ),
        }
    )
    return snapshot.model_copy(
        update={
            "version": snapshot.version + 1,
            "work": work,
            "factor_refs": (),
            "local_blocks": tuple(
                b.model_copy(update={"status": LocalStatus.DISABLED, "version": b.version + 1})
                if b is not None and b.block_id in local_ids
                else b
                for b in snapshot.local_blocks
            ),
            "global_records": tuple(
                r.model_copy(update={"status": RecordStatus.INVALID, "version": r.version + 1})
                if r.record_id in record_ids
                else r
                for r in snapshot.global_records
            ),
            "global_status": LocalStatus.DISABLED if record_ids else snapshot.global_status,
        }
    )
