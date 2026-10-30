"""统一统计实际 token、在线文本副本和参数引用占用。

编译上限检查包含模板的 IDs；文本/W 使用完整稳定序列化，不做字符近似。
"""

from .config import MemoryConfig
from .schema import (
    BudgetExceeded,
    BudgetUsage,
    CompileSpec,
    EncodeText,
    Snapshot,
    TokenIds,
    WorkState,
    stable_json,
)


def count_text_tokens(text: str, encode_text: EncodeText) -> int:
    """返回原始编码 token 数；不添加模板，不改写输入。

    Args:
        text: 完整待计数文本；encode_text 为当前模型 tokenizer 的原始编码函数。
        encode_text: 当前模型 tokenizer 的原始编码函数。

    Returns:
        非负 token 数，不使用字符数、词数或近似换算。
    """
    return len(encode_text(text))


def check_token_limit(token_ids: TokenIds, limit: int, name: str) -> None:
    """检查已模板化的真实输入；等于上限通过，超限抛 BudgetExceeded。

    Args:
        token_ids: 实际输入 IDs。
        limit: 配置中的正整数上限。
        name: 标识具体对象。

    Returns:
        None 表示长度小于或等于上限。

    Raises:
        BudgetExceeded: 超限抛 BudgetExceeded，带 name、实际值和上限；不裁剪。
    """
    if limit <= 0:
        raise BudgetExceeded(f"{name} 上限 {limit} 必须为正整数")
    if len(token_ids) > limit:
        raise BudgetExceeded(f"{name}={len(token_ids)} tokens，超过上限 {limit}")


def prefix_end(text: str, max_tokens: int, encode: EncodeText) -> int:
    """找一个真实编码可装入的原文前缀，返回 Unicode 结束位置。

    不解码 token 生成证据。BPE 长度不保证单调，因此结果不声称是最长前缀，
    返回前仍核对真实长度；调用方保留余下原文。

    Args:
        text: 待保留原文，不通过 token 解码重建。
        max_tokens: 包含调用方模板的正整数额度。
        encode: 实际原始或模板编码函数。

    Returns:
        可装入的非空原文前缀结束位置，单位为 Unicode 码点；不保证最长。

    Raises:
        BudgetExceeded: 额度非正，或搜索无法形成可装入的非空前缀。
    """
    if max_tokens <= 0:
        raise BudgetExceeded(f"原文前缀额度 {max_tokens} 必须为正整数")
    if len(encode(text)) <= max_tokens:
        return len(text)
    low, high = 0, len(text)
    while low + 1 < high:
        mid = (low + high) // 2
        if len(encode(text[:mid])) <= max_tokens:
            low = mid
        else:
            high = mid
    if low == 0:
        raise BudgetExceeded(f"必要字符及模板无法装入 {max_tokens} tokens")
    check_token_limit(encode(text[:low]), max_tokens, "原文前缀")
    return low


def measure_work(work: WorkState, encode_text: EncodeText) -> int:
    """按完整 W 序列化计数，包含目标、元数据和 pending 中的副本。

    Args:
        work: 完整工作状态，包含 pending 的来源与文本副本。
        encode_text: 实际原始 tokenizer 编码。

    Returns:
        稳定序列化后的完整 W token 数，不修改输入。
    """
    return count_text_tokens(stable_json(work.model_dump(mode="json")), encode_text)


def measure_snapshot(
    snapshot: Snapshot, specs: tuple[CompileSpec, ...], encode_text: EncodeText
) -> BudgetUsage:
    """统计来源、副本、卡片、记录、派生索引、回执、W 和因子引用。

    Args:
        snapshot: 提交态或暂存态，不修改任何输入。
        specs: 同一状态全部活跃编译输入；长度直接取真实 IDs。
        encode_text: 与模型相同的原始编码函数。

    Returns:
        完整稳态用量；原始因子字节来自 FactorRef，组合及峰值另报。

    Raises:
        StateError: 来源或依赖结构损坏。
    """
    from .state import build_dependency_index

    persistent = snapshot.model_dump(mode="json", exclude={"work"})
    persistent["dependency_index"] = build_dependency_index(snapshot).model_dump(mode="json")
    # CompiledMemory 也持有编译文本和版本绑定，不能漏计全局序列化文本这一副本。
    # token_ids 是已计入块输入长度的数值载荷，不把整数的十进制展示当成自然语言。
    persistent["compile_inputs"] = [
        spec.model_dump(mode="json", exclude={"token_ids"}) for spec in specs
    ]
    text = stable_json(persistent)
    work_text = stable_json(snapshot.work.model_dump(mode="json"))
    blocks = tuple(block for block in snapshot.local_blocks if block is not None)
    return BudgetUsage(
        local_slots=len(blocks),
        slot_capacity=len(snapshot.local_blocks),
        global_blocks=sum(spec.kind == "global" for spec in specs),
        local_input_tokens=tuple(len(spec.token_ids) for spec in specs if spec.kind == "local"),
        global_input_tokens=tuple(len(spec.token_ids) for spec in specs if spec.kind == "global"),
        local_card_tokens=tuple(count_text_tokens(b.memory_card, encode_text) for b in blocks),
        persistent_text_tokens=count_text_tokens(text, encode_text),
        persistent_text_bytes=len(text.encode("utf-8")),
        working_tokens=count_text_tokens(work_text, encode_text),
        working_bytes=len(work_text.encode("utf-8")),
        content_width=sum(ref.content_width for ref in snapshot.factor_refs),
        factor_bytes=sum(ref.size_bytes for ref in snapshot.factor_refs),
    )


def check_budget(usage: BudgetUsage, limits: MemoryConfig) -> None:
    """统一检查所有稳态容量；不删除、截断或忽略 disabled 块的占用。

    Args:
        usage: 实际统计。
        limits: 唯一的 memory 配置。

    Returns:
        None 表示所有适用上限同时满足；全局可装入性还由 build_compile_specs 检查。

    Raises:
        BudgetExceeded: 任一超限抛 BudgetExceeded；不擅自淘汰记录、增大配置或忽略 disabled 块。
    """
    checks = [
        ("局部槽位总容量", usage.slot_capacity, limits.local_slots),
        ("已占局部槽位", usage.local_slots, limits.local_slots),
        ("全局块数", usage.global_blocks, limits.global_blocks),
        ("持久文本", usage.persistent_text_tokens, limits.persistent_text_tokens),
        ("工作窗口", usage.working_tokens, limits.working_tokens),
    ]
    for name, values, limit in (
        ("局部编译输入", usage.local_input_tokens, limits.local_block_tokens),
        ("全局编译输入", usage.global_input_tokens, limits.global_block_tokens),
        ("局部卡片", usage.local_card_tokens, limits.local_card_tokens),
    ):
        checks.extend((f"{name}[{i}]", value, limit) for i, value in enumerate(values))
    for name, value, limit in checks:
        if value > limit:
            raise BudgetExceeded(f"{name}={value}，超过上限 {limit}")
