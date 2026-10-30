"""统一统计实际 token、在线文本副本和参数引用占用。

编译上限检查包含模板的 IDs；文本/W 使用完整稳定序列化，不做字符近似。
"""

from .schema import (
    BudgetExceeded,
    BudgetUsage,
    CompileSpec,
    EncodeText,
    Snapshot,
    TokenIds,
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


def measure_snapshot(
    snapshot: Snapshot, specs: tuple[CompileSpec, ...], encode_text: EncodeText
) -> BudgetUsage:
    """统计当前工作上下文、完整编译输入、因子元数据和实际张量占用。

    保留所有历史 chunk 的编译输入用于复现；不再设置槽位或持久文本淘汰预算。
    token_ids 的长度另行报告，不把其十进制表示计入自然语言 token 数。
    """
    persistent = snapshot.model_dump(mode="json", exclude={"work"})
    persistent["compile_inputs"] = [
        spec.model_dump(mode="json", exclude={"token_ids"}) for spec in specs
    ]
    text = stable_json(persistent)
    work_text = stable_json(snapshot.work.model_dump(mode="json"))
    return BudgetUsage(
        compiled_chunks=len(specs),
        chunk_input_tokens=tuple(len(spec.token_ids) for spec in specs),
        persistent_text_tokens=count_text_tokens(text, encode_text),
        persistent_text_bytes=len(text.encode("utf-8")),
        working_tokens=count_text_tokens(work_text, encode_text),
        working_bytes=len(work_text.encode("utf-8")),
        content_width=sum(ref.content_width for ref in snapshot.factor_refs),
        factor_bytes=sum(ref.size_bytes for ref in snapshot.factor_refs),
    )
