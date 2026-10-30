"""组织管理可见材料并执行一次管理生成。

不持有独立模型，不修改槽位，不解析或修补生成结果；重提由 Runtime 统一控制。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .budget import check_token_limit
from .config import GenerationConfig, MemoryConfig
from .schema import (
    MANAGER_REPLY_ADAPTER,
    SCHEMA_VERSION,
    Candidate,
    ChatMessage,
    ChatMessages,
    ConfigError,
    GenerationResult,
    Snapshot,
    SourceExcerpt,
    ValidationFeedback,
    stable_json,
)
from .state import online_sources

if TYPE_CHECKING:
    from .d2l import D2LBackend


class AgentManager:
    """复用任务基座，只保存固定管理模板和本模式生成配置。"""

    def __init__(
        self, backend: D2LBackend, prompt_path: Path, generation: GenerationConfig
    ) -> None:
        """读取管理指令并检查实际上下文容量；不加载另一模型或执行生成。

        Args:
            backend: 任务与管理共同使用的同一实例。
            prompt_path: 来自 YAML。
            generation:  generation.manager。

        Returns:
            None，构造一个不持有独立模型、不持有全历史的管理器。

        Raises:
            ConfigError: 模板文件不可读抛 ConfigError；只读取模板，不启动生成、不自行安装适配器。
        """
        self.backend = backend
        self.generation = generation
        try:
            self.prompt = prompt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigError(f"管理提示不可读：{prompt_path}") from exc
        if not self.prompt.strip():
            raise ConfigError("管理提示不能为空")
        if generation.max_input_tokens + generation.max_new_tokens > backend.context_capacity:
            raise ConfigError("generation.manager 的输入加输出超过基座实际上下文容量")

    def build_messages(
        self,
        snapshot: Snapshot,
        candidate: Candidate,
        limits: MemoryConfig,
        source_spans: tuple[SourceExcerpt, ...] = (),
        feedback: ValidationFeedback | None = None,
    ) -> ChatMessages:
        """组织有界已发生材料与原生生成的响应 schema；不带未来问题或 gold。

        Args:
            snapshot: 管理可见状态。
            candidate: 新输入。
            limits: 提供预算。
            source_spans: 只能来自合法补查。
            feedback: 上轮有界错误。

        Returns:
            ChatMessages；输出 schema 由共享类型生成，未来问题、答案和完整 metadata 不得进入。

        Raises:
            StateError: 在线来源的作用域或身份不一致。

        Notes:
            集中构造唯一可见材料清单，使未来问题、gold 与完整管理日志没有第二条注入路径。
        """
        sources = online_sources(snapshot, candidate)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scope_id": snapshot.scope_id,
            "base_state_version": snapshot.version,
            "goal": snapshot.work.goal,
            "candidate": candidate.model_dump(mode="json"),
            "local_slots": [
                None
                if b is None
                else b.model_dump(
                    mode="json",
                    exclude={
                        "compile_text",
                        "factor_key",
                    },
                )
                for b in snapshot.local_blocks
            ],
            "global_records": [r.model_dump(mode="json") for r in snapshot.global_records],
            "global_status": snapshot.global_status.value,
            "budgets": limits.model_dump(mode="json"),
            "available_sources": [
                s.model_dump(mode="json", exclude={"text"}) for s in sources.values()
            ],
            "source_excerpts": [s.model_dump(mode="json") for s in source_spans],
            "feedback": None if feedback is None else feedback.model_dump(mode="json"),
            "response_schema": MANAGER_REPLY_ADAPTER.json_schema(),
        }
        return (
            ChatMessage(role="system", content=self.prompt),
            ChatMessage(role="user", content=stable_json(payload)),
        )

    def generate_reply(self, messages: ChatMessages) -> GenerationResult:
        """编码并检查完整管理输入，生成一次原始响应；不自行重试。

        Args:
            messages: 来自 build_messages。

        Returns:
            原始 GenerationResult；解析、来源补查和重提由 Runtime 决定。

        Raises:
            BudgetExceeded: 完整管理输入超限。
            BackendError: 共享基座生成失败；不自行重试。
        """
        token_ids = self.backend.encode_messages(messages)
        check_token_limit(token_ids, self.generation.max_input_tokens, "管理输入")
        return self.backend.generate(token_ids, self.generation)
