"""将连续上下文按 chunk 直接编译，持续累积全部历史 LoRA 因子。

编译、组合、安装和前向成功后才同时发布新因子与游标；失败保留原文并暂停。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType

from .budget import check_token_limit
from .config import RunConfig
from .d2l import CombinedFactors, CompiledMemory, D2LBackend, RawFactors, empty_compiled
from .events import append_span, build_candidate, finish_candidate
from .run_logging import log_phase
from .schema import (
    BackendError,
    BudgetExceeded,
    Candidate,
    ChatMessage,
    CompileSpec,
    ConfigError,
    GenerationResult,
    MaintenanceCost,
    MaintenanceReceipt,
    MaintenanceResult,
    MaintenanceStatus,
    Snapshot,
    SourceSpan,
    StateError,
    TokenIds,
    Trigger,
    stable_json,
)
from .state import build_compile_spec, empty_snapshot

logger = logging.getLogger(__name__)


class MemoryRuntime:
    """一个历史的直接 D2L 状态机；不执行记忆接纳、摘要或淘汰决策。"""

    def __init__(
        self,
        initial: Snapshot,
        backend: D2LBackend,
        config: RunConfig,
        *,
        record_maintenance: Callable[[MaintenanceResult], None] | None = None,
    ) -> None:
        """装配空历史，检查真实模型容量并清除前一历史的参数绑定。"""
        if initial != empty_snapshot(initial.scope_id, initial.work.goal):
            raise StateError("每份历史必须使用新的空快照")
        self.backend = backend
        self._record_maintenance = record_maintenance
        self.limits, self.task_generation = config.memory, config.generation.task
        if (
            self.task_generation.max_input_tokens + self.task_generation.max_new_tokens
            > backend.context_capacity
        ):
            raise ConfigError("generation.task 的输入加输出超过基座实际上下文容量")
        if self.limits.chunk_tokens > backend.compiler_context_capacity:
            raise ConfigError("memory.chunk_tokens 超过上下文编码器实际容量")
        try:
            self.task_prompt = config.prompts.task_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigError(f"任务提示不可读：{config.prompts.task_path}") from exc
        if not self.task_prompt.strip():
            raise ConfigError("任务提示不能为空")
        self.snapshot, self.compiled, self.status = initial, empty_compiled(), "ready"
        self._cost = MaintenanceCost()
        self._task_input(initial, "")
        with log_phase(logger, "reset"):
            backend.reset()

    def observe(self, span: SourceSpan) -> tuple[MaintenanceResult, ...]:
        """接收连续原文；达到 chunk 上限或再加入一片将超限时直接编译。

        不足一个 chunk 的尾段由历史结束时显式编译。
        暂停时所有尚未成功提交的片段仍留在工作上下文中。
        """
        if self.status != "ready" or span.scope_id != self.snapshot.scope_id:
            raise StateError("当前状态不能接收事件，或片段不属于本作用域")
        check_token_limit(
            self.backend.encode_context(span.text), self.limits.chunk_tokens, "输入片段"
        )
        work = append_span(self.snapshot.work, span)
        self.snapshot = self.snapshot.model_copy(update={"work": work})
        results = []
        while self.snapshot.work.recent_spans:
            candidate = build_candidate(
                self.snapshot.work, Trigger.WINDOW_FULL, self.limits, self.backend.encode_context
            )
            full = len(candidate.source_spans) < len(self.snapshot.work.recent_spans)
            full |= (
                len(self.backend.encode_context(candidate.compile_text)) >= self.limits.chunk_tokens
            )
            if not full:
                break
            result = self.maintain(candidate)
            results.append(result)
            if result.status == MaintenanceStatus.PAUSED:
                break
        return tuple(results)

    def maintain(self, candidate: Candidate) -> MaintenanceResult:
        """在候选关联上下文内完成事务，立即交付结果后才处理下一个候选。"""
        with log_phase(
            logger,
            "maintenance",
            history_id=self.snapshot.scope_id,
            candidate_id=candidate.candidate_id,
            event_ids=",".join(dict.fromkeys(span.event_id for span in candidate.source_spans)),
            base_version=self.snapshot.version,
        ):
            result = self._maintain(candidate)
            # 在事务恢复边界之外写日志；I/O 失败不能被误判为事务失败并重做恢复。
            if self._record_maintenance is not None:
                self._record_maintenance(result)
            return result

    def _maintain(self, candidate: Candidate) -> MaintenanceResult:
        """每个 chunk 编译一次；完整事务失败时恢复当前提交版并保留 pending。

        编译、状态发布与恢复集中在同一异常边界，避免把部分成功当作已消费原文。
        """
        if self.status != "ready":
            raise StateError(f"状态 {self.status} 不允许启动维护")
        spec = build_compile_spec(
            self.snapshot, candidate, self.limits, self.backend.encode_context
        )
        pending = self.snapshot.work.model_copy(update={"pending_candidate": candidate})
        self.snapshot = self.snapshot.model_copy(update={"work": pending})
        self.status, self._cost = "maintaining", MaintenanceCost()
        self._record_factor_peak((), None)
        started, base_version = time.perf_counter(), self.snapshot.version
        errors: list[str] = []
        outcome, reason = MaintenanceStatus.COMMITTED, "直接编译原始 chunk 并追加历史因子"
        try:
            self._reset()
            compiled = self._compile_chunk(spec)
            staged = self.snapshot.model_copy(
                update={
                    "version": base_version + 1,
                    "work": finish_candidate(pending, candidate.candidate_id),
                    "last_maintenance": MaintenanceReceipt(
                        candidate_id=candidate.candidate_id, status=MaintenanceStatus.COMMITTED
                    ),
                }
            )
            self._commit(staged, compiled)
            self.status = "ready"
        except (BackendError, BudgetExceeded, StateError) as exc:
            logger.exception("chunk update failed; restoring committed factors")
            errors.append(str(exc))
            reason = str(exc)
            outcome = self._recover_failure(errors)
        except BaseException:
            self._recover_failure(errors)
            raise
        logger.info(
            "chunk status=%s version=%s->%s", outcome.value, base_version, self.snapshot.version
        )
        return MaintenanceResult(
            status=outcome,
            candidate_id=candidate.candidate_id,
            base_version=base_version,
            final_version=self.snapshot.version,
            reason=reason,
            errors=tuple(errors),
            cost=self._cost,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _add_cost(self, **values: int | float) -> None:
        """累加当前唯一维护账本，不在快照中保存完整成本历史。

        Args:
            **values: 本次操作新增的账本字段用量，单位由字段后缀指定。

        Returns:
            None；只更新当前维护成本，不向任务模式暴露完整历史。
        """
        self._cost = self._cost.model_copy(
            update={name: getattr(self._cost, name) + value for name, value in values.items()}
        )

    def _compile_chunk(self, spec: CompileSpec) -> CompiledMemory:
        """仅生成当前 chunk 的因子；已有因子原样复用，共享项统一组合一次。"""
        if spec.block_key in self.compiled.raw:
            raise StateError("同一个 chunk 不能重复编译提交")
        raw = dict(self.compiled.raw)
        started = time.perf_counter()
        self._add_cost(compile_calls=1, compile_input_tokens=len(spec.token_ids))
        try:
            with log_phase(logger, "compile", block_key=spec.block_key):
                raw[spec.block_key] = self.backend.compile(spec)
            self._record_factor_peak(tuple(raw.values()), None)
        finally:
            self._add_cost(compile_seconds=time.perf_counter() - started)
        combined = self._compose(tuple(raw.values()))
        return CompiledMemory(
            (*self.compiled.specs, spec),
            MappingProxyType(raw),
            combined,
            tuple(block.ref for block in raw.values()),
        )

    def _commit(self, staged: Snapshot, compiled: CompiledMemory) -> None:
        """只允许追加一个 chunk；绑定与真实前向成功后再原子发布状态。"""
        if staged.version != self.snapshot.version + 1 or staged.scope_id != self.snapshot.scope_id:
            raise StateError("待发布版本或作用域不一致")
        keys = tuple(spec.block_key for spec in compiled.specs)
        if compiled.specs[:-1] != self.compiled.specs or tuple(compiled.raw) != keys:
            raise StateError("新状态必须按原顺序保留全部已提交 chunk")
        if any(compiled.raw[key] is not raw for key, raw in self.compiled.raw.items()):
            raise StateError("已经提交的原始因子不能被替换")
        refs = tuple(compiled.raw[key].ref for key in keys)
        if refs != compiled.factor_refs or compiled.combined is None:
            raise StateError("FactorRef 或完整组合因子缺失")
        for spec, ref in zip(compiled.specs, refs, strict=True):
            if (
                ref.block_key != spec.block_key
                or ref.input_fingerprint != spec.input_fingerprint
                or ref.compiler_fingerprint != self.backend.compiler_fingerprint
            ):
                raise StateError("编译产物的输入/编译器指纹不一致")
        self.backend.validate_factors(
            compiled.combined,
            replace(self.backend.layout, width=(len(keys) + 1) * self.backend.layout.rank),
        )
        staged = staged.model_copy(update={"factor_refs": refs})
        self._record_factor_peak(tuple(compiled.raw.values()), compiled.combined)
        self._install_and_prefill(staged, compiled)
        self.snapshot, self.compiled = staged, compiled

    def _recover_failure(self, errors: list[str]) -> MaintenanceStatus:
        """恢复当前提交版；失败不推进游标，即使恢复成功也必须暂停历史交付。"""
        self.status = "paused"
        try:
            self._install_and_prefill(self.snapshot, self.compiled)
        except (BackendError, BudgetExceeded, StateError) as exc:
            logger.exception("committed state recovery failed")
            errors.append(f"恢复已提交状态失败：{exc}")
            try:
                self._reset()
            except BackendError as reset_error:
                errors.append(f"清除失败绑定失败：{reset_error}")
        return MaintenanceStatus.PAUSED

    def _record_factor_peak(
        self, blocks: tuple[RawFactors, ...], combined: CombinedFactors | None
    ) -> None:
        """统计当前与暂存产物共存的真实字节，失败路径也保留已知分配峰值。

        Args:
            blocks: 本次已成功生成或复用的原始内容因子。
            combined: 已成功生成的组合产物，尚未组合时为 None。

        Returns:
            None；只更新维护成本，不发布状态或持有新的因子引用。
        """
        unique_raw = {id(raw): raw.ref.size_bytes for raw in (*self.compiled.raw.values(), *blocks)}
        unique_combined = {
            id(item): item.size_bytes
            for item in (self.compiled.combined, combined)
            if item is not None
        }
        peak = sum(unique_raw.values()) + sum(unique_combined.values())
        self._cost = self._cost.model_copy(
            update={
                "peak_staged_factor_bytes": max(peak, self._cost.peak_staged_factor_bytes),
            }
        )

    def _compose(self, blocks: tuple[RawFactors, ...]) -> CombinedFactors | None:
        """记录一次统一组合的时间，空内容仍按后端契约返回 None。

        Args:
            blocks: 本次全部活跃原始因子。

        Returns:
            后端生成的完整组合产物或 None，同时记录耗时和已知暂存峰值。

        Raises:
            BackendError: 原生组合失败；已知成本仍记录。
        """
        started = time.perf_counter()
        self._add_cost(compose_calls=1)
        self._record_factor_peak(blocks, None)
        try:
            with log_phase(logger, "compose", blocks=len(blocks)):
                combined = self.backend.compose(blocks)
            self._record_factor_peak(blocks, combined)
            return combined
        finally:
            self._add_cost(compose_seconds=time.perf_counter() - started)

    def _reset(self) -> None:
        """记录模式切换成本；顶层调用从不保留外部 KV。

        Returns:
            None；清除模型绑定并累计模式切换时间。

        Raises:
            BackendError: 原生 reset 失败。
        """
        started = time.perf_counter()
        try:
            with log_phase(logger, "reset"):
                self.backend.reset()
        finally:
            self._add_cost(reset_seconds=time.perf_counter() - started)

    def _install_and_prefill(self, snapshot: Snapshot, compiled: CompiledMemory) -> None:
        """安装一份完整状态并以待恢复 W 做真实前向，记录失败操作的成本。

        Args:
            snapshot: 待安装的完整工作状态。
            compiled: 与该状态对应的完整因子。

        Returns:
            None；绑定与真实前向完成，尚不发布快照。

        Raises:
            BudgetExceeded: 工作输入超出任务模式额度。
            BackendError: 安装或前向失败；已发生成本仍计入。
        """
        token_ids = self._task_input(snapshot, "")
        started = time.perf_counter()
        try:
            with log_phase(logger, "install", state_version=snapshot.version):
                self.backend.install(compiled.combined)
        finally:
            self._add_cost(install_seconds=time.perf_counter() - started)
        started = time.perf_counter()
        self._add_cost(prefill_calls=1, prefill_input_tokens=len(token_ids))
        try:
            with log_phase(logger, "prefill", input_tokens=len(token_ids)):
                self.backend.prefill(token_ids)
        finally:
            self._add_cost(prefill_seconds=time.perf_counter() - started)

    def _task_input(self, snapshot: Snapshot, question: str) -> TokenIds:
        """只编码任务规则、尚未编译的工作观察和当前问题。

        Args:
            snapshot: 只向任务模式展示任务规则和未编译的工作上下文。
            question: 当前交付的问题。

        Returns:
            带任务模板的完整 IDs，不包含已经编译的历史原文或其它题答案。

        Raises:
            BudgetExceeded: 完整输入超过任务模式额度。
            BackendError: 模板编码失败。
        """
        payload = {
            "goal": snapshot.work.goal,
            "observations": [
                span.model_dump(
                    mode="json",
                    include={
                        "text",
                        "order",
                        "event_time",
                        "observed_at",
                    },
                )
                for span in snapshot.work.recent_spans
            ],
            "question": question,
        }
        messages = (
            ChatMessage(role="system", content=self.task_prompt),
            ChatMessage(role="user", content=stable_json(payload)),
        )
        ids = self.backend.encode_messages(messages)
        check_token_limit(ids, self.task_generation.max_input_tokens, "任务输入")
        return ids

    def answer(self, question: str) -> GenerationResult:
        """从当前已安装状态回答一题；不把预测写回 W、记录或其它题上下文。

        Args:
            question: 此时已交付的问题。

        Returns:
            原始 GenerationResult；函数不把预测写回长期记忆。

        Raises:
            StateError: 运行时正在维护、已暂停或仍有 pending。
            BudgetExceeded: 完整任务输入超限。
            BackendError: 前向或生成失败。
        """
        if (
            self.status not in {"ready", "frozen"}
            or self.snapshot.work.pending_candidate is not None
        ):
            raise StateError(f"状态 {self.status} 不能回答任务问题")
        ids = self._task_input(self.snapshot, question)
        started = time.perf_counter()
        with log_phase(logger, "prefill", input_tokens=len(ids)):
            self.backend.prefill(ids)
        prefill_seconds = time.perf_counter() - started
        with log_phase(logger, "task_generate"):
            result = self.backend.generate(ids, self.task_generation)
        return result.model_copy(
            update={"prefill_input_tokens": len(ids), "prefill_seconds": prefill_seconds}
        )

    def freeze(self) -> Snapshot:
        """历史全部编译后固定版本；尾段、失败候选或未完成事务均禁止冻结。"""
        if (
            self.status != "ready"
            or self.snapshot.work.pending_candidate is not None
            or self.snapshot.work.recent_spans
        ):
            raise StateError("历史尚未安全处理完成，不能冻结")
        self.status = "frozen"
        return self.snapshot
