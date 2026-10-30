"""串行组织维护、优先失效、编译、安装、发布和安全恢复。

当前快照/因子只在完整安装和前向检查成功后一起切换。管理调用、来源补查
和格式/预算重提共用一次维护账本；完整记录只返回评测外壳。
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from types import MappingProxyType

from .budget import check_budget, check_token_limit, measure_snapshot, measure_work, prefix_end
from .config import RunConfig
from .d2l import CombinedFactors, CompiledMemory, D2LBackend, RawFactors, empty_compiled
from .events import append_span, build_candidate, finish_candidate
from .manager import AgentManager
from .planning import (
    parse_manager_reply,
    resolve_source_request,
    validate_header,
    validate_invalidations,
    validate_plan,
)
from .schema import (
    BackendError,
    BudgetExceeded,
    Candidate,
    ChatMessage,
    ConfigError,
    GenerationResult,
    Invalidation,
    LocalStatus,
    MaintenanceCost,
    MaintenanceReceipt,
    MaintenanceResult,
    MaintenanceStatus,
    PlanValidationError,
    Snapshot,
    SourceRequest,
    SourceSpan,
    StagedUpdate,
    StateError,
    TokenIds,
    Trigger,
    ValidatedPlan,
    ValidationFeedback,
    stable_json,
)
from .state import build_compile_specs, check_candidate, disable_invalidated, online_sources

logger = logging.getLogger(__name__)


class MemoryRuntime:
    """一个任务历史的状态机，任务/管理模式复用同一个后端实例。"""

    def __init__(
        self, initial: Snapshot, backend: D2LBackend, manager: AgentManager, config: RunConfig
    ) -> None:
        """装配空历史并检查容量；清除前一历史的绑定，不保存完整配置或题组。

        Args:
            initial: 此历史的新空快照。
            backend: 与 manager.backend 必须相同。
            manager: backend 与 manager.backend 必须相同。
            config: 在此装配后只保存需要的配置子对象。

        Returns:
            None；初始只读因子集合为空，管理调用账本和当前候选为空。

        Raises:
            StateError: 初始状态非空，或运行时与管理器未共享同一后端。
            ConfigError: 提示缺失或声明容量超过实际模型上限。
            BudgetExceeded: 初始状态或任务输入超出预算。
            BackendError: 初始 reset 失败。

        Notes:
            共同后端身份、两类上下文容量和初始空状态在同一装配边界校验。
        """
        if manager.backend is not backend:
            raise StateError("Runtime 和 AgentManager 必须共享同一个后端")
        if (
            initial.version != 0
            or initial.sources
            or initial.global_records
            or initial.factor_refs
            or any(initial.local_blocks)
            or initial.work.pending_candidate is not None
            or initial.work.recent_spans
        ):
            raise StateError("每份历史必须使用新的空快照")
        self.backend, self.manager = backend, manager
        self.limits, self.task_generation = config.memory, config.generation.task
        if len(initial.local_blocks) != self.limits.local_slots:
            raise StateError("初始固定槽位数与配置不一致")
        if (
            self.task_generation.max_input_tokens + self.task_generation.max_new_tokens
            > backend.context_capacity
        ):
            raise ConfigError("generation.task 的输入加输出超过基座实际上下文容量")
        if (
            max(self.limits.local_block_tokens, self.limits.global_block_tokens)
            > backend.compiler_context_capacity
        ):
            raise ConfigError("memory 编译输入上限超过上下文编码器实际容量")
        try:
            self.task_prompt = config.prompts.task_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigError(f"任务提示不可读：{config.prompts.task_path}") from exc
        if not self.task_prompt.strip():
            raise ConfigError("任务提示不能为空")
        self.snapshot, self.compiled, self.status = initial, empty_compiled(), "ready"
        self._cost = MaintenanceCost()
        self._unsafe_to_restore = False
        check_budget(measure_snapshot(initial, (), backend.encode_text), self.limits)
        self._task_input(initial, "")
        backend.reset()

    def observe(self, span: SourceSpan) -> tuple[MaintenanceResult, ...]:
        """接收当前已发生片段；窗口/候选满时先维护，返回全部实际维护结果。

        Args:
            span: 必须属于本作用域且顺序合法。

        Returns:
            本次实际发生的全部维护结果；未触发维护时返回空 tuple。

        Raises:
            StateError: 当前状态不能接收来源，或来源作用域、身份与顺序错误。
            BudgetExceeded: 片段及候选无法装入没有可先行维护内容的窗口。

        Notes:
            接收前腾出容量与接收后边界维护都在此串行处理，返回全部维护结果以免遗漏成本。
        """
        if self.status != "ready" or span.scope_id != self.snapshot.scope_id:
            raise StateError("当前状态不能接收事件，或片段不属于本作用域")
        existing = online_sources(self.snapshot).get(span.span_id)
        if existing is not None and existing != span:
            raise StateError("新片段与在线来源的身份冲突")
        results = []
        while True:
            try:
                work = append_span(self.snapshot.work, span, self.limits, self.backend.encode_text)
                probe = build_candidate(
                    work, Trigger.WINDOW_FULL, self.limits, self.backend.encode_context
                )
                # pending 的来源和编译副本也计入 W；在接收下一片前为它预留真实容量。
                proposed = work.model_copy(update={"pending_candidate": probe})
                if measure_work(proposed, self.backend.encode_text) > self.limits.working_tokens:
                    raise BudgetExceeded("新片段连同唯一候选将超过工作窗口")
                break
            except BudgetExceeded:
                old = build_candidate(
                    self.snapshot.work,
                    Trigger.WINDOW_FULL,
                    self.limits,
                    self.backend.encode_context,
                )
                if old is None:
                    if results:
                        self.status = "paused"
                        logger.error("新片段无法装入空工作窗口，已停止交付：%s", span.span_id)
                        return tuple(results)
                    raise BudgetExceeded("单个片段及其候选元数据不能装入 W，请修正配置") from None
                result = self.maintain(old)
                results.append(result)
                if result.status == MaintenanceStatus.PAUSED:
                    return tuple(results)
        self.snapshot = self.snapshot.model_copy(update={"work": work})
        while self.snapshot.work.recent_spans:
            candidate = build_candidate(
                self.snapshot.work, Trigger.WINDOW_FULL, self.limits, self.backend.encode_context
            )
            full = len(candidate.source_spans) < len(self.snapshot.work.recent_spans)
            full |= (
                len(self.backend.encode_context(candidate.compile_text))
                >= self.limits.local_block_tokens
            )
            if span.boundary is None and not full:
                break
            if span.boundary is not None and candidate.source_spans[-1] == span:
                candidate = candidate.model_copy(update={"trigger": span.boundary})
            result = self.maintain(candidate)
            results.append(result)
            if result.status == MaintenanceStatus.PAUSED:
                break
        return tuple(results)

    def maintain(self, candidate: Candidate) -> MaintenanceResult:
        """执行一个候选的有限维护事务；任何纠正未完成时保留 pending 并暂停。

        生成/解析/补查/失效/计划检查共用循环；安装或编译失败直接走恢复路径。
        未知异常清理后重抛，预期失败以明确 MaintenanceResult 返回。

        Args:
            candidate: 必须来自当前未处理前缀。

        Returns:
            committed/skipped/aborted/paused 维护结果及实际成本；未完成纠正不确认游标。
        """
        if self.status != "ready":
            raise StateError(f"状态 {self.status} 不允许启动维护")
        check_candidate(self.snapshot, candidate)
        pending = self.snapshot.work.model_copy(update={"pending_candidate": candidate})
        if measure_work(pending, self.backend.encode_text) > self.limits.working_tokens:
            raise BudgetExceeded("登记 pending_candidate 后工作窗口超限")
        self.snapshot = self.snapshot.model_copy(update={"work": pending})
        self.status = "maintaining"
        self._cost, self._unsafe_to_restore = MaintenanceCost(), False
        self._record_factor_peak((), None)
        started, base_version, calls = time.perf_counter(), self.snapshot.version, 0
        errors: list[str] = []
        replies: list[str] = []
        plan_id, feedback, excerpts = None, None, ()
        outcome, reason = MaintenanceStatus.ABORTED, "管理调用额度耗尽"
        try:
            self._reset()
            while calls < self.limits.max_manager_calls:
                remaining = self.limits.model_copy(
                    update={
                        "max_manager_calls": self.limits.max_manager_calls - calls,
                        "source_read_tokens": self.limits.source_read_tokens
                        - self._cost.source_read_tokens,
                    }
                )
                messages = self.manager.build_messages(
                    self.snapshot, candidate, remaining, excerpts, feedback
                )
                calls += 1
                generated = self.manager.generate_reply(messages)
                self._add_cost(
                    manager_input_tokens=generated.input_tokens,
                    manager_output_tokens=generated.output_tokens,
                    manager_seconds=generated.elapsed_seconds,
                )
                replies.append(generated.text)
                try:
                    reply = parse_manager_reply(generated.text)
                    validate_header(self.snapshot, candidate, reply)
                    plan_id = reply.plan_id
                    if isinstance(reply, SourceRequest):
                        read = resolve_source_request(
                            self.snapshot,
                            candidate,
                            reply,
                            remaining.source_read_tokens,
                            self.backend.encode_text,
                        )
                        self._add_cost(source_read_tokens=read.tokens_used)
                        excerpts += read.excerpts
                        feedback = None
                        continue
                    notices = validate_invalidations(self.snapshot, candidate, reply.invalidations)
                    if notices:
                        self._apply_invalidations(notices)
                        # 管理始终在纯基座上生成；安全提交刚安装了任务绑定，需要再次清除。
                        self._reset()
                        feedback = ValidationFeedback(
                            code="safe_state_published",
                            field="base_state_version",
                            message=(
                                f"失效已优先提交；请基于版本 {self.snapshot.version} "
                                "修复并重提计划。"
                            ),
                        )
                        continue
                    checked = validate_plan(
                        self.snapshot,
                        candidate,
                        reply,
                        self.limits,
                        self.backend.encode_text,
                        self.backend.encode_context,
                    )
                    reason = reply.reason
                    if reply.admission == "skip":
                        outcome = MaintenanceStatus.SKIPPED
                        break
                    compiled = self._compile_plan(checked)
                    staged = self._finished(
                        checked.update.snapshot, candidate, MaintenanceStatus.COMMITTED, plan_id
                    )
                    self._commit(staged, compiled)
                    outcome = MaintenanceStatus.COMMITTED
                    break
                except (PlanValidationError, BudgetExceeded) as exc:
                    # 预算重提前释放暂存因子，不能跨重提积累多个新状态的张量。
                    compiled = None
                    errors.append(str(exc))
                    feedback = self._feedback(exc)
            if outcome != MaintenanceStatus.COMMITTED:
                outcome = self._settle_without_commit(candidate, outcome, plan_id)
        except (BackendError, BudgetExceeded, StateError) as exc:
            errors.append(str(exc))
            reason = str(exc)
            outcome = self._recover_failure(candidate, plan_id, errors)
        except BaseException:
            # KeyboardInterrupt 也清理绑定；未知异常不被当作普通候选失败吞掉。
            self._recover_failure(candidate, plan_id, errors)
            self.status = "paused"
            raise
        self.status = "paused" if outcome == MaintenanceStatus.PAUSED else "ready"
        result = MaintenanceResult(
            status=outcome,
            candidate_id=candidate.candidate_id,
            plan_id=plan_id,
            base_version=base_version,
            final_version=self.snapshot.version,
            reason=reason,
            errors=tuple(errors),
            manager_calls=calls,
            cost=self._cost,
            elapsed_seconds=time.perf_counter() - started,
            replies=tuple(replies),
        )
        logger.info(
            "maintenance candidate=%s version=%s->%s status=%s calls=%s",
            candidate.candidate_id,
            base_version,
            self.snapshot.version,
            outcome.value,
            calls,
        )
        return result

    def _add_cost(self, **values: int | float) -> None:
        """累加当前唯一维护账本，不在快照中保存完整成本历史。

        Args:
            **values: 本次操作新增的账本字段用量，单位由字段后缀指定。

        Returns:
            None；只更新当前维护成本，不向 Agent 暴露完整历史。
        """
        self._cost = self._cost.model_copy(
            update={name: getattr(self._cost, name) + value for name, value in values.items()}
        )

    def _feedback(self, error: Exception) -> ValidationFeedback:
        """按管理输出 token 上限收紧反馈；完整错误仅由评测外壳保存。

        Args:
            error: 本次计划解析或预算错误。

        Returns:
            由真实 tokenizer 限长的修复反馈；完整错误只进入外壳日志。

        Raises:
            BudgetExceeded: 反馈无法装入管理输出额度。
        """
        text = str(error)
        end = prefix_end(text, self.manager.generation.max_new_tokens, self.backend.encode_text)
        return ValidationFeedback(
            code=type(error).__name__, field="manager_reply", message=text[:end]
        )

    def _needs_repair(self, candidate: Candidate) -> bool:
        """识别不能按普通失败丢弃的纠正和已停用内容。

        Args:
            candidate: 仍在维护的当前候选。

        Returns:
            是否存在纠正触发或已停用内容；为真时不能按普通跳过确认候选。
        """
        return (
            candidate.trigger == Trigger.CORRECTION
            or self.snapshot.global_status == LocalStatus.DISABLED
            or any(
                b is not None and b.status == LocalStatus.DISABLED
                for b in self.snapshot.local_blocks
            )
        )

    def _finished(
        self,
        snapshot: Snapshot,
        candidate: Candidate,
        status: MaintenanceStatus,
        plan_id: str | None,
    ) -> Snapshot:
        """仅为可终结候选计算新 W 和有界回执；此函数不发布状态。

        Args:
            snapshot: 当前或拟提交的完整状态。
            candidate: 可合法终结的 pending。
            status: 提交、跳过或普通放弃结果。
            plan_id: 最后计划身份，尚无合法计划时为 None。

        Returns:
            已推进游标并保留最后一份回执的新快照，不发布状态。

        Raises:
            StateError: 候选与工作窗口身份不符。
            BudgetExceeded: 终结后的 W 超限。
        """
        work = finish_candidate(
            snapshot.work, candidate.candidate_id, self.limits, self.backend.encode_text
        )
        return snapshot.model_copy(
            update={
                "work": work,
                "last_maintenance": MaintenanceReceipt(
                    candidate_id=candidate.candidate_id,
                    plan_id=plan_id,
                    status=status,
                ),
            }
        )

    def _settle_without_commit(
        self, candidate: Candidate, outcome: MaintenanceStatus, plan_id: str | None
    ) -> MaintenanceStatus:
        """skip/普通放弃也恢复任务绑定；未完成纠正只恢复最新安全子集并暂停。

        Args:
            candidate: 当前待终结候选。
            outcome: 未提交时的跳过或普通放弃结果。
            plan_id: 最后合法计划身份或 None。

        Returns:
            恢复任务绑定后的结果；未完成纠正返回 paused 并保留 pending。

        Raises:
            StateError: 最新安全状态无法恢复。
            BudgetExceeded: 候选终结后的完整状态超限。
        """
        # 管理调用清除了任务绑定，即使记录没变化，也必须完整重装并检查前向。
        self._restore(self.snapshot, self.compiled)
        if self._needs_repair(candidate):
            return MaintenanceStatus.PAUSED
        finished = self._finished(self.snapshot, candidate, outcome, plan_id)
        check_budget(
            measure_snapshot(finished, self.compiled.specs, self.backend.encode_text), self.limits
        )
        self.snapshot = finished
        return outcome

    def _recover_failure(
        self, candidate: Candidate, plan_id: str | None, errors: list[str]
    ) -> MaintenanceStatus:
        """只恢复最近安全前态；安全失效安装本身失败时禁止重装已知错误因子。

        Args:
            candidate: 失败事务的 pending。
            plan_id: 最后合法计划身份或 None。
            errors: 本次维护的错误列表；恢复错误会追加到其中。

        Returns:
            普通失败的 aborted，或不能安全继续时的 paused；不尝试更旧版本。
        """
        if self._unsafe_to_restore or self.status == "paused":
            try:
                self._reset()
            except BackendError as exc:
                errors.append(str(exc))
            self.status = "paused"
            return MaintenanceStatus.PAUSED
        try:
            return self._settle_without_commit(candidate, MaintenanceStatus.ABORTED, plan_id)
        except (BackendError, StateError, BudgetExceeded) as exc:
            errors.append(f"安全恢复失败：{exc}")
            self.status = "paused"
            return MaintenanceStatus.PAUSED

    def _apply_invalidations(self, notices: tuple[Invalidation, ...]) -> None:
        """先停用已知错误并提交安全子集；失败时不得恢复失效前态。

        Args:
            notices: 已由 validate_invalidations 检查。

        Returns:
            None；成功后安全前态与当前版本均已更新，后续普通计划必须基于新版本。

        Raises:
            StateError: 安全状态构造或安装失败；任务必须停止，不能恢复失效前态。
            BackendError: 安全子集的原生组合失败。
        """
        self._unsafe_to_restore = True
        safe = disable_invalidated(self.snapshot, notices)
        update = StagedUpdate(
            base_version=self.snapshot.version,
            snapshot=safe,
            changed_local_ids=(),
            global_changed=False,
        )
        specs = build_compile_specs(update, self.limits, self.backend.encode_context)
        raw = {spec.block_key: self.compiled.raw[spec.block_key] for spec in specs}
        combined = self._compose(tuple(raw.values()))
        compiled = CompiledMemory(
            specs, MappingProxyType(raw), combined, tuple(block.ref for block in raw.values())
        )
        try:
            self._commit(safe, compiled)
        except (BackendError, BudgetExceeded, StateError) as exc:
            raise StateError(f"失效安全状态无法安装，必须停止任务：{exc}") from exc
        self._unsafe_to_restore = False

    def _compile_plan(self, checked: ValidatedPlan) -> CompiledMemory:
        """复用输入及编译器指纹相同的局部块，全局视图变化则重编译所有全局块。

        Args:
            checked: 必须包含真实待提交状态和完整 CompileSpec，不能直接传未校验的 MemoryPlan。

        Returns:
            CompiledMemory，包含新状态全部活跃因子、完整组合产物和来源/输入映射。

        Raises:
            BackendError: 编译/组合失败抛 BackendError；分配单次暂存，不修改当前已提交因子，不保留无
                限缓存。
        """
        raw = {}
        for spec in checked.specs:
            previous = self.compiled.raw.get(spec.block_key)
            reuse = (
                previous is not None
                and previous.ref.input_fingerprint == spec.input_fingerprint
                and previous.ref.compiler_fingerprint == self.backend.compiler_fingerprint
                and not (spec.kind == "global" and checked.update.global_changed)
            )
            if reuse:
                raw[spec.block_key] = previous
                continue
            started = time.perf_counter()
            self._add_cost(compile_calls=1, compile_input_tokens=len(spec.token_ids))
            try:
                raw[spec.block_key] = self.backend.compile(spec)
                self._record_factor_peak(tuple(raw.values()), None)
            finally:
                self._add_cost(compile_seconds=time.perf_counter() - started)
        combined = self._compose(tuple(raw.values()))
        return CompiledMemory(
            checked.specs,
            MappingProxyType(raw),
            combined,
            tuple(block.ref for block in raw.values()),
        )

    def _commit(self, staged: Snapshot, compiled: CompiledMemory) -> None:
        """复核新版本、来源、完整因子和预算，安装及 prefill 成功后原子发布。

        Args:
            staged: 待发布状态。
            compiled: 与全部活跃编译输入匹配。

        Returns:
            None；发布后版本推进，当前状态与实际绑定一致，再释放无引用旧来源和因子。

        Raises:
            StateError: 版本、作用域、来源或编译产物绑定不一致。
            BudgetExceeded: 完整待发布状态超限。
            BackendError: 张量校验、安装或真实前向失败；由 maintain 选择安全恢复。

        Notes:
            全部检查与唯一发布点保持在同一函数；安装或前向失败时不提前切换状态指针。
        """
        if staged.version != self.snapshot.version + 1:
            raise StateError("待发布版本必须等于当前版本加一")
        if staged.scope_id != self.snapshot.scope_id:
            raise StateError("不能向当前 Runtime 提交另一作用域")
        update = StagedUpdate(
            base_version=self.snapshot.version,
            snapshot=staged,
            changed_local_ids=(),
            global_changed=False,
        )
        expected = build_compile_specs(update, self.limits, self.backend.encode_context)
        if compiled.specs != expected or set(compiled.raw) != {s.block_key for s in expected}:
            raise StateError("编译产物与待提交状态的全部活跃输入不一致")
        for spec in expected:
            raw = compiled.raw[spec.block_key]
            if (
                raw.ref.input_fingerprint != spec.input_fingerprint
                or raw.ref.compiler_fingerprint != self.backend.compiler_fingerprint
            ):
                raise StateError("编译产物的输入/编译器指纹不一致")
            self.backend.validate_factors(raw, self.backend.layout)
        refs = tuple(compiled.raw[spec.block_key].ref for spec in expected)
        if refs != compiled.factor_refs:
            raise StateError("FactorRef 与实际原始因子不一致")
        if bool(expected) != (compiled.combined is not None):
            raise StateError("空内容必须为纯基座，非空内容必须有完整组合因子")
        if compiled.combined is not None:
            self.backend.validate_factors(
                compiled.combined,
                replace(
                    self.backend.layout,
                    width=(len(expected) + 1) * self.backend.layout.rank,
                ),
            )
        staged = staged.model_copy(update={"factor_refs": refs})
        self._record_factor_peak(tuple(compiled.raw.values()), compiled.combined)
        check_budget(measure_snapshot(staged, expected, self.backend.encode_text), self.limits)
        self._install_and_prefill(staged, compiled)
        self.snapshot, self.compiled = staged, compiled

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
            self.backend.install(compiled.combined)
        finally:
            self._add_cost(install_seconds=time.perf_counter() - started)
        started = time.perf_counter()
        self._add_cost(prefill_calls=1, prefill_input_tokens=len(token_ids))
        try:
            self.backend.prefill(token_ids)
        finally:
            self._add_cost(prefill_seconds=time.perf_counter() - started)

    def _restore(self, snapshot: Snapshot, compiled: CompiledMemory) -> None:
        """完整恢复最新安全状态；不遍历更旧版本，不确认失败候选已被接纳。

        Args:
            snapshot: snapshot/compiled 必须是最近合法安全前态，二者对应。
            compiled: snapshot/compiled 必须是最近合法安全前态，二者对应。

        Returns:
            None 表示可恢复任务调用；不声称失败的候选已接纳。

        Raises:
            StateError: 重新安装和 prefill；恢复失败抛 StateError，停止任务，不逐个尝试更老版本。
        """
        if self._unsafe_to_restore:
            raise StateError("不能恢复已经确认含错误的旧版本")
        if snapshot != self.snapshot or compiled is not self.compiled:
            raise StateError("只能恢复当前最新安全前态，不能重新安装更早版本")
        try:
            self._install_and_prefill(snapshot, compiled)
        except (BackendError, BudgetExceeded) as exc:
            self.status = "paused"
            raise StateError(f"无法恢复安全状态：{exc}") from exc
        self.snapshot, self.compiled = snapshot, compiled

    def _task_input(self, snapshot: Snapshot, question: str) -> TokenIds:
        """只编码合法目标、工作观察、摘要和当前问题，不暴露管理事实表或卡片。

        Args:
            snapshot: 只向任务模式展示其中的目标、W 和摘要。
            question: 当前交付的问题。

        Returns:
            带任务模板的完整 IDs，不包含 F、局部卡片、管理日志或其它题答案。

        Raises:
            BudgetExceeded: 完整输入超过任务模式额度。
            BackendError: 模板编码失败。
        """
        payload = {
            "goal": snapshot.work.goal,
            "summary": snapshot.work.summary,
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
        self.backend.prefill(ids)
        prefill_seconds = time.perf_counter() - started
        result = self.backend.generate(ids, self.task_generation)
        return result.model_copy(
            update={"prefill_input_tokens": len(ids), "prefill_seconds": prefill_seconds}
        )

    def freeze(self) -> Snapshot:
        """结束历史写入并固定 W/版本；未处理来源、候选、事务或纠正均拒绝冻结。

        Returns:
            不可变 Snapshot，用于逐题版本核对和运行产物记录。

        Raises:
            StateError: 将 Runtime 置为只读，之后 observe/maintain 被拒绝；状态未完成时抛 StateError
                。
        """
        if (
            self.status != "ready"
            or self.snapshot.work.pending_candidate is not None
            or self.snapshot.work.recent_spans
            or self.snapshot.global_status == LocalStatus.DISABLED
            or any(
                b is not None and b.status == LocalStatus.DISABLED
                for b in self.snapshot.local_blocks
            )
        ):
            raise StateError("历史尚未安全处理完成，不能冻结")
        self.status = "frozen"
        return self.snapshot
