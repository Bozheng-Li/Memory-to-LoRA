"""隔离 MAB 历史、问题和 gold，调用固定官方评分并写出真实运行产物。

外壳持有完整数据行；Runtime 只逐片收到已发生历史，冻结后才逐题收到问题。
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import math
import random
import re
import resource
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from . import __version__
from .budget import measure_snapshot
from .config import RunConfig, fingerprint_file
from .d2l import D2LBackend, empty_compiled
from .events import build_candidate, iter_spans
from .manager import AgentManager
from .runtime import MemoryRuntime
from .schema import (
    BackendError,
    BudgetExceeded,
    ConfigError,
    EvalCase,
    EvaluationError,
    Event,
    FrozenModel,
    MaintenanceResult,
    MaintenanceStatus,
    Nonempty,
    Positive,
    Prediction,
    RunReport,
    ScoreFn,
    StateError,
    Trigger,
    stable_json,
)
from .state import empty_snapshot

logger = logging.getLogger(__name__)


class _Selection(FrozenModel):
    """固定 manifest 中一个真实选择项的严格字段。"""

    split: Nonempty
    parquet_file: Nonempty
    source_filter: Nonempty
    row_indices_at_pinned_revision: tuple[int, ...]
    num_histories: Positive
    num_questions: Positive


def load_eval_case(manifest_path: Path, selection: str) -> EvalCase:
    """按 metadata.source 选择历史并校验固定版本的行号和题组对齐。

    首版入口一次选择一份历史；不在多个匹配结果中暗中取第一行。
    metadata 的其它字段可能含答案，只在此外壳读取，不转交 Runtime。

    Args:
        manifest_path: 来自 data.manifest_path。
        selection: 来自 data.selection，必须是 manifest 中真实存在的选择项。

    Returns:
        EvalCase；包含答案的对象只由评测外壳持有，不传给 Runtime 或 AgentManager。

    Raises:
        EvaluationError: 读取本地 JSON/Parquet；版本、选择或结构错误抛 EvaluationError，不按猜测行号
            替代筛选。

    Notes:
        筛选、固定行号核对与题组校验留在数据边界，防止把含 gold 的完整行交给 writer。
    """
    try:
        import pyarrow.parquet as parquet

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if selection not in manifest or manifest["metric"] != "substring_exact_match":
            raise EvaluationError("manifest 选择项不存在或不是支持的官方指标")
        for field in ("dataset_revision", "official_code_revision"):
            if not isinstance(manifest[field], str) or not re.fullmatch(
                r"[0-9a-f]{40}", manifest[field]
            ):
                raise EvaluationError(f"manifest.{field} 必须固定为完整 revision")
        chosen = _Selection.model_validate_json(stable_json(manifest[selection]), strict=True)
        if chosen.num_histories != 1:
            raise EvaluationError("首版单历史入口要求选择项 num_histories=1")
        parquet_path = (manifest_path.parent / chosen.parquet_file).resolve()
        selected = []
        row_index = 0
        for batch in parquet.ParquetFile(parquet_path).iter_batches(
            columns=["context", "questions", "answers", "metadata"],
        ):
            for row in batch.to_pylist():
                if row["metadata"]["source"] == chosen.source_filter:
                    selected.append((row_index, row))
                row_index += 1
        if len(selected) != chosen.num_histories:
            raise EvaluationError(f"source_filter 匹配 {len(selected)} 份历史，与 manifest 不一致")
        if tuple(index for index, _ in selected) != chosen.row_indices_at_pinned_revision:
            raise EvaluationError("按 source 筛选的行号与固定版本校验记录不一致")
        row = selected[0][1]
        case = EvalCase.model_validate_json(
            stable_json(
                {
                    "history_id": chosen.source_filter,
                    "source": chosen.source_filter,
                    "context": row["context"],
                    "questions": row["questions"],
                    "answers": row["answers"],
                    "qa_ids": row["metadata"]["qa_pair_ids"],
                    "dataset_revision": manifest["dataset_revision"],
                    "official_code_revision": manifest["official_code_revision"],
                    "parquet_path": str(parquet_path),
                }
            ),
            strict=True,
        )
        lengths = {len(case.questions), len(case.answers), len(case.qa_ids), chosen.num_questions}
        if len(lengths) != 1 or len(set(case.qa_ids)) != len(case.qa_ids):
            raise EvaluationError("questions/answers/QA ID/manifest 题数不一致或 QA ID 重复")
        if any(not q.strip() for q in case.questions) or any(
            not answers or any(not answer.strip() for answer in answers) for answers in case.answers
        ):
            raise EvaluationError("题目和可接受答案集合不能为空")
        return case
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationError(f"MAB 数据入口失败：{exc}") from exc


def iter_history_events(case: EvalCase) -> Iterator[Event]:
    """按 FactConsolidation 原编号顺序生成事件，保留原文和未知时间。

    Args:
        case: 来自 load_eval_case。

    Returns:
        Event 迭代器；事件时间未知时不伪造时间，编号更新规则来自 YAML 中的共同任务指令。

    Raises:
        EvaluationError: 格式与选定子任务协议不符抛 EvaluationError；不根据测试问题删选事实。
    """
    if not case.source.startswith("factconsolidation_"):
        raise EvaluationError(f"当前事件协议不支持来源 {case.source}")
    matches = iter(re.finditer(r"(?m)^(\d+)\.\s+", case.context))
    previous = next(matches, None)
    if previous is None:
        raise EvaluationError("FactConsolidation context 没有编号事实")
    start = 0
    for match in matches:
        order = int(previous.group(1))
        if int(match.group(1)) <= order:
            raise EvaluationError("历史事实编号必须严格递增")
        yield Event(
            event_id=f"e:{order}",
            scope_id=case.history_id,
            text=case.context[start : match.start()],
            order=order,
            event_time=None,
            observed_at=None,
            boundary=None,
        )
        previous, start = match, match.start()
    yield Event(
        event_id=f"e:{previous.group(1)}",
        scope_id=case.history_id,
        text=case.context[start:],
        order=int(previous.group(1)),
        event_time=None,
        observed_at=None,
        boundary=None,
    )


def load_official_scorer(source_path: Path) -> ScoreFn:
    """载入固定本地官方 substring_exact_match 及其多答案最大匹配函数。

    Args:
        source_path: 来自 data.official_scorer_path，需记录实际文件指纹与 manifest 指定版本。

    Returns:
        ScoreFn；只衔接原生函数参数，不重写归一化或匹配算法。

    Raises:
        EvaluationError: 加载评分模块及其必需依赖；缺失或接口不匹配抛 EvaluationError，不使用旧
            pilot 的近似实现。
    """
    try:
        spec = importlib.util.spec_from_file_location(
            f"param_memory_official_{fingerprint_file(source_path)}",
            source_path,
        )
        if spec is None or spec.loader is None:
            raise EvaluationError("不能从指定路径加载官方评分模块")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metric = module.substring_exact_match_score
        maximum = module.drqa_metric_max_over_ground_truths
        inspect.signature(metric).bind("prediction", "answer")
        inspect.signature(maximum).bind(metric, "prediction", ("answer",))
    except (OSError, ImportError, AttributeError, ValueError, TypeError) as exc:
        raise EvaluationError(f"官方评分代码或依赖不可用：{exc}") from exc

    def score(prediction: str, answers: tuple[str, ...]) -> float:
        """只适配原生调用参数，不另写文本清洗或匹配规则。

        Args:
            prediction: 未经自行清洗的原始回答。
            answers: 本题的可接受答案集合。

        Returns:
            官方多答案最大匹配分数，不重写原生归一化。
        """
        return float(maximum(metric, prediction, answers))

    return score


def score_prediction(prediction: str, answers: tuple[str, ...], scorer: ScoreFn) -> float:
    """对原始回答调用官方分数；空答案或评分失败抛 EvaluationError。

    Args:
        prediction: 未经自行清洗的原始回答。
        answers: 本题可接受答案。
        scorer: 已加载的官方调用。

    Returns:
        原生分数；额外的旧值误用等诊断另列，不改写该分数。

    Raises:
        EvaluationError: 空答案集合或评分错误抛 EvaluationError；不调用 Agent，不回写记忆。
    """
    if not answers or any(not answer.strip() for answer in answers):
        raise EvaluationError("官方评分需要非空可接受答案集合")
    try:
        result = scorer(prediction, answers)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise EvaluationError(f"官方评分失败：{exc}") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise EvaluationError(f"官方评分返回非法分数：{result}")
    return result


def _consume_history(
    case: EvalCase, runtime: MemoryRuntime, records: list[MaintenanceResult]
) -> None:
    """逐片交付历史，循环收尾有限剩余前缀；暂停后停止交付新事件。

    Args:
        case: 外壳持有的数据；只取 context 与合法来源身份。
        runtime: 当前历史的运行时。
        records: 外壳维护列表，按发生顺序追加结果。

    Returns:
        None；完整消费历史后冻结，或在暂停时停止交付。

    Raises:
        StateError: 来源顺序或冻结条件不成立。
        BudgetExceeded: 单个片段无法放入声明的容量。
    """
    for event in iter_history_events(case):
        for span in iter_spans(
            event, runtime.limits.local_block_tokens, runtime.backend.encode_context
        ):
            records.extend(runtime.observe(span))
            if runtime.status == "paused":
                return
    while runtime.status == "ready":
        candidate = build_candidate(
            runtime.snapshot.work,
            Trigger.HISTORY_END,
            runtime.limits,
            runtime.backend.encode_context,
        )
        if candidate is None:
            runtime.freeze()
            return
        records.append(runtime.maintain(candidate))


def _answer_questions(
    case: EvalCase, runtime: MemoryRuntime, scorer: ScoreFn
) -> tuple[Prediction, ...]:
    """逐题读取同一冻结快照，失败题保留 None 分数，不把任何回答回写历史。

    Args:
        case: 与冻结历史对应的题组。
        runtime: 已冻结的运行时。
        scorer: 固定官方评分函数。

    Returns:
        逐题 Prediction；单题失败保留 None 分数及原始错误。

    Raises:
        StateError: 调用期间快照或冻结状态发生变化。
    """
    frozen = runtime.snapshot
    predictions = []
    for question, answers, qa_id in zip(case.questions, case.answers, case.qa_ids, strict=True):
        if runtime.status != "frozen" or runtime.snapshot != frozen:
            raise StateError("所有问题必须读取完全相同的冻结版本与 W")
        generation, score, error = None, None, None
        try:
            generation = runtime.answer(question)
            score = score_prediction(generation.text, answers, scorer)
        except (BackendError, BudgetExceeded, EvaluationError) as exc:
            error = str(exc)
        predictions.append(
            Prediction(
                history_id=case.history_id,
                question_id=qa_id,
                snapshot_version=frozen.version,
                question=question,
                answers=answers,
                generation=generation,
                score=score,
                error=error,
            )
        )
    return tuple(predictions)


def _run_metrics(
    case: EvalCase,
    records: tuple[MaintenanceResult, ...],
    predictions: tuple[Prediction, ...],
    elapsed_seconds: float,
) -> dict[str, object]:
    """以全部题目为运行分母，单列已评分均值、未尝试和失败数。

    Args:
        case: 提供完整题目分母。
        records: 本次全部维护结果。
        predictions: 实际尝试过的题目结果。
        elapsed_seconds: 完整运行的已耗秒数。

    Returns:
        完整题目分母、成功/失败/未尝试数量、已知成本及进程内存的指标字典。

    Notes:
        指标在一处使用相同题目分母；构造字典较长，避免分散维护重复的成功/失败口径。
    """
    scores = [prediction.score for prediction in predictions if prediction.score is not None]
    score_sum = sum(scores)
    costs = (
        {
            field: sum(getattr(record.cost, field) for record in records)
            for field in type(records[0].cost).model_fields
        }
        if records
        else {}
    )
    if records:
        costs["peak_staged_factor_bytes"] = max(r.cost.peak_staged_factor_bytes for r in records)
    return {
        "histories": 1,
        "total_questions": len(case.questions),
        "scored_questions": len(scores),
        "failed_questions": len(case.questions) - len(scores),
        "not_attempted_questions": len(case.questions) - len(predictions),
        "score_sum": score_sum,
        "official_mean_scored": score_sum / len(scores) if scores else None,
        "failure_inclusive_score": score_sum / len(case.questions),
        "failure_inclusive_denominator": len(case.questions),
        "maintenance_counts": {
            status.value: sum(r.status == status for r in records) for status in MaintenanceStatus
        },
        "manager_calls": sum(r.manager_calls for r in records),
        "manager_calls_without_returned_usage": sum(
            r.manager_calls - len(r.replies) for r in records
        ),
        "maintenance_cost_observed": costs,
        "task_input_tokens": sum(p.generation.input_tokens for p in predictions if p.generation),
        "task_output_tokens": sum(p.generation.output_tokens for p in predictions if p.generation),
        "task_generation_seconds": sum(
            p.generation.elapsed_seconds for p in predictions if p.generation
        ),
        "task_calls_without_returned_usage": sum(p.generation is None for p in predictions),
        "task_prefill_input_tokens": sum(
            p.generation.prefill_input_tokens for p in predictions if p.generation
        ),
        "task_prefill_seconds": sum(
            p.generation.prefill_seconds for p in predictions if p.generation
        ),
        "elapsed_seconds": elapsed_seconds,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def run_mab(config: RunConfig) -> RunReport:
    """设种子、装配、流式维护、冻结和官方评分，保留可报告失败及完整分母。

    Args:
        config: 已通过配置和资源校验。

    Returns:
        RunReport，含已发生的完整维护结果、逐题预测、最终状态和失败数量。

    Notes:
        装配、历史、冻结和评测在同一外壳串行执行，以固定 writer/reader/gold 的交付顺序。
    """
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = re.sub(r"[^\w.-]", "_", config.experiment.name)
    run_id = f"{run_name}-{timestamp}-{uuid4().hex}"
    case = load_eval_case(config.data.manifest_path, config.data.selection)
    scorer = load_official_scorer(config.data.official_scorer_path)
    initial = empty_snapshot(
        case.history_id, config.memory.local_slots, config.data.protocol_instruction
    )
    backend, runtime = None, None
    records: list[MaintenanceResult] = []
    predictions: tuple[Prediction, ...] = ()
    status, failure = "failed", None
    try:
        import numpy as np
        import torch

        random.seed(config.experiment.seed)
        # 各库 RNG 的整数宽度不同；按接口宽度映射，接受 YAML 的任意非负种子。
        np.random.seed(config.experiment.seed % (np.iinfo(np.uint32).max + 1))
        torch_seed = config.experiment.seed % (np.iinfo(np.uint64).max + 1)
        torch.manual_seed(torch_seed)
        if config.model.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
            torch.cuda.reset_peak_memory_stats(config.model.device)
        backend = D2LBackend(config.model)
        manager = AgentManager(backend, config.prompts.manager_path, config.generation.manager)
        runtime = MemoryRuntime(initial, backend, manager, config)
        _consume_history(case, runtime, records)
        if runtime.status == "paused":
            status, failure = "paused", "历史维护暂停，未向 reader 交付测试问题"
        else:
            predictions = _answer_questions(case, runtime, scorer)
            status = "completed" if all(p.score is not None for p in predictions) else "partial"
    except (BackendError, BudgetExceeded, ConfigError, StateError, EvaluationError) as exc:
        failure = str(exc)
        status = "paused" if runtime is not None and runtime.status == "paused" else "failed"
        logger.error("MAB execution status=%s: %s", status, failure)
    snapshot = runtime.snapshot if runtime is not None else initial
    compiled = runtime.compiled if runtime is not None else empty_compiled()
    metrics = _run_metrics(case, tuple(records), predictions, time.perf_counter() - started)
    metrics["memory_usage"] = (
        measure_snapshot(snapshot, compiled.specs, backend.encode_text).model_dump(mode="json")
        if backend is not None
        else None
    )
    metrics["steady_factor_bytes"] = compiled.size_bytes
    metrics["allocated_width"] = (
        (len(compiled.raw) + 1) * backend.layout.rank if backend is not None and compiled.raw else 0
    )
    metrics["resident_tensor_bytes"] = backend.resident_bytes if backend is not None else None
    metrics["gpu_peak_allocated_bytes"] = (
        backend._torch.cuda.max_memory_allocated(backend.device)
        if backend is not None and backend.device.type == "cuda"
        else None
    )
    source_root = Path(__file__).parent
    manifest = {
        "package_version": __version__,
        "python": sys.version,
        "status": status,
        "failure": failure,
        "task_readable": runtime is not None and runtime.status == "frozen",
        "dataset_revision_declared": case.dataset_revision,
        "official_code_revision_declared": case.official_code_revision,
        "data_manifest_sha256": fingerprint_file(config.data.manifest_path),
        "parquet_sha256": fingerprint_file(Path(case.parquet_path)),
        "official_scorer_sha256": fingerprint_file(config.data.official_scorer_path),
        "code_sha256": {p.name: fingerprint_file(p) for p in sorted(source_root.glob("*.py"))},
        "prompt_sha256": {
            "manager": fingerprint_file(config.prompts.manager_path),
            "task": fingerprint_file(config.prompts.task_path),
        },
        "model": backend.resource_fingerprints if backend is not None else None,
        "compiler_fingerprint": backend.compiler_fingerprint if backend is not None else None,
        "protocol": "context-only writer; frozen snapshot per question; gold to scorer only",
        "empty_memory_path": "pure_base",
        "compiled_inputs": [s.model_dump(mode="json") for s in compiled.specs],
    }
    return RunReport(
        run_id, config, manifest, tuple(records), predictions, metrics, snapshot, compiled, status
    )


def write_run_report(report: RunReport, output_dir: Path) -> None:
    """在独立子目录写全部复现产物，拒绝覆盖；任何写入错误向上传递。

    原始及完整组合因子使用 safetensors；CPU 副本只在写产物时创建，不进入在线状态。
    manifest.artifacts_complete 只有所有必要文件写完才变为 true。

    Args:
        report: 只包含本次真实执行结果。
        output_dir: 来自 output.directory。

    Returns:
        None 表示所需产物写完；运行本身是否成功以 report.status 为准。

    Notes:
        按文件完整性协议依次落盘，只有全部文件成功才标记 artifacts_complete。
    """
    from safetensors.torch import save_file

    if Path(report.run_id).name != report.run_id or report.run_id in {".", ".."}:
        raise EvaluationError("run_id 必须是单个安全目录名")
    directory = output_dir / report.run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {**report.manifest, "artifacts_complete": False}
    (directory / "manifest.json").write_text(stable_json(manifest) + "\n", encoding="utf-8")
    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(report.config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    for filename, rows in (
        ("maintenance.jsonl", report.maintenance),
        ("predictions.jsonl", report.predictions),
    ):
        with (directory / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(row.model_dump_json() + "\n")
    (directory / "final_snapshot.json").write_text(
        report.final_snapshot.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    tensors = {}
    for key, raw in report.compiled.raw.items():
        for module, values in raw.tensors.items():
            for matrix, tensor in values.items():
                tensors[f"raw.{key}.{module}.{matrix}"] = tensor.detach().cpu().contiguous().clone()
    if report.compiled.combined is not None:
        for module, values in report.compiled.combined.tensors.items():
            for matrix, tensor in values.items():
                tensors[f"combined.{module}.{matrix}"] = tensor.detach().cpu().contiguous().clone()
    save_file(
        tensors,
        str(directory / "memory.safetensors"),
        metadata={"run_id": report.run_id, "status": report.status},
    )
    (directory / "metrics.json").write_text(stable_json(report.metrics) + "\n", encoding="utf-8")
    manifest["artifacts_complete"] = True
    (directory / "manifest.json").write_text(stable_json(manifest) + "\n", encoding="utf-8")
