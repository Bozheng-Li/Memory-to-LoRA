"""纯基座基线:同一份配置、同一批 100 题,但记忆为空(不维护、不装任何 LoRA)。

用途是给主实验提供对照分母;读取路径与主实验完全一致(runtime.answer + 同一任务提示)。
运行:在 GPU 分配内
    srun --jobid=<作业号> --overlap --ntasks=1 --cpus-per-task=8 \
        bash -c 'cd <项目根> && PYTHONPATH=Main-Code/src .venv-d2l/bin/python Main-Code/experiment/run_base_100q.py'
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from param_memory.config import load_config, validate_config, validate_resources
from param_memory.d2l import D2LBackend
from param_memory.evaluation import load_eval_case, load_official_scorer, score_prediction
from param_memory.runtime import MemoryRuntime
from param_memory.state import empty_snapshot
from param_memory.schema import BackendError, BudgetExceeded, EvaluationError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "Main-Code/configs/mab_sh6k.yaml"

cfg = load_config(CONFIG)
validate_config(cfg)
validate_resources(cfg)

run_id = f"base_empty-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
out_dir = cfg.output.directory / run_id
out_dir.mkdir(parents=True, exist_ok=False)
print(f"run_id={run_id}\nout={out_dir}", flush=True)

case = load_eval_case(cfg.data.manifest_path, cfg.data.selection)
scorer = load_official_scorer(cfg.data.official_scorer_path)
backend = D2LBackend(cfg.model)
runtime = MemoryRuntime(
    empty_snapshot(case.history_id, cfg.data.protocol_instruction),
    backend,
    cfg,
)
runtime.freeze()  # 空记忆冻结:纯基座,不安装任何参数

started = time.perf_counter()
rows, scores = [], []
for index, (question, answers, qa_id) in enumerate(
    zip(case.questions, case.answers, case.qa_ids, strict=True), start=1
):
    generation, score, error = None, None, None
    try:
        generation = runtime.answer(question)
        score = score_prediction(generation.text, answers, scorer)
        scores.append(score)
    except (BackendError, BudgetExceeded, EvaluationError) as exc:
        error = str(exc)
    rows.append(
        {
            "question_id": qa_id,
            "question": question,
            "answers": list(answers),
            "generation": None if generation is None else generation.model_dump(mode="json"),
            "score": score,
            "error": error,
        }
    )
    if index % 20 == 0:
        print(f"  {index}/100 scored={len(scores)} mean={sum(scores)/len(scores):.3f}", flush=True)

elapsed = time.perf_counter() - started
metrics = {
    "baseline": "empty_memory_pure_base",
    "total_questions": len(case.questions),
    "scored_questions": len(scores),
    "failed_questions": len(case.questions) - len(scores),
    "score_sum": sum(scores),
    "official_mean_scored": sum(scores) / len(scores) if scores else None,
    "failure_inclusive_score": sum(scores) / len(case.questions),
    "elapsed_seconds": elapsed,
    "task_input_tokens": sum(r["generation"]["input_tokens"] for r in rows if r["generation"]),
    "task_output_tokens": sum(r["generation"]["output_tokens"] for r in rows if r["generation"]),
}
(out_dir / "predictions.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
)
(out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
(out_dir / "final_snapshot.json").write_text(
    runtime.snapshot.model_dump_json(indent=2), encoding="utf-8"
)
print(
    f"\nsubstring-EM(分母固定)={metrics['failure_inclusive_score']:.4f} | "
    f"scored={metrics['scored_questions']}/{metrics['total_questions']} | "
    f"elapsed={elapsed:.0f}s | tokens_in={metrics['task_input_tokens']}",
    flush=True,
)
sys.exit(0)
