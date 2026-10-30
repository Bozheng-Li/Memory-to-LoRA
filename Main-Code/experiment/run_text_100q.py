"""全文上下文基线:同一批 100 题、同一任务提示,但把 455 条历史整段放进 prompt。

与 run_base_100q.py(空记忆)和主实验(参数记忆)构成三档对照,用来判断
"benchmark 本身是否太难",以及参数记忆到底损失了多少可读信息。
运行:在 GPU 分配内
    srun --jobid=<作业号> --overlap --ntasks=1 --cpus-per-task=8 \
        bash -c 'cd <项目根> && PYTHONPATH=Main-Code/src .venv-d2l/bin/python Main-Code/experiment/run_text_100q.py'
"""

import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from param_memory.config import load_config, validate_config, validate_resources
from param_memory.d2l import D2LBackend
from param_memory.evaluation import load_eval_case, load_official_scorer, score_prediction
from param_memory.schema import ChatMessage, stable_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "Main-Code/configs/mab_sh6k.yaml"

cfg = load_config(CONFIG)
validate_config(cfg)
validate_resources(cfg)

run_id = f"text_context-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
out_dir = cfg.output.directory / run_id
out_dir.mkdir(parents=True, exist_ok=False)
print(f"run_id={run_id}\nout={out_dir}", flush=True)

case = load_eval_case(cfg.data.manifest_path, cfg.data.selection)
scorer = load_official_scorer(cfg.data.official_scorer_path)

# 同一批题上的 EM / F1,复用官方实现,便于和 substring-EM 一起看格式影响。
spec = importlib.util.spec_from_file_location("official", str(cfg.data.official_scorer_path))
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)

task_prompt = cfg.prompts.task_path.read_text(encoding="utf-8")
backend = D2LBackend(cfg.model)

payload = {
    "goal": cfg.data.protocol_instruction,
    "context": case.context,
}
started = time.perf_counter()
rows, scores, ems, f1s = [], [], [], []
for index, (question, answers, qa_id) in enumerate(
    zip(case.questions, case.answers, case.qa_ids, strict=True), start=1
):
    user = stable_json({**payload, "question": question})
    ids = backend.encode_messages(
        (ChatMessage(role="system", content=task_prompt), ChatMessage(role="user", content=user))
    )
    generation = backend.generate(ids, cfg.generation.task)
    text = generation.text
    score = score_prediction(text, answers, scorer)
    scores.append(score)
    ems.append(float(official.drqa_metric_max_over_ground_truths(official.drqa_exact_match_score, text, answers)))
    f1s.append(
        float(
            official.drqa_metric_max_over_ground_truths(
                lambda x, y: official.f1_score(x, y)[0], text, answers
            )
        )
    )
    rows.append(
        {
            "question_id": qa_id,
            "question": question,
            "answers": list(answers),
            "generation": generation.model_dump(mode="json"),
            "score": score,
        }
    )
    if index % 20 == 0:
        print(f"  {index}/100 mean={sum(scores)/len(scores):.3f}", flush=True)

elapsed = time.perf_counter() - started
n = len(case.questions)
metrics = {
    "baseline": "full_context_text",
    "total_questions": n,
    "scored_questions": len(scores),
    "substring_em": sum(scores) / n,
    "drqa_em": sum(ems) / n,
    "token_f1": sum(f1s) / n,
    "elapsed_seconds": elapsed,
    "task_input_tokens": sum(r["generation"]["input_tokens"] for r in rows),
    "task_output_tokens": sum(r["generation"]["output_tokens"] for r in rows),
}
(out_dir / "predictions.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
)
(out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(
    f"\n全文上下文: substring-EM={metrics['substring_em']:.4f} drqa-EM={metrics['drqa_em']:.4f} "
    f"F1={metrics['token_f1']:.4f} | 题目 {len(scores)}/{n} | elapsed={elapsed:.0f}s | "
    f"tokens_in={metrics['task_input_tokens']}",
    flush=True,
)
