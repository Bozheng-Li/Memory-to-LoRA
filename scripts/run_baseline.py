#!/usr/bin/env python
"""全量 MAB 基线:空记忆(纯基座)或全文上下文,逐题评分并增量落盘。

用法(在 GPU 分配内):
    PYTHONPATH=Main-Code/src .venv-d2l/bin/python scripts/run_baseline.py \
        --mode base --out result/base_empty
    PYTHONPATH=Main-Code/src .venv-d2l/bin/python scripts/run_baseline.py \
        --mode text --out result/text_context --max-context-tokens 200000 [--resume]

两种模式共用同一任务提示、同一评分器与同一落盘格式,只有是否给 context 不同。
超过 --max-context-tokens 的上下文按"头一半 + 尾一半"截断(插标记并记录),不跳过任何题。
--resume 跳过 predictions.jsonl 里已成功的题,只补失败与未跑的;结束指标始终按文件重算。
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mab_baseline import (  # noqa: E402
    add_prediction,
    build_user_payload,
    finalize,
    load_rows,
    new_metrics,
    write_json,
)

from param_memory.config import load_config, validate_config, validate_resources  # noqa: E402
from param_memory.d2l import D2LBackend  # noqa: E402
from param_memory.schema import ChatMessage, stable_json  # noqa: E402

# 模板与问题本身额外占几百 token,给输入额度留余量,避免截断后仍被拦。
INPUT_HEADROOM = 16384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAB 全量基线")
    parser.add_argument("--mode", choices=("base", "text"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="结果目录")
    parser.add_argument("--config", type=Path, default=ROOT / "Main-Code/configs/mab_sh6k.yaml")
    parser.add_argument("--max-context-tokens", type=int, default=200_000)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题,用于冒烟")
    parser.add_argument("--resume", action="store_true", help="跳过已有成功题,只补失败/未跑")
    return parser.parse_args()


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("official_metric", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_metrics(pred_path: Path) -> tuple[dict, dict]:
    """按文件重算指标:同题取成功行,全量与未截断子集各一份。"""
    best: dict[tuple[str, str], dict] = {}
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["source"], row["question_id"])
        if key not in best or (best[key]["error"] and not row["error"]):
            best[key] = row
    full, untruncated = new_metrics(), new_metrics()
    for row in best.values():
        add_prediction(full, row)
        if not row.get("truncated"):
            add_prediction(untruncated, row)
    return full, untruncated


def main() -> int:
    args = parse_args()
    if args.out.exists() and not args.resume:
        raise SystemExit(f"结果目录已存在,拒绝覆盖(加 --resume 可续跑):{args.out}")

    cfg = load_config(args.config)
    validate_config(cfg)
    validate_resources(cfg)
    official = load_official(cfg.data.official_scorer_path)
    task_prompt = cfg.prompts.task_path.read_text(encoding="utf-8")

    rows = load_rows(ROOT / "resources/data/mab", tuple(args.splits) if args.splits else None)
    backend = D2LBackend(cfg.model)
    gen_cfg = (
        cfg.generation.task.model_copy(
            update={"max_input_tokens": args.max_context_tokens + INPUT_HEADROOM}
        )
        if args.mode == "text"
        else cfg.generation.task
    )

    def fit_context(text: str, cap: int) -> tuple[str, dict]:
        """超出上限时保留头尾各一半并插入截断标记;返回文本与统计。"""
        ids = backend.tokenizer.encode(text, add_special_tokens=False)
        total = len(ids)
        if total <= cap:
            return text, {"context_tokens": total, "kept_tokens": total, "truncated": False}
        marker = "\n...[truncated]...\n"
        marker_tokens = len(backend.tokenizer.encode(marker, add_special_tokens=False))
        budget = max(cap - marker_tokens, 2)
        head = budget // 2
        tail = budget - head
        kept = backend.tokenizer.decode(ids[:head]) + marker + backend.tokenizer.decode(ids[-tail:])
        return kept, {"context_tokens": total, "kept_tokens": head + tail, "truncated": True}

    args.out.mkdir(parents=True, exist_ok=True)
    pred_path = args.out / "predictions.jsonl"
    done_keys: set[tuple[str, str]] = set()
    truncated: list[dict] = []
    if args.resume and pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row["error"]:
                    done_keys.add((row["source"], row["question_id"]))
        print(f"续跑:已有 {len(done_keys)} 道成功题,将跳过", flush=True)

    started_total = time.perf_counter()
    done = 0
    metrics = new_metrics()
    with pred_path.open("a" if args.resume else "w", encoding="utf-8") as stream:
        for row in rows:
            context_text = None
            row_ctx_tokens, row_truncated = None, False
            if args.mode == "text":
                context_text, meta = fit_context(row.context, args.max_context_tokens)
                row_ctx_tokens, row_truncated = meta["context_tokens"], meta["truncated"]
                if meta["truncated"]:
                    truncated.append({"split": row.split, "source": row.source,
                                      "questions": len(row.questions), **meta})
                    print(f"truncate {row.source}: {meta['context_tokens']} -> "
                          f"{meta['kept_tokens']} tok", flush=True)
            for question, answers, qa_id in zip(row.questions, row.answers, row.qa_ids, strict=True):
                if (row.source, qa_id) in done_keys:
                    continue
                if args.limit and done >= args.limit:
                    break
                payload = build_user_payload(cfg.data.protocol_instruction, question, context_text)
                ids = backend.encode_messages(
                    (ChatMessage(role="system", content=task_prompt),
                     ChatMessage(role="user", content=stable_json(payload)))
                )
                t0 = time.perf_counter()
                error = None
                try:
                    generation = backend.generate(ids, gen_cfg)
                    text = generation.text
                except Exception as exc:  # 单题失败不影响整轮,记录后继续
                    generation, text, error = None, "", f"{type(exc).__name__}: {exc}"
                seconds = time.perf_counter() - t0
                if error is None:
                    sub = float(official.drqa_metric_max_over_ground_truths(
                        official.substring_exact_match_score, text, answers))
                    em = float(official.drqa_metric_max_over_ground_truths(
                        official.drqa_exact_match_score, text, answers))
                    f1 = float(official.drqa_metric_max_over_ground_truths(
                        lambda x, y: official.f1_score(x, y)[0], text, answers))
                else:
                    sub = em = f1 = 0.0
                pred = {
                    "split": row.split,
                    "source": row.source,
                    "question_id": qa_id,
                    "context_tokens": row_ctx_tokens,
                    "truncated": row_truncated,
                    "question": question,
                    "answers": list(answers),
                    "prediction": text,
                    "substring_em": sub,
                    "drqa_em": em,
                    "f1": f1,
                    "input_tokens": 0 if generation is None else generation.input_tokens,
                    "output_tokens": 0 if generation is None else generation.output_tokens,
                    "seconds": seconds,
                    "error": error,
                }
                stream.write(json.dumps(pred, ensure_ascii=False) + "\n")
                stream.flush()
                add_prediction(metrics, pred)
                done += 1
                if done % 100 == 0:
                    print(f"  +{done} 题 | 本批 substring-EM="
                          f"{metrics['substring_em_sum'] / max(metrics['scored'], 1):.3f} "
                          f"| {metrics['seconds']:.0f}s", flush=True)
            if args.limit and done >= args.limit:
                break

    full, untruncated = compute_metrics(pred_path)
    result = finalize(full)
    result.update(
        {
            "mode": args.mode,
            "max_context_tokens": args.max_context_tokens if args.mode == "text" else None,
            "wall_seconds": time.perf_counter() - started_total,
            "truncated_rows": truncated,
            "truncated_questions": sum(t["questions"] for t in truncated),
            "config": str(args.config),
            "protocol_instruction": cfg.data.protocol_instruction,
            "source_of_truth": "本文件为全量;metrics_untruncated.json 为未截断子集",
        }
    )
    write_json(args.out / "metrics.json", result)
    if args.mode == "text":
        sub = finalize(untruncated)
        sub.update({"subset": "context_not_truncated",
                    "max_context_tokens": args.max_context_tokens,
                    "source_of_truth": "metrics.json 为全量"})
        write_json(args.out / "metrics_untruncated.json", sub)

    print(json.dumps({k: result[k] for k in
                      ("questions", "scored", "failed", "substring_em_failure_inclusive",
                       "drqa_em_failure_inclusive", "token_f1_failure_inclusive",
                       "wall_seconds", "truncated_questions")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
