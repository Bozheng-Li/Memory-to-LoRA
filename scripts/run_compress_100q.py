#!/usr/bin/env python
"""文本压缩基线:对齐 DSH/Codex 的 checkpoint 压缩机制。

机制要点(与 harness 现状一致,不是"一句话摘要"):
  * 水位阈值:活上下文达到 window * --threshold(默认 0.8)才触发压缩;
  * 保留近段原文:压缩只作用于较早的前缀,最近的 --tail-ratio 部分原样保留;
  * 结构化 checkpoint:固定小节、要点式输出,要求保留精确名称/数字/日期/关系/否定;
  * 合并语义:已有 checkpoint 时"保留仍成立的事实、丢弃过期的、合并新信息";
  * checkpoint 以固定 preamble 注入答题上下文,声明它是自动生成的既有背景。

用法(在 GPU 分配内):
    PYTHONPATH=Main-Code/src .venv-d2l/bin/python scripts/run_compress_100q.py \
        --out result/text_compressed --window 24000 --summary-tokens 2000 [--splits ...]
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
    compute_metrics,
    finalize,
    load_rows,
    new_metrics,
    write_json,
)

from param_memory.config import load_config, validate_config, validate_resources  # noqa: E402
from param_memory.d2l import D2LBackend  # noqa: E402
from param_memory.schema import ChatMessage, stable_json  # noqa: E402

SUMMARY_OPEN_TAG = "<compacted-summary>"

# 对齐 dsh-compaction-basic 的结构:固定小节、要点式、精确值必须保留;
# 只是把"编码任务进度"换成"事实性历史",因为 MAB 的历史是事实而不是对话。
COMPACTION_INSTRUCTION = "\n".join(
    [
        "You are now acting as a compaction engine for a long factual memory. "
        "Condense the material ABOVE into a structured checkpoint that lets another "
        "model answer later questions with no loss of essential facts.",
        "",
        "Output EXACTLY the Markdown structure below: keep every section, in order. "
        'Use terse bullets, not prose paragraphs. Write "(none)" for an empty section '
        "— never drop a section.",
        "",
        "## Entities and Facts",
        "- [entity: attribute = value; one bullet per fact]",
        "",
        "## Updates and Conflicts",
        "- [attribute: the latest value, noting that it supersedes the earlier one]",
        "",
        "## Numbers, Dates and Units",
        "- [exact values, formats, ranges, ordering rules]",
        "",
        "## Task Rules",
        "- [rules needed to interpret the facts, e.g. higher serial number is newer]",
        "",
        "## Open Questions",
        "- [facts that were unclear or contradicted; write \"(none)\" if none]",
        "",
        "Rules:",
        "- Preserve exact entity names, numbers, dates, units, negations and relations; never paraphrase a value.",
        "- When a fact is superseded, keep only the latest value and record that it supersedes the earlier one.",
        "- Do NOT mention this summarization request or that the context was compacted.",
        "- Output only the checkpoint text.",
        f"- If the material contains a {SUMMARY_OPEN_TAG} block, it is a PRIOR checkpoint: do not copy it "
        "verbatim; preserve still-true facts, drop stale ones, and merge newer facts into one consolidated "
        "summary under the same structure.",
    ]
)

CHECKPOINT_PREAMBLE = (
    "This is an automatically generated checkpoint condensing an earlier span of the history. "
    "Treat the captured facts as established background and answer from them together with "
    "the material that follows."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAB 文本压缩基线(checkpoint 式)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "Main-Code/configs/mab_sh6k.yaml")
    parser.add_argument("--window", type=int, default=24_000, help="活上下文的 token 上限")
    parser.add_argument("--threshold", type=float, default=0.8, help="达到窗口的该比例即压缩")
    parser.add_argument("--tail-ratio", type=float, default=0.25, help="保留为原文的最近比例")
    parser.add_argument("--summary-tokens", type=int, default=2_000, help="checkpoint 输出上限")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题,用于冒烟")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true", help="跳过已成功题,只补失败/未跑")
    return parser.parse_args()


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("official_metric", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    encode = lambda t: backend.tokenizer.encode(t, add_special_tokens=False)  # noqa: E731
    decode = backend.tokenizer.decode

    answer_cfg = cfg.generation.task.model_copy(update={"max_input_tokens": args.window + 8_192})
    summary_cfg = cfg.generation.task.model_copy(
        update={"max_input_tokens": args.window + 8_192, "max_new_tokens": args.summary_tokens}
    )
    trigger = int(args.window * args.threshold)
    tail_tokens = max(int(args.window * args.tail_ratio), 1)
    step = max(args.window // 4, 1)

    def compact(material: str) -> str:
        """一次 checkpoint 压缩:material 已包含旧 checkpoint(若有)。"""
        messages = (
            ChatMessage(role="system", content=COMPACTION_INSTRUCTION),
            ChatMessage(role="user", content=material),
        )
        return backend.generate(backend.encode_messages(messages), summary_cfg).text.strip()

    def compress(text: str) -> tuple[str, dict]:
        """按水位触发、保留近段原文、滚动合并 checkpoint。"""
        ids = encode(text)
        stat = {"context_tokens": len(ids), "compress_rounds": 0, "final_tokens": 0}
        if len(ids) <= trigger:
            stat["final_tokens"] = len(ids)
            return text, stat
        buffer, summary = "", ""
        for start in range(0, len(ids), step):
            buffer += decode(ids[start : start + step])
            if len(encode(buffer)) < trigger:
                continue
            # 只压较早的前缀,最近 tail_tokens 原样保留。
            tail_ids = encode(buffer)[-tail_tokens:]
            tail = decode(tail_ids)
            head = buffer[: len(buffer) - len(tail)] if tail else buffer
            summary = compact(f"{summary}\n\n{head}".strip())
            stat["compress_rounds"] += 1
            buffer = f"{summary}\n{tail}"
        while len(encode(buffer)) > args.window:
            tail = decode(encode(buffer)[-tail_tokens:])
            head = buffer[: len(buffer) - len(tail)]
            summary = compact(f"{summary}\n\n{head}".strip())
            stat["compress_rounds"] += 1
            buffer = f"{summary}\n{tail}"
        if summary:
            buffer = f"{CHECKPOINT_PREAMBLE}\n\n{SUMMARY_OPEN_TAG}\n{buffer}\n</compacted-summary>"
        stat["final_tokens"] = len(encode(buffer))
        return buffer, stat

    args.out.mkdir(parents=True, exist_ok=True)
    pred_path = args.out / "predictions.jsonl"
    done_keys: set[tuple[str, str]] = set()
    if args.resume and pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row_done = json.loads(line)
                if not row_done["error"]:
                    done_keys.add((row_done["source"], row_done["question_id"]))
        print(f"续跑:已有 {len(done_keys)} 道成功题,将跳过", flush=True)
    started_total = time.perf_counter()
    done = 0
    metrics = new_metrics()
    row_stats: list[dict] = []

    with pred_path.open("a" if args.resume else "w", encoding="utf-8") as stream:
        for row in rows:
            compressed, stat = compress(row.context)
            row_stats.append({"split": row.split, "source": row.source, **stat})
            print(f"compress {row.source}: {stat['context_tokens']} -> {stat['final_tokens']} tok "
                  f"({stat['compress_rounds']} 轮)", flush=True)
            for question, answers, qa_id in zip(row.questions, row.answers, row.qa_ids, strict=True):
                if (row.source, qa_id) in done_keys:
                    continue
                if args.limit and done >= args.limit:
                    break
                payload = build_user_payload(cfg.data.protocol_instruction, question, compressed)
                ids = backend.encode_messages(
                    (ChatMessage(role="system", content=task_prompt),
                     ChatMessage(role="user", content=stable_json(payload)))
                )
                t0 = time.perf_counter()
                error = None
                try:
                    generation = backend.generate(ids, answer_cfg)
                    text = generation.text
                except Exception as exc:
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
                    "context_tokens": stat["context_tokens"],
                    "final_context_tokens": stat["final_tokens"],
                    "compress_rounds": stat["compress_rounds"],
                    "truncated": False,
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
                    print(f"  {done} 题 | EM={metrics['substring_em_sum'] / max(metrics['scored'], 1):.3f} "
                          f"| {metrics['seconds']:.0f}s", flush=True)
            if args.limit and done >= args.limit:
                break

    full, _ = compute_metrics(pred_path)
    result = finalize(full)
    result.update(
        {
            "mode": "text_compressed_checkpoint",
            "window": args.window,
            "threshold": args.threshold,
            "tail_ratio": args.tail_ratio,
            "summary_tokens": args.summary_tokens,
            "wall_seconds": time.perf_counter() - started_total,
            "row_stats": row_stats,
            "compress_rounds_total": sum(r["compress_rounds"] for r in row_stats),
            "config": str(args.config),
            "protocol_instruction": cfg.data.protocol_instruction,
        }
    )
    write_json(args.out / "metrics.json", result)
    print(json.dumps({k: result[k] for k in
                      ("questions", "scored", "failed", "substring_em_failure_inclusive",
                       "drqa_em_failure_inclusive", "token_f1_failure_inclusive",
                       "wall_seconds", "compress_rounds_total")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
