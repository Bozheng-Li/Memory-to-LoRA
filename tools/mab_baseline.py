"""MAB 基线实验的共享工具:读数据、组提示、评分、落盘。

被 scripts/run_baseline.py 使用;不包含任何模型装配逻辑,便于在 CPU 上单测。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

SPLITS = {
    "Conflict_Resolution": "Conflict_Resolution.parquet",
    "Accurate_Retrieval": "Accurate_Retrieval.parquet",
    "Test_Time_Learning": "Test_Time_Learning.parquet",
    "Long_Range_Understanding": "Long_Range_Understanding.parquet",
}


@dataclass(frozen=True)
class Row:
    """一个历史及其题组。"""

    split: str
    source: str
    context: str
    questions: tuple[str, ...]
    answers: tuple[tuple[str, ...], ...]
    qa_ids: tuple[str, ...]


def load_rows(data_dir: Path, splits: tuple[str, ...] | None = None) -> list[Row]:
    """按分区读取历史行;行数、题数、答案数与上游元数据一致才返回。"""
    names = splits or tuple(SPLITS)
    rows: list[Row] = []
    for name in names:
        table = pq.read_table(Path(data_dir) / SPLITS[name], columns=["context", "questions", "answers", "metadata"])
        meta = table["metadata"].to_pylist()
        for ctx, qs, ans, md in zip(
            table["context"].to_pylist(),
            table["questions"].to_pylist(),
            table["answers"].to_pylist(),
            meta,
            strict=True,
        ):
            if not (len(qs) == len(ans) == len(md["qa_pair_ids"])):
                raise ValueError(f"{name} 行内题目/答案/ID 长度不一致")
            rows.append(
                Row(
                    split=name,
                    source=str(md.get("source")),
                    context=ctx,
                    questions=tuple(qs),
                    answers=tuple(tuple(a) if not isinstance(a, str) else (a,) for a in ans),
                    qa_ids=tuple(md["qa_pair_ids"]),
                )
            )
    return rows


def build_user_payload(goal: str, question: str, context: str | None) -> dict:
    """基线提示的 user 负载;context 为 None 时即空记忆(纯基座)。"""
    payload = {"goal": goal, "question": question}
    if context is not None:
        payload["context"] = context
    return payload


def new_metrics() -> dict:
    """聚合容器:总体与分分区计数、分数、token 与耗时。"""
    return {
        "questions": 0,
        "scored": 0,
        "failed": 0,
        "substring_em_sum": 0.0,
        "drqa_em_sum": 0.0,
        "f1_sum": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "seconds": 0.0,
        "per_source": {},
    }


def add_prediction(metrics: dict, pred: dict) -> None:
    """把单题结果累加进聚合容器(总体 + 分 source)。"""
    for bucket in (metrics, metrics["per_source"].setdefault(pred["source"], {
        "questions": 0, "scored": 0, "failed": 0,
        "substring_em_sum": 0.0, "drqa_em_sum": 0.0, "f1_sum": 0.0,
    })):
        bucket["questions"] += 1
        if pred["error"]:
            bucket["failed"] += 1
        else:
            bucket["scored"] += 1
            bucket["substring_em_sum"] += pred["substring_em"]
            bucket["drqa_em_sum"] += pred["drqa_em"]
            bucket["f1_sum"] += pred["f1"]
    metrics["input_tokens"] += pred["input_tokens"]
    metrics["output_tokens"] += pred["output_tokens"]
    metrics["seconds"] += pred["seconds"]


def finalize(metrics: dict) -> dict:
    """把累加和折算成均值,保留分母固定口径。"""
    def pack(b: dict) -> dict:
        n, s = b["questions"], b["scored"]
        return {
            "questions": n,
            "scored": s,
            "failed": b["failed"],
            "substring_em_failure_inclusive": b["substring_em_sum"] / n if n else None,
            "drqa_em_failure_inclusive": b["drqa_em_sum"] / n if n else None,
            "token_f1_failure_inclusive": b["f1_sum"] / n if n else None,
            "substring_em_scored_only": b["substring_em_sum"] / s if s else None,
        }

    out = pack(metrics)
    out["input_tokens"] = metrics["input_tokens"]
    out["output_tokens"] = metrics["output_tokens"]
    out["seconds"] = metrics["seconds"]
    out["per_source"] = {k: pack(v) for k, v in sorted(metrics["per_source"].items())}
    return out


def write_json(path: Path, payload: dict) -> None:
    """稳定写出 JSON(带换行),不覆盖已有文件由调用方保证。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compute_metrics(pred_path: Path) -> tuple[dict, dict]:
    """按 predictions.jsonl 重算指标:同题取成功行,返回全量与未截断子集两份累加器。"""
    best: dict[tuple[str, str], dict] = {}
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
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
