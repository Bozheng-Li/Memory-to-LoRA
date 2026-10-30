#!/usr/bin/env python
"""Pilot run: MemoryAgentBench FactConsolidation (MAB-CR) with frozen Qwen3-4B + frozen D2L.

This is the *first* real measurement for the PACT proposal. It is deliberately small:
one context, a fixed question subset, greedy decoding, no tuning. It answers one question
that the whole proposal depends on:

    does compiling a 2K-token slice into D2L parameters preserve enough exact-value
    information to answer fact-consolidation questions, compared with (a) putting text
    in the prompt and (b) putting nothing?

Conditions (all share the same reader, prompt template, and decoding):
  no_memory       : question only
  full_context    : the whole context in the prompt (reference; only when it fits)
  text_last2k     : last 2048 tokens in the prompt
  text_bm25_top4  : BM25 top-4 chunks (2048 tokens) in the prompt
  d2l_first2k     : first 2048 tokens compiled into D2L
  d2l_last2k      : last 2048 tokens compiled into D2L
  d2l_bm25_2k     : BM25 top-4 chunks concatenated (2048 tokens) compiled into D2L

Usage:
  CUDA_VISIBLE_DEVICES=2 .venv-d2l/bin/python scripts/_pilot/pilot_mab.py \
      --parquet resources/data/mab/Conflict_Resolution.parquet --row 0 --n 25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
D2L_SRC = ROOT / "resources" / "doc-to-lora" / "src"
CKPT = ROOT / "resources" / "checkpoints" / "qwen_4b_d2l" / "checkpoint-20000" / "pytorch_model.bin"


# --------------------------------------------------------------------------- scoring
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred: str, gold: str) -> dict:
    np_, ng = normalize(pred), normalize(gold)
    return {
        "subem": int(ng in np_) if ng else 0,   # MAB-style substring EM
        "em": int(np_ == ng),
        "f1": token_f1(pred, gold),
    }


# --------------------------------------------------------------------------- retrieval
def chunk_ids(ids: list[int], size: int) -> list[list[int]]:
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def bm25_rank(query: str, chunks: list[str], tok) -> list[int]:
    """Minimal Okapi BM25 (k1=1.5, b=0.75) -- no external dependency."""
    q_terms = [t for t in re.findall(r"[a-z0-9]+", query.lower())]
    docs = [re.findall(r"[a-z0-9]+", c.lower()) for c in chunks]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n)
    df: Counter = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * 2.5 / (tf[t] + 1.5 * (1 - 0.75 + 0.75 * len(d) / avgdl))
        scores.append(s)
    return sorted(range(n), key=lambda i: -scores[i])


# --------------------------------------------------------------------------- model
class Runner:
    def __init__(self, max_new_tokens: int = 64):
        sys.path.insert(0, str(D2L_SRC))
        os.chdir(D2L_SRC.parent)          # base model resolves as a relative symlink
        from ctx_to_lora.model_loading import get_tokenizer
        from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

        t0 = time.time()
        sd = torch.load(str(CKPT), weights_only=False, map_location="cpu")
        self.model = ModulatedPretrainedModel.from_state_dict(
            sd, train=False, use_flash_attn=False, use_sequence_packing=False
        ).eval()
        self.model.reset()
        self.tok = get_tokenizer(self.model.base_model.name_or_path)
        self.max_new = max_new_tokens
        print(f"[runner] loaded in {time.time() - t0:.1f}s", flush=True)

    # ---- generation -------------------------------------------------------
    def ask(self, question: str, context: str | None = None) -> str:
        if context:
            content = (
                "Answer the question using the context. Reply with ONLY the exact entity or "
                "value (at most 5 words), no explanation, no quotes.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
        else:
            content = f"Question: {question}\nAnswer with only the exact entity or value:"
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_special_tokens=False, return_attention_mask=False,
            add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(input_ids=ids, max_new_tokens=self.max_new, do_sample=False)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    # ---- compile ----------------------------------------------------------
    def compile_text(self, text: str) -> int:
        ids = self.tok(text, add_special_tokens=False)["input_ids"][:2048]
        self.model.internalize(self.tok.decode(ids))
        return len(ids)

    def reset(self) -> None:
        self.model.reset()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(ROOT / "resources/data/mab/Conflict_Resolution.parquet"))
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default=str(ROOT / "research/pilot_20261028/raw/results_mab_cr.jsonl"))
    ap.add_argument("--conditions", default="no_memory,text_last2k,text_bm25_top4,full_context,"
                                           "d2l_first2k,d2l_last2k,d2l_bm25_2k")
    args = ap.parse_args()
    args.out = str(Path(args.out).resolve())   # Runner chdir()s into the D2L repo

    df = pd.read_parquet(args.parquet)
    row = df.iloc[args.row]
    context, questions, answers = row["context"], list(row["questions"]), row["answers"]
    n = min(args.n, len(questions))
    print(f"[data] row={args.row} chars={len(context)} questions={len(questions)} using={n}", flush=True)

    runner = Runner()
    ids = runner.tok(context, add_special_tokens=False)["input_ids"]
    chunks = chunk_ids(ids, 512)
    chunk_texts = [runner.tok.decode(c) for c in chunks]
    print(f"[data] {len(ids)} tokens -> {len(chunks)} chunks of 512", flush=True)

    first2k = runner.tok.decode(ids[:2048])
    last2k = runner.tok.decode(ids[-2048:])

    conditions = args.conditions.split(",")
    records = []
    summary = {}

    for cond in conditions:
        t0 = time.time()
        met = {"subem": [], "em": [], "f1": []}
        for qi in range(n):
            q, gold_list = questions[qi], answers[qi]
            gold = str(gold_list[0]) if hasattr(gold_list, "__len__") else str(gold_list)

            if cond == "no_memory":
                runner.reset()
                pred = runner.ask(q)
            elif cond == "full_context":
                runner.reset()
                pred = runner.ask(q, context)
            elif cond == "text_last2k":
                runner.reset()
                pred = runner.ask(q, last2k)
            elif cond in ("text_bm25_top4", "d2l_bm25_2k"):
                top = bm25_rank(q, chunk_texts, runner.tok)[:4]
                ctx = "\n".join(chunk_texts[i] for i in sorted(top))
                runner.reset()
                if cond == "text_bm25_top4":
                    pred = runner.ask(q, ctx)
                else:
                    runner.compile_text(ctx)
                    pred = runner.ask(q)
            elif cond == "d2l_first2k":
                runner.reset(); runner.compile_text(first2k); pred = runner.ask(q)
            elif cond == "d2l_last2k":
                runner.reset(); runner.compile_text(last2k); pred = runner.ask(q)
            else:
                raise ValueError(cond)

            sc = score(pred, gold)
            for k in met:
                met[k].append(sc[k])
            records.append({"cond": cond, "qi": qi, "q": q, "gold": gold,
                            "pred": pred, **sc})

        summary[cond] = {k: round(100 * sum(v) / len(v), 1) for k, v in met.items()}
        print(f"[{cond:16s}] n={n} subem={summary[cond]['subem']:5.1f} "
              f"em={summary[cond]['em']:5.1f} f1={summary[cond]['f1']:5.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        runner.reset()

    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))
    print("\nsummary:", json.dumps(summary, indent=2))
    print(f"records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
