#!/usr/bin/env python
"""B0 kill-gate probe: does the frozen D2L compiler care *what* you compile?

Controlled single-hop NIAH at a 2048-token budget (D2L's native input size). Five
conditions share one reader, one prompt template, greedy decoding:

  text_full     : haystack + needle in the prompt        (the model can answer?)
  text_random   : haystack only, needle removed          (can it guess? should be 0)
  d2l_full      : the SAME text compiled into D2L        (does compiling preserve it?)
  d2l_random    : haystack only, compiled                (content sensitivity control)
  d2l_needle    : only the needle sentence, compiled     (isolates literal fidelity)

Reported: exact match of the code. Delta(text_full - text_random) says the task is
answerable at this budget; delta(d2l_full - d2l_random) is the *content-sensitivity*
number the proposal's Gate B0 depends on.

Usage:
  CUDA_VISIBLE_DEVICES=2 .venv-d2l/bin/python scripts/_pilot/pilot_niah.py --n 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
D2L_SRC = ROOT / "resources" / "doc-to-lora" / "src"
CKPT = ROOT / "resources" / "checkpoints" / "qwen_4b_d2l" / "checkpoint-20000" / "pytorch_model.bin"
HAYSTACK = ROOT / "research/archive/plan-history/BENCHMARK_SURVEY.first-round.2026-10-28.md"


def norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", s.lower())


class Runner:
    def __init__(self, max_new_tokens: int = 24):
        sys.path.insert(0, str(D2L_SRC))
        import os
        os.chdir(D2L_SRC.parent)
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

    def ask(self, question: str, context: str | None) -> str:
        if context:
            content = (
                "Answer the question using the context. Reply with ONLY the exact value, "
                f"no explanation.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
        else:
            content = f"Question: {question}\nAnswer with only the exact value:"
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_special_tokens=False, return_attention_mask=False,
            add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(input_ids=ids, max_new_tokens=self.max_new, do_sample=False)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def compile_text(self, text: str) -> int:
        ids = self.tok(text, add_special_tokens=False)["input_ids"][:2048]
        self.model.internalize(self.tok.decode(ids))
        return len(ids)

    def reset(self) -> None:
        self.model.reset()
        torch.cuda.empty_cache()


def build_haystack(tok, rng: random.Random) -> tuple[str, str, str]:
    """Return (haystack, needle_sentence, answer_code) at ~2048 tokens."""
    raw = HAYSTACK.read_text(encoding="utf-8", errors="ignore")
    paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 120]
    rng.shuffle(paras)
    filler, ntok = [], 0
    for p in paras:
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        if ntok + n > 2000:
            continue
        filler.append(p)
        ntok += n
        if ntok > 1900:
            break
    hay = "\n\n".join(filler)
    code = f"{rng.randrange(1000, 9999)}-{rng.randrange(1000, 9999)}"
    needle = f"The access code for the Sakana vault is {code}."
    # insert the needle near the middle, like a standard NIAH placement
    parts = hay.split("\n\n")
    mid = max(1, len(parts) // 2)
    with_needle = "\n\n".join(parts[:mid] + [needle] + parts[mid:])
    return with_needle, needle, code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "research/pilot_20261028/raw/results_niah.jsonl"))
    args = ap.parse_args()
    args.out = str(Path(args.out).resolve())   # Runner chdir()s into the D2L repo

    runner = Runner()
    rng = random.Random(0)
    question = "What is the access code for the Sakana vault?"

    conditions = ["no_memory", "text_full", "text_random", "d2l_full", "d2l_random", "d2l_needle"]
    records, summary = [], {}
    for cond in conditions:
        t0, hits = time.time(), []
        for i in range(args.n):
            hay_full, needle, code = build_haystack(runner.tok, rng)
            hay_rand = hay_full.replace(needle + "\n\n", "").replace(needle, "")
            runner.reset()
            if cond == "no_memory":
                pred = runner.ask(question, None)
            elif cond == "text_full":
                pred = runner.ask(question, hay_full)
            elif cond == "text_random":
                pred = runner.ask(question, hay_rand)
            elif cond == "d2l_full":
                runner.compile_text(hay_full); pred = runner.ask(question, None)
            elif cond == "d2l_random":
                runner.compile_text(hay_rand); pred = runner.ask(question, None)
            elif cond == "d2l_needle":
                runner.compile_text(needle); pred = runner.ask(question, None)
            else:
                raise ValueError(cond)
            hit = int(norm(code) in norm(pred))
            hits.append(hit)
            records.append({"cond": cond, "i": i, "code": code, "pred": pred, "hit": hit})
        summary[cond] = round(100 * sum(hits) / len(hits), 1)
        print(f"[{cond:12s}] n={args.n} acc={summary[cond]:5.1f} ({time.time() - t0:.0f}s)", flush=True)
        runner.reset()

    Path(args.out).write_text("\n".join(json.dumps(r) for r in records))
    print("\nsummary:", json.dumps(summary, indent=2))
    print(f"records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
