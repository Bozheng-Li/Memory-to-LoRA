#!/usr/bin/env python
"""Dilution curve: how does D2L recall degrade as the *compile length* grows?

The B0 probe found needle-alone = 100% but needle-in-2K = 10%. That points at dilution
inside the perceiver, not at a fixed "literal fidelity" defect. This script measures the
curve directly: same needle, same haystack text, only the compiled window length changes.

For each length L in {256, 512, 1024, 2048}:
  text  : L tokens in the prompt (control, should stay ~100%)
  d2l   : the same L tokens compiled into parameters

Usage:
  CUDA_VISIBLE_DEVICES=2 .venv-d2l/bin/python scripts/_pilot/pilot_dilution.py --n 12
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

        sd = torch.load(str(CKPT), weights_only=False, map_location="cpu")
        self.model = ModulatedPretrainedModel.from_state_dict(
            sd, train=False, use_flash_attn=False, use_sequence_packing=False
        ).eval()
        self.model.reset()
        self.tok = get_tokenizer(self.model.base_model.name_or_path)
        self.max_new = max_new_tokens
        print("[runner] loaded", flush=True)

    def ask(self, question: str, context: str | None) -> str:
        if context:
            content = ("Answer the question using the context. Reply with ONLY the exact "
                       f"value.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:")
        else:
            content = f"Question: {question}\nAnswer with only the exact value:"
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": content}], add_special_tokens=False,
            return_attention_mask=False, add_generation_prompt=True, return_tensors="pt",
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


def build(tok, rng: random.Random, target_len: int) -> tuple[str, str]:
    """Haystack of ~target_len tokens with the needle placed in the middle."""
    raw = HAYSTACK.read_text(encoding="utf-8", errors="ignore")
    paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 80]
    rng.shuffle(paras)
    parts, ntok = [], 0
    for p in paras:
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        if ntok + n > target_len - 40:
            continue
        parts.append(p)
        ntok += n
        if ntok >= target_len - 60:
            break
    code = f"{rng.randrange(1000, 9999)}-{rng.randrange(1000, 9999)}"
    needle = f"The access code for the Sakana vault is {code}."
    mid = max(1, len(parts) // 2)
    text = "\n\n".join(parts[:mid] + [needle] + parts[mid:])
    ids = tok(text, add_special_tokens=False)["input_ids"][:target_len]
    return tok.decode(ids), code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--lens", default="256,512,1024,2048")
    ap.add_argument("--out", default=str(ROOT / "research/pilot_20261028/raw/results_dilution.jsonl"))
    args = ap.parse_args()
    args.out = str(Path(args.out).resolve())

    runner = Runner()
    rng = random.Random(1)
    question = "What is the access code for the Sakana vault?"
    lens = [int(x) for x in args.lens.split(",")]

    records, summary = [], {}
    for L in lens:
        for cond in ("text", "d2l"):
            t0, hits = time.time(), []
            for i in range(args.n):
                text, code = build(runner.tok, rng, L)
                runner.reset()
                if cond == "text":
                    pred = runner.ask(question, text)
                else:
                    runner.compile_text(text)
                    pred = runner.ask(question, None)
                hit = int(norm(code) in norm(pred))
                hits.append(hit)
                records.append({"cond": cond, "len": L, "i": i, "code": code,
                                "pred": pred, "hit": hit})
            summary[f"{cond}_{L}"] = round(100 * sum(hits) / len(hits), 1)
            print(f"[{cond:4s} L={L:4d}] n={args.n} acc={summary[f'{cond}_{L}']:5.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            runner.reset()

    Path(args.out).write_text("\n".join(json.dumps(r) for r in records))
    print("\nsummary:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
