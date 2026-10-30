#!/usr/bin/env python
"""Measure peak VRAM for one D2L internalize+generate pass at a given context size.

    CUDA_VISIBLE_DEVICES=2 .venv-d2l/bin/python scripts/03_mem_probe.py --chunks 1 --len 8192
    CUDA_VISIBLE_DEVICES=2 .venv-d2l/bin/python scripts/03_mem_probe.py --chunks 4 --len 512

One fresh process per configuration gives a clean peak.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "resources" / "checkpoints" / "qwen_4b_d2l" / "checkpoint-20000" / "pytorch_model.bin"
D2L_ROOT = ROOT / "resources" / "doc-to-lora"

FILLER = (
    "The archive records that the northern expedition crossed the river in spring, "
    "carrying grain, tools, and written orders from the capital. "
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--len", type=int, default=8192, help="tokens per chunk")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--no-flash", action="store_true")
    args = ap.parse_args()

    os.chdir(D2L_ROOT)
    sys.path.insert(0, str(D2L_ROOT / "src"))
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
    from ctx_to_lora.model_loading import get_tokenizer

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    sd = torch.load(CKPT, weights_only=False, map_location="cpu")
    model = ModulatedPretrainedModel.from_state_dict(
        sd, train=False, use_flash_attn=not args.no_flash, use_sequence_packing=False
    )
    model.eval()
    model.reset()
    tok = get_tokenizer(model.base_model.name_or_path)
    after_load = torch.cuda.max_memory_allocated() / 2**30

    # build exactly `chunks` x `len` tokens of filler
    unit = tok(FILLER, add_special_tokens=False)["input_ids"]
    need = args.len * args.chunks
    ids = (unit * (need // len(unit) + 1))[:need]
    text = tok.decode(ids)

    t1 = time.time()
    if args.chunks == 1:
        model.internalize(text)
    else:
        ctx = torch.tensor(ids, device=model.device).view(args.chunks, args.len)
        model._internalize_from_ids(ctx, torch.ones_like(ctx))
    t_compile = time.time() - t1
    after_compile = torch.cuda.max_memory_allocated() / 2**30

    prompt = [{"role": "user", "content": "What did the archive record?"}]
    pids = tok.apply_chat_template(
        prompt, add_special_tokens=False, return_attention_mask=False,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    model.generate(input_ids=pids, max_new_tokens=args.gen, do_sample=False)
    peak = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30

    rank = 0
    loras = model.generated_loras or {}
    for v in loras.values():
        if isinstance(v, dict) and "A" in v:
            rank = v["A"].shape[0] * v["A"].shape[2]  # n_ctx * r
            break

    print(
        f"chunks={args.chunks} len={args.len} total={args.chunks*args.len} "
        f"flash={not args.no_flash} rank={rank} | "
        f"load={after_load:.1f} compile+={after_compile-after_load:.1f} "
        f"peak={peak:.1f} reserved={peak_res:.1f} GiB | "
        f"t_compile={t_compile:.1f}s total={time.time()-t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
