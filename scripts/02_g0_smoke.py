#!/usr/bin/env python
"""G0: load the D2L checkpoint, internalize a document, print real tensor shapes.

Verifies the three things Gate 0 needs:
  1. the checkpoint loads against the local base model
  2. `internalize` produces LoRA factors with the expected rank
  3. `reset` really clears the memory (answer degrades)

Run from resources/doc-to-lora with PYTHONPATH=src, or from the repo root:

    CUDA_VISIBLE_DEVICES=2 .venv-d2l/bin/python scripts/02_g0_smoke.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = (
    ROOT / "resources" / "checkpoints" / "qwen_4b_d2l" / "checkpoint-20000" / "pytorch_model.bin"
)
D2L_SRC = ROOT / "resources" / "doc-to-lora" / "src"

DOC = (
    "The Sakana Vault is a storage facility in Tsukuba. "
    "The vault access code is 4815-2209. "
    "The vault was built in 2019 and is guarded by two caretakers, "
    "Mika and Ren. The nearest train station is Kenkyu-gakuen. "
    "Monthly maintenance happens on the first Tuesday. "
)
QUESTION = "What is the vault access code?"


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    return (
        f"alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--doc", default=DOC)
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    import os
    import sys

    sys.path.insert(0, str(D2L_SRC))
    # Run from the D2L repo root: that is where `Qwen/Qwen3-4B-Instruct-2507`
    # exists as a relative symlink, so the hard-coded repo id resolves offline.
    d2l_root = D2L_SRC.parent
    if (d2l_root / "Qwen" / "Qwen3-4B-Instruct-2507").exists():
        os.chdir(d2l_root)
    else:
        print(f"WARN: {d2l_root}/Qwen/Qwen3-4B-Instruct-2507 missing; "
              "base model may be fetched from the hub")

    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
    from ctx_to_lora.model_loading import get_tokenizer

    t0 = time.time()
    print(f"[1/5] torch.load {args.ckpt}")
    sd = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    n_param = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    dtypes = {str(v.dtype) for v in sd.values() if torch.is_tensor(v)}
    print(f"      {len(sd)} entries, {n_param / 1e6:.1f}M params, dtypes={dtypes}")

    print("[2/5] building ModulatedPretrainedModel ...")
    model = ModulatedPretrainedModel.from_state_dict(
        sd,
        train=False,
        use_flash_attn=False,  # sdpa/eager; flip to True once flash-attn is verified
        use_sequence_packing=False,
    )
    model.eval()
    model.reset()
    base = model.base_model
    print(f"      base={base.name_or_path}  attn={base.config._attn_implementation}")
    print(f"      built in {time.time() - t0:.1f}s | {gpu_mem()}")

    tok = get_tokenizer(model.base_model.name_or_path)

    def ask() -> str:
        chat = [{"role": "user", "content": args.question}]
        ids = tok.apply_chat_template(
            chat,
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        out = model.generate(
            input_ids=ids, max_new_tokens=args.max_new_tokens, do_sample=False
        )
        return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()

    print("[3/5] answering WITHOUT memory ...")
    print(f"      -> {ask()!r}")

    print("[4/5] internalize(document) ...")
    t1 = time.time()
    model.internalize(args.doc)
    dt = time.time() - t1
    print(f"      compiled in {dt:.2f}s | {gpu_mem()}")

    loras = model.generated_loras
    print(f"      generated_loras: {type(loras).__name__}, {len(loras)} entries")
    for i, (k, v) in enumerate(loras.items()):
        if i >= 2:
            print(f"      ... {len(loras) - 2} more")
            break
        if torch.is_tensor(v):
            print(f"      {k}: tensor {tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"      {k}: " + ", ".join(f"{kk}{tuple(vv.shape)}" for kk, vv in v.items()))
        else:
            print(f"      {k}: {type(v).__name__}")

    print("[5/5] answering WITH memory ...")
    print(f"      -> {ask()!r}")

    model.reset()
    print("      after reset ->", repr(ask()))
    print(f"\nDONE in {time.time() - t0:.1f}s | {gpu_mem()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
