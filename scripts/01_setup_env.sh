#!/usr/bin/env bash
# Reproduce the D2L environment on this machine (verified 2026-10-28).
#
# Why not just `install.sh`: that pulls vllm / deepspeed / flashinfer / gradio,
# none of which the eval path imports, and vllm 0.8.5 would force its own torch.
# This script installs only what `run_eval.py` + `internalize` actually touch.
#
# Usage:  bash scripts/01_setup_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYPI="https://pypi.tuna.tsinghua.edu.cn/simple"      # pypi.org is unreachable here
TORCH_INDEX="https://download.pytorch.org/whl/cu124" # ~6 MB/s, vs ~2 MB/s on tuna
GH_PROXY="https://ghfast.top/"                       # github release-assets is blocked
HF="https://hf-mirror.com"
FA_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FA_WHL}"

echo "== [1/6] python 3.10 venv"
uv venv --python 3.10 .venv-d2l

echo "== [2/6] torch 2.6.0 + cu124"
UV_HTTP_TIMEOUT=180 uv pip install --python .venv-d2l/bin/python \
  --index-url "$TORCH_INDEX" --extra-index-url "$PYPI" "torch==2.6.0"

echo "== [3/6] pinned stack (versions taken from resources/doc-to-lora/uv.lock)"
UV_HTTP_TIMEOUT=180 uv pip install --python .venv-d2l/bin/python --index-url "$PYPI" \
  "transformers==4.51.3" "datasets==3.6.0" "tokenizers==0.21.0" \
  "peft==0.15.2" "accelerate==1.6.0" "numpy==2.2.6" \
  "bitsandbytes>=0.46.1" setuptools wheel \
  einops jaxtyping safetensors pyarrow pandas

# setuptools is not implicit in a uv venv, but triton (torch dep) imports it.
# bitsandbytes is required because the checkpoint's ctx_encoder_args set
# quantize_ctx_encoder=True, so from_state_dict builds a 4-bit config.

echo "== [4/6] flash-attn prebuilt wheel"
mkdir -p wheels
if [ ! -f "wheels/${FA_WHL}" ]; then
  curl -L --retry 3 -o "wheels/${FA_WHL}" "${GH_PROXY}${FA_URL}"
fi
uv pip install --python .venv-d2l/bin/python --no-deps "wheels/${FA_WHL}"

echo "== [5/6] make the base model resolvable offline"
# checkpoints store base_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507";
# from_pretrained treats that as a relative dir if it exists, so a symlink is
# enough. No re-download, no code patch.
mkdir -p resources/doc-to-lora/Qwen
ln -sfn ../../models/Qwen3-4B-Instruct-2507 resources/doc-to-lora/Qwen/Qwen3-4B-Instruct-2507
grep -qxF 'Qwen/' resources/doc-to-lora/.git/info/exclude 2>/dev/null \
  || echo 'Qwen/' >> resources/doc-to-lora/.git/info/exclude

echo "== [6/6] verify"
cd resources/doc-to-lora
PYTHONPATH=src ../../.venv-d2l/bin/python - <<'PY'
import torch, transformers, datasets, peft, flash_attn
from peft.tuners._buffer_dict import BufferDict
from peft import LoraRuntimeConfig
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "| datasets", datasets.__version__,
      "| peft", peft.__version__, "| flash_attn", flash_attn.__version__)
print("imports OK")
PY

cat <<'EOF'

Environment ready. Smoke test:
  cd resources/doc-to-lora
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src ../../.venv-d2l/bin/python ../../scripts/02_g0_smoke.py

Expected: no-memory answer is a refusal, with-memory answer is "4815-2209",
reset restores the refusal.
EOF
