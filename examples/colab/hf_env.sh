#!/usr/bin/env bash
# A second Python environment for chapter 3's transformers worker: torch 2.12 on CUDA 13.0.
# vLLM 0.30.0 brings torch 2.13, and the flash_attention_2 fallback transformers 5.17.0 loads
# (the hub kernel kernels-community/flash-attn2) has builds up to torch 2.12 only.
# Usage: bash hf_env.sh [dir]   (default /content/hf-torch212); prints the interpreter path.
set -euo pipefail
DIR="${1:-/content/hf-torch212}"
if [ ! -x "$DIR/bin/python" ]; then
  command -v uv >/dev/null || pip install -q uv
  uv venv -q --python "$(command -v python3)" "$DIR"
  uv pip install -q --python "$DIR/bin/python" torch==2.12.1 --torch-backend=cu130
  uv pip install -q --python "$DIR/bin/python" transformers==5.17.0 accelerate kernels
fi
"$DIR/bin/python" -c "import torch, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__)" >&2
echo "$DIR/bin/python"
