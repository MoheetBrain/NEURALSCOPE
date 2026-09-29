#!/usr/bin/env bash
# One-time setup: Python venv + package, web dependencies, model weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f .env ] || cp .env.example .env

echo "==> Python environment (.venv)"
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv .venv --python 3.11
  # CPU wheels by default; set TORCH_INDEX_URL for CUDA builds (e.g. https://download.pytorch.org/whl/cu124)
  uv pip install --python .venv/bin/python --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" torch
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" torch
  .venv/bin/pip install -e ".[dev]"
fi

echo "==> Web dependencies"
(cd apps/web && npm install)

echo "==> Model weights (Hugging Face cache)"
set -a; . ./.env; set +a
.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
import os
for m in {os.environ.get("NEURALSCOPE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"), "Qwen/Qwen2.5-0.5B-Instruct"}:
    if m.startswith("demo/"):
        continue
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "merges.txt", "vocab.json"])
    print("  ", m, "->", p)
PY
echo "Setup complete. Run: make dev"
