#!/usr/bin/env bash
# NeuralScope local development: API (model) + web, with clean shutdown on Ctrl-C.
#   ./scripts/dev.sh            real model (NEURALSCOPE_MODEL, default Qwen/Qwen2.5-1.5B-Instruct)
#   ./scripts/dev.sh --demo     DEMO mode: tiny random-weight fixture model (UI development only)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f .env ] || cp .env.example .env
if [ "${1:-}" = "--demo" ]; then
  export NEURALSCOPE_MODEL="demo/random-qwen2-tiny"
  export NEURALSCOPE_DATA_DIR="${NEURALSCOPE_DATA_DIR:-$ROOT/data/demo}"
  echo "!! DEMO MODE — random-weight fixture model; outputs carry no scientific meaning"
fi
[ -d .venv ] || { echo "Run ./scripts/setup.sh first"; exit 1; }
[ -d apps/web/node_modules ] || (cd apps/web && npm install)

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

"$ROOT/scripts/api.sh" & pids+=($!)
"$ROOT/scripts/web.sh" & pids+=($!)
echo ""
echo "  NeuralScope web  → http://localhost:${WEB_PORT:-3000}      (workspace at /app)"
echo "  NeuralScope API  → http://localhost:${API_PORT:-8000}/docs (model loads in background; see /api/health)"
echo ""
# portable (bash 3.2 has no `wait -n`): exit when either process stops
while kill -0 "${pids[0]}" 2>/dev/null && kill -0 "${pids[1]}" 2>/dev/null; do sleep 1; done
