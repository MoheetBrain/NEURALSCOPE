#!/usr/bin/env bash
# Start the NeuralScope API (FastAPI + model) on $API_PORT (default 8000).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f .env ] && set -a && . ./.env && set +a
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
exec uvicorn neuralscope_api.server:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}" "$@"
