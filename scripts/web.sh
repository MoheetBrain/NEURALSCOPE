#!/usr/bin/env bash
# Start the Next.js frontend on $WEB_PORT (default 3000). /api/* is proxied to NEURALSCOPE_API_URL.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/apps/web"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
export NEURALSCOPE_API_URL="${NEURALSCOPE_API_URL:-http://127.0.0.1:${API_PORT:-8000}}"
# Bind to loopback only: the API's localhost guard cannot tell LAN clients proxied by Next from local ones.
exec npx next dev --hostname "${WEB_HOST:-127.0.0.1}" --port "${WEB_PORT:-3000}"
