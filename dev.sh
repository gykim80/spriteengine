#!/usr/bin/env bash
set -Eeuo pipefail

readonly PORT=2344
readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

# Release the fixed Vite port from a previous development session.
PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$PIDS" ]]; then
  echo "[dev] Stopping previous process on port $PORT: ${PIDS//$'\n'/ }"
  kill $PIDS 2>/dev/null || true

  for _ in {1..20}; do
    if ! lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done

  REMAINING="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$REMAINING" ]]; then
    echo "[dev] Force-stopping process on port $PORT: ${REMAINING//$'\n'/ }"
    kill -9 $REMAINING 2>/dev/null || true
  fi
fi

if command -v wails >/dev/null 2>&1; then
  WAILS_BIN="$(command -v wails)"
elif [[ -x "$(go env GOPATH)/bin/wails" ]]; then
  WAILS_BIN="$(go env GOPATH)/bin/wails"
else
  echo "[dev] Wails CLI is not installed." >&2
  exit 1
fi

echo "[dev] Starting SpriteEngine Studio at http://localhost:$PORT"
exec "$WAILS_BIN" dev
