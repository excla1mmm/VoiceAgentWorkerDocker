#!/usr/bin/env bash
# Runs the worker on a Mac for a station: speech in Docker, the LLM natively on Metal (Docker Desktop has neither the
# GPU nor the memory), a reverse ssh tunnel to the station, and no sleep while it runs.
# Usage: scripts/mac-worker.sh start|stop [ssh-target]     (default: excla1m@176.109.108.176)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TARGET="${2:-excla1m@176.109.108.176}"
RUN=.mac-run
mkdir -p "$RUN"

stop() {
  for name in tunnel llm awake; do
    if [[ -f "$RUN/$name.pid" ]]; then
      kill "$(cat "$RUN/$name.pid")" 2>/dev/null || true
      rm -f "$RUN/$name.pid"
    fi
  done
  pkill -f "127.0.0.1:18080:127.0.0.1:8080" 2>/dev/null || true
  docker compose down
}

start() {
  if [[ ! -f .env ]]; then
    printf 'COMPOSE_FILE=compose.yaml\nWORKER_API_KEY=%s\nLLM_ENABLED=0\nPUBLISH_LLM_PORT=18081\n' "$(openssl rand -hex 24)" > .env
    chmod 600 .env
  fi
  KEY="$(grep ^WORKER_API_KEY= .env | cut -d= -f2)"
  docker compose up -d --build
  nohup llama-server --model models/gpt-oss-20b-MXFP4.gguf --host 127.0.0.1 --port 8080 --api-key "$KEY" --jinja \
    --chat-template-kwargs '{"reasoning_effort":"low"}' --n-gpu-layers 999 --parallel 3 --ctx-size 48000 --no-webui \
    > "$RUN/llm.log" 2>&1 &
  echo $! > "$RUN/llm.pid"
  nohup caffeinate -is -w "$(cat "$RUN/llm.pid")" > /dev/null 2>&1 &
  echo $! > "$RUN/awake.pid"
  nohup bash -c "while true; do ssh -N -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 \
    -o ExitOnForwardFailure=yes -R 127.0.0.1:50051:127.0.0.1:50051 -R 127.0.0.1:18080:127.0.0.1:8080 $TARGET; sleep 5; done" \
    > "$RUN/tunnel.log" 2>&1 &
  echo $! > "$RUN/tunnel.pid"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  *) echo "usage: $0 start|stop [ssh-target]" >&2; exit 2 ;;
esac
