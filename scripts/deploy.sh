#!/usr/bin/env bash
# Copies the worker to a host over ssh and starts it. Usage: scripts/deploy.sh [host]   (default: gpu-pc)
# The first start downloads ~13 GB of models. The key stays in the host's .env; the next runs keep it.
set -euo pipefail
HOST="${1:-gpu-pc}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
COPYFILE_DISABLE=1 tar --no-xattrs --exclude=.git --exclude=.venv --exclude=.env --exclude=models \
  --exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache --exclude=src/yandex -czf - . \
  | ssh "$HOST" 'mkdir -p ~/VoiceAgentWorkerDocker && tar -xzf - -C ~/VoiceAgentWorkerDocker'
ssh "$HOST" bash -s <<'REMOTE'
set -euo pipefail
cd ~/VoiceAgentWorkerDocker
mkdir -p models
if [[ ! -f .env ]]; then
  sed "s|^WORKER_API_KEY=.*|WORKER_API_KEY=$(openssl rand -hex 24)|" .env.example > .env
  chmod 600 .env
fi
docker compose up -d --build
echo
echo "Worker key: $(grep ^WORKER_API_KEY= .env | cut -d= -f2)"
REMOTE
