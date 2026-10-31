#!/usr/bin/env bash
# Generates the Python messages of the SpeechKit API into src/yandex (not committed).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m grpc_tools.protoc -Iproto --python_out=src \
  yandex/cloud/ai/stt/v3/stt.proto yandex/cloud/ai/tts/v3/tts.proto
