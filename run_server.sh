#!/usr/bin/env bash
# Launch llama-server for the autocomplete harness.
# Usage: ./run_server.sh <path-to-gguf> [port] [ctx-size]
set -euo pipefail
MODEL="${1:?path to .gguf required}"
PORT="${2:-8080}"
CTX="${3:-2048}"
exec llama-server \
  --model "$MODEL" \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers 999
