#!/usr/bin/env bash
# Build the base gemma-4-E2B GGUFs from the downloaded safetensors.
# Base ships only as a 10 GB multimodal safetensors with no GGUF, so we convert
# the text tower ourselves (llama.cpp's Gemma4Model) and quantize.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/models/gemma-4-E2B-base"
PY="$ROOT/.venv/bin/python"
F16="$ROOT/models/gemma4-base-f16.gguf"

echo "[1/3] convert safetensors -> f16 GGUF (text tower)"
"$PY" "$ROOT/llama.cpp/convert_hf_to_gguf.py" "$SRC" --outfile "$F16" --outtype f16

echo "[2/3] quantize -> Q4_K_M (primary, ~3 GB, the on-device ship target)"
llama-quantize "$F16" "$ROOT/models/gemma4-base-Q4_K_M.gguf" Q4_K_M

echo "[3/3] quantize -> Q8_0 (for the Q4-vs-Q8 comparison)"
llama-quantize "$F16" "$ROOT/models/gemma4-base-Q8_0.gguf" Q8_0

echo "DONE. GGUFs:"; ls -lh "$ROOT/models/"*.gguf
