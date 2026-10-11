#!/usr/bin/env bash
# Dedicated Ollama server for the hallucination study (port 11435) with
# parallel decoding slots. It shares the model store with the normal
# Ollama install but leaves the default server (11434) untouched.
set -euo pipefail
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11435}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-3}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-60m}"
exec ollama serve
