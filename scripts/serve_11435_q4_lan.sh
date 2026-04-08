#!/usr/bin/env bash
# Serve Qwen3.5-35B-A3B-UD-IQ4_NL on port 11435 for LAN + tool calling.
#
# VRAM layout on RTX 2000E Ada 16 GB (measured via tuning sweep):
#   Total projected:  ~15,295 MiB  (ngl=34, 34/41 layers on GPU)
#   Free headroom:    ~644 MiB     (≥512 MiB fit-target, context maintained)
#   Context:          196,608 tokens (full 196K — fit confirmed this fits)
#   Generation speed: ~26.7 t/s   (vs Q3_K_S: ~42.6 t/s — 37% slower, better tool calling)
#
# CPU offload: 7 layers on 28-core Ryzen 9 9955HX (small, MoE architecture)
# ngl=36 OOM; ngl=34 is maximum that fits.
#
# Batch tuning (vs old -b 512 -ub 128 -tb 24):
#   -b 2048 -ub 512  → GPU gets larger chunks, less CPU scheduling overhead
#   -t 16 -tb 8      → generation/sampling threads + batch threads (no 100% spikes)
set -euo pipefail

cd /home/uniberg/Qwen-3.5-16G-Vram-Local

MODEL=./models/unsloth-gguf/Qwen3.5-35B-A3B-UD-IQ4_NL.gguf

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL"
  echo "Run: scripts/download_ud_iq4nl.sh"
  exit 1
fi

exec ./llama-bin/llama-server \
  -m "$MODEL" \
  --host 0.0.0.0 --port 11435 \
  -ngl 34 \
  -c 196608 \
  --flash-attn on \
  -ctk iq4_nl -ctv iq4_nl \
  -b 2048 -ub 512 \
  -t 16 -tb 8 \
  --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.0 \
  --parallel 1 \
  --fit on --fit-target 256 \
  --reasoning-format deepseek-legacy \
  --reasoning-budget -1 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --metrics
