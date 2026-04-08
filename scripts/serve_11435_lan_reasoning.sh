#!/usr/bin/env bash
set -euo pipefail

cd /home/uniberg/Qwen-3.5-16G-Vram-Local

exec ./llama-bin/llama-server \
  -m ./models/unsloth-gguf/Qwen3.5-35B-A3B-Q3_K_S.gguf \
  --host 0.0.0.0 --port 11435 \
  -c 196608 \
  -ngl 99 \
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
