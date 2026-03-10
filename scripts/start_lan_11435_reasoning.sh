#!/usr/bin/env bash
set -euo pipefail
pkill -f llama-server 2>/dev/null || true
nohup ./llama-bin/llama-server \
  -m ./models/unsloth-gguf/Qwen3.5-35B-A3B-Q3_K_S.gguf \
  --host 0.0.0.0 --port 11435 \
  -c 196608 \
  -ngl 99 \
  --flash-attn on \
  -ctk iq4_nl -ctv iq4_nl \
  -b 1024 -ub 256 \
  --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.0 \
  --parallel 1 \
  --fit on --fit-target 256 \
  --reasoning-format deepseek-legacy \
  --reasoning-budget -1 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --metrics \
  --verbosity 4 \
  > logs/server-11435.log 2>&1 < /dev/null &
echo $! > logs/server-11435.pid
sleep 8
