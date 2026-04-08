#!/usr/bin/env bash
# PRIMARY serve script: Qwen3.5-35B-A3B-UD-IQ3_S on port 11435.
#
# VRAM layout (confirmed, all layers on GPU):
#   Model (ngl=99):   ~12.6 GB on GPU  (all 41 layers)
#   KV cache iq4_nl:  ~1.1 GB  (196K context)
#   Compute buffer:   ~612 MB  (at -ub 512; ~1224 MB at -ub 1024)
#   Free headroom:    ~1.5 GB  (still ~342 MB free at -ub 1024)
#
# vs UD-IQ4_NL (archived in serve_11435_q4_lan.sh):
#   UD-IQ3_S:  43 t/s, 89% GPU util, 0 CPU layers, 7/7 tool calls pass
#   UD-IQ4_NL: 26 t/s, 44% GPU util, 7 CPU layers, 7/7 tool calls pass
#   → UD-IQ3_S is the better choice: same quality, 65% faster, no CPU stalls
#
# UD = unsloth dynamic: attention + embedding layers get higher precision,
#      MoE FFN/expert layers get lower precision. Better than plain Q3_K_S
#      for JSON tool calling despite same file size.
set -euo pipefail

cd /home/uniberg/Qwen-3.5-16G-Vram-Local

MODEL=./models/unsloth-gguf/Qwen3.5-35B-A3B-UD-IQ3_S.gguf

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL"
  echo "Run: scripts/download_ud_iq3s.sh"
  exit 1
fi

exec ./llama-bin/llama-server \
  -m "$MODEL" \
  --host 0.0.0.0 --port 11435 \
  -ngl 99 \
  -c 196608 \
  --flash-attn on \
  -ctk iq4_nl -ctv iq4_nl \
  -b 4096 -ub 1024 \
  -t 16 -tb 16 \
  --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.0 \
  --parallel 1 \
  --fit on --fit-target 256 \
  --reasoning-format deepseek-legacy \
  --reasoning-budget -1 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --metrics
