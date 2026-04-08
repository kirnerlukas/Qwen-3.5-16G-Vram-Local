#!/usr/bin/env bash
# Background launcher for serve_11435_q4_lan.sh (nohup wrapper).
set -euo pipefail

cd /home/uniberg/Qwen-3.5-16G-Vram-Local

pkill -f llama-server 2>/dev/null || true
sleep 1

nohup ./scripts/serve_11435_q4_lan.sh \
  > logs/server-11435.log 2>&1 < /dev/null &

echo $! > logs/server-11435.pid
echo "Started PID $(cat logs/server-11435.pid) — tailing logs/server-11435.log"
sleep 10
tail -20 logs/server-11435.log
