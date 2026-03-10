#!/usr/bin/env bash
set -euo pipefail
PORT=11437
MODEL="Qwen3.5-9B-UD-Q4_K_XL.gguf"
LOG="logs/server-11437.log"
RUNS=3
PROMPT="Write exactly 180 words about GPU memory optimization for llama.cpp."

wait_ready() {
  local port="$1"
  for _ in $(seq 1 180); do
    out=$(curl -sS "http://127.0.0.1:${port}/health" || true)
    if echo "$out" | grep -q '"status":"ok"'; then
      return 0
    fi
    sleep 1
  done
  return 1
}

bench_model() {
  local port="$1"
  local model="$2"
  local sum=0
  echo "=== 9B (${model} @ :${port}) ==="
  for i in $(seq 1 "$RUNS"); do
    start=$(date +%s.%N)
    resp=$(curl -sS "http://127.0.0.1:${port}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"temperature\":0.2,\"max_tokens\":220,\"stream\":false}")
    end=$(date +%s.%N)
    elapsed=$(awk "BEGIN {print $end - $start}")
    ctok=$(echo "$resp" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("usage",{}).get("completion_tokens",0))')
    tps=$(awk "BEGIN { if ($elapsed > 0) print $ctok / $elapsed; else print 0 }")
    sum=$(awk "BEGIN {print $sum + $tps}")
    echo "run_$i: completion_tokens=$ctok elapsed_s=$elapsed tps=$tps"
  done
  avg=$(awk "BEGIN {print $sum / $RUNS}")
  echo "avg_tps_9B=$avg"
}

python server_manager.py stop >/dev/null 2>&1 || true
nohup ./llama-bin/llama-server \
  -m ./models/unsloth-gguf/${MODEL} \
  --host 127.0.0.1 --port ${PORT} \
  -c 262144 -ngl 99 \
  --flash-attn on \
  -ctk q8_0 -ctv q8_0 \
  -b 1024 -ub 256 \
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 \
  --fit on --fit-target 1536 \
  --reasoning-budget 0 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  > "$LOG" 2>&1 < /dev/null &

wait_ready "$PORT"
bench_model "$PORT" "$MODEL"

python server_manager.py stop >/dev/null 2>&1 || true
python server_manager.py start --server coding_openai_11435 >/dev/null 2>&1 || true
