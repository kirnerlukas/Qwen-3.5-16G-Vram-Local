#!/usr/bin/env bash
set -euo pipefail

PORT35=11435
PORT27=11436
MODEL35="Qwen3.5-35B-A3B-Q3_K_S.gguf"
MODEL27="Qwen3.5-27B-Q3_K_S.gguf"
LOG27="logs/server-11436.log"
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
  local label="$3"

  echo "=== ${label} (${model} @ :${port}) ==="
  local sum=0
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
  echo "avg_tps_${label}=$avg"
}

# 35B from configured preset
python server_manager.py stop >/dev/null 2>&1 || true
python server_manager.py start --server coding_openai_11435 >/dev/null 2>&1
wait_ready "$PORT35"
bench_model "$PORT35" "$MODEL35" "35B"

# 27B text-only comparison server
python server_manager.py stop >/dev/null 2>&1 || true
nohup ./llama-bin/llama-server \
  -m ./models/unsloth-gguf/${MODEL27} \
  --host 127.0.0.1 --port ${PORT27} \
  -c 98304 -ngl 99 \
  --flash-attn on \
  -ctk iq4_nl -ctv iq4_nl \
  -b 1024 -ub 256 \
  --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.0 \
  --parallel 1 \
  --fit on --fit-target 256 \
  --reasoning-budget 0 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  > "$LOG27" 2>&1 < /dev/null &
wait_ready "$PORT27"
bench_model "$PORT27" "$MODEL27" "27B"

# restore 35B service
python server_manager.py stop >/dev/null 2>&1 || true
python server_manager.py start --server coding_openai_11435 >/dev/null 2>&1
wait_ready "$PORT35"
echo "restored_35b_on_${PORT35}=ok"
