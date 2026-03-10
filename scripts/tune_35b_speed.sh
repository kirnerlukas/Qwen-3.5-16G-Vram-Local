#!/usr/bin/env bash
set -euo pipefail

BIN=./llama-bin/llama-server
MODEL=./models/unsloth-gguf/Qwen3.5-35B-A3B-Q3_K_S.gguf
PORT=11635
RUNS=3
PROMPT='Write exactly 180 words about GPU memory optimization for llama.cpp.'

mkdir -p logs/tuning

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

bench_once() {
  local port="$1"
  start=$(date +%s.%N)
  resp=$(curl -sS "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen3.5-35B-A3B-Q3_K_S.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0.2,\"max_tokens\":220,\"stream\":false}")
  end=$(date +%s.%N)
  elapsed=$(awk "BEGIN {print $end - $start}")
  ctok=$(echo "$resp" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("usage",{}).get("completion_tokens",0))')
  tps=$(awk "BEGIN { if ($elapsed > 0) print $ctok / $elapsed; else print 0 }")
  echo "$ctok|$elapsed|$tps"
}

run_cfg() {
  local name="$1"; shift
  local log="logs/tuning/${name}.log"
  pkill -f llama-server 2>/dev/null || true
  sleep 1

  nohup "$BIN" \
    -m "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --flash-attn on \
    -ctk iq4_nl -ctv iq4_nl \
    --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.0 \
    --parallel 1 \
    --fit on --fit-target 256 \
    --reasoning-format deepseek-legacy \
    --reasoning-budget -1 \
    --chat-template-kwargs '{"enable_thinking":true}' \
    "$@" > "$log" 2>&1 < /dev/null &

  if ! wait_ready "$PORT"; then
    echo "CFG=$name STATUS=FAILED_TO_START"
    tail -n 5 "$log" || true
    return
  fi

  sum=0
  echo "CFG=$name STATUS=OK"
  for i in $(seq 1 "$RUNS"); do
    line=$(bench_once "$PORT")
    ctok=${line%%|*}; rest=${line#*|}; elapsed=${rest%%|*}; tps=${line##*|}
    sum=$(awk "BEGIN {print $sum + $tps}")
    echo "  run_$i: completion_tokens=$ctok elapsed_s=$elapsed tps=$tps"
  done
  avg=$(awk "BEGIN {print $sum / $RUNS}")
  echo "  avg_tps=$avg"

  echo "  kv=$(grep -m1 'KV buffer size' "$log" || echo n/a)"
  echo "  compute=$(grep -m1 'compute buffer size' "$log" || echo n/a)"
}

echo "=== 35B tuning sweep (reasoning enabled) ==="
run_cfg baseline_ctx196k_b1024_u256 -c 196608 -ngl 99 -b 1024 -ub 256
run_cfg ctx131k_b1024_u256      -c 131072 -ngl 99 -b 1024 -ub 256
run_cfg ctx98k_b1024_u256       -c 98304  -ngl 99 -b 1024 -ub 256
run_cfg ctx196k_b512_u128       -c 196608 -ngl 99 -b 512  -ub 128
run_cfg ctx196k_b512_u128_t20   -c 196608 -ngl 99 -b 512  -ub 128 -t 20 -tb 20
run_cfg ctx196k_b1024_u256_nocache -c 196608 -ngl 99 -b 1024 -ub 256 --no-cache-prompt

pkill -f llama-server 2>/dev/null || true
