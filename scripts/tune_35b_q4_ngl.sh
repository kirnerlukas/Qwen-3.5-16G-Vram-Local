#!/usr/bin/env bash
# Find the best --ngl value for Qwen3.5-35B-A3B-UD-IQ4_NL (17.8 GB) on RTX 2000E Ada 16 GB.
#
# Strategy: sweep ngl from 28 to 36. At each step, record:
#   - VRAM layout (model / KV / compute / free)
#   - Auto-fit context size
#   - Generation speed (t/s) and prompt speed
#
# After the sweep, run the tool-calling benchmark on the best candidate.
# Results go to logs/tuning/q4_ngl_*.log
set -euo pipefail

BIN=./llama-bin/llama-server
MODEL=./models/unsloth-gguf/Qwen3.5-35B-A3B-UD-IQ4_NL.gguf
PORT=11635
RUNS=3
PROMPT='List exactly 5 JSON tool call schemas for a code editor assistant. Be concise.'

mkdir -p logs/tuning

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL"
  echo "Run scripts/download_ud_iq4nl.sh first."
  exit 1
fi

wait_ready() {
  local port="$1"
  for _ in $(seq 1 120); do
    out=$(curl -sS "http://127.0.0.1:${port}/health" 2>/dev/null || true)
    if echo "$out" | grep -q '"status":"ok"'; then return 0; fi
    sleep 1
  done
  return 1
}

bench_once() {
  local port="$1"
  start=$(date +%s%N)
  resp=$(curl -sS "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0.0,\"max_tokens\":300,\"stream\":false}" \
    2>/dev/null)
  end=$(date +%s%N)
  elapsed=$(awk "BEGIN {print ($end - $start) / 1e9}")
  ctok=$(echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("usage",{}).get("completion_tokens",0))' 2>/dev/null || echo 0)
  tps=$(awk "BEGIN { if ($elapsed > 0) print $ctok / $elapsed; else print 0 }")
  echo "$ctok|$elapsed|$tps"
}

run_ngl() {
  local ngl="$1"
  local name="q4_ngl${ngl}"
  local log="logs/tuning/${name}.log"

  pkill -f llama-server 2>/dev/null || true
  sleep 1

  nohup "$BIN" \
    -m "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    -ngl "$ngl" \
    -c 196608 \
    --flash-attn on \
    -ctk iq4_nl -ctv iq4_nl \
    -b 2048 -ub 512 \
    -t 16 -tb 8 \
    --temp 0.0 \
    --parallel 1 \
    --fit on --fit-target 512 \
    --reasoning-format deepseek-legacy \
    --reasoning-budget -1 \
    --chat-template-kwargs '{"enable_thinking":false}' \
    > "$log" 2>&1 < /dev/null &

  if ! wait_ready "$PORT"; then
    echo "NGL=$ngl STATUS=FAILED_TO_START"
    tail -5 "$log" || true
    return
  fi

  # Extract VRAM layout from log
  vram_line=$(grep -m1 "memory breakdown" "$log" || echo "n/a")
  fit_ctx=$(grep -m1 "n_ctx" "$log" | grep -oP 'n_ctx\s*=\s*\K[0-9]+' || echo "n/a")
  kv_buf=$(grep -m1 "KV buffer size" "$log" || echo "n/a")

  echo "NGL=$ngl STATUS=OK"
  echo "  fit_ctx=$fit_ctx"
  echo "  kv=$(echo "$kv_buf" | grep -oP '[0-9]+\.[0-9]+ MiB' | head -1 || echo n/a)"
  echo "  vram=$(echo "$vram_line" | grep -oP '\d+ = .+' | head -1 || echo n/a)"

  sum=0
  for i in $(seq 1 "$RUNS"); do
    line=$(bench_once "$PORT")
    ctok=${line%%|*}; rest=${line#*|}; elapsed=${rest%%|*}; tps=${line##*|}
    sum=$(awk "BEGIN {print $sum + $tps}")
    printf "  run_%d: tokens=%s elapsed=%.2fs tps=%.1f\n" "$i" "$ctok" "$elapsed" "$tps"
  done
  avg=$(awk "BEGIN {print $sum / $RUNS}")
  printf "  avg_tps=%.1f\n" "$avg"
}

echo "=== UD-IQ4_NL ngl sweep (RTX 2000E Ada 16 GB) ==="
echo "Model: $MODEL"
echo ""

# Sweep from conservative to aggressive GPU offload
for ngl in 28 30 32 34 36; do
  run_ngl "$ngl"
  echo ""
done

pkill -f llama-server 2>/dev/null || true

echo "=== Sweep complete. Pick the highest ngl where STATUS=OK and VRAM fits. ==="
echo "Then update serve_11435_q4_lan.sh with the best ngl value."
echo ""
echo "Run the tool-calling benchmark on the winner:"
echo "  scripts/start_11435_q4_lan.sh"
echo "  python tests/benchmark_tool_calling.py 11435"
