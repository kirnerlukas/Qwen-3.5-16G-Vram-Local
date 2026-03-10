#!/usr/bin/env bash
set -euo pipefail
MODEL=./models/unsloth-gguf/Qwen3.5-35B-A3B-Q3_K_S.gguf
MMPROJ=./models/unsloth-gguf/mmproj-35B-F16.gguf
BIN=./llama-bin/llama-server

run_test() {
  name="$1"; port="$2"; ctx="$3"; mmproj_mode="$4"
  log="logs/fit-matrix/${name}.log"
  rm -f "$log"
  pkill -f llama-server 2>/dev/null || true
  sleep 1

  cmd=("$BIN" -m "$MODEL" --host 127.0.0.1 --port "$port" -c "$ctx" -ngl 99 --flash-attn on -ctk iq4_nl -ctv iq4_nl -b 1024 -ub 256 --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.0 --parallel 1 --fit on --fit-target 256 --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}')
  if [ "$mmproj_mode" = "yes" ]; then
    cmd+=(--mmproj "$MMPROJ" --mmproj-offload)
  fi

  nohup "${cmd[@]}" > "$log" 2>&1 < /dev/null &
  pid=$!

  status="UNKNOWN"
  for _ in $(seq 1 120); do
    if grep -q "server is listening" "$log"; then
      status="PASS"
      break
    fi
    if grep -Eqi "cudaMalloc failed|failed to create context|GGML_ASSERT|failed to allocate|out of memory" "$log"; then
      status="OOM_OR_ALLOC_FAIL"
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      status="EXITED"
      break
    fi
    sleep 1
  done

  if [ "$status" = "PASS" ]; then
    health=$(curl -sS "http://127.0.0.1:${port}/health" || true)
  else
    health="-"
  fi

  echo "TEST=$name STATUS=$status HEALTH=$health"
  echo "  KV: $(grep -m1 "KV buffer size" "$log" || echo n/a)"
  echo "  RS: $(grep -m1 "RS buffer size" "$log" || echo n/a)"
  echo "  MODEL: $(grep -m1 "CUDA0 model buffer size" "$log" || echo n/a)"
  echo "  END: $(tail -n 1 "$log" || true)"

  pkill -f llama-server 2>/dev/null || true
  sleep 1
}

mkdir -p logs/fit-matrix
run_test text_only_256k 11501 262144 no
run_test text_only_192k 11502 196608 no
run_test vision_64k 11503 65536 yes
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
