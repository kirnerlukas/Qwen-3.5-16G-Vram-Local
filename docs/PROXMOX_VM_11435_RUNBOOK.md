# Proxmox VM Runbook: Qwen3.5 on `llama.cpp` with OpenAI endpoint on `11435`

## Goal
Run Qwen3.5 locally in a Proxmox VM with NVIDIA GPU passthrough and expose an OpenAI-compatible endpoint on:

- `http://<vm-ip>:11435/v1/...`

Port note:
- `8002` is the default `coding` profile in the general launcher.
- `11435` is the dedicated OpenAI-compatible LAN endpoint documented in this runbook.

This runbook documents exactly what was done in this workspace.

## Environment Used
- OS: Linux VM on Proxmox
- GPU: NVIDIA RTX 2000E Ada 16GB
- Driver/CUDA seen in VM: driver `535.261.03`, CUDA runtime `12.2`, toolkit `12.4`
- Repo root: `/home/uniberg/Qwen-3.5-16G-Vram-Local`

---

## 1) Build `llama.cpp` with CUDA
The latest upstream release did not provide a Linux CUDA binary in this flow, so it was built from source.

```bash
cd /home/uniberg/Qwen-3.5-16G-Vram-Local
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build llama.cpp/build --target llama-server -j
mkdir -p llama-bin
cp llama.cpp/build/bin/llama-server llama-bin/llama-server
chmod +x llama-bin/llama-server
```

GPU detect check:
```bash
./llama-bin/llama-server --version
```

---

## 2) Python env + Hugging Face CLI
```bash
cd /home/uniberg/Qwen-3.5-16G-Vram-Local
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt huggingface_hub
```

> Note: modern CLI command is `hf ...`, not `huggingface-cli ...`.

---

## 3) Download model files
Use repo-local HF cache paths to avoid permission issues in restricted environments.

### 35B (required)
```bash
cd /home/uniberg/Qwen-3.5-16G-Vram-Local
source .venv/bin/activate
mkdir -p .hf-cache .hf-local/share .hf-local/config models/unsloth-gguf
HF_HOME=$PWD/.hf-cache \
HUGGINGFACE_HUB_CACHE=$PWD/.hf-cache/hub \
HF_HUB_CACHE=$PWD/.hf-cache/hub \
HF_XET_CACHE=$PWD/.hf-cache/xet \
XDG_CACHE_HOME=$PWD/.hf-local \
XDG_DATA_HOME=$PWD/.hf-local/share \
XDG_CONFIG_HOME=$PWD/.hf-local/config \
hf download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-Q3_K_S.gguf mmproj-F16.gguf \
  --local-dir ./models/unsloth-gguf/

cp -f models/unsloth-gguf/mmproj-F16.gguf models/unsloth-gguf/mmproj-35B-F16.gguf
```

### Optional comparison models used
```bash
# 27B
HF_HOME=$PWD/.hf-cache HUGGINGFACE_HUB_CACHE=$PWD/.hf-cache/hub HF_HUB_CACHE=$PWD/.hf-cache/hub HF_XET_CACHE=$PWD/.hf-cache/xet \
XDG_CACHE_HOME=$PWD/.hf-local XDG_DATA_HOME=$PWD/.hf-local/share XDG_CONFIG_HOME=$PWD/.hf-local/config \
hf download unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q3_K_S.gguf mmproj-F16.gguf --local-dir ./models/unsloth-gguf/
cp -f models/unsloth-gguf/mmproj-F16.gguf models/unsloth-gguf/mmproj-27B-F16.gguf

# 9B
HF_HOME=$PWD/.hf-cache HUGGINGFACE_HUB_CACHE=$PWD/.hf-cache/hub HF_HUB_CACHE=$PWD/.hf-cache/hub HF_XET_CACHE=$PWD/.hf-cache/xet \
XDG_CACHE_HOME=$PWD/.hf-local XDG_DATA_HOME=$PWD/.hf-local/share XDG_CONFIG_HOME=$PWD/.hf-local/config \
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir ./models/unsloth-gguf/
```

---

## 4) Repo changes made

### `config/config_loader.py`
Linux support was fixed by selecting binary name by platform:
- Windows: `llama-server.exe`
- Linux/macOS: `llama-server`

### `config/servers.yaml`
Added a dedicated endpoint profile/server for port `11435`:
- `coding_openai_11435`
- profile `openai_11435`

Due to 16GB constraints on this GPU, practical stable profile became text-only and less-than-256K context.

---

## 5) Fit findings on this GPU (critical)
Measured on RTX 2000E Ada 16GB:

- `35B text-only @ 256K` -> failed
- `35B text-only @ 192K` -> passed
- `35B + mmproj @ 64K` -> failed

Conclusion:
- For stable local serving on this GPU: **35B text-only**
- Vision projector (`mmproj`) with 35B did not fit reliably here

---

## 6) Throughput comparisons performed
Same prompt/settings, 3 runs each:

- 35B (`Q3_K_S`) avg: **37.99 t/s**
- 27B (`Q3_K_S`) avg: **6.78 t/s**
- 9B (`UD-Q4_K_XL`) avg: **30.77 t/s**

On this machine, 35B MoE was fastest among these tested presets.

---

## 7) Current launch scripts created

- `scripts/start_lan_11435_reasoning.sh`:
  LAN bind + reasoning on baseline
- `scripts/start_best_11435_lan_reasoning.sh`:
  tuned LAN bind + reasoning on (`-b 512 -ub 128 -t 24 -tb 24`)
- `scripts/start_11435_software_fast.sh`:
  software-traffic oriented profile (currently reasoning on, larger batching)
- `scripts/serve_11435_lan_reasoning.sh`:
  foreground launcher intended for `systemd`

Benchmark/diagnostic helpers:
- `scripts/run_fit_matrix.sh`
- `scripts/compare_35b_27b.sh`
- `scripts/bench_9b_only.sh`
- `scripts/tune_35b_speed.sh`

---

## 8) Start/verify commands

### Start software-oriented profile on `11435`
```bash
cd /home/uniberg/Qwen-3.5-16G-Vram-Local
./scripts/start_11435_software_fast.sh
```

### Start LAN reasoning profile on `11435` (classic)
```bash
cd /home/uniberg/Qwen-3.5-16G-Vram-Local
./scripts/start_lan_11435_reasoning.sh
```

### Start LAN reasoning profile in foreground (recommended for persistence)
```bash
cd /home/uniberg/Qwen-3.5-16G-Vram-Local
./scripts/serve_11435_lan_reasoning.sh
```

### Verify on VM
```bash
pgrep -af llama-server
ss -ltnp | grep 11435
curl http://127.0.0.1:11435/health
curl http://10.13.37.16:11435/health
```

### Verify from LAN client
```bash
curl -v http://10.13.37.16:11435/health
```

OpenAI chat test:
```bash
curl -N http://10.13.37.16:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen3.5-35B-A3B-Q3_K_S.gguf",
    "messages":[{"role":"user","content":"Hello"}],
    "stream":true
  }'
```

---

## 9) Why CPU can still be high
Observed in logs:
- very large input requests (`~14k–15k` tokens)
- frequent full prompt reprocessing/checkpoint churn

That can saturate CPU while GPU stays far below 100%, even with full layer offload.

If your app must keep huge prompts and reasoning always on, expect high CPU pressure.

---

## 10) Optional: `systemd` service (not installed in this session)
A service file was prepared conceptually but not installed due permission approval being skipped in-session.

When ready, use `scripts/serve_11435_lan_reasoning.sh` as `ExecStart` in a unit and enable restart-on-failure.

---

## 11) Quick troubleshooting checklist
If LAN clients get `connection refused`:

1. Confirm process exists:
```bash
pgrep -af llama-server
```
2. Confirm listener exists:
```bash
ss -ltnp | grep 11435
```
3. Confirm health locally:
```bash
curl http://127.0.0.1:11435/health
```
4. Confirm health on WG/LAN IP:
```bash
curl http://10.13.37.16:11435/health
```
5. Check recent logs:
```bash
tail -n 120 logs/server-11435.log
```
