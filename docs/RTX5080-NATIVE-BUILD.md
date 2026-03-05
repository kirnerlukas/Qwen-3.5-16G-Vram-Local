# RTX 5080 / 5090 Native Build Guide

> **For SM120 (Blackwell) GPUs only** — RTX 5080, 5090, and future Blackwell cards.
>
> This guide covers building llama.cpp from source with native SM120 support for maximum performance.

---

## Why Build From Source?

Pre-built llama.cpp binaries use CUDA 12.4, which **does not include SM120 support**. They run via PTX JIT compilation (sm_89 → sm_120), which causes:

- **2-3 slow warmup requests** (~12 t/s instead of 125 t/s on first launch)
- **Suboptimal kernels** — not tuned for Blackwell's architecture

A native SM120 build gives you:

| Benefit                    | Impact                                   |
| -------------------------- | ---------------------------------------- |
| No JIT warmup              | Full speed from request 1                |
| Native Blackwell kernels   | ~1% raw throughput gain (125.8 vs 124.8) |
| All Flash Attention quants | Better KV cache options                  |

> **⚠️ IMPORTANT**: The native SM120 build provides only ~1% raw speed improvement (125.8 vs 124.8 t/s). The main benefit is eliminating the 2-3 slow JIT warmup requests on first launch. The **real 10x speedup** (9 t/s → 125 t/s) comes from using `--parallel 1` — see below.

---

## Prerequisites

| Requirement       | Version                | Notes                      |
| ----------------- | ---------------------- | -------------------------- |
| **CUDA Toolkit**  | 12.6+                  | Must include SM120 support |
| **Visual Studio** | 2022 with C++ workload | Windows build              |
| **CMake**         | 3.28+                  | Build system               |
| **Git**           | Latest                 | Clone source               |
| **GPU**           | RTX 5080 / 5090        | SM120 Blackwell            |

### CUDA 12.6+ Installation

1. Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Choose: Windows > x86_64 > 12.6+ > exe (local)
3. Install with default options
4. Verify: `nvcc --version` should show 12.6+

---

## Build Steps

### 1. Clone llama.cpp

```powershell
cd C:\Users\YourName\Projects
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

### 2. Create Build Directory

```powershell
mkdir build
cd build
```

### 3. Configure CMake (SM120 Native)

```powershell
cmake .. `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_CUDA_ARCHITECTURES=120 `
  -DGGML_CUDA=ON `
  -DGGML_CUDA_FA_ALL_QUANTS=ON `
  -DGGML_CUDA_F16=ON `
  -DGGML_FLASH_ATTN=ON `
  -DGGML_CCACHE=OFF `
  -DCMAKE_BUILD_TYPE=Release
```

**Key flags explained:**

| Flag                             | Why                                            |
| -------------------------------- | ---------------------------------------------- |
| `-DCMAKE_CUDA_ARCHITECTURES=120` | Native SM120 (Blackwell) code generation       |
| `-DGGML_CUDA_FA_ALL_QUANTS=ON`   | Compile Flash Attention for ALL KV quant types |
| `-DGGML_CUDA_F16=ON`             | FP16 tensor core optimizations                 |
| `-DGGML_FLASH_ATTN=ON`           | Enable Flash Attention 2                       |

### 4. Build

```powershell
cmake --build . --config Release --parallel 8
```

Build time: ~10-15 minutes on a modern CPU.

### 5. Verify SM120 Support

```powershell
.\bin\Release\llama-cli.exe --version
```

Check that it starts without PTX JIT messages in the console.

---

## Actual Performance Results (March 5, 2026)

Benchmarked on RTX 5080 16GB with Qwen3.5-35B-A3B Q3_K_S, 128K context, `--parallel 1`:

| Metric                 | Pre-built b8196 (PTX JIT) | Native SM120 Build | Difference |
| ---------------------- | :-----------------------: | :----------------: | :--------: |
| First request speed    |       ~12 t/s (JIT)       |      ~125 t/s      | **10x** ✅ |
| Steady state speed     |         124.8 t/s         |     125.8 t/s      | **+0.8%**  |
| Flash Attention quants |         All work          |   All available    |    Same    |
| Warmup time            |       2-3 requests        |        None        | Eliminated |

> **Key takeaway**: The raw throughput gain is only ~1%. The native build's real value is eliminating JIT warmup latency on first requests after server start.
>
> **The actual 10x speedup** (9 t/s → 125 t/s) comes from the `--parallel 1` flag, which is needed for the Gated DeltaNet hybrid architecture. See `results/BENCHMARK_RESULTS.md` for details.

---

## ⚠️ CRITICAL: --parallel 1 for 35B-A3B Models

**This is more important than the SM120 build itself.** The 35B-A3B uses a Gated DeltaNet (GDN) hybrid architecture with recurrent state (RS) buffers. The default `n_parallel=auto` (4 slots) allocates 4x larger RS buffers, causing a **10x generation slowdown**:

| Config                   | RS Buffer | Gen Speed       |
| ------------------------ | --------- | --------------- |
| `--parallel 1`           | 62 MB     | **~125 t/s** ✅ |
| `--parallel 4` (default) | 251 MB    | **~9 t/s** ❌   |

**Always add `--parallel 1` when running any 35B-A3B model.**

---

## KV Cache on SM120

Blackwell has different tensor core characteristics than Ada Lovelace (SM89). Here's what works best:

### For MoE Models (35B-A3B)

| KV Type  | VRAM @ 152K |  Speed  | Recommendation     |
| -------- | :---------: | :-----: | ------------------ |
| `iq4_nl` |   856 MB    | Fastest | ✅ **Best choice** |
| `q8_0`   |  1,712 MB   |  Fast   | Good alternative   |
| `f16`    |  3,424 MB   | Fastest | Too large for 16GB |

The `iq4_nl` is ideal because:

- MoE only has 10 attention layers → small KV
- Dequant overhead is minimal for small caches
- Leaves more VRAM for context

### For Dense Models (9B, 27B)

| KV Type  | VRAM @ 256K (9B) |  Speed  | Recommendation     |
| -------- | :--------------: | :-----: | ------------------ |
| `q8_0`   |     4,352 MB     | Fastest | ✅ **Best choice** |
| `iq4_nl` |     2,176 MB     | Slower  | Bandwidth-bound    |
| `f16`    |     8,704 MB     | Fastest | Won't fit          |

Dense models have 33+ attention layers → large KV. On SM120's high bandwidth (960 GB/s GDDR7), raw read speed wins over dequant savings.

---

## JIT Warmup Explained

### What Happens with Pre-built Binaries

```
Request 1:  PTX → sm_120 JIT compile (CUDA driver)  →  ~12 t/s
Request 2:  Kernel cache warmup                      →  ~80 t/s
Request 3+: Full speed                               →  125 t/s
```

The CUDA driver compiles PTX (portable intermediate) to SM120 machine code on first use. This takes 2-3 inference passes.

### With Native Build

```
Request 1+: Full speed immediately                   →  124+ t/s
```

SM120 machine code is baked into the binary. No JIT needed.

### How to Check

Run this in a separate terminal while the server starts:

```powershell
# Watch for JIT compilation
nvidia-smi -l 1
```

With pre-built: You'll see "Compute M" (compute mode) flicker during first requests.
With native: No flicker — kernels are ready immediately.

---

## Troubleshooting

### "CUDA architecture 120 is not supported"

You're using CUDA < 12.6. Upgrade to CUDA 12.6 or newer.

### "Flash Attention compilation failed"

Make sure you have:

- Visual Studio 2022 with latest updates
- Windows SDK 10.0.22621.0 or newer
- CMake 3.28+

### Build is slow

Use `--parallel N` where N = your CPU core count. Also consider ccache:

```powershell
# Install ccache via scoop or chocolatey
scoop install ccache
# Then add -DGGML_CCACHE=ON to cmake
```

### Still seeing slow first requests

Check that your binary is actually using SM120:

```powershell
# Should show sm_120 in the binary
dumpbin /headers .\bin\Release\llama.dll | findstr "machine"
```

---

## Full Build Script

Save as `build_sm120.ps1`:

```powershell
# RTX 5080/5090 Native Build Script
# Run from llama.cpp root directory

$ErrorActionPreference = "Stop"

Write-Host "=== llama.cpp SM120 Native Build ===" -ForegroundColor Cyan

# Check CUDA version
$nvcc = nvcc --version 2>&1
if (-not ($nvcc -match "release 12\.[6-9]")) {
    Write-Host "ERROR: CUDA 12.6+ required for SM120" -ForegroundColor Red
    Write-Host "Current: $nvcc"
    exit 1
}

Write-Host "CUDA version OK" -ForegroundColor Green

# Create build directory
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
New-Item -ItemType Directory -Path "build" | Out-Null
Set-Location "build"

# Configure
Write-Host "Configuring CMake..." -ForegroundColor Yellow
cmake .. `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_CUDA_ARCHITECTURES=120 `
    -DGGML_CUDA=ON `
    -DGGML_CUDA_FA_ALL_QUANTS=ON `
    -DGGML_CUDA_F16=ON `
    -DGGML_FLASH_ATTN=ON `
    -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "Building (this takes 10-15 minutes)..." -ForegroundColor Yellow
cmake --build . --config Release --parallel 8

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "=== Build Complete ===" -ForegroundColor Green
Write-Host "Binaries: $(Get-Location)\bin\Release\" -ForegroundColor Cyan

# Verify
Write-Host "`nVerifying SM120 support..." -ForegroundColor Yellow
$dumpbin = dumpbin /headers ".\bin\Release\ggml.dll" 2>&1
if ($dumpbin -match "120") {
    Write-Host "SM120 support confirmed!" -ForegroundColor Green
} else {
    Write-Host "WARNING: SM120 not found in binary" -ForegroundColor Yellow
}
```

---

## Files to Copy

After building, copy these to your `llama-bin/` folder:

```
llama.cpp/build/bin/Release/
├── llama-server.exe     ← Main server
├── llama-cli.exe        ← CLI tool
├── llama-bench.exe      ← Benchmarking
├── ggml.dll             ← Core library
├── ggml-cuda.dll        ← CUDA backend
└── *.dll                ← Other dependencies
```

---

## Related

- [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [CUDA 12.6 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [Blackwell architecture whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture)

---

## Contributing

If you build with SM120 and benchmark, please share your results:

| Your GPU | Build type   | Speed vs pre-built | Notes             |
| -------- | ------------ | ------------------ | ----------------- |
| RTX 5080 | SM120 native | +X%                | Your observations |

Open an issue or PR with your data!
