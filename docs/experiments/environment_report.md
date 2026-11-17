# Environment Report

> **Verification Policy:** All hardware, OS, disk, toolchain, `llama.cpp`, and model specifications below were directly inspected via command-line tools on `2026-11-17`.

---

## 1. Host Hardware & Operating System

| Component | Observed Specification |
| :--- | :--- |
| **OS** | Microsoft Windows 10 Pro (Version `10.0.19045`, 64-bit) |
| **CPU** | 13th Gen Intel(R) Core(TM) i5-13500 (`14` physical cores, `20` logical processors, `2500 MHz` base clock) |
| **System RAM** | `15.70 GiB` (~16 GB DDR) |
| **GPU** | Intel(R) UHD Graphics 770 (integrated, `~2.0 GiB` reported AdapterRAM, driver `32.0.101.7088`) — **No discrete GPU** |
| **Disk (`C:`)** | `18.61 GiB` free / `204.36 GiB` used |

---

## 2. Build Toolchain & Runtimes

Located in `C:\msys64\clang64\bin` (and system PATH for Python):

| Tool | Observed Version | Target / Notes |
| :--- | :--- | :--- |
| **Clang** | `22.1.8` (MSYS2 MINGW-packages) | `x86_64-w64-windows-gnu` |
| **CMake** | `4.4.3` | Ninja generator available |
| **Ninja** | `1.13.2` | — |
| **Python** | `3.13.5` | Offline analysis, calibration, & FFI tests |

---

## 3. Local Inference Engine (`llama.cpp`)

- **Path:** `C:\Users\sgi.lm\Desktop\llama.cpp`
- **Git HEAD:** `b79696f` (`ggml-cuda: hip: tune GCN MMQ configs (#30021)`)
- **Build Directory:** `C:\Users\sgi.lm\Desktop\llama.cpp\build`
- **Header Present:** `C:\Users\sgi.lm\Desktop\llama.cpp\include\llama.h` (`True`)
- **Compiled Binaries (`build/bin/`):** `llama-cli.exe`, `llama-bench.exe`, `llama-batched.exe`, `llama-batched-bench.exe`, `llama-completion.exe`, `llama-embedding.exe`, `llama-debug.exe`, etc.
- **Shared Libraries (`build/bin/`):** `libllama.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `libllama-common.dll`, etc.

---

## 4. Local Quantized Models

- **Models Directory:** `C:\Users\sgi.lm\Desktop\models`
- **Available Model:**
  - `qwen2.5-0.5b-instruct-q4_k_m.gguf` — `491,400,032` bytes (`468.64 MiB`)

---

## 5. Engineering Constraints Derived from Environment

1. **CPU-Only Execution:** Without a discrete GPU, all `llama.cpp` inference, representation extraction, and alignment benchmarks must run on the 14C/20T Intel i5-13500 CPU. Training large neural projectors (as in C2C) is infeasible locally; CMCP must prioritize **training-free** (relative anchors, byte-span/MinED alignment, perplexity-pruned fallback) and **closed-form / lightweight** (ridge regression / Procrustes) alignment methods.
2. **RAM & Disk Budget (`15.70 GiB` RAM, `18.61 GiB` free disk):** Concurrent multi-model experiments must use compact quantized GGUF models (e.g., 0.5B–1.5B Q4_K_M, ~470 MiB–1.1 GiB each) so two models + KV caches + OS fit comfortably within 15.70 GiB RAM without paging, and downloaded models do not exhaust the 18.61 GiB free disk space.
3. **Toolchain:** C99/C11/C++20 code compiles cleanly with MSYS2 `clang` (`22.1.8`) + `cmake` (`4.4.3`) + `ninja` (`1.13.2`), linking against `libllama.dll` when `llama.cpp` integration is enabled.
