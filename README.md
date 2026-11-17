# Cross-Model Context Protocol (CMCP)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> [Versión en Español](README.es.md) | **Research Report:** [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md)

**CMCP** is a zero-dependency C99 protocol core, Python SDK, and empirical benchmark suite for **cross-model context transfer** between heterogeneous LLMs (differing in architecture, parameter count, quantization, and tokenizer).

---

## 1. Key Capabilities

- **64-Byte Binary Envelope & Dual CRC32C:** Fixed little-endian header with `"CMCP"` magic, model/tokenizer FNV-1a hashes, and Castagnoli CRC32C checksums (`header_crc32c` + `payload_crc32c`).
- **4-Tier Compatibility Negotiation & KV Guardrails:** `cmcp_check_compat` and `cmcp_check_kv_compat` negotiate `Tier 0` (Exact KV) through `Tier 3` (Text Fallback) and reject subtle mismatches (`n_ctx`, `pos_offset`, `arch`, `rope`) that raw `llama_state_seq_set_data` may accept silently.
- **Typed TLV & Lossless Compression:** `CONCEPT`, `ENTITY`, `RELATION`, `INSTRUCTION`, `CONSTRAINT`, `EVIDENCE`, `REFERENCE`, `DECISION`, `LATENT_TENSOR`, and `KV_COMPAT_DESC`.
- **Decoupled Transport:** In-memory, OS shared-memory (`cmcp_shm_channel.c`), and an optional compile-time-decoupled [Vect-Link](docs/architecture/vect_link_integration.md) design (`CMCP_ENABLE_VECTLINK=OFF` by default).

---

## 2. Repository Structure

```text
Cross-Model-Context-Protocol/
├── core/                   # C99 core library (header, CRC32C, TLV, compat, SHM)
├── representations/        # Python SDK: structured schema, ctypes binary codec, latent projectors
├── adapters/               # Model adapters (Qwen2.5, Llama-3.2) & GGUF registry
├── benchmarks/             # C11 llama.cpp harness & plot/summary aggregator
├── experiments/            # Reproducible scripts for Experiments A–F
├── tests/                  # C99 unit tests (tests/c/) & Python pytest suite
├── results/                # Raw JSON/CSV outputs (results/raw/) & plots (results/plots/)
├── docs/                   # Protocol specs, architecture, methodology, results, roadmap
└── FINAL_RESEARCH_REPORT.md
```

---

## 3. Build & Test

### 3.1 Build C99 Core & C11 `llama.cpp` Harness (Windows MSYS2 Clang64)

```powershell
$env:PATH = "C:\msys64\clang64\bin;" + $env:PATH

# Configure & build C99 core + unit tests (+ optional llama.cpp harness)
cmake -S . -B build -G Ninja `
  -DCMAKE_C_COMPILER=clang `
  -DCMCP_BUILD_TESTS=ON `
  -DCMCP_BUILD_LLAMA_HARNESS=ON `
  -DCMCP_LLAMA_DIR="C:\Users\sgi.lm\Desktop\llama.cpp"

cmake --build build
```

### 3.2 Run Unit Tests

```powershell
# C99 core unit tests (5 test suites)
.\build\test_cmcp_core.exe

# Python SDK & adapter tests (7 tests)
.venv\Scripts\pytest.exe -q
```

Verified output: `7 passed in 1.34s` (`pytest`) and all 5 C99 suites `[PASS]` (`test_cmcp_core.exe`).

---

## 4. Reproduce Experiments (A–F)

```powershell
$env:PATH = "C:\msys64\clang64\bin;C:\Users\sgi.lm\Desktop\llama.cpp\build\bin;" + $env:PATH

.venv\Scripts\python.exe experiments\structured_transfer\run_exp_a_b.py
.venv\Scripts\python.exe experiments\token_reduction\run_exp_c.py
.venv\Scripts\python.exe experiments\latent_alignment\run_exp_d.py
.venv\Scripts\python.exe experiments\state_transfer\run_exp_e.py
.venv\Scripts\python.exe experiments\kv_compatibility\run_exp_f.py
.venv\Scripts\python.exe benchmarks\run_all_benchmarks.py
```

---

## 5. Summary of Observed Results

All values below come directly from [results/raw/](results/raw/benchmark_summary.json) (CPU-only: Intel i5-13500, 15.70 GiB RAM).

| Experiment | Key Comparison | Main Observed Result | Raw Data |
| :--- | :--- | :--- | :--- |
| **Exp A/B** (192 runs) | `nl_conversation` vs `json_structured` vs `cmcp_structured` vs `cmcp_binary` (`Qwen2.5-0.5B` <-> `Llama-3.2-1B`) | `cmcp_structured` had smallest wire size (`274.125 B`), `cmcp_binary` had `0.27–0.35 ms` codec overhead and `1.0` info fidelity, **but `json_structured` achieved higher quality (`0.9083` / `0.8583` vs `0.7438` / `0.7583`) and lower latency (`2653 ms` / `1602 ms`)**. `nl_conversation` was slowest (`4930 ms` / `7021 ms`, includes sender generation). | [exp_ab_results.json](results/raw/exp_ab_results.json) |
| **Exp C** (36 runs) | Multi-turn `full_history` vs `conventional_summary` vs `cmcp_references` vs `cmcp_incremental` | `cmcp_references` achieved **`1.0000` quality** (vs `0.8889` full history, `0.4444` summary) with **-54.6% wire bytes** (`361.67` vs `796.33`) and **-41.7% receiver tokens** (`172.67` vs `296`). | [exp_c_results.json](results/raw/exp_c_results.json) |
| **Exp D** (64/24 split) | Hidden-state alignment (`SmolLM2-135M` -> `Qwen2.5-0.5B`) | Ridge (`mean_pooled`) reached CKA `0.9506` (`0.8789` before), cosine `0.9265`, top-1 `0.8333`; MLP reached top-1 `0.8750`. | [exp_d_latent_alignment.json](results/raw/exp_d_latent_alignment.json) |
| **Exp E** (48/20 split) | Soft-prompt state transfer (`K=8` latents, frozen receiver) | `learned_cmcp` lowered NLL (`2.7108` -> `1.5738`, PPL `15.04` -> `4.82`), **but failed factual exact match (`EM = 0.0`, keyword recall `0.05` vs `0.85` / `0.95` for `full_text`)**. | [exp_e_state_transfer.json](results/raw/exp_e_state_transfer.json) |
| **Exp F** (KV guardrails) | `llama_state_seq_set_data` vs `cmcp_check_kv_compat` | Both accepted `same_model_same_cfg` (`99,008 B`) and rejected `cross_architecture` (`0 B`). `cmcp_check_kv_compat` also caught `n_ctx mismatch` (`512` vs `256`) and `pos_offset mismatch` (`0` vs `16`) that `llama.cpp` accepted. | [exp_f_kv_compat.json](results/raw/exp_f_kv_compat.json) |

---

## 6. Documentation

- [Architecture Overview](docs/architecture/overview.md) | [Design Decisions](docs/architecture/design_decisions.md) | [Vect-Link Integration](docs/architecture/vect_link_integration.md)
- [Protocol Specification](docs/protocol/specification.md) | [Binary Format](docs/protocol/binary_format.md) | [Compatibility](docs/protocol/compatibility.md) | [Security](docs/protocol/security.md)
- [Environment Report](docs/experiments/environment_report.md) | [Methodology](docs/experiments/methodology.md) | [Results](docs/experiments/results.md) | [Limitations](docs/experiments/limitations.md)
- [Roadmap](docs/roadmap.md) | [Final Research Report](FINAL_RESEARCH_REPORT.md)
