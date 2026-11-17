# Phase 11 Independent Audit Report — Cross-Model Context Protocol (CMCP)

**Audit Target:** `C:\Users\sgi.lm\Desktop\Cross-Model-Context-Protocol`  
**External Boundary Verified:** `C:\Users\sgi.lm\Desktop\Vect-Link` was **not modified** and has **zero build-time or runtime coupling** to CMCP.  
**Verification Date:** 2026-11-17  

---

## Executive Summary

The CMCP repository implements a solid, zero-dependency C99 binary framing/TLV/CRC32C/compatibility core (`cmcp.dll`), a clean Python `ctypes` + Pydantic SDK, real `llama.cpp` C11 harness integration (`cmcp_llama_harness.exe`) over local GGUF models, and 6 empirical experiments (Exp A/B, C, D, E, F).

However, an independent five-perspective audit reveals **important gaps between hypotheses/documentation and actual implementation**, **one critical benchmarking asymmetry**, **two C99 integer-overflow/truncation bugs (now fixed and verified)**, and **clear empirical explanations** for why `cmcp_structured` trailed `json_structured` in Exp B and why Exp E lowered NLL without factual transfer.

| Perspective | Verdict | Key Finding |
| :--- | :--- | :--- |
| **1. Architecture** | **Pass with Gaps** | Zero Vect-Link coupling; clean C99/Python boundaries. However, H3 & H5 (`vectlink_adapter.c`, `CMCP_ENABLE_VECTLINK`) are design-doc-only, and `CMCPContext` TLV records embed JSON strings rather than binary fields. |
| **2. Machine Learning** | **Mixed / Rigorous** | Train/eval splits in Exp D (64/24) and Exp E (48/20) are strictly disjoint. H2 holds on SmolLM2→Qwen2.5; H4 fails factual transfer (`0.0` EM in Exp E); `cmcp_structured` loses to JSON in Exp B due to DSL verbosity hitting `max_tokens=96`. |
| **3. Systems (C99)** | **Pass (Fixed 2 Bugs)** | Compiles cleanly under `clang 22.1.8` `-std=c99 -Wall -Wextra -Werror -pedantic` and passes **ASan + UBSan**. Fixed a `DWORD` truncation and `size_t` wrap-around in `cmcp_shm_channel.c` plus defensive bounds guards in `cmcp_tlv.c`. |
| **4. Benchmarking** | **Fairness Defect** | **Unfair comparison:** `nl_conversation` (Exp A/B) and `conventional_summary` (Exp C) include sender LLM generation time (`sender_ms`), whereas `json_structured` / `cmcp_structured` / `cmcp_binary` use pre-built `CMCPContext` objects (`sender_ms = 0`). |
| **5. Skeptical Audit** | **Partially Reproducible** | C99 core, CRC32C, KV-compat checks, and Exp D CKA are genuine and reproducible. H3 and H5 are unimplemented; Exp A/B/C quality relies on substring recall over 8/3 hand-crafted tasks. |

---

## 1. Auditor 1 — Architecture

### 1.1 Vect-Link Independence
- A repository-wide search across `CMakeLists.txt`, `pyproject.toml`, `core/`, `representations/`, `adapters/`, and `tests/` confirms **zero** `#include`s, symbols, or build options referencing `Vect-Link`.
- The only mentions of `vect-link` in code are a string literal inside a test prompt (`experiments/structured_transfer/run_exp_a_b.py#L144-L158`) and the architectural design document (`docs/architecture/vect_link_integration.md`).

### 1.2 Protocol vs. Adapter Boundaries
- **Core Protocol (`core/`):** Self-contained C99 library (`cmcp.dll` / `libcmcp.a`) exposing `cmcp_header_t`, TLV writer/reader, LZ-lite+RLE compression, Castagnoli CRC32C, FNV-1a-64, `cmcp_model_profile_t` compatibility checks, and `cmcp_shm_channel_t`.
- **SDK (`representations/`):** `representations/binary/codec.py` wraps `cmcp.dll` via `ctypes`; `representations/structured/schema.py` defines `CMCPContext`; `representations/latent/projector.py` implements Procrustes, Ridge, MLP, and CKA.
- **Adapters (`adapters/`):** `ModelAdapter` ABC (`adapters/base/adapter.py`) cleanly isolates `LlamaCppSubprocessAdapter`, `QwenAdapter`, and `LlamaAdapter`.

### 1.3 Architectural Gaps & Discrepancies
1. **Unimplemented Design-Doc Interfaces:** `docs/architecture/vect_link_integration.md` specifies `CMCP_ENABLE_VECTLINK`, `include/cmcp/transport.h`, `src/adapters/vectlink_adapter.c`, `cmcp_envelope_t`, and `cmcp_session_t`. None of these exist in `CMakeLists.txt` or `core/`.
2. **TLV Payload Encoding:** `CMCPContext.to_tlv_records()` (`representations/structured/schema.py#L124-L144`) serializes structured entities, relations, constraints, and decisions as **JSON UTF-8 strings** inside each TLV record (`e.model_dump_json(exclude_defaults=True)`). Thus, `cmcp_binary` is a binary framing + LZ-lite/RLE wrapper around per-record JSON rather than a binary field schema.
3. **Disconnected Latent Tags:** `CMCP_TLV_LATENT_TENSOR` (`0x09`) and `CMCP_TLV_KV_COMPAT_DESC` (`0x0A`) are declared in `core/include/cmcp/cmcp.h#L92-L93`, but `CMCPContext` does not serialize them and Experiments D/E bypass the C99 wire format entirely (running in PyTorch).
4. **Hardcoded Local Paths:** `adapters/model_registry/registry.py#L30-L41` hardcodes `C:\Users\sgi.lm\Desktop\models\...` and `adapters/base/adapter.py#L83` hardcodes `C:\msys64\clang64\bin;C:\Users\sgi.lm\Desktop\llama.cpp\build\bin;`.

---

## 2. Auditor 2 — Machine Learning

### 2.1 Validity of Hypotheses H1–H5 (`docs/research/technical_hypotheses.md`)
- **H1 (Protocol Overhead & Compatibility):** **Partially Supported.** In Exp A/B, `cmcp_binary` `codec_ms` mean is `0.3548 ms` (Qwen→Llama) and `0.2736 ms` (Llama→Qwen) — `< 0.1%` of receiver prefill+eval latency — with 100% CRC32C rejection in tests and 100% mismatch rejection in Exp F. However, the `< 50 µs` header microbenchmark across 1 KiB–16 MiB was not run in isolation. Note also that `cmcp_check_kv_compat` is intentionally stricter than `llama.cpp`: in Exp F (`results/raw/exp_f_kv_compat.json`), `llama_state_seq_set_data` accepted `n_ctx=512 → 256` (`99008` bytes), whereas CMCP flagged `CMCP_COMPAT_MISMATCH_CTX`.
- **H2 (Cross-Model Structural Alignment):** **Supported (with caveats).** On `SmolLM2-135M-Instruct` (`d=576`) → `Qwen2.5-0.5B-Instruct` (`d=896`), `results/raw/exp_d_latent_alignment.json` shows raw Linear CKA of `0.8789` (mean-pooled) and `0.9333` (last-token); Ridge raises CKA to `0.9506` / `0.9752` and achieves `0.8333` Top-1 (`0.9583` Top-5) retrieval on mean-pooled vs. `0.0417` Top-1 (`0.1667` Top-5) random. Caveats: small eval set (`n_eval=24`), no `p < 0.01` shuffled-anchor null test, and tested on SmolLM2→Qwen2.5 rather than Qwen2.5↔Llama-3.2.
- **H3 (Cross-Tokenizer Byte-Span Alignment):** **Not Implemented / Untested.** No code or experiment tests UTF-8 byte-span alignment or MinED fallback.
- **H4 (Latent Projection vs. Text Handoff):** **Falsified / Not Achieved.** In Exp E (`results/raw/exp_e_state_transfer.json`), `learned_cmcp` achieved `0.0` exact match and `0.05` keyword recall (identical to `zero_context` and `random_projection`), compared to `0.85` EM and `0.95` keyword recall for `full_text`.
- **H5 (Decoupled Vect-Link Transport):** **Not Implemented / Untested.**

### 2.2 Train/Eval Leakage Checks (Exp D & Exp E)
- **Exp D (`experiments/latent_alignment/run_exp_d.py#L190-L192`):** Enforces `assert set(TRAIN_CONTEXTS).isdisjoint(set(EVAL_CONTEXTS))` across 64 train and 24 eval contexts. Manual inspection confirms distinct topics, algorithms, and code snippets.
- **Exp E (`experiments/state_transfer/run_exp_e.py#L336-L338`):** Enforces `assert {e.context for e in TRAIN_DATA}.isdisjoint({e.context for e in EVAL_DATA})` across 48 train and 20 eval QA pairs. Entities, numbers, and domains are disjoint.

### 2.3 Why `cmcp_structured` Scored Lower Than `json_structured` in Exp B
In `results/raw/exp_ab_results.json`:
- **Qwen2.5-0.5B → Llama-3.2-1B:** `cmcp_structured` mean `quality_score = 0.7438` vs. `json_structured = 0.9083`.
- **Llama-3.2-1B → Qwen2.5-0.5B:** `cmcp_structured` mean `quality_score = 0.7583` vs. `json_structured = 0.8583`.

Inspection of per-task outputs in `exp_ab_results.json` shows the root cause:
1. **Out-of-distribution DSL triggers verbose scaffolding:** `CMCPContext.to_wire_prompt()` renders a custom line format (`CMCP_CTX:\nTASK:...\nENT:...\nCON:...\nDEC[...]=...`). Small 0.5B–1B instruction-tuned models are heavily pre-trained on JSON, so `json_structured` produces direct answers (`mean eval_ms = 2076.14 ms` on Llama-3.2-1B), whereas `cmcp_structured` triggers verbose markdown headers, step-by-step preambles, or raw `CMCP_CTX:` echoing (`T1` on Qwen2.5-0.5B).
2. **`max_tokens=96` truncation:** On Qwen→Llama, `cmcp_structured` `eval_ms` jumps to `3000.28 ms` mean (`3496.18 ms` median, hitting the 96-token generation cap). The verbose preamble exhausts `max_tokens=96` before later `expected_items` are emitted (e.g., `T1`, `T4`, `T5`, `T6`).

### 2.4 Why Exp E Lowered NLL (`2.7108 → 1.5738`) Without Factual Transfer (`0.0` EM)
In `results/raw/exp_e_state_transfer.json`:
- `zero_context`: `mean_nll = 2.7108`, `PPL = 15.0412`, `EM = 0.0`, `KW = 0.05`
- `random_projection`: `mean_nll = 2.5347`, `PPL = 12.613`, `EM = 0.0`, `KW = 0.05`
- `learned_cmcp`: `mean_nll = 1.5738`, `PPL = 4.8249`, `EM = 0.0`, `KW = 0.05`
- `full_text`: `mean_nll = 0.3717`, `PPL = 1.4503`, `EM = 0.85`, `KW = 0.95`

**Root Cause:** `SoftPromptAdapter` (`576 → 576 → 896`, `K=8` chunk-pooled last-layer vectors) was trained for 24 steps (`batch_size=8`) on 48 synthetic single-fact sentences using cross-entropy loss on answer tokens (`" 8443."`, `" 14 days."`, etc.). With only 48 training examples and a frozen receiver, the adapter learned **answer-format & prior calibration** (shifting the receiver's output distribution toward short numeric/unit completions, which lowers token NLL), **not** entity-specific factual binding. Arbitrary unseen entities and numbers (`Beacon-11` → `437.5`, `Obsidian` → `14`) cannot be decoded by a frozen Qwen2.5-0.5B input embedding layer from 8 mean-pooled SmolLM2-135M vectors without joint pretraining or explicit token-level alignment.

---

## 3. Auditor 3 — Systems (C99 Core)

### 3.1 Compiler & Sanitizer Verification
- **Strict C99 Build:** `clang 22.1.8` (`x86_64-w64-windows-gnu`) with `-std=c99 -Wall -Wextra -Werror -pedantic` compiled all 5 core C files and `tests/c/test_cmcp_core.c` with **zero warnings**.
- **ASan + UBSan Run:** Compiled and executed with `-g -O1 -fsanitize=address,undefined` — **zero memory errors, zero undefined behavior**, all 5 C99 tests (`test_crc32c`, `test_header_roundtrip_and_corruption`, `test_tlv_and_compression`, `test_compat_and_kv_validation`, `test_shm_channel`) passed.

### 3.2 Memory Safety, Endianness, and Bounds Review
- **Endianness (`cmcp_header.c`):** Uses explicit byte-by-byte little-endian helpers (`write_u16_le`, `read_u64_le`, etc.) on `uint8_t *` — immune to host endianness and unaligned-access UB.
- **Varint & Decompression (`cmcp_tlv.c`):** `uleb128_decode_u64` guards against shift >= 64 (`shift >= 64u || (shift == 63u && (byte & 0x7Fu) > 1u)`). `cmcp_decompress` validates `raw_len <= dst_cap`, literal/RLE/match bounds, `dist > 0 && dist <= out_pos`, and exact `out_pos == raw_len && in_pos == src_len`.

### 3.3 Defects Found and Fixed
1. **Integer Truncation in `core/transport/cmcp_shm_channel.c` (`channel_alloc_common`):**  
   `DWORD total_size = (DWORD)(capacity_bytes + 4u);` could silently truncate `size_t` to 32-bit `DWORD` on 64-bit Windows if `capacity_bytes > 0xFFFFFFFFu - 4u`, allocating a smaller SHM mapping than `chan->capacity`.  
   **Fix:** Added `if (capacity_bytes > (size_t)0xFFFFFFFFu - 4u) return CMCP_ERR_LIMIT_EXCEEDED;`.
2. **Integer Wrap-Around in `core/transport/cmcp_shm_channel.c` (`cmcp_shm_channel_send_frame`):**  
   `total_len = CMCP_HEADER_SIZE + payload_len;` could wrap `size_t` if `payload_len` was near `SIZE_MAX`.  
   **Fix:** Added `if (payload_len > CMCP_MAX_PAYLOAD_BYTES) return CMCP_ERR_LIMIT_EXCEEDED;` and `if (payload_len > chan->capacity - CMCP_HEADER_SIZE) return CMCP_ERR_BUFFER_TOO_SMALL;` before addition.
3. **Defensive Cursor Guards in `core/serialization/cmcp_tlv.c`:**  
   Added `w->pos > w->cap` in `cmcp_tlv_write` and `r->pos > r->len` in `cmcp_tlv_next` before unsigned subtraction.
4. **Regression Tests:** Added overflow/limit checks to `tests/c/test_cmcp_core.c` and re-verified CTest (`1/1 passed`), ASan+UBSan (`All CMCP core C99 tests passed`), and `pytest` (`7/7 passed`).

### 3.4 Remaining Non-Critical Systems Observations
- `cmcp_crc32c.c`: Lazy `g_crc32c_table` initialization (`g_crc32c_init`) is not thread-safe under concurrent first calls.
- `cmcp_shm_channel.c`: `CMCP_CHANNEL_SHM` has no inter-process synchronization (mutex/event/seqlock), so concurrent multi-process read/write on the same named SHM mapping is unsafe without external synchronization.
- `cmcp_llama_harness.c`: Uses `atoi()` without range checks for CLI flags and unchecked `realloc` on `out_text`.

---

## 4. Auditor 4 — Benchmarking

### 4.1 Unfair Comparison Call-Out (Sender Generation Asymmetry)
- **Exp A/B (`experiments/structured_transfer/run_exp_a_b.py#L346-L386`):**  
  `nl_conversation` runs `sender.generate(handoff_prompt, max_tokens=128)` and includes `sender_ms` (`2136.5 ms` mean on Qwen→Llama; `5254.4 ms` on Llama→Qwen) in `total_latency_ms`. By contrast, `json_structured`, `cmcp_structured`, and `cmcp_binary` serialize **pre-authored `task.cmcp_ctx` objects** with `sender_gen_tokens = 0` and `sender_ms = 0.0`.
- **Exp C (`experiments/token_reduction/run_exp_c.py#L178-L205`):**  
  `conventional_summary` runs `qwen.generate(sum_prompt, max_tokens=120)`, whereas `cmcp_references` and `cmcp_incremental` use pre-authored `CMCPContext` objects (`sender_tokens = 0`, `sender_ms = 0.0`).
- **Implication:** End-to-end `total_latency_ms` comparisons between `nl_conversation`/`conventional_summary` and the structured/binary conditions are **not apples-to-apples**; they compare *sender LLM summarization + receiver inference* against *pre-structured serialization + receiver inference*. Fair comparison should isolate **receiver-only latency** (`tokenize_ms + prefill_ms + eval_ms + codec_ms`) or include an LLM-based structured-extraction cost for the sender.

### 4.2 Subprocess Overhead, Warm-Up, Variance, and Sample Sizes
- **Subprocess Reloading:** `LlamaCppSubprocessAdapter` spawns a fresh `cmcp_llama_harness.exe` per `generate()` call, reloading the GGUF model every time. In-process `tokenize_ms + prefill_ms + eval_ms` are measured inside C, excluding process startup/model load, whereas `wall_ms` includes it.
- **Warm-Up:** Only 1 warm-up call per model (`max_tokens=4`) is run before the benchmark loop.
- **Sample Sizes & Variance:** Exp A/B has `n=24` per condition/direction (8 tasks × 3 reps) with high latency variance (`std` up to `899.19 ms` due to variable generation length); Exp C has `n=9` (3 scenarios × 3 reps); Exp D has `n_eval=24`; Exp E has `n_eval=20`. No confidence intervals or significance tests are reported.

### 4.3 RSS Measurement Semantics
- `get_rss_mib()` (`benchmarks/cmcp_llama_harness.c#L24-L32`) reads Windows `PROCESS_MEMORY_COUNTERS.WorkingSetSize` (current working set at process exit, **not** `PeakWorkingSetSize`).
- In Exp A/B, `max_rss = max(sender_rss, r_res.rss_mib)` reports `~1317 MiB` (dominated by `Llama-3.2-1B-Instruct-Q4_K_M.gguf` memory-mapped pages) vs. `~475 MiB` when `Qwen2.5-0.5B` is the sole model (`Llama→Qwen` structured conditions).

---

## 5. Auditor 5 — Skeptical Auditor

### 5.1 What Holds Up Under Scrutiny
1. **C99 Core & Integrity:** 64-byte header, Castagnoli CRC32C, ULEB128 TLV, LZ-lite+RLE compression, and 4-tier compatibility checks are real, tested, and ASan/UBSan-clean.
2. **Real GGUF / `llama.cpp` Execution:** `cmcp_llama_harness.exe` executes real local GGUF inference and `llama_state_seq_get_data`/`set_data` KV-cache checks (`exp_f_kv_compat.json`), proving that `cmcp_check_kv_compat` catches `n_ctx` mismatches that `llama.cpp` itself accepts.
3. **Exp D Latent Geometry:** Real hidden states from `SmolLM2-135M` and `Qwen2.5-0.5B` on a disjoint 64/24 split show genuine linear CKA (`0.8789` raw, `0.9506` Ridge) and `0.8333` Top-1 retrieval.

### 5.2 What Is Overstated, Brittle, or Unreproduced
1. **Unimplemented Hypotheses:** H3 (byte-span tokenizer alignment) and H5 (`cmcp_vtlk_adapter`) have no implementation or test results.
2. **Brittle Quality Metric:** `score_response()` (`run_exp_a_b.py#L275-L284`) is case-insensitive substring matching on 8 tasks (Exp A/B) and 3 scenarios (Exp C). Moreover, `cmcp_structured` underperformed `json_structured` (`0.7438` vs. `0.9083`).
3. **No End-to-End Latent Transfer:** Exp E demonstrates `0.0` EM factual transfer; NLL reduction (`2.71 → 1.57`) reflects format calibration, not semantic context transfer.

### 5.3 Reproducibility Commands (Verified)
```powershell
# 1. Build C99 core & run CTest (from repo root)
$env:PATH = "C:\msys64\clang64\bin;" + $env:PATH
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure

# 2. Run ASan + UBSan on C99 core
clang -std=c99 -Wall -Wextra -Werror -pedantic -g -O1 "-fsanitize=address,undefined" `
  -Icore/include core/validation/cmcp_crc32c.c core/protocol/cmcp_header.c `
  core/serialization/cmcp_tlv.c core/validation/cmcp_compat.c `
  core/transport/cmcp_shm_channel.c tests/c/test_cmcp_core.c -o build/test_cmcp_asan.exe
.\build\test_cmcp_asan.exe

# 3. Run Python protocol & live adapter tests
.\.venv\Scripts\python.exe -m pytest -v

# 4. Reproduce experiments (requires local GGUFs & HF cache)
.\.venv\Scripts\python.exe experiments/structured_transfer/run_exp_a_b.py
.\.venv\Scripts\python.exe experiments/token_reduction/run_exp_c.py
.\.venv\Scripts\python.exe experiments/latent_alignment/run_exp_d.py
.\.venv\Scripts\python.exe experiments/state_transfer/run_exp_e.py
.\.venv\Scripts\python.exe experiments/kv_compatibility/run_exp_f.py
.\.venv\Scripts\python.exe benchmarks/run_all_benchmarks.py
```
