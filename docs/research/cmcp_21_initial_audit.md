# CMCP 2.1 Initial Audit

**HEAD:** `7493409` (`main` — `docs: add research, protocol spec, 5-perspective audit, and FINAL_RESEARCH_REPORT`)  
**Audited Hardware:** Windows 10 Pro x64 (`10.0.19045`), Intel Core i5-13500 (`14C/20T`), `15.70 GiB` RAM, Intel UHD Graphics 770 (no discrete GPU), `15.36 GiB` free disk on `C:`, Clang `22.1.8`, CMake `4.4.3`, Ninja `1.13.2`, Python `3.13.5` (`.venv` with `torch 2.15.0+cpu`, `transformers 5.23.0`, `numpy 2.5.5`, `scipy 1.18.1`, `pytest 9.2.0`, `matplotlib 3.11.2`).  
**Local Models:**
- GGUF (`llama.cpp` CPU): `qwen2.5-0.5b-instruct-q4_k_m.gguf` (`468.6 MiB`), `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (`770.3 MiB`)
- HuggingFace (`transformers` CPU): `HuggingFaceTB/SmolLM2-135M-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`

---

## Component Status Classification

| Component | Path | Status | Evidence / Defect Observed |
|---|---|---|---|
| 64B Header & Little-Endian Codec | `core/protocol/cmcp_header.c` | **Implemented & Tested** | `test_cmcp_core` (`1/1`) & `pytest` (`7/7`) pass; bounds & CRC32C verified. |
| TLV & Varint/RLE-LZ Compression | `core/serialization/cmcp_tlv.c` | **Implemented & Tested (Partial Semantics)** | Compiles `-std=c99 -Wall -Wextra -Werror -pedantic`. However, `CMCPContext.to_tlv_records()` embeds JSON strings inside TLV payloads (`schema.py`), not typed binary structs. |
| CRC32C Castagnoli | `core/validation/cmcp_crc32c.c` | **Defective (Thread-Safety)** | Uses non-atomic lazy runtime initialization (`g_crc32c_init`, `g_crc32c_table[256]`) — data race under concurrent first-call threads. Must be compile-time `static const` table + concurrency test. |
| SHM / File / Memory Transport | `core/transport/cmcp_shm_channel.c` | **Partial / Defective** | Windows SHM lacks `ERROR_ALREADY_EXISTS` / DACL isolation and cross-process sync; POSIX `#else` falls back to `calloc()` (process-private, not real POSIX SHM). |
| Model Identity & KV Compat | `core/validation/cmcp_compat.c` | **Partial** | Checks `arch_hash`, `weights_hash`, `tokenizer_hash`, `n_layer`, `n_embd`, `n_head_kv`, `rope_type`, `rope_freq_scale`, `kv_dtype`, `n_ctx`, `pos_offset`. In Python `registry.py`, `weights_hash` is a placeholder/constant rather than file-derived hash; `llama_harness` hashes `model_path` string rather than GGUF header/file metadata. |
| Python SDK & `ctypes` Bindings | `representations/binary/codec.py`, `representations/structured/schema.py` | **Implemented & Tested** | 7/7 `pytest` pass. `CMCP_CTX:` DSL underperformed `json_structured` on 0.5B/1B models (`0.74` vs `0.91`). |
| GGUF Adapters (`llama.cpp`) | `adapters/qwen/`, `adapters/llama/`, `benchmarks/cmcp_llama_harness.c` | **Implemented & Tested** | Real CPU inference via subprocess (`libllama.dll`). Reloads model per subprocess call in `run_exp_a_b.py` / `run_exp_c.py`. |
| Context Transfer Exp A/B/C | `experiments/structured_transfer/`, `experiments/token_reduction/` | **Partial (Methodological Flaws)** | Only 8 tasks (Exp A/B) and 3 scenarios (Exp C); `nl_conversation` & `conventional_summary` include sender generation time while structured conditions use pre-built `CMCPContext` (`sender_ms = 0`); preparation cost not measured. |
| Latent Alignment & State Transfer (Exp D/E) | `experiments/latent_alignment/`, `experiments/state_transfer/` | **Implemented & Executed** | Exp D showed CKA `0.95` / Top-1 `83–87.5%` retrieval; Exp E showed NLL drop (`2.71 -> 1.57`) but `EM = 0.0` (refuted factual transfer via frozen `inputs_embeds`). |
| Cross-Arch KV Transfer (Phase 6) | `experiments/kv_compatibility/run_exp_f.py` | **Pending (Only Compat Check Done)** | Exp F only verified that `llama_state_seq_set_data` rejects cross-arch blobs (`24` vs `16` layers). No K/V tensor extraction, RoPE/GQA inspection, or projected K/V injection via `past_key_values` has been executed yet. |
| Vect-Link Integration | `docs/architecture/vect_link_integration.md` | **Documented Only (Not Implemented)** | Design doc only; `Vect-Link` remains untouched and independent. |
| Leakage Audit Script | — | **Pending** | No automated `experiments/audit/validation_report.json` gate exists. |

---

## Sequential Execution Plan (CMCP 2.1)

1. **Checkpoint 1 (SOTA):** Concise analysis of C2C, K-V Cache Alignment, and Vector Translation (`docs/research/sota_and_baselines.md`).
2. **Checkpoint 5 (Core Audit Fixes):** Fix `cmcp_crc32c.c` compile-time table + multi-thread test, harden `cmcp_shm_channel.c` & `cmcp_compat.c` (real GGUF file/header fingerprinting), add typed binary TLV encoding option, verify CTest + ASan/UBSan + pytest.
3. **Checkpoint 2 (Priority Context & Reference Benchmark):** 30+ synthetic controlled tasks (dev/tune/eval split), equivalent information across baselines A–E, separate transport vs inference + full context-preparation cost, 3+ reps, raw CSV/JSON/logs.
4. **Checkpoint 3 (Latent Transfer):** Offline tensor extraction (`SmolLM2-135M` + `Qwen2.5-0.5B`) across embeddings/mid/last layers, Ridge/Procrustes/low-rank/MLP, negative controls (random/permuted), functional test.
5. **Checkpoint 4 (Cross-Arch KV Transfer):** PyTorch `past_key_values` extraction & experimental K/V projection/injection (`H1`–`H5`) between `SmolLM2-135M` and `Qwen2.5-0.5B` with strict refutation criteria.
6. **End-to-End Comparison + Leakage Audit + `FINAL_RESEARCH_REPORT_V2.md`**.
