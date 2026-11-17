# CMCP Architecture & Design Decisions

This document records the core engineering and scientific design decisions made in CMCP, their rationale, and what empirical experiments validated or challenged them.

---

## 1. Zero-Dependency C99 Core + Strict Compiler Flags

- **Decision:** Implement the protocol core (`core/`) in strict C99 (`C_STANDARD 99`, `C_EXTENSIONS OFF`, `-Wall -Wextra -Werror -pedantic` on Clang/GCC, `/W4 /WX` on MSVC) with no third-party dependencies.
- **Rationale:** Enables embedding CMCP into C/C++ inference engines (`llama.cpp`, `Vect-Link`) and exposing a stable C ABI (`cmcp.dll`) to Python via `ctypes` without dependency conflicts.
- **Validation:** Compiles cleanly on MSYS2 Clang `22.1.8` and passes all 5 C99 unit test suites in [test_cmcp_core.c](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/tests/c/test_cmcp_core.c) and all 7 Python FFI/adapter tests in [test_protocol_and_adapters.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/tests/test_protocol_and_adapters.py).

---

## 2. Fixed 64-Byte Little-Endian Envelope with Dual Castagnoli CRC32C

- **Decision:** Frame every CMCP message with a 64-byte fixed little-endian header containing `header_crc32c` (covering bytes `0..59`) and `payload_crc32c` (covering the wire payload), alongside `sender_model_hash`, `receiver_model_hash`, and `tokenizer_hash` (FNV-1a 64-bit).
- **Rationale:** Guarantees zero silent corruption across memory, file, or shared-memory boundaries and allows rejecting malformed headers before allocating or decompressing `payload_bytes`.
- **Measured Overhead:** In Experiment A/B, binary encode/decode (`codec_ms`) averaged **0.3548 ms** (`Qwen2.5-0.5B->Llama-3.2-1B`) and **0.2736 ms** (`Llama-3.2-1B->Qwen2.5-0.5B`) end-to-end from Python via `ctypes`, representing `< 0.02%` of total latency (`1925.95 ms`–`3560.51 ms`).

---

## 3. Stricter KV-Cache Guardrails Than `llama.cpp` (`cmcp_check_kv_compat`)

- **Decision:** Require exact matches on `arch`, `weights_hash`, `tokenizer_hash`, `n_layer`, `n_embd`, `n_head_kv`, `rope_type`, `rope_freq_scale`, `kv_dtype`, **`n_ctx`**, and **`pos_offset`** for `CMCP_TIER_0_EXACT_KV`.
- **Rationale:** During Experiment F ([exp_f_kv_compat.json](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_f_kv_compat.json)), `llama_state_seq_set_data` in `llama.cpp` rejected cross-architecture injection (`qwen2` -> `llama`, returning `0` bytes), **but accepted** a `99,008`-byte KV state from `n_ctx=512` into `n_ctx=256` on the same model. `cmcp_check_kv_compat` explicitly caught and rejected both `n_ctx mismatch` (reason code `7`) and `pos_offset mismatch` (reason code `8`), preventing silent context/positional misalignment.

---

## 4. 4-Tier Compatibility Negotiation & Decoupled Vect-Link Boundary

- **Decision:** Negotiate capabilities across four tiers (`Tier 0 Exact KV`, `Tier 1 Same-Family`, `Tier 2 Cross-Model`, `Tier 3 Text Fallback`) and keep `Vect-Link` strictly optional (`CMCP_ENABLE_VECTLINK=OFF` default).
- **Rationale:** `Vect-Link` enforces strict model identity for bit-exact zero-copy KV sharing (`max_logit_diff = 0.000000`). CMCP complements `Vect-Link` by handling heterogeneous model pairs (Tiers 1–3) while delegating Tier 0 to `Vect-Link` only when an optional adapter is enabled.

---

## 5. Structured Context Representation vs. Compact `CMCP_CTX` Rendering

- **Decision:** Model structured context as typed records (`Entity`, `Relation`, `Constraint`, `Decision`, `Evidence`, `Reference`, `Instruction`) that serialize to TLV for binary transport and render to a compact `CMCP_CTX` prompt block for LLM ingestion.
- **Empirical Lesson (Honest Tradeoff):**
  - In **Experiment A/B**, `cmcp_structured` achieved the smallest wire size (**274.125 B** vs. **371.75 B** for `json_structured`) and `info_fidelity_score = 1.0`, **but scored lower on downstream `quality_score`** (`0.7438` vs. `0.9083` on `Qwen2.5-0.5B->Llama-3.2-1B`; `0.7583` vs. `0.8583` on `Llama-3.2-1B->Qwen2.5-0.5B`). Small 0.5B/1B models parsed standard JSON more reliably than the custom `CMCP_CTX` notation.
  - Conversely, in **Experiment C** (multi-turn context with `Reference` pointers), `cmcp_references` achieved **1.0000** `quality_score_mean` while cutting wire bytes by **54.6%** (`361.67 B` vs. `796.33 B`) and receiver input tokens by **41.7%** (`172.67` vs. `296`).
- **Implication:** Typed `Reference` and incremental delta semantics provide strong token/quality benefits in multi-turn workflows, while single-turn prompt rendering for small LLMs should favor JSON-aligned or schema-familiar formatting over terse custom syntax.

---

## 6. CPU-Feasible Latent & State Transfer Choices (Exp D & Exp E)

- **Decision:** Evaluate closed-form (`Procrustes`, `Ridge`) and lightweight (`2-layer MLP`) projectors for hidden-state alignment (Exp D), and a lightweight adapter mapping `K=8` chunk-pooled latents into frozen receiver `inputs_embeds` (Exp E).
- **Rationale:** With no discrete GPU (`i5-13500` CPU, `15.70 GiB` RAM), large-scale end-to-end projector pretraining (as in C2C) is infeasible.
- **Empirical Lesson:** Linear/ridge alignment recovered high geometric similarity (CKA up to `0.9752`, top-1 retrieval `0.8333`–`0.8750` in Exp D), and the soft-prompt adapter lowered receiver NLL from `2.7108` to `1.5738` (Exp E), **but failed factual exact match (`EM = 0.0`)**. Direct latent state transfer cannot replace explicit symbolic/structured context for exact factual handoff at this scale without substantially larger training corpora or deeper KV/LoRA adaptation.
