# CMCP Architecture Overview

The **Cross-Model Context Protocol (CMCP)** is a layered, architecture-independent protocol and SDK for exchanging context between heterogeneous Large Language Models (LLMs) that may differ in architecture, parameter count, quantization, and tokenizer.

---

## 1. Layered Architecture

```text
+-------------------------------------------------------------------------+
| 1. Experiment & Benchmark Layer (experiments/, benchmarks/)             |
|    - Exp A/B (Structured Transfer), Exp C (Token Reduction),            |
|      Exp D (Latent Alignment), Exp E (Soft-Prompt State Transfer),      |
|      Exp F (KV-Cache Compatibility), C11 llama.cpp Harness              |
+-------------------------------------------------------------------------+
| 2. Model Adapters & Registry (adapters/)                                |
|    - ModelAdapter, LlamaCppSubprocessAdapter, QwenAdapter, LlamaAdapter |
|    - MODEL_PROFILES (Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct)      |
+-------------------------------------------------------------------------+
| 3. Representations & Codecs (representations/)                          |
|    - Structured: CMCPContext (Entity, Relation, Constraint, Decision,   |
|      Evidence, Reference, Instruction)                                  |
|    - Binary: Python ctypes bindings to cmcp.dll (encode/decode/compat)  |
|    - Latent: LatentProjector (Procrustes, Ridge, MLP, Random) + CKA     |
+-------------------------------------------------------------------------+
| 4. CMCP C99 Core Library (core/)                                        |
|    - Protocol Framing: 64-byte LE header (cmcp_header.c)                |
|    - Integrity: Castagnoli CRC32C (cmcp_crc32c.c)                       |
|    - Serialization: TLV + Varint/RLE/LZ-lite (cmcp_tlv.c)               |
|    - Negotiation: 4-tier compat & KV guardrails (cmcp_compat.c)         |
|    - Transport: OS SHM channel (cmcp_shm_channel.c)                     |
+-------------------------------------------------------------------------+
| 5. Optional External Integration (Decoupled)                            |
|    - Vect-Link SHM arena adapter (design in vect_link_integration.md;   |
|      CMCP_ENABLE_VECTLINK=OFF by default, zero build coupling)          |
+-------------------------------------------------------------------------+
```

---

## 2. Repository Components

### 2.1 C99 Core Library (`core/`)
Built with CMake (`CMakeLists.txt`) into both static (`libcmcp.a`) and shared (`cmcp.dll` / `libcmcp.so`) libraries under strict C99 (`-Wall -Wextra -Werror -pedantic` / `/W4 /WX`) with zero external dependencies:
- **Public API:** [cmcp.h](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/core/include/cmcp/cmcp.h)
- **Header Framing:** [cmcp_header.c](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/core/protocol/cmcp_header.c) — 64-byte fixed little-endian envelope with magic `"CMCP"`, protocol version `1`, `compat_tier`, 64-bit FNV-1a model/tokenizer hashes, `sequence_id`, `payload_bytes`, `raw_bytes`, and dual CRC32C checksums.
- **CRC32C Integrity:** [cmcp_crc32c.c](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/core/validation/cmcp_crc32c.c) — Castagnoli CRC32C over header bytes `0..59` (`header_crc32c`) and payload (`payload_crc32c`).
- **TLV & Compression:** [cmcp_tlv.c](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/core/serialization/cmcp_tlv.c) — Typed Length-Value records (`CONCEPT`, `ENTITY`, `RELATION`, `INSTRUCTION`, `CONSTRAINT`, `EVIDENCE`, `REFERENCE`, `DECISION`, `LATENT_TENSOR`, `KV_COMPAT_DESC`) and lossless compression (`cmcp_compress` / `cmcp_decompress`).
- **Compatibility & KV Guardrails:** [cmcp_compat.c](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/core/validation/cmcp_compat.c) — 4-tier negotiation (`cmcp_check_compat`) and strict KV-cache validation (`cmcp_check_kv_compat`).
- **Shared Memory Channel:** [cmcp_shm_channel.c](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/core/transport/cmcp_shm_channel.c) — Cross-platform OS shared-memory ring/channel transport.

### 2.2 C11 `llama.cpp` Harness (`benchmarks/cmcp_llama_harness.c`)
Optional CMake target (`-DCMCP_BUILD_LLAMA_HARNESS=ON -DCMCP_LLAMA_DIR=...`) linking `cmcp` against `libllama.dll` to:
1. Run deterministic single-process inference with detailed telemetry (`tokenize_ms`, `prefill_ms`, `eval_ms`, `rss_mib`, token counts).
2. Execute real `llama_state_seq_get_data` / `llama_state_seq_set_data` KV-cache transfer checks (`--mode kv-compat`) alongside `cmcp_check_kv_compat`.

### 2.3 Python SDK & Adapters (`representations/`, `adapters/`)
- **Structured Context:** [schema.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/representations/structured/schema.py) — Pydantic-validated `CMCPContext` with TLV conversion and compact `CMCP_CTX` prompt rendering.
- **Binary Codec:** [codec.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/representations/binary/codec.py) — `ctypes` bindings to `cmcp.dll` for framing, TLV serialization, and compatibility checks.
- **Latent Alignment:** [projector.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/representations/latent/projector.py) — Procrustes, Ridge Regression, 2-layer MLP, and Random projectors plus linear CKA, cosine similarity, and MSE metrics.
- **Model Adapters:** [adapter.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/adapters/base/adapter.py), [qwen/adapter.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/adapters/qwen/adapter.py), [llama/adapter.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/adapters/llama/adapter.py), and [registry.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/adapters/model_registry/registry.py).

---

## 3. Related Protocol & Integration Documents

- [specification.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/protocol/specification.md) — Protocol lifecycle, message types, tiers, and error codes.
- [binary_format.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/protocol/binary_format.md) — Exact 64-byte header layout and TLV wire format.
- [compatibility.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/protocol/compatibility.md) — Compatibility decision matrix and reason codes.
- [security.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/protocol/security.md) — Threat model, integrity checks, and bounds enforcement.
- [vect_link_integration.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/architecture/vect_link_integration.md) — Decoupled optional integration design for `Vect-Link`.
