# CMCP v1 Protocol Specification

**Status:** Draft Standard (v1.0)  
**Scope:** Cross-Model Context Protocol (CMCP) control plane, envelope framing, compatibility negotiation, and transport abstraction.

---

## 1. Design Goals

The Cross-Model Context Protocol (CMCP) enables heterogeneous LLM agents (varying in architecture, parameter count, quantization, and tokenizer) to exchange context safely and efficiently over local IPC or network links:

1. **Zero Silent Corruption:** Every CMCP frame carries a fixed 64-byte little-endian header protected by Castagnoli CRC32C (`header_crc32c` and `payload_crc32c`), plus explicit model and tokenizer identity hashes.
2. **Tiered Compatibility Negotiation:** Agents negotiate the highest-fidelity representation supported by both sender and receiver, from zero-copy KV-cache snapshots (Tier 0) down to portable structured/text context (Tier 3).
3. **Zero External Dependencies:** The core C99 library (`libcmcp`) requires only the C99 standard library (`stdint.h`, `stddef.h`, `string.h`) and optional OS shared-memory primitives.
4. **Decoupled Transport:** CMCP defines a byte-oriented envelope and channel interface that operates over in-memory buffers, files, OS shared memory, or optionally [Vect-Link](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/architecture/vect_link_integration.md) when enabled at build time.

---

## 2. Protocol Architecture

```text
+-------------------------------------------------------------------+
| Application / Agent Runtime (llama.cpp, vLLM, Python, C/C++)      |
+-------------------------------------------------------------------+
| CMCP Negotiation & Validation (cmcp_check_compat / kv_compat)     |
+-------------------------------------------------------------------+
| CMCP Serialization (TLV Records + Varint/RLE/LZ-lite Compression) |
+-------------------------------------------------------------------+
| CMCP Framing (64-Byte Fixed LE Envelope + Dual CRC32C)            |
+-------------------------------------------------------------------+
| Transport (In-Memory / File / OS SHM / Optional Vect-Link)        |
+-------------------------------------------------------------------+
```

---

## 3. Message Types

| Value | Name | Direction | Purpose |
| :--- | :--- | :--- | :--- |
| `0x0001` | `CMCP_MSG_HELLO` | Bidirectional | Capability advertisement (`cmcp_model_profile_t` / `KV_COMPAT_DESC`). |
| `0x0002` | `CMCP_MSG_CONTEXT_TRANSFER` | Sender -> Receiver | Structured TLV context payload (entities, instructions, evidence, etc.). |
| `0x0003` | `CMCP_MSG_KV_SNAPSHOT` | Sender -> Receiver | Tier 0 raw KV-cache state (e.g. `llama_state_seq_get_data` or handle ref). |
| `0x0004` | `CMCP_MSG_LATENT_TRANSFER` | Sender -> Receiver | Tier 1/2 projected hidden-state or anchor-relative tensor payload. |
| `0x0005` | `CMCP_MSG_ACK` | Receiver -> Sender | Positive acknowledgment of `sequence_id` and negotiated `compat_tier`. |
| `0x0006` | `CMCP_MSG_REJECT` | Receiver -> Sender | Rejection of incompatible or corrupted frame with reason code. |

---

## 4. Compatibility Tiers

| Tier | Constant | Meaning |
| :--- | :--- | :--- |
| **0** | `CMCP_TIER_0_EXACT_KV` | Identical architecture, weights, tokenizer, KV geometry, RoPE config, dtype, and `pos_offset`. Direct KV-cache transfer is valid. |
| **1** | `CMCP_TIER_1_SAME_FAMILY` | Same model family / architecture & embedding geometry, but different weights, quantization, or `pos_offset`. Requires re-prefill or affine latent projection. |
| **2** | `CMCP_TIER_2_CROSS_MODEL` | Different model families/architectures. Uses cross-model anchor-relative / ridge-projected latent vectors or byte-span-aligned structured context. |
| **3** | `CMCP_TIER_3_TEXT_FALLBACK` | Universal fallback carrying compressed structured TLV text (`INSTRUCTION`, `EVIDENCE`, `CONCEPT`, etc.). |
| **65535** | `CMCP_TIER_INCOMPATIBLE` | Hard rejection (e.g., strict KV mode requested on mismatched models). |

See [compatibility.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/protocol/compatibility.md) for the exact decision matrix.

---

## 5. Session & Handshake Lifecycle

1. **Discovery (`HELLO`):** Sender and receiver exchange `cmcp_model_profile_t` descriptors (via `CMCP_TLV_KV_COMPAT_DESC`).
2. **Negotiation:** Both sides evaluate `cmcp_check_compat(&src, &dst, &res)` to agree on `compat_tier`.
3. **Transfer (`CONTEXT_TRANSFER` / `KV_SNAPSHOT` / `LATENT_TRANSFER`):**
   - Sender builds payload (raw KV bytes or TLV records).
   - If `CMCP_FLAG_COMPRESSED` is set, sender compresses payload with `cmcp_compress()`.
   - Sender encodes the 64-byte header with `sender_model_hash`, `receiver_model_hash`, `tokenizer_hash`, `payload_bytes`, `raw_bytes`, `sequence_id`, `payload_crc32c`, and `header_crc32c`.
4. **Verification & Ingestion:**
   - Receiver validates `magic == "CMCP"`, `version == 1`, and `header_crc32c`.
   - Receiver checks `payload_bytes <= max_allowed` and verifies `payload_crc32c`.
   - Receiver verifies `compat_tier` and `receiver_model_hash` / `tokenizer_hash` against local profile.
   - If `CMCP_FLAG_COMPRESSED` is set, receiver decompresses `payload_bytes` -> `raw_bytes` and verifies exact decompressed length.
   - Receiver parses TLV records or restores KV snapshot, then replies with `CMCP_MSG_ACK` (or `CMCP_MSG_REJECT` on failure).

---

## 6. Error Codes

| Code | Name | Meaning |
| :--- | :--- | :--- |
| `0` | `CMCP_OK` | Operation succeeded. |
| `-1` | `CMCP_ERR_INVALID_ARG` | Null pointer or invalid parameter. |
| `-2` | `CMCP_ERR_BUFFER_TOO_SMALL` | Caller-provided output buffer is insufficient. |
| `-3` | `CMCP_ERR_BAD_MAGIC` | Header magic is not `CMCP`. |
| `-4` | `CMCP_ERR_UNSUPPORTED_VERSION` | Protocol `version != 1`. |
| `-5` | `CMCP_ERR_HEADER_CRC` | `header_crc32c` mismatch. |
| `-6` | `CMCP_ERR_PAYLOAD_CRC` | `payload_crc32c` mismatch. |
| `-7` | `CMCP_ERR_CORRUPT_DATA` | Malformed TLV or compressed stream. |
| `-8` | `CMCP_ERR_INCOMPATIBLE` | Model or KV-cache compatibility check failed. |
| `-9` | `CMCP_ERR_IO` | Transport/channel I/O or OS SHM error. |
| `-10` | `CMCP_ERR_LIMIT_EXCEEDED` | Payload exceeds configured security/size limit. |
