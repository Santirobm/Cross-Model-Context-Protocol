# CMCP Security & Integrity Model

## 1. Threat Model

CMCP is designed for local IPC (shared memory, pipes, files) and trusted/semi-trusted multi-agent RPC links. It protects against:
1. **Accidental Cross-Model Tensor Corruption:** Loading KV caches or latent tensors from an incompatible model architecture, quantization, or tokenizer.
2. **Transport Corruption / Truncation:** Partial writes, torn SHM reads, or bit-flips in transit.
3. **Malformed Input / Parser Attacks:** Integer overflows, out-of-bounds reads, zip-bombs (`raw_bytes` vs `payload_bytes` spoofing), and truncated TLV records.

> **Note on Confidentiality:** CMCP v1 focuses on **integrity and compatibility**. When crossing untrusted networks, CMCP frames should be transported over TLS/mTLS or encrypted at the transport layer.

---

## 2. Integrity Verification

- **Header Integrity (`header_crc32c`):** Covers bytes `[0..59]` of the 64-byte header, including `payload_crc32c`, `payload_bytes`, `raw_bytes`, and all model/tokenizer hashes. Any single-bit header corruption is detected before payload allocation or parsing.
- **Payload Integrity (`payload_crc32c`):** Covers `payload[0..payload_bytes-1]`. Verified before decompression or TLV parsing.

---

## 3. Parser & Decompression Safety Guarantees

1. **Bounded Allocations:** `CMCP_MAX_PAYLOAD_BYTES` (default 64 MiB) caps `payload_bytes` and `raw_bytes`. Frames exceeding this limit are rejected with `CMCP_ERR_LIMIT_EXCEEDED` before allocation.
2. **Decompression Bomb Protection:** `cmcp_decompress` requires the caller-supplied `dst_cap` and verifies `stream_raw_len == header.raw_bytes <= dst_cap`. Any attempt to write past `dst_cap` or read past `src_len` aborts immediately with `CMCP_ERR_CORRUPT_DATA` or `CMCP_ERR_BUFFER_TOO_SMALL`.
3. **Zero-Trust TLV Iteration:** `cmcp_tlv_next` validates that the ULEB128 varint terminates within 5 bytes, does not overflow `uint32_t`, and that `cursor + length <= buf_len` (using overflow-safe subtraction `buf_len - cursor >= length`).
4. **Shared Memory Isolation:** `cmcp_shm_channel` validates frame length against channel capacity before reading or writing, preventing out-of-bounds SHM access.
