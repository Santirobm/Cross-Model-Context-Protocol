# CMCP v1 Binary Wire Format

All multi-byte integer fields in CMCP v1 are encoded in **explicit little-endian** byte order, independent of host CPU endianness.

---

## 1. Fixed 64-Byte Envelope Header

Every CMCP frame begins with a 64-byte header (`CMCP_HEADER_SIZE = 64`), followed immediately by `payload_bytes` of payload data.

| Offset | Size | Type | Field | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0` | `4` | `char[4]` | `magic` | ASCII `'C'`, `'M'`, `'C'`, `'P'` (`0x50434D43` LE). |
| `4` | `2` | `uint16` | `version` | Protocol version. Must be `1`. |
| `6` | `2` | `uint16` | `flags` | Bitmask (`CMCP_FLAG_COMPRESSED = 0x0001`, `CMCP_FLAG_HAS_KV_DESC = 0x0002`, `CMCP_FLAG_STRICT_KV = 0x0004`). |
| `8` | `2` | `uint16` | `msg_type` | Message type (`cmcp_msg_type_t`). |
| `10` | `2` | `uint16` | `compat_tier` | Negotiated compatibility tier (`0..3`, or `0xFFFF`). |
| `12` | `8` | `uint64` | `sender_model_hash` | 64-bit hash of sender model weights/config identity. |
| `20` | `8` | `uint64` | `receiver_model_hash` | 64-bit hash of intended receiver model (`0` if broadcast/any). |
| `28` | `8` | `uint64` | `tokenizer_hash` | 64-bit hash of sender tokenizer vocabulary & merges. |
| `36` | `4` | `uint32` | `payload_bytes` | Byte length of the wire payload immediately following this header. |
| `40` | `4` | `uint32` | `raw_bytes` | Uncompressed payload length in bytes (equals `payload_bytes` when uncompressed). |
| `44` | `8` | `uint64` | `sequence_id` | Monotonic frame/conversation sequence identifier. |
| `52` | `4` | `uint32` | `reserved` | Reserved for future use; must be `0` in v1. |
| `56` | `4` | `uint32` | `payload_crc32c` | Castagnoli CRC32C of `payload[0 .. payload_bytes - 1]` (`0` if `payload_bytes == 0`). |
| `60` | `4` | `uint32` | `header_crc32c` | Castagnoli CRC32C of header bytes `[0 .. 59]` (covers `payload_crc32c` as well). |

**Total Header Size:** `4 + 2 + 2 + 2 + 2 + 8 + 8 + 8 + 4 + 4 + 8 + 4 + 4 + 4 = 64` bytes.

### CRC32C Parameters (Castagnoli)
- **Polynomial:** `0x1EDC6F41` (reflected: `0x82F63B78`)
- **Initial Value:** `0xFFFFFFFF`
- **Final XOR:** `0xFFFFFFFF`
- **Reflect Input/Output:** `true`

---

## 2. Compact TLV Record Format

When `msg_type` carries structured context (`CMCP_MSG_CONTEXT_TRANSFER`, `CMCP_MSG_HELLO`, etc.), the uncompressed payload is a sequence of contiguous TLV records:

```text
+----------+----------+-------------------+------------------------+
| tag (1B) | flags(1B)| length (ULEB128)  | value (length bytes)   |
+----------+----------+-------------------+------------------------+
```

- `tag` (`uint8_t`): Record semantic type:
  - `0x01`: `CMCP_TLV_CONCEPT`
  - `0x02`: `CMCP_TLV_ENTITY`
  - `0x03`: `CMCP_TLV_RELATION`
  - `0x04`: `CMCP_TLV_INSTRUCTION`
  - `0x05`: `CMCP_TLV_CONSTRAINT`
  - `0x06`: `CMCP_TLV_EVIDENCE`
  - `0x07`: `CMCP_TLV_REFERENCE`
  - `0x08`: `CMCP_TLV_DECISION`
  - `0x09`: `CMCP_TLV_LATENT_TENSOR`
  - `0x0A`: `CMCP_TLV_KV_COMPAT_DESC`
- `flags` (`uint8_t`): Record-level flags (default `0x00`).
- `length` (`ULEB128`): Unsigned variable-length integer (7 bits per byte, MSB `0x80` continuation bit, max 5 bytes for `uint32_t`).
- `value`: `length` raw bytes (UTF-8 text for semantic tags, or binary struct/tensor for `LATENT_TENSOR` / `KV_COMPAT_DESC`).

---

## 3. Lossless Compression Format (`CMCP_FLAG_COMPRESSED`)

CMCP includes a self-contained, dependency-free lossless codec (`cmcp_compress` / `cmcp_decompress`) combining ULEB128 framing, RLE, and LZ77-lite back-references:

### Stream Header
- `raw_len` (`ULEB128`): Uncompressed byte count (must match `header.raw_bytes`).

### Token Stream
Each token begins with a 1-byte control tag `op`:
- **Literal Run (`0x00 .. 0x3F`):** `count = (op & 0x3F) + 1` (1..64) literal bytes follow directly.
- **RLE Run (`0x40 .. 0x7F`):** `count = (op & 0x3F) + 3` (3..66) repetitions of the single next byte `val`.
- **LZ77 Short Match (`0x80 .. 0xBF`):** `len = (op & 0x3F) + 4` (4..67) bytes copied from `out_pos - offset`, where `offset` is a 1-byte unsigned value `1..255` (`offset_byte + 1`, wait: `offset` encoded as ULEB128 or 2-byte LE; in CMCP v1 we use ULEB128 `offset >= 1` for both short and long matches so arbitrary window sizes up to 64 KiB are supported cleanly!).
  - Specifically, in CMCP v1:
    - `0x00 .. 0x3F`: Literal run of `(op & 0x3F) + 1` bytes.
    - `0x40 .. 0x7F`: RLE run of `(op & 0x3F) + 3` copies of next byte `b`.
    - `0x80 .. 0xFF`: LZ77 match of `len = (op & 0x7F) + 4` bytes (4..131), followed by `ULEB128(offset)` (`offset >= 1`).

The decoder strictly checks that:
1. Input reads never exceed `src_len`.
2. Output writes never exceed `raw_len` or `dst_cap`.
3. `offset >= 1` and `offset <= out_pos`.
4. Overlapping LZ77 copies (where `offset < len`) are copied byte-by-byte in forward order.
