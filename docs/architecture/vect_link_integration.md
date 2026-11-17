# Vect-Link Integration Architecture (Decoupled & Optional)

> **Scope & Boundary:** This design is based on a **read-only** inspection of `C:\Users\sgi.lm\Desktop\Vect-Link` (`README.md`, `include/vectlink_kv.h`, `include/vectlink/protocol.h`). CMCP **must not** depend on Vect-Link at build time or runtime; Vect-Link integration is strictly an optional transport/storage adapter.

---

## 1. Architectural Separation of Concerns

| Concern | **Vect-Link** (Existing Project) | **CMCP** (This Project) |
| :--- | :--- | :--- |
| **Primary Role** | Same-model Copy-on-Write (CoW) KV-cache SHM store & zero-copy IPC | Cross-model semantic context negotiation, representation translation, & portable envelope |
| **Model Compatibility** | **Strict identity match** (`model_arch_hash`, `rope_config_hash`, `lora_adapter_hash`, `kv_dtype`, `layer_count`, `head_dim`, `kv_head_count`) | **Heterogeneous models** via tiered negotiation (Exact $\to$ Same-Family $\to$ Cross-Model $\to$ Text Fallback) |
| **Data Plane** | 64B-aligned paged SHM arena (`vtlk_kv_arena_t`), L1/L2/L3 LRU tiering, `WaitOnAddress`/`futex` | `cmcp_envelope_t` carrying raw KV, latent/anchor vectors, or compressed text + metadata |
| **Control Plane** | `vtlk_header_t` (64B) / `vtlk_header_v2_t` (96B) + SPSC descriptor rings | `cmcp_session_t` capability negotiation & translation pipeline |

**Key Insight:** Vect-Link explicitly *rejects* cross-model sharing at its identity boundary to guarantee bit-exactness (`max_logit_diff = 0.000000`). CMCP sits **above** or **beside** Vect-Link:
1. When two agents run the **exact same model** (Tier 0), CMCP can delegate directly to Vect-Link's `vtlk_kv_snapshot_create()` / `vtlk_kv_fork()` / `vtlk_kv_restore()` for ~5 µs zero-copy branching.
2. When two agents run **different models** (Tiers 1–3), CMCP translates the context into a target-compatible representation and can *optionally* use Vect-Link's SHM arena or v2 transport (`VTLK_QUANT_RAW`) as a high-speed local IPC carrier for the CMCP envelope.

---

## 2. Zero-Dependency Decoupled Adapter Pattern

CMCP defines a minimal, self-contained transport interface (`cmcp_transport_t`) in `include/cmcp/transport.h` with **no Vect-Link headers**:

```c
typedef struct cmcp_transport_ops {
    int (*send_envelope)(void *ctx, const uint8_t *buf, size_t len, uint64_t *out_handle);
    int (*recv_envelope)(void *ctx, uint64_t handle, const uint8_t **out_buf, size_t *out_len);
    void (*release_handle)(void *ctx, uint64_t handle);
} cmcp_transport_ops_t;
```

### Build-Time Option (`CMakeLists.txt`)
- `CMCP_ENABLE_VECTLINK=OFF` (default): Core CMCP library, translation codecs, `llama.cpp` adapter, and all unit/benchmark tests compile with **zero** reference to Vect-Link. Built-in in-memory / file / POSIX-or-Windows pipe transports are used.
- `CMCP_ENABLE_VECTLINK=ON`: Compiles an optional adapter module (`src/adapters/vectlink_adapter.c`) against `VECTLINK_INCLUDE_DIR` and `VECTLINK_LIB`.

---

## 3. Mapping CMCP onto Vect-Link Primitives (When Enabled)

1. **Tier 0 (Exact Same-Model Passthrough):**
   - CMCP inspects `cmcp_model_desc_t` for source and target. If `model_arch_hash`, `rope_config_hash`, `lora_adapter_hash`, `kv_dtype`, and geometry match, CMCP emits a `CMCP_PAYLOAD_VTLK_SNAPSHOT_REF` envelope containing only the `uint64_t snapshot_id` and `vtlk_kv_identity_t`.
   - The receiver calls `vtlk_kv_restore()` directly—achieving zero-copy SHM handoff without serializing KV tensors through CMCP buffers.

2. **Tiers 1–3 (Translated / Portable CMCP Envelopes over Vect-Link SHM):**
   - When models differ, CMCP serializes a `cmcp_envelope_t` (header + payload) into a Vect-Link SHM block (`vtlk_kv_block_alloc()` -> `vtlk_kv_block_data()` -> `vtlk_kv_block_seal()`), or frames it with `vtlk_header_v2_t` (`quant_scheme = VTLK_QUANT_RAW`, `codec = VTLK_CODEC_NONE`).
   - The receiver obtains a zero-copy `const uint8_t *` pointer via `vtlk_kv_block_data()`, decodes the CMCP envelope, and applies the target-side translation/injection.

---

## 4. Safety & Isolation Guarantees

- **No ABI/Struct Coupling:** CMCP never embeds `vtlk_kv_arena_t` or `vtlk_header_v2_t` in public CMCP structs; the adapter holds them behind an opaque `void *ctx`.
- **Fallback Safety:** If Vect-Link initialization fails at runtime (e.g., SHM quota exceeded), the adapter cleanly returns a non-fatal error so CMCP can fall back to standard file/memory transport.
- **No Modification to Vect-Link:** All integration code lives strictly inside `Cross-Model-Context-Protocol`.
