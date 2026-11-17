# CMCP Compatibility & KV-Cache Validation

## 1. Model Profile (`cmcp_model_profile_t`)

Every CMCP endpoint advertises a `cmcp_model_profile_t` containing:

| Field | Type | Purpose |
| :--- | :--- | :--- |
| `arch_name` | `char[64]` | Model architecture string (e.g., `"qwen2"`, `"llama"`). |
| `arch_hash` | `uint64_t` | Hash of `arch_name`. |
| `weights_hash` | `uint64_t` | Hash of model weights/GGUF identity. |
| `tokenizer_hash` | `uint64_t` | Hash of vocabulary and merge rules. |
| `n_layer` | `uint32_t` | Number of transformer layers. |
| `n_embd` | `uint32_t` | Hidden embedding dimension. |
| `n_head_kv` | `uint32_t` | Number of key-value attention heads. |
| `rope_type` | `int32_t` | RoPE scaling/type identifier (e.g., `LLAMA_ROPE_TYPE_NORM`, `NEOX`). |
| `rope_freq_scale` | `float` | RoPE frequency scale factor. |
| `kv_dtype` | `uint32_t` | KV-cache element type (e.g., `F16`, `Q8_0`). |
| `n_ctx` | `uint32_t` | Context window size allocated for the session. |
| `pos_offset` | `uint32_t` | Starting sequence position offset. |

---

## 2. Strict KV-Cache Compatibility (`cmcp_check_kv_compat`)

Tier 0 (`CMCP_TIER_0_EXACT_KV`) is **only** permitted when **all** of the following hold:
1. `src->arch_hash == dst->arch_hash` (and `arch_name` matches)
2. `src->weights_hash == dst->weights_hash`
3. `src->tokenizer_hash == dst->tokenizer_hash`
4. `src->n_layer == dst->n_layer`
5. `src->n_embd == dst->n_embd`
6. `src->n_head_kv == dst->n_head_kv`
7. `src->rope_type == dst->rope_type`
8. `|src->rope_freq_scale - dst->rope_freq_scale| <= 1e-6f`
9. `src->kv_dtype == dst->kv_dtype`
10. `src->n_ctx == dst->n_ctx`
11. `src->pos_offset == dst->pos_offset`

If **any** of these 11 conditions fails, `cmcp_check_kv_compat` returns `kv_compatible = false` with a specific `reason_code`:
- `CMCP_COMPAT_MISMATCH_ARCH`
- `CMCP_COMPAT_MISMATCH_WEIGHTS`
- `CMCP_COMPAT_MISMATCH_TOKENIZER`
- `CMCP_COMPAT_MISMATCH_GEOMETRY` (`n_layer`, `n_embd`, or `n_head_kv`)
- `CMCP_COMPAT_MISMATCH_ROPE` (`rope_type` or `rope_freq_scale`)
- `CMCP_COMPAT_MISMATCH_KV_DTYPE`
- `CMCP_COMPAT_MISMATCH_CTX` (`n_ctx`)
- `CMCP_COMPAT_MISMATCH_POS_OFFSET` (`pos_offset`)

> **Empirical Grounding:** In `llama.cpp`, `llama_state_seq_set_data` validates sequence state metadata and rejects mismatched architectures/configurations by returning `0`. Even when `n_ctx` or `pos_offset` differs on the same model, raw KV state reuse can either fail validation or produce corrupted positional attention if RoPE offsets diverge. CMCP enforces strict equality at the protocol boundary.

---

## 3. 4-Tier Negotiation Matrix (`cmcp_check_compat`)

| Condition | Assigned Tier | Recommended Payload |
| :--- | :--- | :--- |
| All 11 KV checks pass | `CMCP_TIER_0_EXACT_KV` | `CMCP_MSG_KV_SNAPSHOT` (or Vect-Link snapshot ref) |
| Same `arch_hash` & `n_embd`, but different `weights_hash`, `kv_dtype`, `n_ctx`, or `pos_offset` | `CMCP_TIER_1_SAME_FAMILY` | `CMCP_MSG_LATENT_TRANSFER` (affine-projected) or `CONTEXT_TRANSFER` |
| Different `arch_hash` or `n_embd`, but valid profile | `CMCP_TIER_2_CROSS_MODEL` | `CMCP_MSG_LATENT_TRANSFER` (anchor-relative / ridge-projected) + TLV |
| No latent alignment available / legacy endpoint | `CMCP_TIER_3_TEXT_FALLBACK` | `CMCP_MSG_CONTEXT_TRANSFER` (compressed TLV text) |
| Null/invalid profile or `strict_kv` requested on non-Tier-0 pair | `CMCP_TIER_INCOMPATIBLE` | `CMCP_MSG_REJECT` |
