# CMCP Roadmap

Prioritized next steps grounded in the empirical findings from Experiments A–F:

---

## 1. Immediate Priorities (Protocol & Representation Quality)

1. **Close the `cmcp_structured` vs. `json_structured` Quality Gap on Small Models:**
   - **Finding:** `cmcp_binary` already provides `0.27–0.35 ms` codec overhead, CRC32C integrity, and compact TLV wire transport (`325.875 B`), while `json_structured` achieved higher receiver `quality_score` (`0.9083` / `0.8583` vs. `0.7438` / `0.7583`).
   - **Action:** Keep TLV + 64B header on the wire, and test rendering decoded TLV records into **compact JSON** (or JSON-schema-aligned key-value blocks) at the receiver prompt boundary rather than custom `CMCP_CTX` line syntax.

2. **Benchmark End-to-End Sender Extraction Cost:**
   - **Action:** Measure the latency and accuracy of a lightweight sender-side extractor (rule-based/NER or constrained JSON decoding) converting raw prose into `CMCPContext`, so `nl_conversation` and `cmcp_binary` can be compared end-to-end under identical starting inputs.

---

## 2. Near-Term Engineering (Integration & Missing Hypotheses)

3. **Implement & Benchmark Optional `Vect-Link` Adapter (H5):**
   - **Action:** Implement `cmcp_vtlk_adapter` behind `-DCMCP_ENABLE_VECTLINK=ON` as specified in [vect_link_integration.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/architecture/vect_link_integration.md), validating Tier 0 `vtlk_kv_restore()` passthrough and Tier 1–3 zero-copy SHM envelope handoff.

4. **Implement Cross-Tokenizer Byte-Span Alignment Benchmark (H3):**
   - **Action:** Build and benchmark UTF-8 byte-span + MinED token alignment between Qwen2.5 (`151,936` vocab) and Llama-3.2 (`128,256` vocab) across prose, code, and JSON payloads.

---

## 3. Longer-Term Research (GPU-Scale State Transfer)

5. **Re-evaluate Soft-Prompt / Latent State Transfer (Exp E) at Scale:**
   - **Finding:** `learned_cmcp` (`K=8`, 48 train examples, frozen receiver) lowered NLL (`2.7108` -> `1.5738`) but failed factual EM (`0.0`).
   - **Action:** On GPU hardware, test (a) larger training corpora (`1k–10k+` examples), (b) per-layer KV projection or LoRA-assisted receiver adaptation (C2C-style), and (c) hybrid transfers combining `K` latent tokens for global framing with sparse `CMCP_TLV_ENTITY` / `REFERENCE` tokens for exact factual bindings.
