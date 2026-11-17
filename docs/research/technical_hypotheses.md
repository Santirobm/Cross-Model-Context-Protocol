# Technical Hypotheses (H1–H5)

All hypotheses below are formulated as falsifiable engineering/scientific claims testable within our local CPU + `llama.cpp` + quantized GGUF environment ([environment_report.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/experiments/environment_report.md)).

---

## Hypothesis H1: Protocol Negotiation & Tiered Compatibility Overhead
- **Statement:** A self-describing C99/C++ CMCP binary envelope (`cmcp_envelope_t`) with CRC32C integrity verification and capability negotiation adds negligible serialization/validation overhead (< 50 µs for metadata + < 1% of TTFT) compared to `llama.cpp` prefill latency, while preventing silent cross-model tensor corruption.
- **Independent Variable:** Payload size (1 KiB to 16 MiB) and compatibility tier (Tier 0 Exact vs. Tier 2 Translated vs. Tier 3 Text Fallback).
- **Dependent Variables:** Envelope encode/decode latency (µs), CRC32C throughput (GB/s), and mismatch rejection rate (100% on incompatible tuples).
- **Baseline:** Raw unvalidated memcpy / JSON text serialization.
- **Success Criteria:** Encode + validate + decode < 50 µs for header/metadata; 0 silent corruptions across fuzz/mismatch tests.
- **Failure Criteria:** Overhead > 5% of baseline TTFT or any undetected model/tokenizer mismatch passing validation.
- **Limitations:** Measures protocol/control-plane overhead only, not downstream generation quality.

## Hypothesis H2: Cross-Model Structural Alignment on Quantized Small LLMs
- **Statement:** Even among small quantized GGUF models (< 2B parameters, Q4_K_M), intermediate and final hidden-state representations on shared anchor prompts exhibit non-trivial linear CKA (> 0.50) and relative-representation rank correlation ($\rho > 0.50$) across different model families (e.g., Qwen2.5-0.5B vs. a second distinct small GGUF model).
- **Independent Variable:** Layer depth ($l \in [0, L]$), model pair (same-family vs. cross-family), and anchor set size ($|A| \in \{16, 32, 64, 128\}$).
- **Dependent Variables:** Linear CKA score, Spearman rank correlation of pairwise cosine similarity matrices.
- **Baseline:** Random/untrained projection matrix baseline and shuffled-anchor null baseline.
- **Success Criteria:** Peak cross-model layer CKA $\ge 0.50$ and statistically significant ($p < 0.01$) above shuffled-anchor null baseline.
- **Failure Criteria:** CKA $< 0.30$ across all layers, indicating insufficient shared geometry at 0.5B–1B Q4_K_M scale for linear/relative alignment.
- **Limitations:** High CKA is necessary but not sufficient for generative fidelity; functional utility must be tested by H4.

## Hypothesis H3: Cross-Tokenizer Byte-Span Alignment
- **Statement:** Mapping token sequences between heterogeneous tokenizers via UTF-8 byte-span overlap (augmented with MinED fallback for partial subwords) preserves $\ge 95\%$ exact span boundary alignment for downstream context segmentation without requiring model weights.
- **Independent Variable:** Tokenizer pair (e.g., Qwen2.5 BPE vs. Llama/SentencePiece BPE) and text domain (English prose, code, structured JSON).
- **Dependent Variables:** Byte-span alignment coverage (%), boundary-split rate (%), and mapping latency (µs/1k tokens).
- **Baseline:** Naive 1:1 token-index alignment and pure string-level MinED without byte-span tracking.
- **Success Criteria:** $\ge 95\%$ byte-exact span reconstruction and $< 1\text{ ms}$ mapping latency per 1k tokens on CPU.
- **Failure Criteria:** Coverage $< 85\%$ or failure on multi-byte UTF-8 / code tokens.
- **Limitations:** Aligns token *positions* and *spans*, not internal attention weights across split subwords.

## Hypothesis H4: Lightweight Linear / Affine Latent Projection vs. Text-Only Handoff
- **Statement:** Under a fixed inter-agent communication bandwidth/token budget, transferring a compressed CMCP representation (anchor-relative or low-rank ridge-regressed projection fitted offline on probe prompts) achieves lower perplexity / higher task accuracy on a receiver GGUF model than naive truncated text handoff, while reducing receiver prefill tokens.
- **Independent Variable:** Transfer mode (Cold Full-Text Prefill [upper bound], Truncated/Pruned Text [LLMLingua-style baseline], CMCP Anchor-Relative / Projected Embedding).
- **Dependent Variables:** Receiver next-token perplexity (PPL), downstream QA/instruction match rate, and receiver TTFT (ms).
- **Baseline:** (a) Cold full-text prefill (quality ceiling); (b) Equal-byte/equal-token truncated or perplexity-pruned text handoff.
- **Success Criteria:** Beats equal-budget truncated text handoff on PPL/accuracy with statistical significance ($p < 0.05$) and reduces receiver TTFT by $\ge 1.5\times$ vs. full-text prefill.
- **Failure Criteria:** Worse PPL than equal-budget pruned text, or CPU projection overhead exceeds prefill savings.
- **Limitations:** `llama.cpp` public C API injects embeddings via `llama_batch` (`embd` field) rather than arbitrary mid-layer KV injection without custom hooks; initial tests will evaluate input-embedding/soft-prompt injection and `llama_state` compatibility.

## Hypothesis H5: Decoupled Vect-Link Transport Integration
- **Statement:** An optional, compile-time-decoupled `cmcp_vtlk_adapter` can store and exchange CMCP envelopes (and Tier 0 exact KV snapshots) over Vect-Link's SHM arena with zero-copy read access and zero hard build dependency when Vect-Link is absent.
- **Independent Variable:** Build configuration (`CMCP_ENABLE_VECTLINK=OFF` vs. `ON`) and payload size.
- **Dependent Variables:** IPC transfer latency (µs), zero-copy validation (`copied_bytes == 0` at adapter boundary), and clean build/test pass rate without Vect-Link headers.
- **Baseline:** File-based / pipe-based envelope exchange.
- **Success Criteria:** `CMCP_ENABLE_VECTLINK=OFF` builds and passes 100% of unit tests with zero Vect-Link symbols; `ON` achieves $< 25\text{ }\mu\text{s}$ handoff latency for envelopes up to 4 MiB.
- **Failure Criteria:** Any build coupling to Vect-Link in core CMCP headers, or data corruption across process boundaries.
- **Limitations:** Tier 0 zero-copy KV sharing via Vect-Link applies only when source and target models match Vect-Link's strict `vtlk_kv_identity_t` contract.
