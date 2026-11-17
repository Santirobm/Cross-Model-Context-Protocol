# Research Gaps & Open Problems

Based on our review of the state of the art ([state_of_the_art.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/research/state_of_the_art.md)) and comparative taxonomy ([related_work.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/research/related_work.md)), we identify five concrete research and engineering gaps that motivate **CMCP**.

---

## Gap 1: Zero-Training / Low-Compute Cross-Model Transfer on CPU
- **Observation:** State-of-the-art cross-model KV transfer (**C2C**) requires training pair-specific neural projectors and layer gates on GPU clusters. Conversely, training-free KV methods (**DroidSpeak**, **KVCOMM**, **CacheBlend**) assume identical base architectures or identical models.
- **Gap:** There is no lightweight, CPU-feasible protocol for transferring context between *heterogeneous small/quantized models* (e.g., Qwen2.5-0.5B $\leftrightarrow$ Llama-3.2-1B / SmolLM2) without GPU fine-tuning.

## Gap 2: Lack of a Standardized, Self-Describing Cross-Model Context Envelope
- **Observation:** Existing KV-cache protocols (including Vect-Link's `vtlk_kv_identity_t` and `vtlk_header_t`) are designed to *reject* mismatched models via strict hash checks (`model_arch_hash`, `rope_config_hash`). Research prototypes like C2C and LatentMAS pass raw PyTorch tensors in-memory without a portable, versioned, checksummed wire format.
- **Gap:** There is no standardized binary/wire envelope that encodes **source model identity, tokenizer fingerprint, representation type (raw KV, hidden state, relative anchor, or compressed text), RoPE/position metadata, and integrity checksums** so a receiver can negotiate or translate incoming context safely.

## Gap 3: Cross-Tokenizer & Subword-Boundary Alignment for Sequence States
- **Observation:** **MinED** aligns vocabularies for *logit* distillation, and **Relative Representations** align *single-vector* embeddings. However, multi-token context caches (KV or per-token hidden states) have different sequence lengths $T_A \neq T_B$ when tokenizers split text differently.
- **Gap:** Practical, deterministic span-alignment rules (mapping token spans $[i..j]$ in tokenizer $A$ to byte offsets in UTF-8 and then to token spans $[i'..j']$ in tokenizer $B$) have not been standardized for cross-model context handoff.

## Gap 4: RoPE & Prefix-Offset Mismatch in Translated Context
- **Observation:** **KVCOMM** and **CacheBlend** highlight that RoPE positional encodings and prefix cross-attention break naive KV reuse even *within* the same model. When translating representations across models with different `rope_theta`, `head_dim`, or context offsets, positional coupling becomes even more severe.
- **Gap:** Pre-RoPE key/value or position-invariant hidden-state representations are rarely separated cleanly from post-RoPE cached tensors in inference runtimes like `llama.cpp`.

## Gap 5: Absence of CPU-First C/C++ Reference Implementations & Reproducible Baselines
- **Observation:** Nearly all SOTA papers (C2C, DroidSpeak, CacheBlend, LatentMAS, ICAE, 500xCompressor) are implemented in Python/PyTorch targeting CUDA GPUs.
- **Gap:** Edge and local multi-agent deployments running quantized GGUF models via `llama.cpp` on CPU (e.g., 16 GB RAM, no discrete GPU) lack a deterministic C/C++ protocol library, benchmark harness, and rigorous comparison against cold prefill and perplexity-pruned text handoff.
