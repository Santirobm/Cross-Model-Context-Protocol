# Related Work & Comparative Taxonomy

This document synthesizes the literature reviewed in [state_of_the_art.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/research/state_of_the_art.md) along four engineering axes relevant to **CMCP (Cross-Model Context Protocol)**.

---

## 1. Taxonomy of Context & Representation Transfer

| Work | Year | Transfer Scope | Tokenizer Requirement | Training Required? | Primary Artifact Transferred | Target Hardware |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **Vect-Link** (local) | 2026 | Same model (bit-exact) | Identical | None | Paged CoW KV blocks (`SHM`) | CPU / Local IPC |
| **LMCache** | 2025 | Same model (distributed) | Identical | None | Offloaded KV tensors | GPU + DRAM/Disk |
| **CacheBlend** | 2024 | Same model (non-prefix RAG) | Identical | None | Chunk KV + ~15% recompute | GPU |
| **KVCOMM** (Ye et al.) | 2025 | Same model (cross-context) | Identical | None (online anchors) | Prefix-adjusted KV cache | GPU |
| **KVComm** (Shi et al.) | 2025 | Compatible agents | Identical / aligned | None | Attention-selected KV pairs | GPU |
| **DroidSpeak** | 2024 | Same base family (fine-tuned) | Identical | Profiling only | Layer-selective KV + ~11% recompute | GPU |
| **LatentMAS** | 2025 | Compatible latent space | Identical / shared | None (same family) | Last-layer hidden thoughts + KV | GPU |
| **C2C** | 2025 | Cross-model (heterogeneous) | Projector-bridged | Yes (neural fuser) | Projected + gated KV cache | GPU |
| **ICAE / 500x / Gist** | 2023–24 | Single model (self-compression) | Identical | Yes (LoRA / FT) | Soft prompts / compressed KV | GPU |
| **LLMLingua** | 2023 | Universal (any model) | Any (text output) | None (small LM scorer) | Pruned discrete text | GPU / CPU |
| **Relative Reps** | 2022–23 | Cross-model / cross-modal | Any (anchor-level) | Anchor precomputation | Cosine-similarity vectors | GPU / CPU |
| **MinED** | 2024 | Cross-tokenizer | Any (vocab table) | None (edit distance) | Token-aligned distributions | GPU / CPU |

---

## 2. Key Structural Insights from Related Work

1. **Same-Model vs. Cross-Model Boundary:**
   Systems like **Vect-Link**, **LMCache**, **CacheBlend**, and **KVCOMM** achieve massive TTFT reductions (2×–9×) with zero or near-zero quality loss, *strictly because* the producer and consumer share identical model weights, RoPE geometry, and tokenizers. As **DroidSpeak** demonstrates, even fine-tuning the *same* base architecture causes >50% F1 degradation if KV caches are shared naively without selective layer recomputation.

2. **The Text Bottleneck vs. Latent Expressiveness:**
   **C2C** and **LatentMAS** independently establish that forcing rich internal model activations through discrete text tokens loses semantic nuance and incurs autoregressive decoding overhead. However, C2C requires GPU-trained pair-specific neural projectors, and LatentMAS assumes compatible latent spaces.

3. **Coordinate-Free Geometry as a Bridge:**
   **Platonic Representation Hypothesis**, **CKA**, **Model Stitching**, and **Relative Representations** collectively show that while absolute neuron coordinates are incompatible across models, *pairwise relational geometry* over shared anchor samples is largely preserved—even across different architectures and tokenizers.

4. **Positioning CMCP:**
   CMCP is **not** a replacement for same-model zero-copy KV stores like Vect-Link, nor is it a GPU-only distributed serving engine like LMCache. Instead, CMCP is a **tiered, model-agnostic context-exchange protocol and translation runtime** that selects the highest-fidelity transfer tier supported by a `(source_model, target_model)` pair:
   - **Tier 0 (Exact):** Bit-exact KV passthrough (delegated to Vect-Link or raw KV).
   - **Tier 1 (Same-Family):** Selective-layer KV reuse + critical-layer recomputation (inspired by DroidSpeak / CacheBlend).
   - **Tier 2 (Cross-Model / Cross-Tokenizer):** Relative anchor representations or lightweight linear/affine latent projectors (inspired by Relative Reps / Model Stitching / C2C).
   - **Tier 3 (Universal Fallback):** Perplexity-guided compressed text / structured semantic summary (inspired by LLMLingua).
