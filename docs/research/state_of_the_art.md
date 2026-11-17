# State of the Art: Cross-Model Context & Representation Transfer

> **Scientific Rigor Policy:** This document strictly separates **Published Results** (reported by the original authors in peer-reviewed venues or arXiv preprints), **Reproduced Results** (none executed yet in Phases 1–3; explicitly marked), **Own Hypotheses**, and **Engineering Decisions**. Every citation and URL was verified via web search.

---

## 1. Cross-Model Representation Alignment

### 1.1 The Platonic Representation Hypothesis (PRH)
1. **Title & Authors (Year / Venue):** *The Platonic Representation Hypothesis* — Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola (2024, ICML Position Paper).
2. **Verified URL / Identifier:** [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)
3. **Category:** Cross-Model Representation Alignment (Theoretical & Empirical).
4. **Problem Addressed:** Whether independently trained neural networks across different architectures, objectives, and modalities learn fundamentally different or convergent internal representations of the world.
5. **Core Method / Mechanism:** Measures representational alignment (via mutual nearest-neighbor metrics and kernel alignment) across vision, language, and multimodal models of varying scale and training objectives.
6. **Key Published Results:** Larger, higher-performing models exhibit monotonically increasing representational alignment across architectures and even across modalities (e.g., vision-to-language), suggesting convergence toward a shared statistical model of reality.
7. **Tokenizer / Architecture Assumptions:** Metric-space comparisons operate on matched data samples (e.g., identical concepts/sentences/images) rather than requiring identical tokenizers or layer dimensions.
8. **Training / Calibration Requirements:** Evaluation-only (requires a shared probe/anchor dataset to compute pairwise similarity matrices).
9. **Known Limitations:** Alignment is strongest at large model scales and coarse semantic levels; small/quantized models (e.g., 0.5B–1.5B) exhibit weaker convergence and higher idiosyncratic variance.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Hypothesis:* Even small open-weight LLMs share enough relative geometric structure on anchor prompts to support low-rank linear or Procrustes-style latent alignment. *Engineering Decision:* CMCP will include an optional anchor-based alignment calibration step rather than assuming zero-shot raw vector compatibility.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 1.2 Relative Representations
1. **Title & Authors (Year / Venue):** *Relative Representations Enable Zero-Shot Latent Space Communication* — Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, Emanuele Rodolà (2022 arXiv / ICLR 2023 Notable-Top 5%).
2. **Verified URL / Identifier:** [arXiv:2209.15430](https://arxiv.org/abs/2209.15430) | [GitHub: lucmos/relreps](https://github.com/lucmos/relreps)
3. **Category:** Cross-Model Representation Alignment / Zero-Shot Latent Communication.
4. **Problem Addressed:** Latent spaces of independently trained models differ by unknown quasi-isometric transformations (rotations, reflections, rescalings), preventing direct vector reuse.
5. **Core Method / Mechanism:** Encodes each latent vector $z$ as a vector of cosine similarities to a fixed set of shared anchor samples $A = \{a_1, \dots, a_k\}$, producing a coordinate-free relative representation invariant to orthogonal transformations and isotropic scaling.
6. **Key Published Results:** Enables zero-shot model stitching and latent space comparison across different random seeds, architectures, and modalities without training a bridge network.
7. **Tokenizer / Architecture Assumptions:** Requires a shared semantic anchor set that both models can encode into comparable sequence/sentence-level embeddings; does not directly resolve token-granularity mismatches for token-by-token KV caches.
8. **Training / Calibration Requirements:** Training-free at inference; requires pre-computing anchor embeddings for each participating model.
9. **Known Limitations:** Cosine-similarity relative representations discard magnitude information and assume near-isometric latent geometry; performance degrades if anchor sets are too small or out-of-distribution.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* Relative anchor-similarity representations are a primary candidate for CMCP's model-agnostic intermediate representation (IR) payload format (`CMCP_PAYLOAD_REL_ANCHOR`), especially for heterogeneous-tokenizer handoffs.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 1.3 Centered Kernel Alignment (CKA)
1. **Title & Authors (Year / Venue):** *Similarity of Neural Network Representations Revisited* — Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton (2019, ICML).
2. **Verified URL / Identifier:** [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
3. **Category:** Representational Similarity Metric.
4. **Problem Addressed:** Canonical Correlation Analysis (CCA) fails in high-dimensional settings ($d \ge n$) by reporting spurious high similarity between random representations.
5. **Core Method / Mechanism:** Uses normalized Hilbert-Schmidt Independence Criterion (HSIC) over centered Gram matrices (linear or RBF kernels) to measure representational similarity in $[0, 1]$, invariant to orthogonal transformations and isotropic scaling.
6. **Key Published Results:** Reliably identifies layer-wise correspondences across different random initializations, widths, and architectures where CCA fails.
7. **Tokenizer / Architecture Assumptions:** Requires paired sample representations (same $n$ inputs evaluated on both models); independent of hidden dimension $d_1 \neq d_2$.
8. **Training / Calibration Requirements:** Training-free diagnostic metric computed over a calibration probe batch.
9. **Known Limitations:** Sensitive to outlier activations in transformer hidden states; measures structural similarity rather than directly providing an invertible mapping for generation.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* CMCP will use linear CKA as the standard offline diagnostic metric to select which layer pairs $(L_{\text{src}}, L_{\text{dst}})$ exhibit maximal structural alignment before fitting a lightweight projector.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 1.4 Model Stitching
1. **Title & Authors (Year / Venue):** *Revisiting Model Stitching to Compare Neural Representations* — Yamini Bansal, Preetum Nakkiran, Boaz Barak (2021, NeurIPS).
2. **Verified URL / Identifier:** [arXiv:2106.07682](https://arxiv.org/abs/2106.07682)
3. **Category:** Cross-Model Representation Compatibility.
4. **Problem Addressed:** Static similarity metrics (like CKA) do not directly test whether representations from model $A$ are functionally usable by model $B$.
5. **Core Method / Mechanism:** Freezes bottom layers of model $A$ and top layers of model $B$, inserting a minimal trainable linear/1x1-conv stitching layer to evaluate functional compatibility.
6. **Key Published Results:** Independently trained models (even supervised vs. self-supervised) can be stitched with minimal accuracy loss ("stitching connectivity"), and stronger bottom representations can boost weaker top models.
7. **Tokenizer / Architecture Assumptions:** Evaluated primarily on vision architectures with matching spatial resolutions; for LLMs, requires token-sequence alignment.
8. **Training / Calibration Requirements:** Requires training a lightweight linear stitching layer on labeled or next-token data.
9. **Known Limitations:** Does not work zero-shot across arbitrary dimensions without fitting the stitch; autoregressive LLM KV-cache stitching is more sensitive than single-pass classification.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Hypothesis:* A single linear (or low-rank affine) stitch between matched layers of two small GGUF LLMs can transfer semantic context with significantly lower perplexity than cold text-only handoff under constrained token budgets.
    - *Local Reproduction Status:* Not yet reproduced locally.

---

## 2. Cross-Model KV-Cache & Activation Transfer

### 2.1 Cache-to-Cache (C2C)
1. **Title & Authors (Year / Venue):** *Cache-to-Cache: Direct Semantic Communication Between Large Language Models* — Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang (2025, arXiv / ICLR submission).
2. **Verified URL / Identifier:** [arXiv:2510.03215](https://arxiv.org/abs/2510.03215) | [GitHub: thu-nics/C2C](https://github.com/thu-nics/C2C)
3. **Category:** Cross-Model KV-Cache Transfer.
4. **Problem Addressed:** Text-to-text (T2T) multi-LLM communication suffers from semantic information loss (bottlenecked by discrete tokens) and high autoregressive decoding/re-encoding latency.
5. **Core Method / Mechanism:** Projects and fuses a sharer LLM's KV cache directly into a receiver LLM's KV cache using a lightweight neural projector/fuser and learnable layer-wise gating to inject context only into beneficial target layers.
6. **Key Published Results:** Reports **8.5%–10.5% higher average accuracy** than standalone models, **3%–5% higher accuracy** than T2T multi-LLM communication, and **2×–2.5× average speedup** by eliminating intermediate text generation.
7. **Tokenizer / Architecture Assumptions:** Supports cross-model transfer across distinct models when a trained C2C projector/fuser is learned for the specific `(sharer, receiver)` pair.
8. **Training / Calibration Requirements:** Requires training the neural projector and layer gates on paired multi-LLM interaction data (GPU training).
9. **Known Limitations:** Pair-specific trained projectors do not generalize zero-shot to unseen model pairs without retraining.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* C2C validates that KV-cache-level semantic transfer outperforms text handoff; CMCP will support learned linear/MLP projectors as an optional adapter profile alongside training-free anchor/summary modes so it remains practical on CPU-only environments.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 2.2 KVCOMM (Online Cross-Context KV-Cache Communication)
1. **Title & Authors (Year / Venue):** *KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems* — Hancheng Ye, Zhengqi Gao, Mingyuan Ma, Qinsi Wang, Yuzhe Fu, Ming-Yu Chung, Yueqian Lin, Zhijian Liu, Jianyi Zhang, Danyang Zhuo, Yiran Chen (2025, arXiv / NeurIPS).
2. **Verified URL / Identifier:** [arXiv:2510.12872](https://arxiv.org/abs/2510.12872)
3. **Category:** Cross-Context KV-Cache Reuse in Multi-Agent Systems.
4. **Problem Addressed:** In multi-agent workflows, agents share overlapping text segments preceded by different prefixes/contexts, invalidating standard prefix-only KV caching due to RoPE and attention offset variance.
5. **Core Method / Mechanism:** Training-free framework maintaining an online pool of "anchors" (cached examples tracking KV deviations under varying prefixes) to dynamically estimate and align KV caches for shared segments across different contexts.
6. **Key Published Results:** Achieves **>70% KV reuse rates** across RAG, math, and coding tasks, and up to **7.8× TTFT reduction** without fine-tuning.
7. **Tokenizer / Architecture Assumptions:** Assumes agents share the same underlying model architecture/weights (cross-context, same-model or tightly coupled variants).
8. **Training / Calibration Requirements:** Training-free; relies on online anchor statistics collected during serving.
9. **Known Limitations:** Does not bridge across fundamentally different model families or mismatched tokenizers.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* Highlights the critical importance of explicit `pos_offset` / RoPE metadata in CMCP headers when reusing or translating cached context across divergent prefixes.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 2.3 KVComm (Selective KV Sharing)
1. **Title & Authors (Year / Venue):** *KVComm: Enabling Efficient LLM Communication through Selective KV Sharing* — Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire Jr., Dejan Kostić (2025, arXiv / ICLR).
2. **Verified URL / Identifier:** [arXiv:2510.03346](https://arxiv.org/abs/2510.03346)
3. **Category:** Selective KV-Cache Inter-Agent Communication.
4. **Problem Addressed:** Natural language communication incurs high token-generation latency, while naive hidden-state sharing suffers from concentration bias.
5. **Core Method / Mechanism:** Selectively shares subsets of KV pairs guided by attention importance scores and priors, bypassing full text generation.
6. **Key Published Results:** Preserves rich semantic context while substantially reducing inter-agent communication overhead compared to natural-language baselines.
7. **Tokenizer / Architecture Assumptions:** Targets homogeneous or compatible LLM agents where attention-selected KV pairs map directly into receiver attention.
8. **Training / Calibration Requirements:** Uses attention-score selection heuristics/priors without full model retraining.
9. **Known Limitations:** Direct KV injection still assumes compatible key/value head dimensions and positional conventions unless paired with a projection layer.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* Supports storing attention-importance masks or sparse KV-selection metadata in CMCP envelopes to reduce CPU/memory bandwidth during context transfer.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 2.4 DroidSpeak
1. **Title & Authors (Year / Venue):** *DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving* — Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, Junchen Jiang, Shan Lu, Madan Musuvathi, Esha Choukse (2024, arXiv).
2. **Verified URL / Identifier:** [arXiv:2411.02820](https://arxiv.org/abs/2411.02820)
3. **Category:** Cross-LLM KV-Cache Sharing (Same Base Family).
4. **Problem Addressed:** Fine-tuned variants of the same base LLM (e.g., task-specialized agents) cannot naively share KV caches without severe accuracy drops (>50% F1 loss) due to weight drift.
5. **Core Method / Mechanism:** Profiles layer sensitivity for a model pair, reuses KV caches for insensitive layers, and selectively recomputes contiguous blocks of critical sensitive layers (~11% of layers on average), pipelining recomputation with KV loading.
6. **Key Published Results:** Up to **4× serving throughput** and **1.7×–3.1× TTFT reduction** with negligible loss in F1, ROUGE-L, or code similarity.
7. **Tokenizer / Architecture Assumptions:** Strictly requires models to share the same base architecture and tokenizer (fine-tuned variants of one base model).
8. **Training / Calibration Requirements:** Offline layer-sensitivity profiling per model pair; no model retraining.
9. **Known Limitations:** Cannot transfer across distinct model families (e.g., Qwen to Llama) or different hidden dimensions/tokenizers.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* DroidSpeak defines CMCP's "Tier 1 (Same-Family / Fine-Tuned)" compatibility class: partial KV reuse + selective critical-layer recomputation.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 2.5 CacheBlend
1. **Title & Authors (Year / Venue):** *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion* — Jiayi Yao, Hanchen Li, Yuhan Liu, et al. (2024, arXiv / EuroSys).
2. **Verified URL / Identifier:** [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) | [GitHub: YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend)
3. **Category:** Non-Prefix KV Cache Fusion for RAG.
4. **Problem Addressed:** Precomputed KV caches for retrieved RAG chunks cannot be naively concatenated because they lack cross-attention with preceding chunks.
5. **Core Method / Mechanism:** Concatenates precomputed chunk KV caches and selectively recomputes only ~15% of tokens (those with highest KV deviation / boundary tokens) to restore cross-attention consistency.
6. **Key Published Results:** **2.2×–3.3× lower TTFT** and **2.8×–5× higher throughput** vs. full prefill, keeping quality drop within 1%–3%.
7. **Tokenizer / Architecture Assumptions:** Single-model (identical weights and tokenizer) across chunk precomputation and fusion.
8. **Training / Calibration Requirements:** Training-free; uses lightweight runtime deviation checks.
9. **Known Limitations:** Assumes identical model weights; does not address cross-model or cross-tokenizer transfer.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* Selective boundary-token recomputation is directly applicable when fusing imported CMCP context chunks into a local `llama.cpp` KV cache.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 2.6 LMCache
1. **Title & Authors (Year / Venue):** *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference* — Yihua Cheng, Yuhan Liu, Jiayi Yao, Yuwei An, Xiaokun Chen, Shaoting Feng, Yuyang Huang, Samuel Shen, Kuntai Du, Junchen Jiang (2025, arXiv).
2. **Verified URL / Identifier:** [arXiv:2510.09665](https://arxiv.org/abs/2510.09665) | [GitHub: LMCache/LMCache](https://github.com/LMCache/LMCache)
3. **Category:** Enterprise KV-Cache Offloading & Disaggregated Serving.
4. **Problem Addressed:** GPU VRAM limits prevent retaining large prefix/document KV caches across requests and instances.
5. **Core Method / Mechanism:** Pluggable KV connector for vLLM/SGLang enabling hierarchical offloading (CPU DRAM, local disk, Redis/remote) and Prefill-Decode (PD) disaggregation with pipelined I/O.
6. **Key Published Results:** Enables cross-instance KV reuse and significant TTFT savings in GPU serving clusters.
7. **Tokenizer / Architecture Assumptions:** Same-model KV sharing across serving instances.
8. **Training / Calibration Requirements:** Training-free systems layer.
9. **Known Limitations:** Python/GPU-cluster-centric (vLLM/SGLang); not a C99 CPU/SHM engine and does not perform cross-model semantic translation.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* Reinforces the separation between **transport/storage** (LMCache / Vect-Link) and **cross-model semantic translation** (CMCP).
    - *Local Reproduction Status:* Not yet reproduced locally.

### 2.7 LatentMAS
1. **Title & Authors (Year / Venue):** *Latent Collaboration in Multi-Agent Systems* — Jiaru Zou, Ruizhong Qiu, Gaotang Li, Xiyuan Yang, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, Ling Yang (2025, arXiv).
2. **Verified URL / Identifier:** [arXiv:2511.20639](https://arxiv.org/abs/2511.20639) | [GitHub: Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)
3. **Category:** Latent-Space Multi-Agent Collaboration.
4. **Problem Addressed:** Text-based multi-agent collaboration loses information when projecting high-dimensional hidden states into discrete tokens and incurs heavy autoregressive generation cost.
5. **Core Method / Mechanism:** Training-free framework where agents generate autoregressive "latent thoughts" (last-layer hidden embeddings) and exchange them alongside KV-cache working memory without decoding to text.
6. **Key Published Results:** Across 9 reasoning/coding benchmarks, reports up to **14.6% higher accuracy**, **70.8%–83.7% fewer output tokens**, and **4×–4.3× faster** end-to-end inference.
7. **Tokenizer / Architecture Assumptions:** Requires agents to share compatible hidden-state/KV spaces (typically homogeneous base models or aligned latent spaces).
8. **Training / Calibration Requirements:** Training-free when agents share the same model/latent space.
9. **Known Limitations:** Heterogeneous-architecture and cross-tokenizer latent exchange remains unaddressed without an external alignment layer.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* CMCP's protocol envelope should support passing **last-layer hidden states / latent thought vectors** in addition to KV caches or compressed text.
    - *Local Reproduction Status:* Not yet reproduced locally.

---

## 3. Prompt / Context Compression & Soft Prompts

### 3.1 In-Context Autoencoder (ICAE)
1. **Title & Authors (Year / Venue):** *In-context Autoencoder for Context Compression in a Large Language Model* — Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, Furu Wei (2023 arXiv / ICLR 2024).
2. **Verified URL / Identifier:** [arXiv:2307.06945](https://arxiv.org/abs/2307.06945)
3. **Category:** Soft-Prompt / Memory-Slot Context Compression.
4. **Problem Addressed:** Long raw-text contexts consume excessive KV-cache memory and prefill compute.
5. **Core Method / Mechanism:** A LoRA-adapted encoder compresses long text into compact continuous "memory slots" (soft prompt embeddings) consumed by a frozen target LLM decoder, trained via autoencoding + language modeling.
6. **Key Published Results:** Achieves ~**4× context compression** on LLaMA/Mistral with <1% extra parameters.
7. **Tokenizer / Architecture Assumptions:** Encoder and decoder are coupled to the same base LLM embedding space.
8. **Training / Calibration Requirements:** Requires pretraining + instruction fine-tuning of the LoRA encoder.
9. **Known Limitations:** Compressed memory slots are model-specific and cannot be directly fed into a different LLM family without re-projection.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* Demonstrates that compact continuous memory slots can replace long token sequences; CMCP can treat memory slots as a compressible payload type when paired with a cross-model projector.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 3.2 500xCompressor
1. **Title & Authors (Year / Venue):** *500xCompressor: Generalized Prompt Compression for Large Language Models* — Zongqian Li, Yixuan Su, Nigel Collier (2024, arXiv / ACL).
2. **Verified URL / Identifier:** [arXiv:2408.03094](https://arxiv.org/abs/2408.03094)
3. **Category:** Extreme KV-Value Prompt Compression.
4. **Problem Addressed:** Soft-prompt embedding compression (like ICAE) loses fidelity at extreme compression ratios (>10×).
5. **Core Method / Mechanism:** Compresses long contexts into the **KV-cache values** of a tiny number of special tokens (6× to ~480×–500×) using ~0.3% trainable LoRA encoder parameters while keeping the decoder frozen.
6. **Key Published Results:** Retains 62%–73% (up to 84% on specific EM metrics) of uncompressed LLM QA capability at extreme compression ratios.
7. **Tokenizer / Architecture Assumptions:** Encoder and decoder share the same frozen LLM architecture.
8. **Training / Calibration Requirements:** Requires LoRA encoder training on Arxiv / ArxivQA.
9. **Known Limitations:** Model-coupled KV representations; not directly portable across different model architectures.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Own Insight:* Confirms that KV-cache tensors carry higher information density than input-layer soft-prompt embeddings at high compression ratios.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 3.3 Gist Tokens
1. **Title & Authors (Year / Venue):** *Learning to Compress Prompts with Gist Tokens* — Jesse Mu, Xiang Lisa Li, Noah Goodman (2023, NeurIPS).
2. **Verified URL / Identifier:** [arXiv:2304.08467](https://arxiv.org/abs/2304.08467)
3. **Category:** Prompt Compression via Attention Masking.
4. **Problem Addressed:** Repeatedly prefilling long system prompts wastes context window and FLOPs.
5. **Core Method / Mechanism:** Modifies Transformer attention masks during instruction fine-tuning so prompt tokens are compressed into a small set of learned "gist tokens" whose KV cache is saved and reused.
6. **Key Published Results:** Up to **26× prompt compression** and **40% FLOPs reduction** with minimal quality degradation.
7. **Tokenizer / Architecture Assumptions:** Requires fine-tuning the model itself to recognize gist tokens; single-model only.
8. **Training / Calibration Requirements:** Instruction fine-tuning with custom attention masks.
9. **Known Limitations:** Cannot be applied zero-shot to off-the-shelf GGUF models without fine-tuning.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* Since CMCP targets off-the-shelf GGUF models on CPU without weight fine-tuning, gist-token training is out of scope for core CMCP, though CMCP can transport gist KV blocks if a model supports them.
    - *Local Reproduction Status:* Not yet reproduced locally.

### 3.4 LLMLingua
1. **Title & Authors (Year / Venue):** *LLMLingua: Compressing Prompts for Accelerated Inference* — Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu (2023, EMNLP).
2. **Verified URL / Identifier:** [arXiv:2310.05736](https://arxiv.org/abs/2310.05736)
3. **Category:** Discrete Token-Level Prompt Compression.
4. **Problem Addressed:** Long prompts in ICL, CoT, and RAG increase latency and API cost.
5. **Core Method / Mechanism:** Coarse-to-fine token pruning using a small open-source LM's perplexity to score token redundancy, guided by a dynamic budget controller and iterative conditional probability estimation.
6. **Key Published Results:** Up to **20× compression** on GSM8K, BBH, ShareGPT, and Arxiv-March23 with competitive downstream accuracy.
7. **Tokenizer / Architecture Assumptions:** Outputs discrete natural language tokens—**100% cross-tokenizer and cross-architecture compatible**.
8. **Training / Calibration Requirements:** Training-free (with optional instruction tuning for distribution alignment).
9. **Known Limitations:** At high compression ratios, discrete token pruning can drop critical syntactic/numerical details; still requires the target model to re-prefill the compressed text.
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* Perplexity-guided discrete compression is the universal **Tier 3 Fallback** in CMCP when two models share neither architecture nor calibrated latent projectors, providing a strong, verifiable baseline on CPU.
    - *Local Reproduction Status:* Not yet reproduced locally.

---

## 4. Cross-Tokenizer Alignment

### 4.1 MinED (Minimum Edit Distance Vocabulary Alignment)
1. **Title & Authors (Year / Venue):** *Knowledge Fusion of Large Language Models* — Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan, Wei Bi, Shuming Shi (2024, ICLR).
2. **Verified URL / Identifier:** [arXiv:2401.10491](https://arxiv.org/abs/2401.10491)
3. **Category:** Cross-Tokenizer Vocabulary & Distribution Alignment.
4. **Problem Addressed:** Models with different tokenizers (e.g., BPE vs. SentencePiece, different vocab sizes) cannot align token-level logits or token-indexed caches 1:1.
5. **Core Method / Mechanism:** **MinED** greedily aligns tokens across two vocabularies by minimizing string-level Levenshtein edit distance between decoded token strings, enabling approximate likelihood/distribution matching across mismatched vocabularies.
6. **Key Published Results:** Enables effective knowledge fusion across heterogeneous open-source LLMs (e.g., Llama-2, Mistral, MPT) with different tokenizers.
7. **Tokenizer / Architecture Assumptions:** Requires access to both models' vocabulary tables (token-ID-to-UTF8 string mappings), which are embedded in GGUF metadata.
8. **Training / Calibration Requirements:** Purely string/vocabulary-based mapping (no GPU training required to build the token alignment table).
9. **Known Limitations:** Greedy 1:1 edit-distance alignment is lossy when one tokenizer splits a word into 3 subwords and another into 1 token (many-to-one / many-to-many boundary mismatches).
10. **Relevance to CMCP (Own Analysis & Engineering Decision):** *Engineering Decision:* GGUF files expose exact vocabulary tables; CMCP can use character/byte-span overlap + MinED-style alignment in C/Python to map token-level attention importance or align probe sequences across heterogeneous tokenizers.
    - *Local Reproduction Status:* Not yet reproduced locally.
