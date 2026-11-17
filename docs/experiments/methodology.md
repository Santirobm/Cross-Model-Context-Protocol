# Experimental Methodology

All experiments were executed locally on the hardware and software stack documented in [environment_report.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/experiments/environment_report.md).

---

## 1. Environment & Models

### 1.1 Hardware & Toolchain
- **OS:** Microsoft Windows 10 Pro (`10.0.19045`, 64-bit)
- **CPU:** 13th Gen Intel Core i5-13500 (`14` physical cores, `20` logical processors, CPU-only, no discrete GPU)
- **RAM:** `15.70 GiB`
- **C/C++ Toolchain:** MSYS2 Clang `22.1.8`, CMake `4.4.3`, Ninja `1.13.2`
- **Python:** `3.13.5` (`.venv`)
- **Inference Engine:** `llama.cpp` (Git HEAD `b79696f`, `libllama.dll`) for Exp A, B, C, F; HuggingFace `transformers` + PyTorch (CPU) for Exp D, E.

### 1.2 Models Evaluated
| Role / Experiments | Model | Format / Architecture | Key Specs |
| :--- | :--- | :--- | :--- |
| Exp A, B, C, F | `Qwen2.5-0.5B-Instruct` | GGUF `Q4_K_M` (`491,400,032` B), `qwen2` | 24 layers, `d=896`, 2 KV heads, RoPE=2, vocab `151,936` |
| Exp A, B, C, F | `Llama-3.2-1B-Instruct` | GGUF `Q4_K_M` (`807,694,464` B), `llama` | 16 layers, `d=2048`, 8 KV heads, RoPE=0, vocab `128,256` |
| Exp D, E (Sender) | `HuggingFaceTB/SmolLM2-135M-Instruct` | HF PyTorch CPU (`LlamaForCausalLM`) | `d=576` |
| Exp D, E (Receiver) | `Qwen/Qwen2.5-0.5B-Instruct` | HF PyTorch CPU (`Qwen2ForCausalLM`) | `d=896` |

---

## 2. Experiment Designs

### 2.1 Experiments A & B: Cross-Model Structured Context Transfer
- **Script:** [run_exp_a_b.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/experiments/structured_transfer/run_exp_a_b.py)
- **Design:** 8 deterministic multi-step tasks (`T1`–`T8`) × 3 repetitions = **24 runs per condition**, tested bidirectionally (`Qwen2.5-0.5B->Llama-3.2-1B` and `Llama-3.2-1B->Qwen2.5-0.5B`), totaling **192 runs**.
- **Conditions:**
  1. `nl_conversation`: Sender generates a natural-language handoff summary (`max_tokens=128`), which is passed to the receiver. **Note:** `total_latency_ms` includes `sender_ms` (sender generation time).
  2. `json_structured`: Pre-constructed structured context serialized as compact JSON (`sender_gen_tokens = 0`).
  3. `cmcp_structured`: Pre-constructed `CMCPContext` rendered as compact `CMCP_CTX` text (`sender_gen_tokens = 0`).
  4. `cmcp_binary`: Same `CMCPContext` encoded into a 64-byte header + compressed TLV binary frame via `cmcp.dll`, decoded on receiver, and rendered to `CMCP_CTX` (`sender_gen_tokens = 0`).
- **Metrics:** `message_bytes`, `sender_gen_tokens`, `receiver_input_tokens`, `tokenize_ms`, `prefill_ms`, `eval_ms`, `codec_ms`, `total_latency_ms`, `max_rss_mib`, `throughput_tok_s`, `quality_score` (deterministic alias recall in `[0, 1]`), and `info_fidelity_score` (key-item preservation in transmitted payload).

### 2.2 Experiment C: Multi-Turn Token Reduction
- **Script:** [run_exp_c.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/experiments/token_reduction/run_exp_c.py)
- **Design:** 3 multi-turn scenarios (`S1_protocol_debugging`, `S2_latency_budget`, `S3_kv_guardrails`) × 3 repetitions = **9 runs per condition** (**36 runs total**), direction `Qwen2.5-0.5B->Llama-3.2-1B`.
- **Conditions:**
  1. `full_history`: Full multi-turn conversation history passed to receiver.
  2. `conventional_summary`: Sender (`Qwen2.5-0.5B`) generates a natural-language summary (includes sender generation time in `total_latency_ms`).
  3. `cmcp_references`: CMCP binary frame carrying entities, constraints, and resolved `Reference` records.
  4. `cmcp_incremental`: CMCP binary frame carrying only the incremental delta (`Evidence` + `Constraint`) for the latest turn.

### 2.3 Experiment D: Cross-Model Shared Latent Representation Alignment
- **Script:** [run_exp_d.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/experiments/latent_alignment/run_exp_d.py)
- **Design:** Extracts final-layer hidden states from `SmolLM2-135M-Instruct` (`d=576`) and `Qwen2.5-0.5B-Instruct` (`d=896`) on a strictly disjoint **64-train / 24-eval** context split across technical, reasoning, and code domains.
- **Variables:** 2 pooling strategies (`mean_pooled`, `last_token`) × 4 projection methods (`random`, `procrustes`, `ridge`, `mlp`).
- **Metrics:** Linear CKA (`cka_before`, `cka_after`), `eval_mse`, `eval_cosine_sim`, `eval_top1_acc`, `eval_top5_acc`, `train_time_ms`, `eval_infer_time_ms`.

### 2.4 Experiment E: Approximate Internal State / Soft-Prompt Transfer
- **Script:** [run_exp_e.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/experiments/state_transfer/run_exp_e.py)
- **Design:** Trains a lightweight adapter to project `K=8` chunk-pooled hidden states from `SmolLM2-135M-Instruct` (`d=576`) into `Qwen2.5-0.5B-Instruct`'s (`d=896`) `inputs_embeds` space with the receiver frozen. Evaluated on **20 unseen factual QA examples** disjoint from the **48 training examples**.
- **Conditions:** `zero_context` (question only), `random_projection` (`K=8` unlearned embeddings + question), `learned_cmcp` (`K=8` trained embeddings + question), `full_text` (original context + question).
- **Metrics:** `mean_nll`, `perplexity`, `exact_match_rate`, `keyword_recall`, `mean_receiver_input_tokens`, `mean_latency_ms`.

### 2.5 Experiment F: Real `llama.cpp` KV-Cache Compatibility Verification
- **Script:** [run_exp_f.py](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/experiments/kv_compatibility/run_exp_f.py)
- **Design:** Runs `cmcp_llama_harness.exe --mode kv-compat` (`--n-ctx 512 --threads 4`) with `qwen2.5-0.5b-instruct-q4_k_m.gguf` and `Llama-3.2-1B-Instruct-Q4_K_M.gguf` to compare `llama_state_seq_set_data` acceptance against `cmcp_check_kv_compat` across 4 cases (`same_model_same_cfg`, `same_model_diff_n_ctx`, `same_model_diff_pos_offset`, `cross_architecture`), plus Python `ctypes` SDK verification.
