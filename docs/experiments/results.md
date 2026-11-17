# Experimental Results

All tables and numbers below are taken directly from the raw outputs in [results/raw/](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/benchmark_summary.json).

---

## 1. Experiments A & B: Structured Cross-Model Context Transfer

Raw data: [exp_ab_results.json](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_ab_results.json), [exp_ab_results.csv](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_ab_results.csv).  
Chart: [exp_ab_comparison.png](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/plots/exp_ab_comparison.png).

### 1.1 Direction: `Qwen2.5-0.5B -> Llama-3.2-1B` (`n_runs = 24` per condition)

| Condition | Wire Bytes (mean) | Sender Gen Tokens | Receiver In Tokens | Prefill (ms) | Codec (ms) | Total Latency (ms, mean / p95) | Quality Score | Info Fidelity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nl_conversation` | 317.125 | 85.125 | 141.75 | 500.648 | 0.0000 | 4929.7055 / 6391.6925 | 0.8167 | 0.8833 |
| `json_structured` | 371.750 | 0.000 | 165.75 | 576.786 | 0.0848 | 2653.2407 / 3716.0293 | **0.9083** | **1.0000** |
| `cmcp_structured` | **274.125** | 0.000 | 161.375 | 561.176 | 0.0158 | 3561.6944 / 4279.9410 | 0.7438 | **1.0000** |
| `cmcp_binary` | 325.875 | 0.000 | 161.375 | 564.156 | 0.3548 | 3560.5143 / 4288.0181 | 0.7438 | **1.0000** |

### 1.2 Direction: `Llama-3.2-1B -> Qwen2.5-0.5B` (`n_runs = 24` per condition)

| Condition | Wire Bytes (mean) | Sender Gen Tokens | Receiver In Tokens | Prefill (ms) | Codec (ms) | Total Latency (ms, mean / p95) | Quality Score | Info Fidelity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nl_conversation` | 471.750 | 127.125 | 192.50 | 588.347 | 0.0000 | 7020.7347 / 7502.7004 | 0.6937 | 0.8083 |
| `json_structured` | 371.750 | 0.000 | 167.875 | 549.743 | 0.0744 | **1602.1010 / 2361.6699** | **0.8583** | **1.0000** |
| `cmcp_structured` | **274.125** | 0.000 | 163.500 | 529.724 | 0.0148 | 1936.6377 / 2482.5240 | 0.7583 | **1.0000** |
| `cmcp_binary` | 325.875 | 0.000 | 163.500 | 519.425 | 0.2736 | 1925.9530 / 2471.4389 | 0.7583 | **1.0000** |

**Key Observations (Exp A & B):**
- `cmcp_structured` achieved the smallest wire footprint (`274.125` bytes) and `cmcp_binary` added only `0.27`–`0.35 ms` codec overhead while guaranteeing CRC32C integrity and `info_fidelity_score = 1.0`.
- **However, `cmcp_structured` and `cmcp_binary` scored lower on `quality_score` (`0.7438` and `0.7583`) than `json_structured` (`0.9083` and `0.8583`)** on these 0.5B/1B models, because the compact `CMCP_CTX` text format triggered longer/less accurate completions (`eval_ms` `3000.28 ms` vs `2076.14 ms` on Llama; `1406.55 ms` vs `1051.92 ms` on Qwen).
- `nl_conversation` had the highest total latency (`4929.71 ms` and `7020.73 ms`) because it includes sender generation (`85.125` and `127.125` tokens), and lost information (`info_fidelity_score` `0.8833` and `0.8083`).

---

## 2. Experiment C: Multi-Turn Token Reduction (`Qwen2.5-0.5B -> Llama-3.2-1B`)

Raw data: [exp_c_results.json](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_c_results.json), [exp_c_results.csv](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_c_results.csv).  
Chart: [exp_c_tokens_latency.png](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/plots/exp_c_tokens_latency.png).

| Condition (`n_runs = 9`) | Wire Bytes (mean) | Receiver In Tokens (mean) | Prefill (ms, mean) | Total Latency (ms, mean / p95) | Quality Score (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `full_history` | 796.33 | 296.00 | 1186.558 | 3915.091 / 5790.839 | 0.8889 |
| `conventional_summary` | 398.33 | 169.33 | 668.629 | 6383.302 / 8993.516 | 0.4444 |
| `cmcp_references` | 361.67 | 172.67 | 687.696 | 3802.966 / 6251.347 | **1.0000** |
| `cmcp_incremental` | **248.33** | **128.33** | **509.690** | **3664.379 / 5849.488** | 0.7778 |

**Key Observations (Exp C):**
- `cmcp_references` achieved **1.0000** quality recall (higher than `full_history` at `0.8889` and more than 2x `conventional_summary` at `0.4444`) while reducing wire bytes by **54.6%** and receiver input tokens by **41.7%**.
- `cmcp_incremental` achieved the lowest tokens (`128.33`, **-56.6%**) and lowest latency (`3664.379 ms`), with `0.7778` quality recall.

---

## 3. Experiment D: Cross-Model Latent Representation Alignment

Raw data: [exp_d_latent_alignment.json](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_d_latent_alignment.json), [exp_d_latent_alignment.csv](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_d_latent_alignment.csv).  
Pair: `SmolLM2-135M-Instruct` (`d=576`) -> `Qwen2.5-0.5B-Instruct` (`d=896`), `n_train = 64`, `n_eval = 24`. Extraction times: sender `62902.12 ms`, receiver `172854.64 ms`.

| Pooling | Method | CKA Before | CKA After | Eval MSE | Eval Cosine Sim | Top-1 Acc | Top-5 Acc | Train (ms) | Eval Infer (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mean_pooled` | `random` | 0.878888 | 0.878888 | 46.203339 | -0.014406 | 0.041667 | 0.166667 | 0.001 | 0.250 |
| `mean_pooled` | `procrustes` | 0.878888 | 0.878888 | 10.729951 | 0.876393 | 0.833333 | **1.000000** | 99.037 | 0.345 |
| `mean_pooled` | `ridge` | 0.878888 | **0.950601** | **6.266980** | **0.926500** | 0.833333 | 0.958333 | 1102.290 | 0.357 |
| `mean_pooled` | `mlp` | 0.878888 | 0.938037 | 7.232878 | 0.914000 | **0.875000** | 0.958333 | 3578.530 | 1.338 |
| `last_token` | `random` | 0.933289 | 0.933289 | 91.901520 | 0.010059 | 0.083333 | 0.125000 | 0.002 | 0.254 |
| `last_token` | `procrustes` | 0.933289 | 0.933289 | 20.834435 | 0.864029 | 0.666667 | 0.916667 | 95.505 | 0.340 |
| `last_token` | `ridge` | 0.933289 | **0.975185** | **10.852974** | **0.929830** | 0.666667 | 0.875000 | 19.388 | 0.388 |
| `last_token` | `mlp` | 0.933289 | 0.945502 | 13.107480 | 0.919964 | 0.500000 | 0.875000 | 3359.143 | 1.277 |

---

## 4. Experiment E: Approximate Internal State / Soft-Prompt Transfer

Raw data: [exp_e_state_transfer.json](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_e_state_transfer.json), [exp_e_state_transfer.csv](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_e_state_transfer.csv).  
Pair: `SmolLM2-135M-Instruct` -> `Qwen2.5-0.5B-Instruct`, `n_latents = 8`, `n_train = 48`, `n_eval = 20`. Adapter train time: `111657.14 ms`.

| Condition (`n_eval = 20`) | Mean NLL | Perplexity | Exact Match Rate | Keyword Recall | Mean Receiver In Tokens | Mean Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero_context` | 2.7108 | 15.0412 | 0.00 | 0.05 | 13.5 | 2154.25 |
| `random_projection` | 2.5347 | 12.6130 | 0.00 | 0.05 | 21.5 | 2234.97 |
| `learned_cmcp` | 1.5738 | 4.8249 | **0.00** | **0.05** | 21.5 | 2692.36 |
| `full_text` | **0.3717** | **1.4503** | **0.85** | **0.95** | 30.8 | 3311.11 |

**Critical Finding:** Although `learned_cmcp` reduced NLL (`2.7108` -> `1.5738`) and perplexity (`15.04` -> `4.82`), it **failed factual exact match (`0.0`) and keyword recall (`0.05`)** on unseen entities, showing that NLL improvement on a small frozen-receiver soft-prompt adapter did not translate into factual recall.

---

## 5. Experiment F: Real `llama.cpp` KV-Cache Compatibility

Raw data: [exp_f_kv_compat.json](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/results/raw/exp_f_kv_compat.json).  
Primary model: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (`qwen2`); secondary: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (`llama`). Harness RSS: `1427.72 MiB`.

| Case | KV State Bytes | `llama_state_seq_set_data` | `cmcp_kv_compatible` | CMCP Reason (Code) |
| :--- | ---: | :--- | :--- | :--- |
| `same_model_same_cfg` | 99,008 | Accepted (`99,008` B) | `true` | `exact KV-cache match (Tier 0)` (`0`) |
| `same_model_diff_n_ctx` (`512` vs `256`) | 99,008 | Accepted (`99,008` B) | `false` | `n_ctx mismatch` (`7`) |
| `same_model_diff_pos_offset` (`0` vs `16`) | — | (not tested in harness) | `false` | `pos_offset mismatch` (`8`) |
| `cross_architecture` (`qwen2` vs `llama`) | 99,008 | Rejected (`0` B) | `false` | `architecture mismatch` (`1`, non-strict fallback Tier `2`) |
