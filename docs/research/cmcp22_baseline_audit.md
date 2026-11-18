# CMCP 2.2 Baseline Audit (Statistical & Methodological Review of CMCP 2.1)

**Audited Commit:** `5b23076` (`main`)  
**Audited Artifact:** `experiments/context_transfer/results/context_benchmark_runs.csv` (`270` rows = `18` eval tasks × `3` reps × `5` methods)

---

## 1. Independence & Intra-Task Variability

- In CMCP 2.1, inference used **greedy decoding** (`llama_sampler_init_greedy()`). Consequently, `intra-task std(exact_match) = 0.0000` across all 3 repetitions on every method.
- The `54` rows per method are **not** 54 independent quality samples; the true statistical unit is **`n = 18` tasks** (repeated 3 times only to measure CPU timing variance: `mean intra-task std(end_to_end_ms) = 49.88–109.60 ms`).

---

## 2. Task-Level Clustered Bootstrap (10,000 resamples, 95% CI, `n = 18`)

| Method | Recv Tokens [95% CI] | End-to-End ms [95% CI] | Exact Match [95% CI] | F1 [95% CI] | KW Recall [95% CI] | Relation Recovery [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| `full_history` | 233.67 [232.67, 234.83] | 1,893.32 [1,814.37, 1,972.81] | 0.6111 [0.3889, 0.8333] | 0.4843 [0.3900, 0.5792] | 0.8704 [0.7963, 0.9445] | 0.4444 [0.2500, 0.6389] |
| `json_structured` | 409.83 [408.39, 411.28] | 2,777.56 [2,753.47, 2,803.03] | 0.1667 [0.0000, 0.3333] | 0.2545 [0.1948, 0.3221] | 0.6667 [0.5741, 0.7593] | 0.7500 [0.5833, 0.8889] |
| `cmcp_textual` | 308.83 [307.39, 310.28] | 2,370.01 [2,317.89, 2,406.82] | 0.8889 [0.7222, 1.0000] | 0.4844 [0.4220, 0.5434] | 0.9630 [0.9074, 1.0000] | 0.1667 [0.0556, 0.2778] |
| `cmcp_binary_tlv` | 308.83 [307.39, 310.22] | 2,378.03 [2,325.51, 2,421.61] | 0.8889 [0.7222, 1.0000] | 0.4844 [0.4215, 0.5440] | 0.9630 [0.9074, 1.0000] | 0.1667 [0.0556, 0.2778] |
| `cmcp_incremental` | 124.83 [123.83, 125.72] | 1,694.10 [1,676.13, 1,712.69] | 0.8333 [0.6667, 1.0000] | 0.4549 [0.3722, 0.5253] | 0.8889 [0.7778, 1.0000] | 0.0833 [0.0000, 0.1667] |

### Paired Comparison: `cmcp_incremental` minus `full_history` (`n = 18`)

| Metric | Mean Diff | 95% Bootstrap CI | Wins / Ties / Losses | Verdict |
|---|---:|---:|---:|---|
| `receiver_input_tokens` | -108.83 (-46.6%) | [-109.61, -108.17] | 0 / 0 / 18 (fewer tokens) | **Survives** (conditional on pre-structured facts) |
| `end_to_end_ms` | -199.22 (-10.5%) | [-274.68, -124.69] | 2 / 0 / 16 (faster) | **Survives** (excludes LLM extraction cost) |
| `exact_match` | +0.2222 | [+0.0000, +0.4444] | 5 / 12 / 1 | **Borderline** (CI touches 0.0; small `n=18`) |
| `f1` | -0.0294 | [-0.1278, +0.0775] | 8 / 0 / 10 | **No significant difference** |
| `keyword_recall` | +0.0185 | [-0.1111, +0.1296] | 5 / 10 / 3 | **No significant difference** |
| `relation_recovery` | **-0.3611** | **[-0.5556, -0.1667]** | **0 / 10 / 8** | **Statistically significant degradation** |

---

## 3. Root Causes & Methodological Gaps Identified

1. **Why `relation_recovery` dropped (`0.0833` vs `0.4444`):**
   - `cmcp_incremental` in CMCP 2.1 omitted the `relations` list from the persisted delta (only persisting `active_facts`), AND `Answer concisely:` led `Llama-3.2-1B` to emit key-value answers (`"Active port: 7243 Owner: Team-B13"`) omitting the source entity (`Node-*`/`Shard-*`) needed by the substring relation scorer.
2. **Missing Extraction Cost:** CMCP 2.1 measured `prep_extract_ms` from already-structured Python dicts (`0.07 ms`). A real multi-agent system must pay either deterministic parsing cost or LLM extraction cost.
3. **Single-Turn Contexts Only:** CMCP 2.1 only tested 1-shot static tasks (`n = 18`), not evolving multi-turn conversations with updates, corrections, tombstones, and conflicts.
4. **C99 Core & Latent/KV Status:** C99 core (`test_cmcp_core` Release + ASan/UBSan) and Python (`7/7` pytest) pass. Cross-architecture KV transfer (`SmolLM2-135M -> Qwen2.5-0.5B`) was cleanly refuted (`PPL = 493,744`, `EM = 0.0`).
