# CMCP Optimization Mission — GATE 2.5: Factorial Ablation & Unified Scaling Report

**Branch:** `exp/gate2-llama-ablation` (branched from `exp/gate2-adapters` @ `ed53ee5`; `main` untouched)
**Date:** 2026-11-20
**Dataset & Protocol:** `eval` split (`n = 50` scenarios × 4 queries × 3 reps = `600` rows/variant, `3,600` rows total), `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, real `llama_tokenize()` counts, `0` extra LLM calls.

---

## 1. 2×2 Factorial Ablation on `Llama-3.2-1B` (`gate2_llama_ablation`)

| Variant | Dedup (`; attr=val`) | Order | In Tok | Out Tok | Net Tok | E2E (ms) | EM | Ent Rec | Rel Rec | Constr Sat | Obs Rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_r3` | No (full store) | Original | 156.0 | 61.1 | 217.1 | 2,967.7 | 0.695 | 0.908 | 0.850 | 0.850 | 0.000 |
| `ref_var_a` | No (full `r3_canon`) | Original | 151.5 | **59.8** | 211.3 | 2,902.1 | **0.710** | 0.907 | 0.855 | **0.880** | 0.000 |
| `abl_A` | No | Original | 107.1 | 61.3 | 168.5 | 2,783.6 | 0.600 | **0.910** | **0.865** | 0.760 | 0.000 |
| `abl_B` | Yes | Original | **103.6** | 62.0 | 165.6 | 2,775.6 | 0.520 | 0.897 | 0.845 | 0.750 | 0.000 |
| `abl_C` | No | Unified | 107.1 | 60.7 | 167.8 | **2,735.2** | 0.560 | 0.892 | 0.835 | 0.745 | 0.000 |
| `abl_D` (`var_d_unified`) | Yes | Unified | **103.6** | 61.4 | **165.0** | 2,748.4 | 0.490 | 0.888 | 0.830 | 0.750 | 0.000 |

### Paired Differences vs. `baseline_r3` (95% Scenario-Clustered Bootstrap CI)
- **`ref_var_a`:** `EM +0.015 [-0.020, +0.050]`, `Constr Sat +0.030 [0.000, +0.060]`, `E2E -65.7 ms [-95.2, -39.2]`.
- **`abl_A`:** `EM -0.095 [-0.150, -0.040]`, `Constr Sat -0.090 [-0.145, -0.040]`, `In Tok -48.9`, `E2E -184.2 ms`.
- **`abl_B`:** `EM -0.175 [-0.245, -0.100]`, `Constr Sat -0.100 [-0.155, -0.050]`, `In Tok -52.4`, `E2E -192.1 ms`.
- **`abl_C`:** `EM -0.135 [-0.205, -0.070]`, `Constr Sat -0.105 [-0.155, -0.055]`, `In Tok -48.9`, `E2E -232.5 ms`.
- **`abl_D`:** `EM -0.205 [-0.295, -0.120]`, `Constr Sat -0.100 [-0.150, -0.050]`, `In Tok -52.4`, `E2E -219.3 ms`.

---

## 2. Root-Cause Attribution (Additive Decomposition)

The `-0.205` EM drop of `abl_D` (`var_d_unified`) vs `baseline_r3` decomposes cleanly and additively into three factors:
1. **Task-selection pruning (`abl_A` vs `baseline_r3`):** **`-0.095` EM** and **`-0.090` Constr Sat** (removing surrounding store context hurts `Llama-3.2-1B` even when 100% of gold keywords are present).
2. **Entity attribute deduplication (`abl_B` vs `abl_A`: `-0.080`; `abl_D` vs `abl_C`: `-0.070`):** **`-0.075` EM mean effect** while saving only `3.54` input tokens.
3. **Unified section order `<facts>` before `<constraints>` (`abl_C` vs `abl_A`: `-0.040`; `abl_D` vs `abl_B`: `-0.030`):** **`-0.035` EM mean effect**.

**Cross-Validation Decision (Step 2.2):** Because no intermediate cell (`abl_A`, `abl_B`, `abl_C`) recovered `EM` or `Constr Sat` within the GATE 2 `<= 0.02` drop threshold on `Llama-3.2-1B`, cross-validation on `Qwen2.5-0.5B` and `SmolLM2-360M` was not triggered.

---

## 3. Direct 40-Turn Scaling of `var_d_unified` (`gate2_scaling`, `Llama-3.2-1B`, `n_ctx=2048`)

| Horizon (Turns) | `baseline_r3` In Tok | `var_d_unified` In Tok | Token Reduction | `baseline_r3` Prefill (ms) | `var_d_unified` Prefill (ms) | `var_d_unified` Select (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 145 | **87** | -40.0% | 456.0 | **277.5** | 0.30 |
| 10 | 208 | **87** | -58.2% | 650.1 | **279.1** | 0.16 |
| 20 | 334 | **87** | -73.9% | 1,056.8 | **282.2** | 0.38 |
| 40 | 586 | **87** | **-85.2%** | 1,914.9 | **286.3** (`6.69×`) | 0.62 |

---

## 4. Verification & Decision Summary

- **Tests:** `ctest --test-dir build`, `ctest --test-dir build-san`, and `pytest tests/ -v` all passing.
- **Architectural Takeaway:**
  - `ref_var_a` (`r3_canon`, canonical full state) is the **only** configuration that preserves/improves `Llama-3.2-1B` accuracy (`+0.015 EM`, `+0.030 Constr Sat`, `-65.7 ms` E2E), at the cost of linear `O(T)` token growth on long horizons.
  - `var_d_unified` delivers constant `O(1)` **`87` tokens** at 40 turns (`-85.2%`) and passes all 9 thresholds on `SmolLM2-360M` and `Qwen2.5-0.5B`, **but** on `Llama-3.2-1B` selection + deduplication + reordering together cause `-0.205 EM` (`abl_A` without dedup/reorder reduces the loss by more than half to `-0.095 EM`).
