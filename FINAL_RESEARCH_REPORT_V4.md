# Cross-Model Context Protocol (CMCP) 2.2b — External Validation, Semantic Portability & Unstructured Extraction

**Repository:** `Cross-Model-Context-Protocol` (`main`)
**Baseline Commit:** `a244429` (CMCP 2.2)
**Audit Status:** `experiments/cmcp22b/audit/validation_report_cmcp22b.json` (`overall_passed: true`, `max_diff: 0.0` across 4,984 rows)

---

## 1. Executive Summary & Answers to the 5 Core Questions

1. **Clean From-Scratch Reproduction & Latency Variance (Gate 1, `--reps 3`):**
   - **Exact reproducibility confirmed:** `repro_hetero` (1,800 rows) and `repro_homo` (1,800 rows) reproduced **bit-for-bit** all token counts and quality scores from `FINAL_RESEARCH_REPORT_V3.md` (`exact_match`, `relation_recovery`, `input_tokens`), with CPU `e2e_accum_ms` within ~1.0–1.8% of V3 (`repro_hetero`: `full_history` `2559.5 ms` vs `2516.6 ms`; `repro_homo`: `cmcp_incremental` `1825.3 ms` vs `1793.1 ms`).
2. **Did Semantic Re-Rendering Solve Verbosity & `relation_recovery` on `Qwen -> Llama`? (Gate 2, Pair 1):**
   - **NO (`H_port_1` REFUTED on `Llama-3.2-1B`).** On `tune`, XML-lite `R3` outperformed `R1` and `R2` (`EM 0.75`, `constr_sat 0.875`), and on `eval` (`n = 50, reps = 3`), `cmcp_r3` raised `exact_match` to `0.61` (`-0.01 [-0.08, +0.06]` vs `full_history`'s `0.62`) and cut `prefill_ms` by `-257.6 ms`. However, `Llama-3.2-1B` still generated `+17.78 [+15.83, +19.65]` extra `output_tokens` (`+442.1 ms [+363.7, +516.3]` E2E) and suffered `-0.30 [-0.36, -0.23]` on `relation_recovery` due to greedy repetition loops on Turn 8.
3. **Did the Result Generalize to a Second Heterogeneous Pair Without Ad-Hoc Tuning? (Gate 2, Pair 2):**
   - **PARCIALMENTE DEMOSTRADO.** On `Qwen2.5-0.5B -> SmolLM2-360M-Instruct` (`eval`, `n = 50, reps = 3`), frozen `cmcp_r3` **eliminated verbosity** (`-8.51` `output_tokens`), achieved a **net E2E speedup** (`-396.04 ms [-436.38, -357.37]`), improved `relation_recovery` (`+0.15 [+0.08, +0.22]`), and reduced `obsolete_info_rate` from `0.28` to `0.00`, **but** lost `exact_match` (`-0.15 [-0.21, -0.09]`, `0.32` vs `0.47`).
4. **Real Precision & Cost Frontier on Unstructured Natural-Language Dialogues (Gate 3, `n = 30`):**
   - `full_history_baseline` collapsed to `0.00 EM` (`0.80` `obsolete_info_rate`) under informal revisions (`"scratch X"`, `"forget the previous limit"`, `"retiring TK-10"`).
   - `E1` (`oracle_structured`, simulating native `tool_use` capture): `state_precision = 1.00`, `state_recall = 1.00`, `0.039 ms` extraction, `0.2667 EM` (`+0.2667` vs `full_history`), `0.90` `constraint_satisfaction`.
   - `E2` (`deterministic_nlp_rules`): `0.109 ms` extraction, **but** phrasing variation caused `state_precision = 0.5555`, `state_recall = 0.5555`, `obsolete_pruning_accuracy = 0.2417`, `0.1333 EM`.
   - `E3` (`hybrid_gated` with `Qwen2.5-0.5B`): added `3,704.0 ms` extraction latency without recovering missed facts (`state_recall = 0.5555`, `e2e_accum_ms = +4,700.2 ms` vs `full_history`).
5. **Final Verdict on CMCP Maturity:**
   - **DEMOSTRADO** for **wire/token compression** (`-63.6%` to `-91.5%` wire bytes, `-41.5%` to `-42.2%` input tokens), **homogeneous small-model workflows** (`Qwen2.5-0.5B -> Qwen2.5-0.5B`: `+0.22 EM`, `-101.3 ms E2E`), and **native structured/tool_use state capture** (`E1`).
   - **PARCIALMENTE DEMOSTRADO** for **heterogeneous zero-tuning portability** (`SmolLM2-360M` gains `-396 ms` E2E & `+0.15` `rel_rec` but loses `-0.15 EM`; `Llama-3.2-1B` ties `EM` at `-0.01` but loses E2E due to decode verbosity).
   - **REFUTADO** for **0.5B LLM-assisted extraction from unstructured text** (`hybrid_gated` / `cmcp_inc_llm_ext`) and **cross-architecture KV/latent transfer** (CMCP 2.1).

---

## 2. Models Verified (Hashes, Quantization, Sizes)

| Role / Pair | Model File | Size (Bytes) | SHA-256 Prefix | Quant |
|---|---|---:|---|---|
| Agent A / Homo B | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491,400,032 | `74a4da8c9fdbcd15` | Q4_K_M |
| Pair 1 Agent B | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807,694,464 | `6f85a640a97cf2bf` | Q4_K_M |
| Pair 2 Agent B | `SmolLM2-360M-Instruct-Q4_K_M.gguf` | 270,590,880 | `2fa3f013dcdd7b99` | Q4_K_M |

All experiments ran on Windows 10 Pro x64, Intel Core i5-13500 (CPU-only, `threads=4`, greedy decoding, `bootstrap_seed=42`, sequential model loading/unloading within 15.70 GiB RAM).

---

## 3. Gate 1: Clean Reproduction & OS Latency Variance (`--reps 3`, `n = 50`)

| Run & Method | Input Tok | EM | Rel Rec | Prefill Mean (P50 / P95 / Std) ms | E2E Mean (P50 / P95 / Std) ms |
|---|---:|---:|---:|---:|---:|
| **`repro_hetero` (`Qwen -> Llama`)** | | | | | |
| `full_history` | 235.96 | 0.62 | 1.00 | 849.6 (868.1 / 1013.2 / 138.6) | **2559.5** (2274.4 / 3582.5 / 937.8) |
| `cmcp_incremental` | **136.38** | 0.57 | 0.74 | **489.7** (519.0 / 590.7 / 76.4) | 2789.0 (3047.0 / 3157.4 / 587.2) |
| **`repro_homo` (`Qwen -> Qwen`)** | | | | | |
| `full_history` | 272.86 | 0.53 | 0.95 | 863.7 (929.7 / 1203.3 / 188.3) | 1926.7 (2174.5 / 2408.2 / 453.9) |
| `cmcp_incremental` | **159.68** | **0.75** | 0.90 | **499.2** (528.1 / 693.5 / 98.2) | **1825.3** (1823.5 / 1973.9 / 114.2) |

---

## 4. Gate 2: Semantic Rendering Portability (`R1` / `R2` / `R3`)

### 4.1 `tune` Split Selection (`port_tune`, `n = 8`, `Qwen2.5-0.5B -> Llama-3.2-1B`)
- **`cmcp_r1` (Key-value tags):** `EM = 0.6875`, `rel_rec = 0.6875`, `constr_sat = 0.6875`, `out_tok = 58.81`, `e2e = 2856.1 ms`.
- **`cmcp_r2` (Compact prose):** `EM = 0.5625`, `rel_rec = 0.7500`, `constr_sat = 0.6250`, `out_tok = 61.06`, `e2e = 2929.8 ms`.
- **`cmcp_r3` (XML-lite typed `<relations><entities><constraints><decisions><facts>`):** **`EM = 0.7500`**, `rel_rec = 0.7500`, **`constr_sat = 0.8750`**, `out_tok = 56.88`, `e2e = 2856.8 ms`. -> **`R3` frozen for `eval`.**

### 4.2 Frozen `eval` Comparison (`n = 50`, `--reps 3`, 600 rows/pair)

| Pair & Method | Wire B | In Tok | Out Tok | Prefill ms | Decode ms | E2E ms | EM | Rel Rec | Constr Sat | Obs Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Pair 1 (`Qwen -> Llama-3.2-1B`)** | | | | | | | | | | |
| `full_history` | 686.4 | 236.0 | **42.4** | 845.6 | **1725.1** | **2571.0** | **0.62** | **1.00** | **0.85** | 0.02 |
| `cmcp_r3` | **250.0** | **164.4** | 60.2 | **588.0** | 2424.5 | 3013.1 | 0.61 | 0.70 | 0.74 | **0.00** |
| **Pair 2 (`Qwen -> SmolLM2-360M`)** | | | | | | | | | | |
| `full_history` | 686.4 | 303.1 | 62.1 | 888.0 | 942.5 | 1831.0 | **0.47** | 0.61 | **0.56** | 0.28 |
| `cmcp_r3` | **250.0** | **218.1** | **53.5** | **630.8** | **803.4** | **1435.0** | 0.32 | **0.76** | 0.51 | **0.00** |

---

## 5. Gate 3: Unstructured Natural-Language Extraction (`n = 30` Dialogues, `Llama-3.2-1B` Receiver)

| Extractor | Extract ms | State Prec | State Rec | Obs Prune Acc | In Tok | E2E ms | EM | Rel Rec | Constr Sat | Obs Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `E1 oracle_structured` | **0.039** | **1.000** | **1.000** | **0.733** | 141.0 | 2740.5 | **0.267** | **0.133** | **0.900** | **0.067** |
| `E2 deterministic_nlp_rules` | 0.109 | 0.556 | 0.556 | 0.242 | **140.0** | 2930.0 | 0.133 | **0.133** | 0.333 | 0.700 |
| `E3 hybrid_gated` | 3703.96 | 0.556 | 0.556 | 0.242 | **140.0** | 6637.3 | 0.133 | **0.133** | 0.333 | 0.700 |
| `E4 full_history_baseline` | 0.000 | 1.000 | 1.000 | 0.000 | 227.9 | **1937.2** | 0.000 | 0.000 | 0.433 | 0.800 |

---

## 6. Reproducibility & Verification

- C99 unit tests: `ctest --test-dir build` (`1/1`) & `ctest --test-dir build-san` (`1/1`).
- Python suite: `pytest tests/ -v` (`13/13` passed).
- Audit script: `experiments/cmcp22b/audit/validate_cmcp22b.py` -> `validation_report_cmcp22b.json` (`overall_passed: true`, `max_diff: 0.0`).
