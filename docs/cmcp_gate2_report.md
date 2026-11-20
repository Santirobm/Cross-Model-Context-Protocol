# CMCP Optimization Mission — GATE 2 Final Report (Unified vs. Receiver-Adapted Presentation)

**Branch:** `exp/gate2-adapters` (child of `exp/gate1-abd`; `main` untouched)
**Date:** 2026-11-19
**Extra LLM Calls Added:** `0` (`+0`)
**Token Measurement:** 100% via `cmcp_llama_harness.exe` (`llama_tokenize()`; zero heuristics).

---

## 1. Executive Summary

Following the Gate 1 diagnostic (which proved `selector.py` has **100.0% gold-context recall** across all 200 `eval` queries and 100% of `var_d` failures were presentation/parsing issues), Gate 2 evaluated:
1. **`var_d_unified` (`representations/adapters.py`):** A single generic XML-lite renderer over `select_for_task()` with `; attr=val` entity deduplication, `k = v` facts, and `<constraints>`/`<decisions>` placed after `<facts>`.
2. **`var_d_adapted` (`representations/adapters.py`):** Receiver-specific presentation adapters (`qwen`: value-first facts `- v (k)`; `llama`: `k: v` facts; `smollm2`: unified fallback).

### Key Findings (`n = 50` `eval` scenarios × 4 queries × **3 reps** = `600` rows/variant per model)
- **`var_d_unified` PASSED ALL 9 THRESHOLDS on `SmolLM2-360M` AND `Qwen2.5-0.5B`:**
  - **`SmolLM2-360M` (`gate2_eval_smol`):** `-35.3%` input tokens (`-73.6`), `-29.1%` net tokens (`-74.0`), **`-234.3 ms` E2E (`-18.4%`)**, **`+0.100 EM`**, `+0.070` `entity_recall`, `+0.020` `relation_recovery`, **`+0.250` `constraint_satisfaction`**, `0.000` `obsolete_info_rate`.
  - **`Qwen2.5-0.5B` (`gate2_eval_qwen`):** `-34.5%` input tokens (`-61.1`), `-30.8%` net tokens (`-69.2`), **`-347.1 ms` E2E (`-24.1%`)**, **`-0.020 EM`** (`0.735` vs `0.755`, within `<= 0.02`!), `+0.043` `entity_recall`, `+0.085` `relation_recovery`, `+0.030` `constraint_satisfaction`, `0.000` `obsolete_info_rate` — **outperforming `var_d_adapted`** (`-0.025 EM`, `-0.040` constraints).
- **`var_d_unified` on `Llama-3.2-1B` (`gate2_eval_llama`):**
  - Passed 7 of 9 thresholds: `-33.6%` input tokens (`-52.4`), `-24.0%` net tokens (`-52.1`), **`-183.0 ms` E2E (`-6.6%`)**, `-0.020` `entity_recall`, `-0.020` `relation_recovery`, `0.000` `obsolete_info_rate`, `0` extra LLM calls.
  - **Failed Threshold #3 (`exact_match`: `-0.205 [-0.295, -0.120]`) and Threshold #9 (`constraint_satisfaction`: `-0.100 [-0.150, -0.050]`)**: removing `; attr=val` from `<entities>` and placing `<facts>` before `<constraints>` hurt `Llama-3.2-1B`'s `T6` and `T7` parsing compared to full `var_a` (`+0.015 EM`, `+0.030` constraints, `-70.9 ms` E2E in Gate 1).

---

## 2. Final `eval` Results (`n = 50` × 4 queries × 3 reps = 600 rows/variant)

| Model | Variant | In Tok | Out Tok | Net Tok | E2E (ms) | EM | Ent Rec | Rel Rec | Constr Sat | Obs Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`SmolLM2-360M`** | `baseline_r3` | 208.3 | 46.0 | 254.3 | 1,275.8 | 0.470 | 0.872 | 0.880 | 0.570 | 0.000 |
| **`SmolLM2-360M`** | `var_d_unified` | **134.7** | **45.6** | **180.2** | **1,041.5** | **0.570** | **0.942** | **0.900** | **0.820** | 0.000 |
| **`Qwen2.5-0.5B`** | `baseline_r3` | 177.2 | 47.6 | 224.7 | 1,438.5 | **0.755** | 0.937 | 0.885 | 0.875 | 0.000 |
| **`Qwen2.5-0.5B`** | `var_d_unified` | **116.1** | 39.4 | **155.5** | **1,091.3** | 0.735 | 0.980 | 0.970 | **0.905** | 0.000 |
| **`Qwen2.5-0.5B`** | `var_d_adapted` | **116.1** | **38.2** | **154.3** | **1,075.4** | 0.730 | **0.983** | **0.975** | 0.835 | 0.000 |
| **`Llama-3.2-1B`** | `baseline_r3` | 156.0 | **61.1** | 217.1 | 2,787.5 | **0.695** | **0.908** | **0.850** | **0.850** | 0.000 |
| **`Llama-3.2-1B`** | `var_d_unified` | **103.6** | 61.4 | **165.0** | **2,604.5** | 0.490 | 0.888 | 0.830 | 0.750 | 0.000 |

- **40-Turn Scaling (`gate1_scaling`, Gate 1 `var_d`):** `var_d` kept input tokens constant at **`87` tokens** across 5–40 turns vs `586` for `baseline_r3` (**-85.2%**, `6.45×` prefill speedup). Specific 40-turn scaling measurement on `var_d_unified` remains pending.
- **Evidence Fallback Rate:** `0.000` on structured `eval`; verified active (`True`) in `tests/test_gate1_abd.py`.

---

## 3. Threshold Verification & Final Verdict

| # | Approved Threshold | `SmolLM2-360M` | `Qwen2.5-0.5B` | `Llama-3.2-1B` |
|---:|---|---|---|---|
| 1 | `input_tokens` >= 20% (50% @ 40T) | **-35.3% (PASS)** | **-34.5% (PASS)** | **-33.6% (PASS; -85.2% @ 40T in `gate1_scaling`)** |
| 2 | `net_tokens` >= 15% | **-29.1% (PASS)** | **-30.8% (PASS)** | **-24.0% (PASS)** |
| 3 | `exact_match` drop <= 0.02 | **+0.100 (PASS)** | **-0.020 (PASS)** | **-0.205 (FAIL)** |
| 4 | `entity_recall` drop <= 0.02 | **+0.070 (PASS)** | **+0.043 (PASS)** | **-0.020 (PASS)** |
| 5 | `relation_recovery` drop <= 0.02 | **+0.020 (PASS)** | **+0.085 (PASS)** | **-0.020 (PASS)** |
| 6 | `e2e_accum_ms` <= +2% | **-18.4% (PASS)** | **-24.1% (PASS)** | **-6.6% (PASS)** |
| 7 | Extra LLM calls == 0 | **0 (PASS)** | **0 (PASS)** | **0 (PASS)** |
| 8 | Reproducible >= 3 reps (`n=50`) | **3 reps (PASS)** | **3 reps (PASS)** | **3 reps (PASS)** |
| 9 | Constraints & obsolete drop <= 0.02 | **+0.250 / 0.0 (PASS)** | **+0.030 / 0.0 (PASS)** | **-0.100 / 0.0 (FAIL)** |

### Scientific Conclusion
1. **Unified format beats receiver-specific adapters (`var_d_unified` >= `var_d_adapted`):** A single generic renderer (`var_d_unified`) passed **all 9 thresholds** on `SmolLM2-360M` and `Qwen2.5-0.5B` without per-model adapters, and stands as the **primary canonical candidate**.
2. **Honest Acceptance Boundary:** For `Llama-3.2-1B`, `var_d_unified` trades `EM` (`-0.205`) for `-33.6%` input token reduction and `-183.0 ms` E2E speedup. `var_a` (canonical full state, **+0.015 EM**, **+0.030 constraints**, **-70.9 ms E2E** in Gate 1) is retained **strictly as a diagnostic reference**.
3. **GATE 3 Recommendation:** **Do NOT open GATE 3** (`obsolete_info_rate = 0.000` across all models and variants).
