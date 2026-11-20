# CMCP GATE 3 Technical Report: Cross-Matrix Anchored, 1-Hop Graph Expansion, and Deterministic Structured Ingestion

**Branch:** `exp/gate3-enhancements` (branched from `main` @ `469b744`; `main` untouched)  
**Date:** 2026-11-20  
**Local Models Available:** `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (`807.7 MB`), `qwen2.5-0.5b-instruct-q4_k_m.gguf` (`491.4 MB`), `SmolLM2-360M-Instruct-Q4_K_M.gguf` (`270.6 MB`). *(No 7B/8B mid-range GGUF model is present in the local environment.)*  
**Protocol:** `eval` split (`n = 50` scenarios × 4 queries × 3 reps = `600` rows/variant), greedy decoding, real `llama_tokenize()` counts, `0` extra LLM calls.

---

## 1. Deliverable 1 & 2: Full Cross-Matrix (`eval`, n = 50, 3 reps)

### 1.1 `SmolLM2-360M-Instruct-Q4_K_M.gguf` (`gate3_eval_smol`)

| Variant | In Tok | Out Tok | Net Tok | E2E (ms) | Select (ms) | EM | Ent Rec | Rel Rec | Constr Sat | Obs Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_r3` | 208.25 | 46.03 | 254.28 | 1,285.1 | 0.016 | 0.4700 | 0.8717 | 0.8800 | 0.5700 | 0.0000 |
| `compact_h0` (`mode="compact"`, default) | **134.69** | **45.56** | **180.25** | **1,045.9** | 0.218 | **0.5700** | **0.9417** | **0.9000** | **0.8200** | 0.0000 |
| `compact_h1` (`expand_hops=1`) | 160.12 | 47.30 | 207.42 | 1,147.8 | 0.251 | 0.3600 | 0.9417 | 0.9000 | 0.8150 | 0.0000 |
| `anchored` (`mode="anchored"`) | 201.50 | 46.70 | 248.21 | 1,265.9 | 0.030 | 0.5200 | 0.9367 | **0.9000** | 0.6900 | 0.0000 |

- **Paired vs. `baseline_r3` (95% Bootstrap CI):**
  - `compact_h0`: `In Tok -73.56`, `E2E -239.2 ms`, `EM +0.1000 [+0.010, +0.185]`, `Constr Sat +0.2500 [+0.205, +0.295]`.
  - `compact_h1`: `In Tok -48.12`, `E2E -137.3 ms`, `EM -0.1100 [-0.165, -0.055]`, `Constr Sat +0.2450 [+0.195, +0.295]`.
  - `anchored`: `In Tok -6.75`, `E2E -19.2 ms`, `EM +0.0500 [0.000, +0.095]`, `Constr Sat +0.1200 [+0.075, +0.160]`.

### 1.2 `qwen2.5-0.5b-instruct-q4_k_m.gguf` (`gate3_eval_qwen`)

| Variant | In Tok | Out Tok | Net Tok | E2E (ms) | Select (ms) | EM | Ent Rec | Rel Rec | Constr Sat | Obs Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_r3` | 177.15 | 47.55 | 224.71 | 1,462.8 | 0.014 | 0.7550 | 0.9367 | 0.8850 | 0.8750 | 0.0000 |
| `compact_h0` (`mode="compact"`, default) | **116.08** | **39.38** | **155.47** | **1,102.7** | 0.184 | 0.7350 | **0.9800** | **0.9700** | **0.9050** | 0.0000 |
| `compact_h1` (`expand_hops=1`) | 136.93 | 46.16 | 183.09 | 1,303.9 | 0.211 | **0.7600** | **0.9800** | **0.9700** | **0.9050** | 0.0000 |
| `anchored` (`mode="anchored"`) | 172.65 | 45.58 | 218.23 | 1,406.4 | 0.026 | 0.7550 | 0.9400 | 0.8900 | 0.8850 | 0.0000 |

- **Paired vs. `baseline_r3` (95% Bootstrap CI):**
  - `compact_h0`: `In Tok -61.06`, `E2E -360.1 ms`, `EM -0.0200 [-0.045, +0.005]`, `Constr Sat +0.0300 [+0.005, +0.055]`.
  - `compact_h1`: `In Tok -40.23`, `E2E -158.9 ms`, `EM +0.0050 [-0.010, +0.020]`, `Constr Sat +0.0300 [+0.005, +0.055]`.
  - `anchored`: `In Tok -4.50`, `E2E -56.3 ms`, `EM +0.0000 [-0.015, +0.015]`, `Constr Sat +0.0100 [-0.010, +0.030]`.

### 1.3 `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (`gate3_eval_llama`)

| Variant | In Tok | Out Tok | Net Tok | E2E (ms) | Select (ms) | EM | Ent Rec | Rel Rec | Constr Sat | Obs Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_r3` | 156.00 | 61.08 | 217.08 | 2,773.3 | 0.013 | 0.6950 | **0.9083** | 0.8500 | 0.8500 | 0.0000 |
| `compact_h0` (`mode="compact"`, default) | **103.58** | 61.37 | **164.96** | **2,588.4** | 0.178 | 0.4900 | 0.8883 | 0.8300 | 0.7500 | 0.0000 |
| `compact_h1` (`expand_hops=1`) | 122.78 | 60.23 | 183.00 | 2,618.8 | 0.204 | 0.4550 | 0.8883 | 0.8300 | 0.7050 | 0.0000 |
| `anchored` (`mode="anchored"`) | 151.50 | **59.84** | 211.34 | 2,714.4 | 0.025 | **0.7100** | 0.9067 | **0.8550** | **0.8800** | 0.0000 |

- **Paired vs. `baseline_r3` (95% Bootstrap CI):**
  - `compact_h0`: `In Tok -52.41`, `E2E -184.9 ms`, `EM -0.2050 [-0.295, -0.120]`, `Constr Sat -0.1000 [-0.150, -0.050]`.
  - `compact_h1`: `In Tok -33.23`, `E2E -154.5 ms`, `EM -0.2400 [-0.310, -0.170]`, `Constr Sat -0.1450 [-0.205, -0.090]`.
  - `anchored`: `In Tok -4.50`, `E2E -58.9 ms`, `EM +0.0150 [-0.020, +0.050]`, `Constr Sat +0.0300 [0.000, +0.060]`.

---

## 2. Long-Horizon Scaling (5, 10, 20, 40 Turns) Across All 3 Local Models

| Model | Horizon | `baseline_r3` Tok (Prefill) | `anchored` Tok (Prefill) | `compact_h0` Tok (Prefill / Sel) | `compact_h1` Tok (Prefill / Sel) |
|---|---:|---:|---:|---:|---:|
| `Llama-3.2-1B` | 5T | 145 (459.9 ms) | 145 (474.2 ms) | **87** (282.3 ms / 0.422 ms) | 100 (318.8 ms / 0.234 ms) |
| `Llama-3.2-1B` | 10T | 208 (678.1 ms) | 208 (725.6 ms) | **87** (306.8 ms / 0.160 ms) | 100 (335.1 ms / 0.183 ms) |
| `Llama-3.2-1B` | 20T | 334 (1,115.9 ms) | 334 (1,088.1 ms) | **87** (286.3 ms / 0.219 ms) | 100 (339.6 ms / 0.235 ms) |
| `Llama-3.2-1B` | 40T | 586 (2,037.9 ms) | 586 (1,891.5 ms) | **87** (288.3 ms / 0.370 ms) | 100 (331.0 ms / 0.374 ms) |
| `Qwen2.5-0.5B` | 5T | 156 (560.6 ms) | 156 (485.5 ms) | **88** (269.6 ms / 0.303 ms) | 101 (578.8 ms / 0.272 ms) |
| `Qwen2.5-0.5B` | 10T | 234 (749.3 ms) | 234 (805.1 ms) | **89** (380.6 ms / 0.344 ms) | 102 (405.0 ms / 0.332 ms) |
| `Qwen2.5-0.5B` | 20T | 402 (1,276.9 ms) | 402 (1,229.5 ms) | **89** (271.6 ms / 0.246 ms) | 102 (348.3 ms / 0.345 ms) |
| `Qwen2.5-0.5B` | 40T | 738 (2,450.9 ms) | 738 (2,356.2 ms) | **89** (289.3 ms / 0.389 ms) | 102 (323.4 ms / 0.455 ms) |
| `SmolLM2-360M` | 5T | 181 (459.8 ms) | 181 (512.5 ms) | **106** (302.3 ms / 0.313 ms) | 121 (336.6 ms / 0.253 ms) |
| `SmolLM2-360M` | 10T | 266 (778.9 ms) | 266 (793.1 ms) | **107** (309.8 ms / 0.181 ms) | 122 (345.5 ms / 0.199 ms) |
| `SmolLM2-360M` | 20T | 448 (1,357.3 ms) | 448 (1,324.6 ms) | **107** (333.2 ms / 0.249 ms) | 122 (358.1 ms / 0.275 ms) |
| `SmolLM2-360M` | 40T | 812 (2,524.5 ms) | 812 (2,976.4 ms) | **107** (308.6 ms / 0.401 ms) | 122 (349.3 ms / 0.433 ms) |

### Promotion Decision for `expand_hops=1`
- Although `compact_h1` preserves sub-millisecond selection (`0.37–0.46 ms` at 40T) and `O(1)` scaling (`+13` to `+15` tokens vs `compact_h0`), it **does not recover** `Llama-3.2-1B` (`EM` `0.4900 -> 0.4550`, `Constr Sat` `0.7500 -> 0.7050`) and **degrades** `SmolLM2-360M` (`EM` `0.5700 -> 0.3600`).
- **Decision:** Keep **`expand_hops=0` as the default** for `select_for_task` and `serialize`, and expose `expand_hops` strictly as an **explicit opt-in parameter** (useful e.g. on `Qwen2.5-0.5B` where `compact_h1` lifts `EM` from `0.7350` to `0.7600`).

---

## 3. Deliverable 3: Deterministic Structured Ingestion Layer (`ingestion/structured_hooks.py`)

- **`ingest_tool_output(store, payload, ...)`:** Flattens JSON/dict payloads deterministically, infers `CMCPStateStore` categories via declarative mapping/rules, enforces idempotency (re-ingesting identical payloads emits `0` ops and preserves `state_hash()`), and updates/invalidates changed or revoked keys (`obsolete_info_rate = 0.0`).
  - **Measured Latency (500 runs):** `mean = 0.0548 ms`, `p95 = 0.0569 ms` (`0` LLM calls).
- **`extract_explicit_constraints(text, store=None, ...)`:** Auditable rule/pattern extractor for prohibitions, numeric/SLA limits, named thresholds, and mandatory versions/formats, with zero false positives on neutral prose.
  - **Measured Latency (500 runs):** `mean = 0.0470 ms`, `p95 = 0.0505 ms` (`0` LLM calls).
- **Unit & Integration Coverage:** 4 dedicated tests in `tests/test_gate3_enhancements.py` covering idempotency, monotonic updates/invalidations, zero false positives, and end-to-end `serialize(..., mode="compact"|"anchored")`.

---

## 4. Objective Promotion Recommendation to `main`

1. **Promote to `main`:**
   - The deterministic ingestion module (`ingestion/structured_hooks.py`) + its tests (`tests/test_gate3_enhancements.py`).
   - The `expand_hops: int = 0` optional parameter in `select_for_task` / `serialize` (preserving `expand_hops=0` as default).
   - Full GATE 3 empirical artifacts (`docs/cmcp_gate3_report.md` and `experiments/gate3/`).
2. **Keep Optional / Non-Default:**
   - `expand_hops=1` remains opt-in (`expand_hops=0` default) because it failed the cross-model promotion criterion.
   - `mode="anchored"` remains the recommended fallback whenever `Llama-3.2-1B`-class receivers require full-store canonical accuracy (`EM +0.015`, `Constr Sat +0.030` vs `baseline_r3` across all 3 models: `>= baseline_r3` on Llama, Qwen, and SmolLM2).
