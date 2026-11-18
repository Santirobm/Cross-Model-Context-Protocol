# Cross-Model Context Protocol (CMCP) 2.2 — Real-World Multi-Agent Context Transfer Validation

**Status:** Experimental Validation Complete (`eval_hetero`, `eval_homo`, `scaling_v1`, `validation_report.json` passed)
**Baseline Commit:** `5b23076` (CMCP 2.1) -> Audit `9c3957d` -> CMCP 2.2

---

## 1. Executive Summary & Core Hypothesis

**Hypothesis Tested:**
> *An incremental structured context protocol (CMCP) allows heterogeneous and homogeneous LLM agents to share, update, and reuse working state more efficiently end-to-end than full history, rolling summary, or full JSON—including extraction cost—without degrading task quality.*

### Verdict Summary
| Claim / Question | Classification | Evidence Summary |
|---|---|---|
| **Incremental wire & token efficiency** | **DEMOSTRADO** | `cmcp_incremental` cuts wire bytes by **63.6%** (`686.4 B -> 250.0 B` on 8-turn eval) and **91.5%** cumulative at 40 turns (`94,625 B -> 8,005 B`), with constant `O(1)` per-turn update (`311 B`) and **41.5–42.2%** fewer input tokens (`p < 0.05`, 95% CI). |
| **End-to-end speedup & quality gain (Homogeneous `Qwen2.5-0.5B -> Qwen2.5-0.5B`)** | **DEMOSTRADO** | `cmcp_incremental` outperforms `full_history` in both latency (`-123.36 ms [-156.52, -92.74]`) and quality (`exact_match +0.22 [+0.14, +0.30]`, `task_completion_rate +0.34 [+0.27, +0.41]`, `obsolete_info_rate 0.00` vs `0.14`). |
| **End-to-end speedup & zero quality loss (Heterogeneous `Qwen2.5-0.5B -> Llama-3.2-1B`)** | **PARCIALMENTE DEMOSTRADO** | `cmcp_incremental` matches `full_history` on `exact_match` (`-0.05 [-0.12, +0.02]`), cuts `input_tokens` by `-99.58 [-100.28, -98.91]`, and cuts `prefill_ms` by **42.3%** (`847.3 ms -> 488.7 ms`), **but** `Llama-3.2-1B` generates more output tokens when reading structured lines, making `e2e_accum_ms` `+222.98 ms [+118.68, +326.06]` slower and lowering `relation_recovery` (`-0.26 [-0.33, -0.19]`). |
| **LLM-assisted extraction / Rolling Summary on small local models (`0.5B`)** | **REFUTADO** | Both `rolling_summary` (`+5,725 ms E2E`, `-0.22 EM`) and `cmcp_inc_llm_ext` (`+4,477 ms E2E`, `-0.42 EM`) severely degrade both latency and accuracy compared to `full_history` and deterministic extraction (`0.064 ms`). |
| **Cross-architecture KV-cache / latent transfer** | **REFUTADO (CMCP 2.1)** | Not re-run per scope; already refuted in CMCP 2.1 (`PPL = 493,744`, `EM = 0.0`). |

---

## 2. Experimental Setup & Hardware

- **Machine:** Windows 10 Pro x64, Intel Core i5-13500 (14C / 20T), 15.70 GiB RAM, CPU-only inference.
- **Inference Engine:** `llama.cpp` via C99 `build/cmcp_llama_harness.exe --mode persistent` (`threads=4`, greedy decoding, `n_ctx=1024` for 8-turn eval, `n_ctx=2048` for 5–40 turn scaling).
- **Models:**
  - Agent A (`CMCP_MODEL_A`): `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468.6 MiB)
  - Agent B (`CMCP_MODEL_B`): `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (770.3 MiB) in `eval_hetero`; `qwen2.5-0.5b-instruct-q4_k_m.gguf` in `eval_homo`.
- **Sequential Execution:** Agent A runs first and unloads before Agent B loads, preventing RAM contention.
- **Statistical Methodology:** `n = 50` disjoint held-out `eval` scenarios across 5 unseen domains (`fin_ledger`, `med_triage`, `supply_chain`, `ci_release`, `energy_grid`), with 2 evaluation queries per scenario (`Turn 4` intermediate handoff + `Turn 8` composite final query = `100` rows per method, `600` rows total per run). All 95% confidence intervals are **scenario-clustered bootstrap** (`5,000` resamples over the `50` scenarios).

---

## 3. Dataset Splits & Anti-Leakage Discipline

- **`dev` (8 scenarios):** `telecom_core`, `iot_fleet` — initial smoke verification (`dev_smoke`).
- **`tune` (8 scenarios):** `telecom_core`, `iot_fleet` — prompt rendering adjustment (`tune_check`, `--max-tokens 64`).
- **`eval` (50 scenarios):** Disjoint domains (`fin_ledger`, `med_triage`, `supply_chain`, `ci_release`, `energy_grid`) covering all 10 hard cases: entity/value updates, invalidation/tombstones, near-duplicate distractors, ternary relations, numeric constraints, temporal ordering, conflict resolution, and composite multi-hop queries.
- **Zero-Leakage Guarantee:** `DeterministicExtractor` inspects **only** `TurnEvent.utterance` (verified by AST check in `experiments/cmcp22/audit/validate_cmcp22.py`: 0 `.ops` attribute accesses, 0 extraction errors across 858 extracted ops).

---

## 4. Versioned Conflict-Aware State Store (`representations/state_store.py`)

`CMCPStateStore` implements:
1. **Idempotent `UPSERT` / `INVALIDATE` ops** keyed by `(fact_id, seq, version, agent_id, digest)`.
2. **Out-of-order stale rejection** and **tombstone resurrection prevention** (an `INVALIDATE` at `seq_2` blocks stale `UPSERT` at `seq_1 < seq_2`).
3. **Concurrent conflict policies** (`lww` and `reject`) recording `ConflictRecord` entries.
4. **Reference resolution & time-travel:** `save_snapshot(ref_id)`, `resolve_reference(ref_id)` (raising `ReferenceMissingError` on missing refs), and `checkout_seq(target_seq)`.
5. **Compaction:** `compact(prune_tombstones_before_seq=...)` trimming per-entry history and expired tombstones.
6. **Unit Tests:** `tests/test_state_store_consistency.py` (6/6 passing).

---

## 5. Main Results: Heterogeneous (`Qwen2.5-0.5B -> Llama-3.2-1B`, `eval_hetero`, `n=50`)

| Method | Wire Bytes | Input Tok | Extract ms | Prefill ms | E2E ms | EM | Rel Rec | Constr Sat | Temp Cons | Task Comp | Obs Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_history` | 686.4 | 236.0 | 0.00 | 847.3 | **2516.6** | **0.62** | **1.00** | **0.85** | 0.98 | **0.77** | 0.02 |
| `rolling_summary` | 709.8 | **108.6** | 5997.51 | **386.7** | 8241.2 | 0.40 | 0.50 | 0.56 | 0.91 | 0.40 | 0.09 |
| `json_structured` | 799.6 | 286.7 | 0.06 | 1018.2 | 2759.9 | 0.40 | **1.00** | 0.58 | **1.00** | 0.47 | **0.00** |
| `cmcp_full` | 697.7 | 136.4 | 0.06 | 488.7 | 2738.2 | 0.57 | 0.74 | 0.72 | **1.00** | 0.58 | **0.00** |
| `cmcp_incremental` | **250.0** | 136.4 | 0.06 | 488.7 | 2739.6 | 0.57 | 0.74 | 0.72 | **1.00** | 0.58 | **0.00** |
| `cmcp_inc_llm_ext` | 474.6 | 218.6 | 4308.70 | 777.8 | 6993.6 | 0.20 | 0.61 | 0.55 | 0.88 | 0.20 | 0.12 |

### Paired Differences vs `full_history` (`eval_hetero`, 95% Scenario-Clustered Bootstrap CI)
- **`cmcp_incremental`:**
  - `input_tokens`: **-99.58** `[-100.28, -98.91]` (-42.2%)
  - `e2e_accum_ms`: **+222.98 ms** `[+118.68, +326.06]` (prefill is 358.6 ms faster, but `Llama-3.2-1B` generates longer answers on structured lines)
  - `exact_match`: **-0.05** `[-0.12, +0.02]` (statistically tied)
  - `relation_recovery`: **-0.26** `[-0.33, -0.19]`
  - `constraint_satisfaction`: **-0.13** `[-0.22, -0.04]` (beats `json_structured` `-0.27` and `rolling_summary` `-0.29`)
- **`json_structured`:** `+50.72` tokens, `+243.33 ms`, `-0.22 EM`, `-0.27` constraint satisfaction.
- **`rolling_summary`:** `+5724.66 ms` E2E, `-0.22 EM`, `-0.50` relation recovery.
- **`cmcp_inc_llm_ext`:** `+4476.98 ms` E2E, `-0.42 EM`, `-0.57` task completion.

---

## 6. Homogeneous Comparison (`Qwen2.5-0.5B -> Qwen2.5-0.5B`, `eval_homo`, `n=50`)

| Method | Wire Bytes | Input Tok | Prefill ms | E2E ms | EM | Rel Rec | Constr Sat | Temp Cons | Task Comp | Obs Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_history` | 686.4 | 272.9 | 874.5 | 1916.5 | 0.53 | **0.95** | 0.59 | 0.86 | 0.54 | 0.14 |
| `rolling_summary` | 709.8 | **121.1** | **381.1** | 7647.2 | 0.29 | 0.50 | 0.63 | 0.80 | 0.29 | 0.20 |
| `json_structured` | 799.6 | 311.5 | 985.6 | 2199.6 | 0.50 | 0.91 | 0.59 | **1.00** | 0.53 | **0.00** |
| `cmcp_full` | 697.7 | 159.7 | 497.5 | 1801.5 | **0.75** | 0.90 | **0.80** | **1.00** | **0.88** | **0.00** |
| `cmcp_incremental` | **250.0** | 159.7 | 491.7 | **1793.1** | **0.75** | 0.90 | **0.80** | **1.00** | **0.88** | **0.00** |
| `cmcp_inc_llm_ext` | 474.6 | 236.5 | 744.6 | 6360.8 | 0.24 | 0.59 | 0.50 | 0.86 | 0.24 | 0.14 |

### Paired Differences vs `full_history` (`eval_homo`, 95% CI)
- **`cmcp_incremental`:**
  - `input_tokens`: **-113.18** `[-113.84, -112.54]` (-41.5%)
  - `e2e_accum_ms`: **-123.36 ms** `[-156.52, -92.74]` (faster end-to-end!)
  - `exact_match`: **+0.22** `[+0.14, +0.30]`
  - `constraint_satisfaction`: **+0.21** `[+0.12, +0.30]`
  - `task_completion_rate`: **+0.34** `[+0.27, +0.41]`
  - `relation_recovery`: **-0.05** `[-0.11, 0.00]` (tied)

**Key Takeaway:** Smaller/weaker receiver models (`Qwen2.5-0.5B`) suffer heavily from obsolete facts in `full_history` (`obsolete_info_rate = 0.14`, `EM = 0.53`), so CMCP's tombstone-aware state store yields huge accuracy **and** latency wins. Stronger/more verbose receivers (`Llama-3.2-1B`) handle raw `full_history` well (`EM = 0.62`, `obs = 0.02`) and need concise formatting prompts so structured context doesn't trigger extra generation tokens.

---

## 7. Long-Horizon Scaling (5 to 40 Turns, `Llama-3.2-1B`, `n_ctx=2048`, `scaling_v1`)

| Horizon | Method | Cum Wire Bytes | Last Wire Bytes | Input Tokens | Prefill ms |
|---:|---|---:|---:|---:|---:|
| **5** | `full_history` | 1,649 | 553 | 192 | 587.6 |
| **5** | `json_structured` | 3,352 | 888 | 312 | 957.0 |
| **5** | `cmcp_incremental` | **1,119** | **304** | **117** | **371.4** |
| **10** | `full_history` | 6,170 | 1,129 | 365 | 1,152.3 |
| **10** | `json_structured` | 10,011 | 1,467 | 516 | 1,621.8 |
| **10** | `cmcp_incremental` | **2,103** | **312** | **180** | **539.5** |
| **20** | `full_history` | 23,955 | 2,299 | 711 | 2,272.3 |
| **20** | `json_structured` | 32,253 | 2,659 | 924 | 2,973.0 |
| **20** | `cmcp_incremental` | **4,069** | **311** | **306** | **1,095.8** |
| **40** | `full_history` | 94,625 | 4,639 | 1,403 | 5,396.3 |
| **40** | `json_structured` | 112,497 | 5,043 | 1,740 | 6,946.4 |
| **40** | `cmcp_incremental` | **8,005** | **311** | **558** | **2,054.3** |

At **40 turns**:
- `cmcp_incremental` cuts cumulative wire bytes by **91.5%** vs `full_history` and **92.9%** vs `json_structured`, keeping per-turn wire size constant (`311 B`).
- `cmcp_incremental` cuts prefill latency by **2.63x** vs `full_history` (`2,054 ms` vs `5,396 ms`) and **3.38x** vs `json_structured` (`6,946 ms`).
- `checkout_seq(target_seq=30)` completed in **0.0777 ms**; `compact()` pruned **38** obsolete history/tombstone ops.

---

## 8. Reproducibility & Validation Artifacts

- `experiments/cmcp22/dataset.py` — 66 deterministic multi-turn scenarios (`dev`, `tune`, `eval`)
- `experiments/cmcp22/extractors.py` — `DeterministicExtractor` & `LLMExtractor`
- `experiments/cmcp22/run_multiturn_benchmark.py` — Main benchmark runner
- `experiments/cmcp22/run_scaling_benchmark.py` — 5–40 turn scaling benchmark
- `experiments/cmcp22/audit/validate_cmcp22.py` -> `experiments/cmcp22/audit/validation_report.json` (`overall_passed: true`)
- Raw CSVs & Summaries: `experiments/cmcp22/results/multiturn_runs_{eval_hetero,eval_homo,tune_check,dev_smoke}.csv` and `scaling_runs_scaling_v1.csv`.
