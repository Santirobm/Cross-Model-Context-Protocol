# CMCP Optimization Mission — GATE 0: Audit, Diagnosis & Minimal Design (A+B+D)

**Repository:** `Cross-Model-Context-Protocol` (`main`)
**Audited Commit:** `e33d026` (`docs: refine CMCP README with clear overview and protocol design`)
**Working Tree at Audit:** Clean (`git status --short` empty; 0 uncommitted changes)
**Constraint:** No source code, tests, benchmarks, formats, or configs modified in GATE 0.

---

## 1. Examined Components & Baseline State

| Component | Path | Current Role |
|---|---|---|
| Versioned State Store | `representations/state_store.py` | `CMCPStateStore`, `StateOp` (`UPSERT`, `INVALIDATE`), tombstones, `lww`/`reject`, `checkout_seq`, `compact`, `render_for_llm(style="r1"\|"r2"\|"r3")` |
| Binary TLV Codec | `representations/binary/` | zlib-compressed typed TLV frames + CRC32C (`~250 B` incremental delta) |
| Structured Schema | `representations/structured/schema.py` | `CMCPContext`, `to_wire_prompt`, `to_typed_tlv_records` |
| C99 Core & Harness | `core/`, `benchmarks/cmcp_llama_harness.c` | `cmcp.dll` + persistent `llama.cpp` runner returning exact `llama_tokenize()` counts (`n_input_tokens`, `n_gen_tokens`), `tokenize_ms`, `prefill_ms`, `eval_ms`, `rss_mib` |
| Multi-Turn Dataset | `experiments/cmcp22/dataset.py` | `8 dev`, `8 tune` (`telecom_core`, `iot_fleet`), `50 eval` (5 disjoint domains, 4 queries/scenario: T4, T6, T7, T8) |
| Extractors | `experiments/cmcp22/extractors.py` | `DeterministicExtractor` (`0.064 ms`, 0 errors on structured turns) & `LLMExtractor` (`4,309 ms`, `-0.42 EM`) |
| Test Suite | `tests/` | `13` pytest tests (`test_protocol_and_adapters.py`, `test_state_store_consistency.py`) + CTest `test_cmcp_core` |

---

## 2. Technical Diagnosis & Evidence of Real Problems

### 2.1 Existing Heterogeneous Baseline Results (CMCP 2.2 & 2.2b)
- **Pair 1 (`Qwen2.5-0.5B -> Llama-3.2-1B`, `eval_pair1_r3`, `n = 50` × 3 reps, T4+T8):**
  - `full_history`: `236.0` input tok, `42.4` output tok, `845.6 ms` prefill, `1,725.1 ms` decode, **`2,571.0 ms` E2E**, `0.62 EM`, `1.00` `relation_recovery`, `0.85` `constraint_satisfaction`.
  - `cmcp_r3`: `164.4` input tok (`-30.3%`), **`60.2` output tok (`+17.8`)**, `588.0 ms` prefill (`-257.6 ms`), **`2,424.5 ms` decode (`+699.4 ms`)**, **`3,013.1 ms` E2E (`+442.1 ms` regression)**, `0.61 EM`, **`0.70` `relation_recovery` (`-0.30`)**, `0.74` `constraint_satisfaction`.
- **Pair 2 (`Qwen2.5-0.5B -> SmolLM2-360M`, `eval_pair2_r3`, `n = 50` × 3 reps, T4+T8):**
  - `full_history`: `303.1` input tok, `62.1` output tok, `1,831.0 ms` E2E, `0.47 EM`, `0.61` `relation_recovery`.
  - `cmcp_r3`: `218.1` input tok (`-28.0%`), `53.5` output tok (`-8.5`), `1,435.0 ms` E2E (`-396.0 ms`), **`0.32 EM` (`-0.15` regression)**, `0.76` `relation_recovery` (`+0.15`).
- **40-Turn Scaling (`scaling_v1`, `Llama-3.2-1B`):**
  - `cmcp_incremental` still sent `558` input tokens (`2,054.3 ms` prefill) because `active_view()` rendered **all 40+ active items** (including `shard_1..shard_40` telemetry) regardless of the target task.

### 2.2 Root Causes Identified in Code Inspection
1. **Unconditional Full-Store Rendering (Lack of Task-Conditioned Selection — Gap D):**
   - `render_for_llm()` dumps every active entry in `CMCPStateStore`. For `T4` (*route code + TEMP validity*), `T6` (*SLA + decision + audit tag*), and `T7` (*initial anchor*), 50–85% of transmitted state items are irrelevant to the query.
2. **Inline Noise & Redundancy in Canonical Values (Gap A):**
   - `ent:primary` embeds `primary; anchor=ANCHOR-Q408` while `fact:anchor` separately stores `ANCHOR-Q408` (duplication).
   - `dec:mode` embeds `tier_mode=strict_v2(from:tier_mode=standard)` (leaking superseded decision text into active state, increasing tokens and confusing small receivers).
   - `StateEntry` lacks `evidence_ref` (source turn/span pointer) and `confidence`, forcing either raw-history retention or loss of provenance.
3. **Flat Relation Storage Without Typed Edges (Gap B):**
   - `rel:path3` is stored as a flat string `Vault-408 -> Clearing-412 -> Settlement-417` alongside `ent:twin` (`Vault-408B: standby_backup_only`) without explicit typed edges (`ROUTES_TO`, `CONSTRAINS`, `DEPENDS_ON`, `REPLACES`, `PART_OF`, `CONTRADICTS`), contributing to `Llama-3.2-1B` Turn 8 repetition loops (`Vault-408 -> Vault-408B -> ...`) and preventing graph-based dependency closure during selection.
4. **Extraction Cost Boundary (Verified in CMCP 2.2b):**
   - Calling a `0.5B` LLM for extraction (`LLMExtractor` / `hybrid_gated`) costs `3,704–4,477 ms` and degrades quality (`-0.42 EM`), so GATE 1 must **not** add per-transfer LLM calls.

---

## 3. Minimal Technical Design for GATE 1 (A + B + D)

All additions to `CMCPStateStore` / `StateOp` will remain **100% backward-compatible** (optional fields with defaults; existing 13 tests must continue passing unchanged).

### Component A — Canonical State & Recoverable Evidence (`representations/state_store.py`)
- Add optional metadata fields on `StateOp` and `StateEntry`: `source: str = "system"`, `confidence: float = 1.0`, `evidence_ref: Optional[str] = None` (e.g., `"turn:3#sha256_8"`).
- Add a lightweight `EvidenceStore` (`dict[str, str]` mapping `evidence_ref -> raw_excerpt`) kept **off the wire** by default and retrieved only on explicit `resolve_evidence(ref)` (when uncertainty or missing detail requires original text).
- Normalize canonical values cleanly (e.g., `tier_mode=strict_v2` without inline `(from:...)` clutter; history remains in `entry.history` / `checkout_seq()`).

### Component B — Typed Semantic Relations (`representations/state_store.py`)
- Add `TypedRelation(src_id, rel_type, dst_id, seq, active=True)` supporting typed edges:
  `DEPENDS_ON`, `REPLACES`, `CONSTRAINS`, `CONTRADICTS`, `PRECEDES`, `EXCEPTION_TO`, `PART_OF`, `CONDITIONED_BY`, `ROUTES_TO`.
- Populate typed edges deterministically from `StateOp`s (e.g. `REPLACES` on state mutation, `CONSTRAINS` linking active constraints to primary entity, `ROUTES_TO` for multi-hop paths), enabling both **dependency-aware selection** and **explicit relation rendering** without `-> standby` confusion.

### Component D — Task-Conditioned Selection (`representations/selector.py`)
- Deterministic, zero-LLM-call `select_for_task(query: str, store: CMCPStateStore, max_items: Optional[int] = None) -> SelectedContext`:
  1. **Query Intent & Entity Matching:** Tokenize query into normalized terms/entities/categories (constraints, decisions, relations, anchors, validity/tombstone checks).
  2. **Typed Relation Closure:** Traverse `TypedRelation` edges (`ROUTES_TO`, `CONSTRAINS`, `DEPENDS_ON`, `REPLACES`) from matched entities so required constraints, decisions, and multi-hop relations are preserved even without literal keyword overlap.
  3. **Tombstone / Contradiction Check:** If the query mentions an entity/token present in `store._entries` with `tombstone=True`, emit an explicit compact `INVALIDATED: <key>` marker instead of omitting it silently.
  4. **Evidence Fallback Trigger:** Only attach `EvidenceStore` excerpt if `confidence < 0.7` or required slot is unresolved.

### Estimated Overhead & Risks
- **Overhead:** `< 0.2 ms` CPU per transfer (pure Python dict/set/regex), `0` extra LLM calls, `0` extra output tokens.
- **Risks:** Over-pruning an implicit constraint or historical anchor (`T7`) if query-to-category mapping misses a synonym. Mitigated by always including active constraints/decisions linked via `CONSTRAINS`/`DEPENDS_ON` edges to any matched entity, and evaluating on **all 4 query types (T4, T6, T7, T8)** on a separate `eval` split.

---

## 4. Experimental Plan & Token Measurement Rule

1. **Strict Token Rule:** Every input/output token count will be measured using `cmcp_llama_harness.exe` (`llama_tokenize()` of each real GGUF model). **Zero** byte/char heuristics (`len(text)/4`).
2. **Splits:** Selection rules designed/verified only on `dev` + `tune` (`16` scenarios); frozen before evaluating on `eval` (`50` scenarios × 5 unseen domains, all 4 queries T4, T6, T7, T8 = `200` queries/rep, `3` reps = `600` rows/variant) and `40`-turn scaling.
3. **Model Pairs:**
   - Pair 1: `Qwen2.5-0.5B -> Llama-3.2-1B`
   - Pair 2: `Qwen2.5-0.5B -> SmolLM2-360M`
4. **Mandatory Comparisons:**
   - **Baseline:** Current CMCP (`R3` full active store)
   - **Variant A:** Canonical normalized state + recoverable evidence (`EvidenceStore`)
   - **Variant B:** A + typed semantic relations
   - **Variant D:** A + B + task-conditioned selection (plus individual ablation of D alone if isolated cleanly)
5. **Economic Cost:** Local GGUF models have no API tariff (`$0`); we will report exact input tokens, output tokens, model call counts, and end-to-end latency separately without inventing prices.

---

## 5. Proposed Quantitative Acceptance Thresholds (For Human Approval)

Compared against **Baseline (Current CMCP `R3`)** on `eval` (`n = 50` scenarios, 95% scenario-clustered bootstrap CI) across **both** heterogeneous pairs:

| Criterion | Proposed Threshold (vs. Baseline CMCP `R3`) |
|---|---|
| 1. Minimum receiver `input_tokens` reduction | **>= 20%** on targeted/multi-query eval (`T4+T6+T7+T8`) and **>= 50%** on 40-turn scaling |
| 2. Minimum net token savings (incl. extraction/selection metadata & output tokens) | **>= 15%** net token reduction (`(in_tok + out_tok)`), with `0` extra LLM calls |
| 3. Max tolerable `exact_match` drop | **<= 0.02** (95% CI lower bound `>= -0.04`) |
| 4. Max tolerable `entity_recall` / fact recall drop | **<= 0.02** |
| 5. Max tolerable `relation_recovery` drop | **<= 0.02** (target: **>= +0.10** recovery on `Llama-3.2-1B`) |
| 6. Max tolerable `e2e_accum_ms` increase | **<= 0 ms** (must not regress E2E latency; target net reduction) |
| 7. Additional LLM calls allowed | **0** extra calls (`+0`) |
| 8. Statistical reproducibility | `>= 3` reps on `eval` (`n = 50`), scenario-clustered bootstrap `5,000` resamples (`seed=42`), passing on **both** `Llama-3.2-1B` and `SmolLM2-360M` |

---

## 6. Pending Human Approval Before GATE 1

Please confirm:
1. Approval of the **GATE 1 scope** (implementing **A + B + D** only, keeping backward compatibility and zero extra LLM calls).
2. Approval or adjustment of the **8 quantitative acceptance/rejection thresholds** in Section 5.
