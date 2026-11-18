# Cross-Model Context Protocol (CMCP)

An experimental research protocol and C99/Python implementation for **incremental structured context transfer** between heterogeneous LLM agents.

> **Research Scope & Honesty Note:** CMCP evaluates *contextual state transfer* (structured facts, relations, constraints, decisions, and incremental references), **not** cross-architecture KV-cache transfer. Controlled experiments in CMCP 2.1 (`SmolLM2-135M -> Qwen2.5-0.5B`) showed that linear/Ridge projection of KV-caches across different architectures fails functionally (`PPL = 493,744`, `EM = 0.0`), and latent `inputs_embeds` projection without end-to-end receiver tuning fails factual transfer (`EM = 0.0`).

---

## 1. What Problem Does CMCP Solve?

When multi-agent pipelines collaborate over long or evolving tasks, re-sending the full conversation history (`full_history`) or verbose JSON on every handoff increases token count, prefill latency, and exposure to stale/contradictory turns.

CMCP separates:
- **Contextual Transfer (Level A — Working):** Typed binary TLV framing (`cmcp.dll` / `libcmcp.a`), CRC32C integrity, versioned state updates with tombstones/invalidation, and incremental references.
- **Internal State / KV Transfer (Level B — Refuted for Cross-Arch Without E2E Training):** Investigated and documented as non-viable via linear adapters across heterogeneous models; `cmcp_check_kv_compat()` explicitly rejects incompatible KV states.

---

## 2. Demonstrated Results & Limitations (CMCP 2.2 Multi-Turn Validation, `n = 50` Disjoint Eval Scenarios)

Evaluated across 50 held-out 8-turn scenarios (5 unseen domains, 2 queries/scenario = 100 eval points/method) with scenario-clustered bootstrap 95% CIs ([FINAL_RESEARCH_REPORT_V3.md](FINAL_RESEARCH_REPORT_V3.md)):

| Setup & Method | Wire Bytes | Input Tok | E2E ms | Exact Match | Rel Rec | Constr Sat | Task Comp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hetero (`Qwen2.5-0.5B -> Llama-3.2-1B`)** | | | | | | | |
| `full_history` | 686.4 | 236.0 | **2,516.6** | **0.62** | **1.00** | **0.85** | **0.77** |
| `rolling_summary` | 709.8 | **108.6** | 8,241.2 | 0.40 | 0.50 | 0.56 | 0.40 |
| `json_structured` | 799.6 | 286.7 | 2,759.9 | 0.40 | **1.00** | 0.58 | 0.47 |
| `cmcp_incremental` (det. ext) | **250.0** | 136.4 | 2,739.6 | 0.57 | 0.74 | 0.72 | 0.58 |
| `cmcp_inc_llm_ext` (0.5B LLM ext) | 474.6 | 218.6 | 6,993.6 | 0.20 | 0.61 | 0.55 | 0.20 |
| **Homo (`Qwen2.5-0.5B -> Qwen2.5-0.5B`)** | | | | | | | |
| `full_history` | 686.4 | 272.9 | 1,916.5 | 0.53 | **0.95** | 0.59 | 0.54 |
| `cmcp_incremental` (det. ext) | **250.0** | **159.7** | **1,793.1** | **0.75** | 0.90 | **0.80** | **0.88** |

**What worked:**
- **Wire & Token Efficiency:** `cmcp_incremental` cuts wire bytes by **63.6%** on 8-turn eval and **91.5%** cumulative at 40 turns (`94,625 B -> 8,005 B`, constant `311 B` per turn), reducing input tokens by **41.5–42.2%** and prefill latency by **42.3%** (`2.63x` at 40 turns).
- **Homogeneous Quality & Speedup:** On `Qwen2.5-0.5B -> Qwen2.5-0.5B`, `cmcp_incremental` beats `full_history` in both E2E latency (`-123.4 ms [-156.5, -92.7]`) and quality (`EM +0.22 [+0.14, +0.30]`, `task_completion +0.34 [+0.27, +0.41]`).

**Honest limitations & negative results:**
- **Heterogeneous E2E & Relation Gap (`Llama-3.2-1B`):** On `Llama-3.2-1B`, `cmcp_incremental` matches `full_history` on `EM` (`-0.05 [-0.12, +0.02]`), and XML-lite `R3` raises `EM` to `0.61` (`-0.01` vs `0.62`), **but** `Llama-3.2-1B` still generates more output tokens (`+17.8` tokens, `+442.1 ms` E2E) and trails on `relation_recovery` (`-0.30`). On a second pair (`Qwen2.5-0.5B -> SmolLM2-360M`), `cmcp_r3` achieves `-396.0 ms` E2E speedup and `+0.15` `relation_recovery`, but loses `-0.15 EM` ([FINAL_RESEARCH_REPORT_V4.md](FINAL_RESEARCH_REPORT_V4.md)).
- **Unstructured Natural-Language Extraction:** On 30 unstructured dialogues with informal revisions, `full_history` collapses (`0.00 EM`, `0.80` obsolete rate), `oracle_structured` (`tool_use` upper bound) reaches `0.267 EM` / `0.90` constraints, while CPU `deterministic_nlp_rules` drops to `0.556` state precision/recall and `0.5B` `hybrid_gated` adds `3,704 ms` without recovering missed facts ([FINAL_RESEARCH_REPORT_V4.md](FINAL_RESEARCH_REPORT_V4.md)).
- **0.5B LLM Extraction & Rolling Summary:** Both `rolling_summary` (`+5,725 ms E2E`, `-0.22 EM`) and `cmcp_inc_llm_ext` (`+4,477 ms E2E`, `-0.42 EM`) are slower and much less accurate than `full_history` and deterministic extraction (`0.06 ms`).
- **Cross-Arch KV & Latent Transfer:** Refuted for factual transfer without end-to-end training ([FINAL_RESEARCH_REPORT_V2.md](FINAL_RESEARCH_REPORT_V2.md)).

---

## 3. Build, Test & Reproduce

```sh
# 1. Build C99 core & tests (requires CMake, Ninja, Clang/GCC)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

# 2. Optional: Build with llama.cpp harness
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMCP_BUILD_LLAMA_HARNESS=ON -DCMCP_LLAMA_DIR=/path/to/llama.cpp
cmake --build build

# 3. Python tests & CMCP 2.2 validation
.venv\Scripts\pytest.exe tests/ -v
.venv\Scripts\python.exe -m experiments.cmcp22.run_multiturn_benchmark --split eval --reps 1 --max-tokens 64 --model-a $CMCP_MODEL_A --model-b $CMCP_MODEL_B --tag eval_hetero
.venv\Scripts\python.exe -m experiments.cmcp22.run_scaling_benchmark --model $CMCP_MODEL_B --tag scaling_v1 --turns 5 10 20 40
.venv\Scripts\python.exe experiments/cmcp22/audit/validate_cmcp22.py
```

---

## 4. License

Apache License 2.0 — see [LICENSE](LICENSE).
