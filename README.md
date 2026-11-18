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

## 2. Demonstrated Results & Limitations (CMCP 2.1 Baseline, `n = 18` Eval Tasks × 3 Reps)

Evaluated on `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (CPU, `llama.cpp` persistent harness) with task-level clustered bootstrap 95% CIs ([docs/research/cmcp22_baseline_audit.md](docs/research/cmcp22_baseline_audit.md)):

| Method | Wire Bytes | Recv Input Tokens | End-to-End (ms) | Exact Match | F1 | Relation Recovery |
|---|---:|---:|---:|---:|---:|---:|
| `full_history` | 709.67 | 233.67 | 1,893.32 | 0.6111 | 0.4843 | **0.4444** |
| `json_structured` | 1,364.00 | 409.83 | 2,777.56 | 0.1667 | 0.2545 | **0.7500** |
| `cmcp_binary_tlv` | 709.17 | 308.83 | 2,378.03 | **0.8889** | 0.4844 | 0.1667 |
| `cmcp_incremental` | **236.17** (+151 B store) | **124.83** (`-46.6%`) | **1,694.10** (`-10.5%`) | 0.8333 | 0.4549 | **0.0833** |

**What worked:**
- `cmcp_incremental` reduced receiver input tokens by **46.6%** (`[-109.6, -108.2]` 95% CI) and end-to-end latency by **10.5%** when facts were pre-structured.

**Known limitations:**
- **Relation Recovery Drop:** `cmcp_incremental` dropped to `0.0833` vs `0.4444` (`full_history`) because relations were omitted from the incremental payload and concise prompting suppressed source-entity naming.
- **Extraction Cost Excluded:** CMCP 2.1 timed conversion from pre-structured objects (`0.07 ms`); real-world LLM extraction cost was not yet included.
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

# 3. Python tests & benchmarks
.venv\Scripts\pytest.exe tests/test_protocol_and_adapters.py -v
.venv\Scripts\python.exe experiments/context_transfer/run_context_benchmark.py --split eval --reps 3
.venv\Scripts\python.exe experiments/audit/validate_cmcp21.py
```

---

## 4. License

Apache License 2.0 — see [LICENSE](LICENSE).
