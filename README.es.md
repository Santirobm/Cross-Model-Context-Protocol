# Cross-Model Context Protocol (CMCP)

Protocolo de investigación e implementación en C99/Python para **transferencia incremental de contexto estructurado** entre agentes LLM heterogéneos.

> **Alcance y Nota de Honestidad:** CMCP evalúa *transferencia contextual* (hechos estructurados, relaciones, restricciones, decisiones y referencias incrementales), **no** transferencia de KV-cache entre arquitecturas distintas. Los experimentos controlados de CMCP 2.1 (`SmolLM2-135M -> Qwen2.5-0.5B`) demostraron que la proyección lineal/Ridge de KV-cache entre arquitecturas diferentes colapsa funcionalmente (`PPL = 493,744`, `EM = 0.0`), y la inyección latente en `inputs_embeds` sin entrenamiento conjunto no transfiere hechos (`EM = 0.0`).

---

## 1. Qué problema resuelve CMCP

En flujos multi-agente, reenviar todo el historial (`full_history`) o JSON verboso en cada traspaso incrementa los tokens de entrada, la latencia de prefill y el riesgo de arrastrar hechos obsoletos o contradictorios.

CMCP separa:
- **Transferencia Contextual (Nivel A — Funcional):** Framing binario TLV tipado (`cmcp.dll` / `libcmcp.a`), integridad CRC32C, estado versionado con tombstones/invalidación y referencias incrementales.
- **Transferencia de Estado Interno / KV (Nivel B — Refutada sin entrenamiento E2E):** Documentada como inviable vía adaptadores lineales entre arquitecturas heterogéneas; `cmcp_check_kv_compat()` rechaza de forma segura estados KV incompatibles.

---

## 2. Resultados Observados y Limitaciones (CMCP 2.1, `n = 18` tareas × 3 reps)

Evaluado sobre `Llama-3.2-1B-Instruct-Q4_K_M.gguf` en CPU con bootstrap agrupado por tarea e IC 95% ([docs/research/cmcp22_baseline_audit.md](docs/research/cmcp22_baseline_audit.md)):

| Método | Bytes Wire | Tokens Receptor | End-to-End (ms) | Exact Match | F1 | Relation Recovery |
|---|---:|---:|---:|---:|---:|---:|
| `full_history` | 709.67 | 233.67 | 1,893.32 | 0.6111 | 0.4843 | **0.4444** |
| `json_structured` | 1,364.00 | 409.83 | 2,777.56 | 0.1667 | 0.2545 | **0.7500** |
| `cmcp_binary_tlv` | 709.17 | 308.83 | 2,378.03 | **0.8889** | 0.4844 | 0.1667 |
| `cmcp_incremental` | **236.17** (+151 B store) | **124.83** (`-46.6%`) | **1,694.10** (`-10.5%`) | 0.8333 | 0.4549 | **0.0833** |

**Qué funcionó:**
- `cmcp_incremental` redujo tokens de entrada en **46.6%** (`[-109.6, -108.2]` IC 95%) y latencia end-to-end en **10.5%** partiendo de hechos estructurados.

**Limitaciones:**
- **Caída en Relation Recovery:** `cmcp_incremental` bajó a `0.0833` frente a `0.4444` (`full_history`) porque omitía las relaciones explícitas del delta y la respuesta telegráfica omitía la entidad origen.
- **Costo de extracción excluido:** CMCP 2.1 no midió el costo de extracción asistida por LLM desde texto libre.
- **KV / Latente Cross-Arch:** Refutado para transferencia factual sin entrenamiento end-to-end ([FINAL_RESEARCH_REPORT_V2.md](FINAL_RESEARCH_REPORT_V2.md)).

---

## 3. Compilación, Tests y Reproducción

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

.venv\Scripts\pytest.exe tests/test_protocol_and_adapters.py -v
.venv\Scripts\python.exe experiments/context_transfer/run_context_benchmark.py --split eval --reps 3
.venv\Scripts\python.exe experiments/audit/validate_cmcp21.py
```

---

## 4. Licencia

Apache License 2.0 — ver [LICENSE](LICENSE).
