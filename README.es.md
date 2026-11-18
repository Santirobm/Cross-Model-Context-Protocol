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

## 2. Resultados Observados y Limitaciones (Validación Multi-Turno CMCP 2.2, `n = 50` Escenarios Eval Disjuntos)

Evaluado en 50 escenarios de 8 turnos (5 dominios no vistos, 2 consultas/escenario = 100 puntos/método) con bootstrap agrupado por escenario e IC 95% ([FINAL_RESEARCH_REPORT_V3.md](FINAL_RESEARCH_REPORT_V3.md)):

| Configuración y Método | Bytes Wire | Tokens Entrada | E2E ms | Exact Match | Rel Rec | Constr Sat | Task Comp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hetero (`Qwen2.5-0.5B -> Llama-3.2-1B`)** | | | | | | | |
| `full_history` | 686.4 | 236.0 | **2,516.6** | **0.62** | **1.00** | **0.85** | **0.77** |
| `rolling_summary` | 709.8 | **108.6** | 8,241.2 | 0.40 | 0.50 | 0.56 | 0.40 |
| `json_structured` | 799.6 | 286.7 | 2,759.9 | 0.40 | **1.00** | 0.58 | 0.47 |
| `cmcp_incremental` (ext. det.) | **250.0** | 136.4 | 2,739.6 | 0.57 | 0.74 | 0.72 | 0.58 |
| `cmcp_inc_llm_ext` (ext. LLM 0.5B) | 474.6 | 218.6 | 6,993.6 | 0.20 | 0.61 | 0.55 | 0.20 |
| **Homo (`Qwen2.5-0.5B -> Qwen2.5-0.5B`)** | | | | | | | |
| `full_history` | 686.4 | 272.9 | 1,916.5 | 0.53 | **0.95** | 0.59 | 0.54 |
| `cmcp_incremental` (ext. det.) | **250.0** | **159.7** | **1,793.1** | **0.75** | 0.90 | **0.80** | **0.88** |

**Qué funcionó:**
- **Eficiencia Wire y Tokens:** `cmcp_incremental` reduce bytes en red en **63.6%** en eval de 8 turnos y **91.5%** acumulado a 40 turnos (`94,625 B -> 8,005 B`, `311 B` constantes por turno), reduciendo tokens de entrada en **41.5–42.2%** y prefill en **42.3%** (`2.63x` a 40 turnos).
- **Ganancia Homogénea:** En `Qwen2.5-0.5B -> Qwen2.5-0.5B`, `cmcp_incremental` supera a `full_history` tanto en latencia E2E (`-123.4 ms [-156.5, -92.7]`) como en calidad (`EM +0.22 [+0.14, +0.30]`, `task_completion +0.34 [+0.27, +0.41]`).

**Limitaciones y resultados negativos:**
- **Brecha Heterogénea en E2E y Relaciones (`Llama-3.2-1B`):** `cmcp_incremental` empata en `EM` (`-0.05 [-0.12, +0.02]`) y supera a `json_structured` y `rolling_summary`, pero `Llama-3.2-1B` genera más tokens sobre líneas estructuradas (`+223.0 ms` E2E) y cae en `relation_recovery` (`-0.26`).
- **Extracción LLM 0.5B y Rolling Summary:** `rolling_summary` (`+5,725 ms E2E`, `-0.22 EM`) y `cmcp_inc_llm_ext` (`+4,477 ms E2E`, `-0.42 EM`) son más lentos y mucho menos precisos que `full_history` y la extracción determinista (`0.06 ms`).
- **KV / Latente Cross-Arch:** Refutado para transferencia factual sin entrenamiento end-to-end ([FINAL_RESEARCH_REPORT_V2.md](FINAL_RESEARCH_REPORT_V2.md)).

---

## 3. Compilación, Tests y Reproducción

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

.venv\Scripts\pytest.exe tests/ -v
.venv\Scripts\python.exe -m experiments.cmcp22.run_multiturn_benchmark --split eval --reps 1 --max-tokens 64 --model-a $CMCP_MODEL_A --model-b $CMCP_MODEL_B --tag eval_hetero
.venv\Scripts\python.exe -m experiments.cmcp22.run_scaling_benchmark --model $CMCP_MODEL_B --tag scaling_v1 --turns 5 10 20 40
.venv\Scripts\python.exe experiments/cmcp22/audit/validate_cmcp22.py
```

---

## 4. Licencia

Apache License 2.0 — ver [LICENSE](LICENSE).
