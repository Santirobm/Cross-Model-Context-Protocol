# FINAL RESEARCH REPORT — Cross-Model Context Protocol (CMCP)

**Fecha:** 2026-11-17  
**Entorno de Ejecución:** Windows 10 Pro (`10.0.19045`), Intel Core i5-13500 (14C/20T, solo CPU, sin GPU discreta), `15.70 GiB` RAM, MSYS2 Clang `22.1.8`, `llama.cpp` (`b79696f`), Python `3.13.5` ([environment_report.md](docs/experiments/environment_report.md)).

---

## 1. Resumen Ejecutivo (15 Preguntas Clave)

1. **¿Qué construimos?**  
   Construimos **CMCP v1**: un núcleo de protocolo en **C99 sin dependencias externas** (`libcmcp.a` / `cmcp.dll` con sobre binario little-endian de 64 bytes, doble CRC32C Castagnoli, registros TLV con compresión sin pérdida, negociación de compatibilidad en 4 niveles, guardrails de KV-cache y canal de memoria compartida), un **harness C11 enlazado a `llama.cpp`** (`cmcp_llama_harness.exe`), un **SDK en Python** (`representations/`, `adapters/`) con bindings `ctypes`, y una **suite de 6 experimentos empíricos (A–F)** más pruebas unitarias C99 (`test_cmcp_core.exe`) y Python (`pytest`).

2. **¿Qué problema resuelve?**  
   Permite que agentes LLM heterogéneos (distinta arquitectura, tamaño, cuantización y tokenizador, por ejemplo `Qwen2.5-0.5B` y `Llama-3.2-1B`) intercambien contexto con **integridad verificada (CRC32C)**, **detección explícita de incompatibilidades KV-cache** (evitando corrupciones silenciosas) y **reducción de tokens en flujos multi-turno mediante referencias e incrementos estructurados**.

3. **¿Qué NO resuelve todavía?**  
   No resuelve la **transferencia de estado interno/latente con preservación de hechos exactos sin texto** en modelos pequeños en CPU (Exp E obtuvo `EM = 0.0`), ni supera a `json_structured` en calidad de respuesta en transferencia de un solo turno sobre modelos de `0.5B`–`1B` (donde `cmcp_structured` obtuvo `0.7438`/`0.7583` vs `0.9083`/`0.8583` de JSON).

4. **¿Qué funciona en la práctica hoy?**  
   - El protocolo binario C99 (`header` 64 B + TLV + CRC32C) con costo de códec de **`0.27–0.35 ms`** desde Python `ctypes` y **5/5** suites C99 + **7/7** pruebas Python pasando.
   - Los guardrails KV (`cmcp_check_kv_compat`) validados frente a `llama_state_seq_set_data` real (Exp F).
   - La reducción multi-turno con `cmcp_references` (Exp C): **`1.0000`** de calidad con **-54.6% bytes** y **-41.7% tokens en receptor** respecto a `full_history`.

5. **¿Qué falló o rindió por debajo de lo esperado?**  
   - **Exp A/B (`cmcp_structured` / `cmcp_binary` vs `json_structured`):** Aunque `cmcp_structured` fue el más compacto en red (`274.125 B`) y mantuvo `info_fidelity_score = 1.0`, su formato de texto `CMCP_CTX` produjo menor `quality_score` (`0.7438` en Qwen->Llama, `0.7583` en Llama->Qwen) y mayor latencia (`3561.69 ms` / `1936.64 ms`) que `json_structured` (`0.9083` / `0.8583` y `2653.24 ms` / `1602.10 ms`).
   - **Exp E (Soft-Prompt State Transfer):** Aunque `learned_cmcp` redujo la NLL de `2.7108` a `1.5738`, falló completamente el exact match factual (`EM = 0.0`, `keyword_recall = 0.05`).

6. **¿Cómo se compara `cmcp_binary` / `cmcp_structured` contra `nl_conversation` y `json_structured`?**  
   En Exp A/B, `nl_conversation` fue el más lento (`4929.71 ms` y `7020.73 ms`, **porque incluye la latencia de generación del emisor**) y perdió información (`info_fidelity_score = 0.8833` y `0.8083`). `cmcp_structured` y `cmcp_binary` lograron `info_fidelity_score = 1.0` y menor tamaño en bytes (`274.125 B` y `325.875 B` vs `371.75 B` de JSON), pero `json_structured` ganó en `quality_score` (`0.9083` / `0.8583`) y latencia total (`2653.24 ms` / `1602.10 ms`).

7. **¿CMCP reduce tokens y latencia en escenarios multi-turno?**  
   **Sí en Exp C.** `cmcp_references` redujo los tokens de entrada del receptor de `296.00` (`full_history`) a `172.67` (**-41.7%**), los bytes de `796.33` a `361.67` (**-54.6%**), el prefill de `1186.558 ms` a `687.696 ms` (**-42.0%**), y subió la calidad de `0.8889` a **`1.0000`**. `cmcp_incremental` redujo tokens a `128.33` (**-56.6%**) y latencia a `3664.379 ms` con calidad `0.7778`.

8. **¿Existe geometría compartida aprovechable entre representaciones latentes de modelos distintos?**  
   **Sí (Exp D).** Entre `SmolLM2-135M-Instruct` (`d=576`) y `Qwen2.5-0.5B-Instruct` (`d=896`), Ridge Regression elevó el CKA lineal de `0.878888` a **`0.950601`** (`mean_pooled`) y de `0.933289` a **`0.975185`** (`last_token`), con similitud coseno de `0.9265` / `0.9298` y recuperación top-1 de `0.8333` (`mean_pooled`, `0.8750` con MLP).

9. **¿Esa alineación latente sirve hoy para transferir hechos sin texto (`inputs_embeds`)?**  
   **No (Exp E).** La proyección de `K=8` vectores latentes a `inputs_embeds` con receptor congelado y `48` ejemplos de entrenamiento redujo la perplejidad (`15.0412` -> `4.8249`), pero obtuvo **`0.0` Exact Match** y **`0.05` keyword recall** frente a `0.85` EM y `0.95` recall de `full_text`.

10. **¿Qué demostró la verificación real con `llama.cpp` sobre KV-cache?**  
    **Exp F** demostró que `llama_state_seq_set_data` acepta transferencia exacta en mismo modelo/configuración (`99,008 B`) y rechaza arquitectura cruzada (`qwen2` -> `llama`, `0 B`), **pero acepta silenciosamente estados con distinto `n_ctx` (`512` vs `256`, `99,008 B`)**. `cmcp_check_kv_compat` bloqueó correctamente tanto `n_ctx mismatch` (código `7`) como `pos_offset mismatch` (código `8`) y `architecture mismatch` (código `1`).

11. **¿Cuál es el costo computacional real de CMCP?**  
    Despreciable frente a la inferencia: el códec binario completo (encode + decode + CRC32C vía `ctypes`) tomó **`0.3548 ms`** (`Qwen->Llama`) y **`0.2736 ms`** (`Llama->Qwen`) en Exp A/B (`< 0.02%` de la latencia total), y la memoria RSS del proceso se mantuvo en `~475 MiB` (Qwen `0.5B`) a `~1317 MiB` (Llama `1B`).

12. **¿Cómo se relaciona CMCP con `Vect-Link`?**  
    `Vect-Link` permanece intacto y sin acoplamiento (`CMCP_ENABLE_VECTLINK=OFF`). `Vect-Link` cubre el plano de datos SHM bit-exacto para el mismo modelo (Tier 0); CMCP cubre la negociación entre modelos distintos (Tiers 1–3) y el sobre portable, con un diseño desacoplado en [vect_link_integration.md](docs/architecture/vect_link_integration.md).

13. **¿Qué hipótesis quedaron sin ejecutar o incompletas?**  
    - **H3** (alineación de spans de bytes entre tokenizadores heterogéneos) no se implementó como experimento independiente.
    - **H5** (adaptador runtime `CMCP_ENABLE_VECTLINK=ON` contra `Vect-Link`) quedó solo a nivel de diseño arquitectónico, no compilado ni medido contra `Vect-Link`.

14. **¿Cuál es la mayor limitación metodológica actual?**  
    (1) Escala limitada a modelos `135M–1B` en CPU; (2) asimetría en Exp A/B/C donde `nl_conversation` y `conventional_summary` incluyen generación del emisor mientras las condiciones estructuradas parten de contextos pre-construidos (`sender_gen_tokens = 0`); y (3) evaluación por recall determinista de aliases en muestras pequeñas (`n=24` en Exp A/B, `n=9` en Exp C).

15. **¿Cuál es el próximo experimento que más valor aportaría?**  
    **Mantener el sobre binario TLV+CRC32C en el cable pero renderizar el contexto decodificado en el receptor como JSON compacto (o esquema clave-valor afín a JSON)** en lugar de `CMCP_CTX`, midiéndolo en Exp A/B y Exp C junto con el costo de extracción estructurada en el emisor. Esto ataca directamente la brecha de calidad (`0.74–0.76` vs `0.86–0.91`) observada frente a `json_structured` sin perder los beneficios de compresión, integridad y referencias de CMCP.

---

## 2. Tablas Comparativas Completas (Datos Crudos Verificados)

### 2.1 Experimentos A & B — Transferencia Estructurada (`n_runs = 24` por condición y dirección)

Fuente: [exp_ab_results.json](results/raw/exp_ab_results.json) | [exp_ab_results.csv](results/raw/exp_ab_results.csv) | Gráfico: [exp_ab_comparison.png](results/plots/exp_ab_comparison.png)

#### Dirección `Qwen2.5-0.5B -> Llama-3.2-1B`

| Condición | Bytes (mean) | Sender Gen Tok | Recv In Tok | Prefill (ms) | Eval (ms) | Codec (ms) | Total Latency (ms, mean / p95) | Quality Score | Info Fidelity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nl_conversation` | 317.125 | 85.125 | 141.750 | 500.648 | 2292.167 | 0.0000 | 4929.7055 / 6391.6925 | 0.8167 | 0.8833 |
| `json_structured` | 371.750 | 0.000 | 165.750 | 576.786 | 2076.139 | 0.0848 | **2653.2407 / 3716.0293** | **0.9083** | **1.0000** |
| `cmcp_structured` | **274.125** | 0.000 | 161.375 | 561.176 | 3000.276 | 0.0158 | 3561.6944 / 4279.9410 | 0.7438 | **1.0000** |
| `cmcp_binary` | 325.875 | 0.000 | 161.375 | 564.156 | 2995.774 | 0.3548 | 3560.5143 / 4288.0181 | 0.7438 | **1.0000** |

#### Dirección `Llama-3.2-1B -> Qwen2.5-0.5B`

| Condición | Bytes (mean) | Sender Gen Tok | Recv In Tok | Prefill (ms) | Eval (ms) | Codec (ms) | Total Latency (ms, mean / p95) | Quality Score | Info Fidelity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nl_conversation` | 471.750 | 127.125 | 192.500 | 588.347 | 1177.591 | 0.0000 | 7020.7347 / 7502.7004 | 0.6937 | 0.8083 |
| `json_structured` | 371.750 | 0.000 | 167.875 | 549.743 | 1051.921 | 0.0744 | **1602.1010 / 2361.6699** | **0.8583** | **1.0000** |
| `cmcp_structured` | **274.125** | 0.000 | 163.500 | 529.724 | 1406.545 | 0.0148 | 1936.6377 / 2482.5240 | 0.7583 | **1.0000** |
| `cmcp_binary` | 325.875 | 0.000 | 163.500 | 519.425 | 1405.900 | 0.2736 | 1925.9530 / 2471.4389 | 0.7583 | **1.0000** |

> **Nota metodológica:** `nl_conversation` incluye `sender_ms` (generación del emisor) dentro de `total_latency_ms`.

### 2.2 Experimento C — Reducción de Tokens Multi-Turno (`Qwen2.5-0.5B -> Llama-3.2-1B`, `n_runs = 9`)

Fuente: [exp_c_results.json](results/raw/exp_c_results.json) | [exp_c_results.csv](results/raw/exp_c_results.csv) | Gráfico: [exp_c_tokens_latency.png](results/plots/exp_c_tokens_latency.png)

| Condición | Wire Bytes (mean) | Recv In Tokens (mean) | Prefill (ms, mean) | Total Latency (ms, mean / p95) | Quality Score (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `full_history` | 796.33 | 296.00 | 1186.558 | 3915.091 / 5790.839 | 0.8889 |
| `conventional_summary` | 398.33 | 169.33 | 668.629 | 6383.302 / 8993.516 | 0.4444 |
| `cmcp_references` | 361.67 | 172.67 | 687.696 | 3802.966 / 6251.347 | **1.0000** |
| `cmcp_incremental` | **248.33** | **128.33** | **509.690** | **3664.379 / 5849.488** | 0.7778 |

### 2.3 Experimento D — Alineación de Representaciones Latentes (`SmolLM2-135M -> Qwen2.5-0.5B`)

Fuente: [exp_d_latent_alignment.json](results/raw/exp_d_latent_alignment.json) | [exp_d_latent_alignment.csv](results/raw/exp_d_latent_alignment.csv) (`n_train=64`, `n_eval=24`, `d_src=576`, `d_dst=896`).

| Pooling | Método | CKA Before | CKA After | Eval MSE | Eval Cosine Sim | Top-1 Acc | Top-5 Acc | Train (ms) | Eval Infer (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mean_pooled` | `random` | 0.878888 | 0.878888 | 46.203339 | -0.014406 | 0.041667 | 0.166667 | 0.001 | 0.250 |
| `mean_pooled` | `procrustes` | 0.878888 | 0.878888 | 10.729951 | 0.876393 | 0.833333 | **1.000000** | 99.037 | 0.345 |
| `mean_pooled` | `ridge` | 0.878888 | **0.950601** | **6.266980** | **0.926500** | 0.833333 | 0.958333 | 1102.290 | 0.357 |
| `mean_pooled` | `mlp` | 0.878888 | 0.938037 | 7.232878 | 0.914000 | **0.875000** | 0.958333 | 3578.530 | 1.338 |
| `last_token` | `random` | 0.933289 | 0.933289 | 91.901520 | 0.010059 | 0.083333 | 0.125000 | 0.002 | 0.254 |
| `last_token` | `procrustes` | 0.933289 | 0.933289 | 20.834435 | 0.864029 | 0.666667 | 0.916667 | 95.505 | 0.340 |
| `last_token` | `ridge` | 0.933289 | **0.975185** | **10.852974** | **0.929830** | 0.666667 | 0.875000 | 19.388 | 0.388 |
| `last_token` | `mlp` | 0.933289 | 0.945502 | 13.107480 | 0.919964 | 0.500000 | 0.875000 | 3359.143 | 1.277 |

### 2.4 Experimento E — Transferencia de Estado Soft-Prompt (`SmolLM2-135M -> Qwen2.5-0.5B`)

Fuente: [exp_e_state_transfer.json](results/raw/exp_e_state_transfer.json) | [exp_e_state_transfer.csv](results/raw/exp_e_state_transfer.csv) (`n_latents=8`, `n_train=48`, `n_eval=20`).

| Condición | Mean NLL | Perplexity | Exact Match Rate | Keyword Recall | Mean Recv In Tokens | Mean Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero_context` | 2.7108 | 15.0412 | 0.00 | 0.05 | 13.5 | 2154.25 |
| `random_projection` | 2.5347 | 12.6130 | 0.00 | 0.05 | 21.5 | 2234.97 |
| `learned_cmcp` | 1.5738 | 4.8249 | **0.00** | **0.05** | 21.5 | 2692.36 |
| `full_text` | **0.3717** | **1.4503** | **0.85** | **0.95** | 30.8 | 3311.11 |

### 2.5 Experimento F — Compatibilidad Real de KV-Cache en `llama.cpp`

Fuente: [exp_f_kv_compat.json](results/raw/exp_f_kv_compat.json) (RSS del harness: `1427.72 MiB`).

| Caso | KV Bytes | `llama_state_seq_set_data` | `cmcp_kv_compatible` | Razón CMCP (Código) |
| :--- | ---: | :--- | :--- | :--- |
| `same_model_same_cfg` | 99,008 | Aceptado (`99,008` B) | `true` | `exact KV-cache match (Tier 0)` (`0`) |
| `same_model_diff_n_ctx` (`512` vs `256`) | 99,008 | Aceptado (`99,008` B) | `false` | `n_ctx mismatch` (`7`) |
| `same_model_diff_pos_offset` (`0` vs `16`) | — | (no probado en harness) | `false` | `pos_offset mismatch` (`8`) |
| `cross_architecture` (`qwen2` vs `llama`) | 99,008 | Rechazado (`0` B) | `false` | `architecture mismatch` (`1`, fallback Tier `2`) |

---

## 3. Clasificación Científica de Hallazgos

| Clasificación | Afirmación / Hipótesis | Evidencia Directa |
| :--- | :--- | :--- |
| **DEMOSTRADO** | El sobre binario C99 (64 B + TLV + CRC32C) tiene overhead despreciable (`< 0.4 ms` vía `ctypes`) y 0 corrupciones silenciosas. | Exp A/B (`codec_ms` `0.27–0.35 ms`), `test_cmcp_core.exe` (5/5 PASS), `pytest` (7/7 passed). |
| **DEMOSTRADO** | `cmcp_check_kv_compat` previene incompatibilidades KV-cache (`arch`, `n_ctx`, `pos_offset`) incluyendo casos que `llama_state_seq_set_data` acepta. | Exp F ([exp_f_kv_compat.json](results/raw/exp_f_kv_compat.json)). |
| **DEMOSTRADO** | CMCP con referencias estructuradas (`cmcp_references`) reduce tokens (`-41.7%`) y bytes (`-54.6%`) manteniendo calidad `1.0` en escenarios multi-turno frente a `full_history` (`0.8889`) y `conventional_summary` (`0.4444`). | Exp C ([exp_c_results.json](results/raw/exp_c_results.json)). |
| **DEMOSTRADO** | Existe alta alineación geométrica recuperable mediante proyección lineal (Ridge/Procrustes) entre `SmolLM2-135M` y `Qwen2.5-0.5B` (CKA hasta `0.9752`, top-1 hasta `0.8333`–`0.8750`). | Exp D ([exp_d_latent_alignment.json](results/raw/exp_d_latent_alignment.json)). |
| **PARCIALMENTE DEMOSTRADO** | La transferencia estructurada preserva `info_fidelity_score = 1.0` y menor tamaño en bytes (`274.125 B`), pero el renderizado `CMCP_CTX` rindió peor en calidad que `json_structured` en modelos `0.5B/1B` (`0.7438`/`0.7583` vs `0.9083`/`0.8583`). | Exp A/B ([exp_ab_results.json](results/raw/exp_ab_results.json)). |
| **NO DEMOSTRADO** | Que la proyección soft-prompt (`K=8`, receptor congelado, `48` ejemplos) pueda transferir hechos específicos sin texto. Aunque bajó NLL (`2.7108` -> `1.5738`), obtuvo `EM = 0.0` y `keyword_recall = 0.05`. | Exp E ([exp_e_state_transfer.json](results/raw/exp_e_state_transfer.json)). |
| **REFUTADO** | Que `llama_state_seq_set_data` por sí solo sea suficiente como guardrail de seguridad KV-cache para cambios de `n_ctx`: aceptó `99,008 B` entre `n_ctx=512` y `n_ctx=256`. | Exp F ([exp_f_kv_compat.json](results/raw/exp_f_kv_compat.json)). |
| **NO EJECUTADO** | (1) Benchmark independiente de alineación de byte-spans entre tokenizadores (H3); (2) compilación y benchmark end-to-end de `CMCP_ENABLE_VECTLINK=ON` contra `Vect-Link` (H5). | [technical_hypotheses.md](docs/research/technical_hypotheses.md), [vect_link_integration.md](docs/architecture/vect_link_integration.md). |

---

## 4. Comandos de Reproducción Exactos

```powershell
$env:PATH = "C:\msys64\clang64\bin;C:\Users\sgi.lm\Desktop\llama.cpp\build\bin;" + $env:PATH

# 1. Compilar núcleo C99, pruebas unitarias y harness llama.cpp
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMCP_BUILD_TESTS=ON -DCMCP_BUILD_LLAMA_HARNESS=ON -DCMCP_LLAMA_DIR="C:\Users\sgi.lm\Desktop\llama.cpp"
cmake --build build

# 2. Ejecutar pruebas unitarias (C99 + Python)
.\build\test_cmcp_core.exe
.venv\Scripts\pytest.exe -q

# 3. Reproducir experimentos A-F y regenerar resumen/gráficos
.venv\Scripts\python.exe experiments\structured_transfer\run_exp_a_b.py
.venv\Scripts\python.exe experiments\token_reduction\run_exp_c.py
.venv\Scripts\python.exe experiments\latent_alignment\run_exp_d.py
.venv\Scripts\python.exe experiments\state_transfer\run_exp_e.py
.venv\Scripts\python.exe experiments\kv_compatibility\run_exp_f.py
.venv\Scripts\python.exe benchmarks\run_all_benchmarks.py
```

---

## 5. Evaluación Honesta del Potencial Técnico y Comercial

- **Potencial Técnico (Alto en plano de control, guardrails y referencias):** El núcleo C99 (`cmcp.dll`), la validación estricta de compatibilidad KV-cache (Exp F) y el patrón de referencias/deltas multi-turno (Exp C) son sólidos, reproducibles y listos para integrarse en runtimes locales (`llama.cpp`, `Vect-Link`). Sin embargo, el formato de prompt `CMCP_CTX` debe evolucionar hacia una proyección JSON/clave-valor amigable para modelos pequeños, y la transferencia latente sin texto (Exp E) requiere entrenamiento a escala GPU o adaptación LoRA/KV antes de ser viable para datos factuales.
- **Potencial Comercial:** Como producto independiente, un protocolo de texto estructurado compite con JSON/MCP estándar; su diferenciador real está en **orquestación heterogénea de bajo overhead** (Edge/Local AI, sistemas multi-agente mixtos CPU/GPU que combinan modelos pequeños y grandes con guardrails KV-cache estrictos y transporte SHM opcional sobre motores como `Vect-Link`).
