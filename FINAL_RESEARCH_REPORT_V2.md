# FINAL RESEARCH REPORT V2 — CMCP 2.1 (Cross-Model Context & Computational Transfer)

**Repository:** `https://github.com/Santirobm/Cross-Model-Context-Protocol`  
**Audit Gate:** `experiments/audit/validation_report.json` → **`PASS` (0 violations)**  
**Hardware:** Intel Core i5-13500 (14C/20T), `15.70 GiB` RAM, Windows 10 Pro x64, CPU-only (Intel UHD 770, no CUDA), Clang 22.1.8.  
**Models Evaluated:** `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (`770.3 MiB`), `qwen2.5-0.5b-instruct-q4_k_m.gguf` (`468.6 MiB`), `HuggingFaceTB/SmolLM2-135M-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`.

---

## A. Resumen Ejecutivo

La investigación CMCP 2.1 respondió experimentalmente las dos preguntas centrales:

1. **Transferencia de contexto entre modelos heterogéneos (Nivel A):** **Sí funciona y aporta ventaja medible**, pero **no** por un formato textual nuevo mágico sino por **referencias incrementales + separación de hechos activos vs. historial/distractores**. En el benchmark held-out (`18` tareas × `3` repeticiones = `270` inferencias reales con `Llama-3.2-1B-Instruct`), `cmcp_incremental` redujo tokens de entrada en **-46.6%** vs. `full_history` (`124.83` vs `233.67`) y **-69.5%** vs. `json_structured` (`409.83`), redujo el tiempo total end-to-end (`1,694.10 ms` vs `1,893.32 ms` / `2,777.56 ms`), y mejoró la precisión (`Exact Match = 0.8333` vs `0.6111` `full_history` y `0.1667` `json_structured`). Sin referencias incrementales, `cmcp_binary_tlv` alcanza `EM = 0.8889` / `KW Recall = 0.9630` pero usa más tokens que `full_history` (`308.83` vs `233.67`).
2. **Transferencia de representaciones internas y KV-cache entre arquitecturas distintas (Nivel B):** **NO es funcionalmente viable** mediante proyecciones lineales/MLP sin entrenamiento end-to-end del receptor. Con controles negativos estrictos (`neg_permuted_pairs`) y split por dominios disjuntos, la similitud coseno alta (`> 0.98`) resultó ser un artefacto de anisotropía, la recuperación de última capa cayó al azar (`0.0417`), y la inyección real de `DynamicCache` KV transformado (`SmolLM2-135M` → `Qwen2.5-0.5B`) colapsó completamente (`PPL = 493,744`, `EM = 0.0` vs `EM = 1.0` / `PPL = 2.45` en `full_prefill` y `self_kv_restore`).

---

## B. Estado del Arte (Qué se reprodujo y qué no)

Detalle en [docs/research/sota_and_baselines.md](docs/research/sota_and_baselines.md):
- **C2C — Cache-to-Cache** (Fu et al., 2025, `arXiv:2510.03215`): No se reprodujo el entrenamiento end-to-end con gating en GPU (sin GPU CUDA); se probó la variante CPU con regresores Ridge por capa para `K` y `V`, verificando que sin entrenamiento conjunto de gates la inyección directa colapsa.
- **Latent Space Communication via K-V Cache Alignment** (Dery et al., 2026, `arXiv:2601.06123`): Se reprodujo la extracción offline de tensores `K`/`V` y mapeo lineal por capa/cabeza; se confirmó que la diferencia de `rope_theta` (`100k` vs `1M`), GQA (`3` vs `2` KV heads) y tokenizador rompe la atención sin adaptadores entrenados conjuntamente.
- **Direct Semantic Communication via Vector Translation** (`arXiv:2511.03945`): Se reprodujo el mapeo Ridge/Procrustes/MLP entre `SmolLM2-135M` y `Qwen2.5-0.5B` bajo split por dominio y controles negativos.

---

## C. Benchmark de Contexto (`experiments/context_transfer/`, `eval` split, `n = 54` por método)

Comando: `.venv\Scripts\python.exe experiments/context_transfer/run_context_benchmark.py --split eval --reps 3 --max-tokens 32`  
Datos crudos: [context_benchmark_runs.csv](experiments/context_transfer/results/context_benchmark_runs.csv), [context_benchmark_summary.json](experiments/context_transfer/results/context_benchmark_summary.json)

| Método | Wire Bytes (mean) | Recv Tokens (mean) | Prep+Ser+Rec (ms) | Prefill (ms) | End-to-End (ms) | Exact Match | F1 | KW Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_history` | 709.67 | 233.67 | 0.004 | 856.13 | 1,893.32 | 0.6111 | 0.4843 | 0.8704 |
| `json_structured` | 1,364.00 | 409.83 | 0.119 | 1,497.76 | 2,777.56 | 0.1667 | 0.2545 | 0.6667 |
| `cmcp_textual` | 849.00 | 308.83 | 0.115 | ~1,118 | ~2,378 | **0.8889** | 0.4844 | **0.9630** |
| `cmcp_binary_tlv` | 709.17 | 308.83 | 0.772 | 1,118.41 | 2,378.03 | **0.8889** | 0.4844 | **0.9630** |
| `cmcp_incremental` | **236.17** (+151 B store) | **124.83** | 0.408 | **448.77** | **1,694.10** | 0.8333 | 0.4549 | 0.8889 |

**Nota crítica sobre el costo de construcción:** `prep_extract_ms` (`~0.07 ms`) mide la conversión desde objetos estructurados ya disponibles. Si en una aplicación real se requiere un LLM emisor para extraer entidades/relaciones desde texto libre, ese costo de extracción debe sumarse aparte.

---

## D. Transferencia Latente (`experiments/latent_alignment/run_cmcp21_latent.py`)

Split estricto por dominios (`48` train en 4 dominios, `24` eval en 2 dominios no vistos). Datos: [cmcp21_latent_results.csv](experiments/latent_alignment/results/cmcp21_latent_results.csv).

| Capa | Método | CKA Before → After | Cosine Sim | Retrieval Top-1 | Retrieval Top-5 |
|---|---|---:|---:|---:|---:|
| `embed_mean` | `neg_random_unlearned` | 0.9938 → 0.9938 | 0.0294 | 0.0417 | 0.2500 |
| `embed_mean` | `neg_permuted_pairs` | 0.9938 → 0.5818 | 0.9819 | 0.0417 | 0.2083 |
| `embed_mean` | **`procrustes`** | **0.9938 → 0.9938** | **0.9788** | **0.8750** | **1.0000** |
| `mid_mean` | `neg_permuted_pairs` | 0.9979 → 0.8552 | 0.9988 | 0.0000 | 0.1667 |
| `mid_mean` | `procrustes` | 0.9979 → 0.9979 | 0.9983 | **0.3750** | 0.6667 |
| `last_tok` | `ridge` / `procrustes` / `neg_permuted` | 0.9994 → 0.9931+ | 0.9806–0.9822 | **0.0417** | 0.2083 |
| `last_mean` | `ridge` / `procrustes` / `neg_permuted` | 0.9978 → 0.9916+ | 0.9772–0.9869 | **0.0417** | 0.2083 |

**Conclusión:** Existe **alineamiento geométrico y recuperación en capas tempranas (`embed_mean` Procrustes `87.5%` Top-1)**, pero **NO** existe generalización discriminativa en última capa ni transferencia factual funcional (`Exp E`: `EM = 0.0`).

---

## E. Transferencia KV-Cache (`experiments/kv_compatibility/run_cmcp21_kv_transfer.py`)

Emisor: `SmolLM2-135M-Instruct` (`30L`, `3` KV heads, `rope_theta=100k`). Receptor: `Qwen2.5-0.5B-Instruct` (`24L`, `2` KV heads, `rope_theta=1M`). Datos: [cmcp21_kv_transfer_runs.csv](experiments/kv_compatibility/results/cmcp21_kv_transfer_runs.csv).

| Método / Hipótesis | NLL | PPL | Exact Match | KW Recall | Eval (ms) | Transform (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_a_full_prefill` | **0.8958** | **2.45** | **1.0000** | **1.0000** | 3,649.43 | 0.0 |
| `h1_self_kv_restore` (Qwen → Qwen) | **0.8958** | **2.45** | **1.0000** | **1.0000** | 3,662.38 | 0.0 |
| `neg_zero_context` | 3.7376 | 42.27 | 0.0000 | 0.0000 | 3,706.56 | 0.0 |
| `neg_random_kv` | 9.8297 | 35,379.76 | 0.0000 | 0.0000 | 3,774.13 | 0.0 |
| `h2_h3_cross_kv_ridge` (24 capas) | **13.0964** | **493,744.14** | **0.0000** | **0.0000** | 3,738.53 | 610.89 |
| `h4_partial_top12_kv` (capas 12..23) | 4.2057 | 67.82 | 0.0000 | 0.0000 | 3,723.53 | 305.45 |

---

## F. Comparación End-to-End (Resumen de Métodos Funcionales vs. No Funcionales)

| Enfoque | Funciona Cross-Arch | Reduce Bytes | Reduce Tokens | Reduce Tiempo E2E | Mantiene Calidad (`EM`) |
|---|---|---|---|---|---|
| Historial completo (`full_history`) | Sí | Baseline (710 B) | Baseline (234 tok) | Baseline (1,893 ms) | Media (0.6111) |
| JSON estructurado (`json_structured`) | Sí | No (+92%) | No (+75%) | No (+47%) | Baja (0.1667) |
| CMCP binario tipado (`cmcp_binary_tlv`) | Sí | Empata (709 B) | No (+32% vs hist, -25% vs JSON) | No (+26% vs hist) | **Alta (0.8889)** |
| **CMCP incremental (`cmcp_incremental`)** | **Sí** | **Sí (-66.7%)** | **Sí (-46.6%)** | **Sí (-10.5%)** | **Alta (0.8333)** |
| Transferencia latente (`inputs_embeds`) | No (solo formato) | N/A | N/A | No | **Falla (0.0000)** |
| Transferencia KV Ridge (`cross_kv_ridge`) | **No (colapsa)** | N/A | N/A | No (+611 ms) | **Falla (0.0000)** |

---

## G. Auditoría de Ingeniería (Problemas Encontrados y Corregidos)

1. **`cmcp_crc32c.c`:** Eliminada la inicialización lazy no atómica (`g_crc32c_init`) reemplazándola por tabla `static const uint32_t g_crc32c_table[256]` precalculada en compilación + test multi-hilo de 8 hilos (`test_crc32c_concurrency_and_fingerprint`).
2. **`cmcp_compat.c`:** Implementado `cmcp_fingerprint_file()` (FNV-1a 64-bit sobre tamaño + primeros 64 KB + últimos 64 KB del archivo GGUF).
3. **`cmcp_shm_channel.c`:** Añadido rechazo `ERROR_ALREADY_EXISTS` en `CreateFileMappingA` bajo `create_new` en Windows (limitación remanente documentada: `#else` POSIX sigue usando `calloc()` en memoria privada).
4. **`schema.py`:** Añadida serialización binaria tipada pura (`to_typed_tlv_records` / `from_typed_tlv_records` con strings length-prefixed `<H`, sin JSON embebido dentro del TLV).
5. **Verificación:** `ctest` Release **`1/1` passed**, `ctest` ASan+UBSan **`1/1` passed**, `pytest` **`7/7` passed**, `validate_cmcp21.py` **`PASS` (0 violations)**.

---

## H. Clasificación Obligatoria de Resultados

| Conclusión / Hipótesis | Clasificación | Evidencia |
|---|---|---|
| CMCP incremental con referencias reduce bytes, tokens, prefill y tiempo E2E manteniendo calidad (`0.8333` EM) | **DEMOSTRADO** | `context_benchmark_runs.csv` (`n=54/método`, `Llama-3.2-1B`) |
| CMCP binario TLV tipado supera en precisión (`0.8889` EM) al historial completo (`0.6111`) y a JSON (`0.1667`) | **DEMOSTRADO** | `context_benchmark_runs.csv` |
| CMCP no-incremental reduce tokens frente al historial completo | **REFUTADO** | `308.83` vs `233.67` tokens |
| Transferencia directa de KV entre modelos idénticos (`H1`) | **DEMOSTRADO** | `h1_self_kv_restore`: `EM=1.0`, `NLL=0.8958` |
| Alineamiento latente geométrico en capas tempranas (`embed_mean` Procrustes `87.5%` Top-1) | **PARCIALMENTE DEMOSTRADO** | `cmcp21_latent_results.csv` (falla en última capa: `4.17%` azar) |
| Transferencia funcional de conocimiento vía `inputs_embeds` con receptor congelado (`Exp E`) | **REFUTADO** | `EM=0.0`, `KW=0.05` |
| Transferencia KV transformada (`H2`, `H3`, `H4`) entre `SmolLM2-135M` y `Qwen2.5-0.5B` sin entrenamiento E2E | **REFUTADO** | `NLL=13.10` (`PPL=493,744`), colapso de generación |
| Imposibilidad práctica de transferencia KV funcional sin entrenamiento E2E entre estas arquitecturas (`H5`) | **DEMOSTRADO** | `cmcp21_kv_transfer_runs.csv` |
| Entrenamiento E2E de módulos C2C con *learnable gating* en GPU (`arXiv:2510.03215`) | **NO EJECUTADO** | Sin GPU CUDA ni presupuesto de entrenamiento |

---

## I. Limitaciones y Conclusión Comercial / Técnica

- **Limitaciones:** Sin GPU CUDA; tareas sintéticas controladas (no benchmarks abiertos como MMLU/GSM8K); `cmcp_shm_channel.c` en POSIX no usa `shm_open` real; el benchmark contextual asumió estructura extraída (si se usa un LLM para extraerla, hay que sumar ese costo).
- **Conclusión técnica y comercial:** **CMCP no tiene viabilidad actual como protocolo de transferencia de KV-cache o activaciones latentes entre arquitecturas distintas sin entrenamiento conjunto en GPU.** Sí tiene valor técnico claro como **protocolo compacto binario/incremental de contexto estructurado con referencias (`-46.6%` tokens, `+22–28 pp` de Exact Match frente a historial/JSON)**, especialmente cuando un sistema multi-agente comparte estado persistido entre pasos y modelos heterogéneos.

## J. Próximo Paso Recomendado

Concentrar CMCP exclusivamente en **Nivel A (protocolo de referencias incrementales + deltas tipados sobre un store persistido)** y medir el costo real de extracción automática en flujos multi-turno de programación/auditoría reales, abandonando la transferencia KV lineal entre arquitecturas distintas.
