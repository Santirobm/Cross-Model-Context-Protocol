# Cross-Model Context Protocol (CMCP)

[![Licencia: Apache-2.0](https://img.shields.io/badge/Licencia-Apache--2.0-blue.svg)](LICENSE)

> [English Version](README.md) | **Reporte Final de Investigación:** [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md)

**CMCP** es un núcleo de protocolo en C99 sin dependencias externas, un SDK en Python y una suite de benchmarks empíricos para **transferencia de contexto entre modelos LLM heterogéneos** (distinta arquitectura, tamaño, cuantización y tokenizador).

---

## 1. Capacidades Principales

- **Sobre Binario de 64 Bytes y Doble CRC32C:** Cabecera little-endian fija con magic `"CMCP"`, hashes FNV-1a de modelo/tokenizador y sumas de verificación Castagnoli CRC32C (`header_crc32c` + `payload_crc32c`).
- **Negociación de Compatibilidad en 4 Niveles y Guardrails KV:** `cmcp_check_compat` y `cmcp_check_kv_compat` negocian desde `Tier 0` (KV exacto) hasta `Tier 3` (fallback de texto) y rechazan incompatibilidades sutiles (`n_ctx`, `pos_offset`, `arch`, `rope`) que `llama_state_seq_set_data` puede aceptar silenciosamente.
- **Registros TLV Tipados y Compresión sin Pérdida:** `CONCEPT`, `ENTITY`, `RELATION`, `INSTRUCTION`, `CONSTRAINT`, `EVIDENCE`, `REFERENCE`, `DECISION`, `LATENT_TENSOR` y `KV_COMPAT_DESC`.
- **Transporte Desacoplado:** Memoria, memoria compartida del SO (`cmcp_shm_channel.c`) y un diseño opcional desacoplado en compilación para [Vect-Link](docs/architecture/vect_link_integration.md) (`CMCP_ENABLE_VECTLINK=OFF` por defecto).

---

## 2. Estructura del Repositorio

```text
Cross-Model-Context-Protocol/
├── core/                   # Librería C99 (cabecera, CRC32C, TLV, compat, SHM)
├── representations/        # SDK Python: esquema estructurado, codec binario ctypes, proyectores latentes
├── adapters/               # Adaptadores de modelos (Qwen2.5, Llama-3.2) y registro GGUF
├── benchmarks/             # Harness C11 llama.cpp y agregador de gráficos/resumen
├── experiments/            # Scripts reproducibles para Experimentos A–F
├── tests/                  # Pruebas unitarias C99 (tests/c/) y suite pytest en Python
├── results/                # Datos crudos JSON/CSV (results/raw/) y gráficos (results/plots/)
├── docs/                   # Especificación, arquitectura, metodología, resultados, roadmap
└── FINAL_RESEARCH_REPORT.md
```

---

## 3. Compilación y Pruebas

### 3.1 Compilar el Núcleo C99 y el Harness C11 `llama.cpp` (Windows MSYS2 Clang64)

```powershell
$env:PATH = "C:\msys64\clang64\bin;" + $env:PATH

cmake -S . -B build -G Ninja `
  -DCMAKE_C_COMPILER=clang `
  -DCMCP_BUILD_TESTS=ON `
  -DCMCP_BUILD_LLAMA_HARNESS=ON `
  -DCMCP_LLAMA_DIR="C:\Users\sgi.lm\Desktop\llama.cpp"

cmake --build build
```

### 3.2 Ejecutar Pruebas Unitarias

```powershell
# Pruebas unitarias C99 (5 suites)
.\build\test_cmcp_core.exe

# Pruebas SDK Python y adaptadores (7 pruebas)
.venv\Scripts\pytest.exe -q
```

Resultado verificado: `7 passed in 1.34s` (`pytest`) y las 5 suites C99 en `[PASS]` (`test_cmcp_core.exe`).

---

## 4. Reproducir Experimentos (A–F)

```powershell
$env:PATH = "C:\msys64\clang64\bin;C:\Users\sgi.lm\Desktop\llama.cpp\build\bin;" + $env:PATH

.venv\Scripts\python.exe experiments\structured_transfer\run_exp_a_b.py
.venv\Scripts\python.exe experiments\token_reduction\run_exp_c.py
.venv\Scripts\python.exe experiments\latent_alignment\run_exp_d.py
.venv\Scripts\python.exe experiments\state_transfer\run_exp_e.py
.venv\Scripts\python.exe experiments\kv_compatibility\run_exp_f.py
.venv\Scripts\python.exe benchmarks\run_all_benchmarks.py
```

---

## 5. Resumen de Resultados Observados

Todos los valores provienen directamente de [results/raw/](results/raw/benchmark_summary.json) (ejecución solo en CPU: Intel i5-13500, 15.70 GiB RAM).

| Experimento | Comparación Principal | Resultado Principal Observado | Datos Crudos |
| :--- | :--- | :--- | :--- |
| **Exp A/B** (192 corridas) | `nl_conversation` vs `json_structured` vs `cmcp_structured` vs `cmcp_binary` (`Qwen2.5-0.5B` <-> `Llama-3.2-1B`) | `cmcp_structured` obtuvo el menor tamaño en red (`274.125 B`), `cmcp_binary` tuvo `0.27–0.35 ms` de códec y `1.0` de fidelidad, **pero `json_structured` logró mayor calidad (`0.9083` / `0.8583` vs `0.7438` / `0.7583`) y menor latencia (`2653 ms` / `1602 ms`)**. `nl_conversation` fue el más lento (`4930 ms` / `7021 ms`, incluye generación del emisor). | [exp_ab_results.json](results/raw/exp_ab_results.json) |
| **Exp C** (36 corridas) | Multi-turno `full_history` vs `conventional_summary` vs `cmcp_references` vs `cmcp_incremental` | `cmcp_references` alcanzó **`1.0000` de calidad** (vs `0.8889` historial completo, `0.4444` resumen) con **-54.6% de bytes** (`361.67` vs `796.33`) y **-41.7% de tokens en receptor** (`172.67` vs `296`). | [exp_c_results.json](results/raw/exp_c_results.json) |
| **Exp D** (split 64/24) | Alineación latente (`SmolLM2-135M` -> `Qwen2.5-0.5B`) | Ridge (`mean_pooled`) alcanzó CKA `0.9506` (desde `0.8789`), coseno `0.9265`, top-1 `0.8333`; MLP alcanzó top-1 `0.8750`. | [exp_d_latent_alignment.json](results/raw/exp_d_latent_alignment.json) |
| **Exp E** (split 48/20) | Transferencia de estado soft-prompt (`K=8` latentes, receptor congelado) | `learned_cmcp` redujo NLL (`2.7108` -> `1.5738`, PPL `15.04` -> `4.82`), **pero falló el exact match factual (`EM = 0.0`, recall `0.05` vs `0.85` / `0.95` de `full_text`)**. | [exp_e_state_transfer.json](results/raw/exp_e_state_transfer.json) |
| **Exp F** (Guardrails KV) | `llama_state_seq_set_data` vs `cmcp_check_kv_compat` | Ambos aceptaron `same_model_same_cfg` (`99,008 B`) y rechazaron `cross_architecture` (`0 B`). `cmcp_check_kv_compat` también detectó `n_ctx mismatch` (`512` vs `256`) y `pos_offset mismatch` (`0` vs `16`) que `llama.cpp` aceptó. | [exp_f_kv_compat.json](results/raw/exp_f_kv_compat.json) |

---

## 6. Documentación

- [Visión General de Arquitectura](docs/architecture/overview.md) | [Decisiones de Diseño](docs/architecture/design_decisions.md) | [Integración Vect-Link](docs/architecture/vect_link_integration.md)
- [Especificación del Protocolo](docs/protocol/specification.md) | [Formato Binario](docs/protocol/binary_format.md) | [Compatibilidad](docs/protocol/compatibility.md) | [Seguridad](docs/protocol/security.md)
- [Reporte de Entorno](docs/experiments/environment_report.md) | [Metodología](docs/experiments/methodology.md) | [Resultados](docs/experiments/results.md) | [Limitaciones](docs/experiments/limitations.md)
- [Hoja de Ruta](docs/roadmap.md) | [Reporte Final de Investigación](FINAL_RESEARCH_REPORT.md)
