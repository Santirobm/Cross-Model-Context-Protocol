# SOTA & Baselines: Cross-Model Context & KV Transfer

Concise implementation-oriented review of the three target research lines.

## Comparative Summary

| Trabajo | Representación | Método | Requisito | Experimento reproducible |
|---|---|---|---|---|
| **C2C — Cache-to-Cache** (Fu et al., ICLR / Oct 2025, `arXiv:2510.03215`) | KV-cache (`K`, `V` por capa) | Proyección neuronal + fusión con *learnable gating* por capa (`Sharer -> Receiver`) | Entrenamiento de módulos proyectores/gates sobre pares de modelos en GPU; acceso directo a `past_key_values` | Parcial: extracción `past_key_values` + proyección lineal/MLP por capa en PyTorch CPU (`SmolLM2-135M -> Qwen2.5-0.5B`) |
| **Latent Space Communication via K-V Cache Alignment** (Dery et al., Google DeepMind, Jan 2026, `arXiv:2601.06123`) | KV-cache alineado a un espacio latente compartido | Adaptadores ligeros que mapean KV-caches de modelos heterogéneos a un espacio común sin modificar pesos base | Entrenamiento de adaptadores de alineamiento KV; acceso por capa/cabeza a tensores KV | Parcial: alineamiento Ridge/Procrustes/low-rank de tensores K y V por separado en PyTorch CPU |
| **Direct Semantic Communication via Vector Translation** (`arXiv:2511.03945`) | Vectores latentes continuos / hidden states | Traductor dual-encoder entre espacios de representación de modelos distintos (p.ej. Llama <-> Mistral) | Pares paralelos de entrenamiento para el traductor + mecanismo de inyección en el receptor | Sí (nivel retrieval/probe y `inputs_embeds` con receptor congelado en PyTorch CPU) |

## Technical Notes for CMCP 2.1

1. **Qué se transfiere:**
   - En **C2C** y **K-V Cache Alignment**, se transfieren tensores `K`/`V` por capa (o una proyección intermedia), evitando generar tokens de texto en el emisor.
   - En **Vector Translation**, se transfieren estados ocultos comprimidos o vectores de secuencia traducidos entre espacios latentes.
2. **Por qué falla la transferencia ingenua entre arquitecturas:**
   - **RoPE y posiciones:** `K` está rotado por RoPE (`rope_theta`, `head_dim`, `pos_offset`) mientras `V` no lo está; mezclar ambos o ignorar `rope_theta` rompe el producto punto `Q·K^T`.
   - **GQA / MQA / MHA:** `SmolLM2-135M` (`30` capas, `3` KV heads, `head_dim=64`, `rope_theta=100000`) vs `Qwen2.5-0.5B` (`24` capas, `2` KV heads, `head_dim=64`, `rope_theta=1000000`) difieren en profundidad, número de cabezas KV, frecuencia RoPE, `hidden_size` (`576` vs `896`) y tokenizador (`49,152` vs `151,936`).
   - **Métrica funcional:** CKA o Cosine Similarity altos no demuestran que el receptor pueda responder preguntas factuales ni continuar una tarea. Se requiere evaluar **Exact Match / F1 / NLL sobre datos no vistos** frente a controles negativos (pares permutados, proyección aleatoria, cero contexto).
3. **Lo que ya probó CMCP v1 vs. lo que falta:**
   - CMCP v1 probó recuperación latente (`Exp D`: CKA `0.95`, Top-1 `83–87.5%`), refutó transferencia factual vía `inputs_embeds` congelados (`Exp E`: `EM = 0.0` pese a bajar NLL), y probó rechazo de blobs KV en `llama.cpp` (`Exp F`).
   - **Brecha experimental de CMCP 2.1:** (a) benchmark de contexto con costo de preparación incluido y 30+ tareas equivalentes, (b) controles negativos estrictos en alineamiento latente multi-capa, y (c) prueba experimental real de extracción/transformación/inyección de `past_key_values` (`K` y `V` por separado) en PyTorch.
