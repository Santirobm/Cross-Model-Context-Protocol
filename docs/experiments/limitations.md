# Experimental & Architectural Limitations

This document records the known limitations of the current CMCP implementation and experimental evaluation so that results are interpreted accurately.

---

## 1. Hardware & Model-Scale Constraints
- **CPU-Only Execution:** All experiments ran on a single Intel Core i5-13500 (`15.70 GiB` RAM, no discrete GPU). Consequently, inference and representation extraction were restricted to compact models (`135M`, `0.5B`, and `1B` parameters).
- **Generalization to Larger Models:** Behavior on `7B`–`70B` models—especially whether larger receivers parse `CMCP_CTX` better or whether soft-prompt transfer improves with higher hidden capacity—was not tested.

## 2. `cmcp_structured` Quality Regression vs. `json_structured` (Exp A/B)
- On both `Qwen2.5-0.5B->Llama-3.2-1B` and `Llama-3.2-1B->Qwen2.5-0.5B`, `cmcp_structured` and `cmcp_binary` achieved lower `quality_score` (`0.7438` and `0.7583`) than `json_structured` (`0.9083` and `0.8583`), despite `1.0000` `info_fidelity_score` and smaller wire size (`274.125 B` vs `371.75 B`).
- Small instruction-tuned models are heavily pretrained on JSON syntax and handled `json_structured` more accurately and concisely than the custom `CMCP_CTX` line format.

## 3. Latency & Sender-Cost Asymmetry (Exp A, B, C)
- In Exp A/B, `nl_conversation` `total_latency_ms` (`4929.71 ms` and `7020.73 ms`) **includes sender generation time** (`85.125` and `127.125` generated tokens), whereas `json_structured`, `cmcp_structured`, and `cmcp_binary` used pre-constructed structured contexts (`sender_gen_tokens = 0`).
- Similarly, in Exp C, `conventional_summary` (`6383.302 ms`) includes sender summarization latency, whereas `cmcp_references` (`3802.966 ms`) and `cmcp_incremental` (`3664.379 ms`) used pre-built structured payloads. Thus, latency comparisons measure **wire + receiver + optional sender-summarization**, not sender-side structured extraction cost from raw prose.

## 4. Soft-Prompt / Internal State Transfer Failure on Factual Recall (Exp E)
- In Exp E (`SmolLM2-135M-Instruct` -> `Qwen2.5-0.5B-Instruct`), `learned_cmcp` (`K=8` latents, `48` training examples, frozen receiver) reduced NLL from `2.7108` to `1.5738` (perplexity `15.0412` -> `4.8249`), **but scored `0.0` `exact_match_rate` and `0.05` `keyword_recall`** on the 20 unseen factual QA pairs (identical to `zero_context` and `random_projection`), while `full_text` achieved `0.85` EM and `0.95` keyword recall.
- With only 48 training pairs and `K=8` pooled vectors injected at `inputs_embeds`, the adapter learned formatting/distributional cues rather than arbitrary unseen factual bindings.

## 5. Evaluation Scope & Unexecuted Components
- **Deterministic Keyword/Alias Scoring:** Exp A, B, and C use deterministic alias recall (`quality_score`), which is reproducible and fast but does not capture stylistic fluency or subtle semantic errors outside the expected alias list.
- **Small Sample Sizes:** Exp A/B had `n_runs = 24` per condition/direction (`8` tasks × `3` reps); Exp C had `n_runs = 9` (`3` scenarios × `3` reps); Exp D had `n_eval = 24`; Exp E had `n_eval = 20`.
- **Cross-Tokenizer Byte-Span Alignment (H3):** Formulated in [technical_hypotheses.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/research/technical_hypotheses.md) as Hypothesis H3, but not implemented or benchmarked as a standalone experiment.
- **Vect-Link Runtime Adapter (H5):** The decoupled integration architecture is specified in [vect_link_integration.md](file:///C:/Users/sgi.lm/Desktop/Cross-Model-Context-Protocol/docs/architecture/vect_link_integration.md), but `CMCP_ENABLE_VECTLINK=ON` was not compiled or benchmarked end-to-end against `Vect-Link`.
