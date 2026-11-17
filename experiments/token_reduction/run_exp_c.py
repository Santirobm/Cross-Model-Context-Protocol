"""Experiment C: Full History vs Conventional Summary vs CMCP References vs CMCP Incremental."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from adapters.llama import LlamaAdapter
from adapters.qwen import QwenAdapter
from experiments.structured_transfer.run_exp_a_b import score_response
from representations.binary.codec import decode_frame, encode_frame
from representations.structured import (
    CMCPContext,
    Constraint,
    Decision,
    Entity,
    Evidence,
    Reference,
)


@dataclass
class MultiTurnScenario:
    scenario_id: str
    full_history: str
    full_ctx: CMCPContext
    needed_ref_id: str
    incremental_ctx: CMCPContext
    query: str
    expected_items: List[List[str]]


def build_scenarios() -> List[MultiTurnScenario]:
    return [
        MultiTurnScenario(
            scenario_id="S1_protocol_debugging",
            full_history=(
                "Turn 1 User: We are building the CMCP C99 core library in MSYS2 Clang64.\n"
                "Turn 1 Assistant: Confirmed. CMake builds cmcp.dll and test_cmcp_core.exe with -Wall -Wextra -Werror.\n"
                "Turn 2 User: Let's review the 64-byte header layout. Bytes 0..3 are 'CMCP', version=1, followed by flags, msg_type, compat_tier, "
                "sender_model_hash, receiver_model_hash, tokenizer_hash, payload_bytes, raw_bytes, sequence_id, reserved, payload_crc32c, header_crc32c.\n"
                "Turn 2 Assistant: All multi-byte integers are little-endian and header_crc32c covers bytes 0..59.\n"
                "Turn 3 User: Also remember our storage spec: GGUF models live in C:\\Users\\sgi.lm\\Desktop\\models and Qwen2.5-0.5B is 491400032 bytes.\n"
                "Turn 3 Assistant: Noted. And Llama-3.2-1B is 807694464 bytes.\n"
                "Turn 4 User: Now we hit a bug: a test frame failed with CMCP_ERR_PAYLOAD_CRC (-6) even though header_crc32c passed. "
                "We traced it to a transport truncation where payload_bytes in the header was 128 but only 120 bytes arrived."
            ),
            full_ctx=CMCPContext(
                task_id="S1_protocol_debugging",
                entities=[
                    Entity(name="cmcp.dll", kind="library", value="C99 MSYS2 Clang64"),
                    Entity(name="Qwen2.5-0.5B", kind="model", value="491400032 bytes"),
                    Entity(name="Llama-3.2-1B", kind="model", value="807694464 bytes"),
                ],
                references=[
                    Reference(id="REF_BUILD", uri="cmcp://build/cmake", digest="c99-clang64", resolved_text="CMake builds cmcp.dll with -Wall -Wextra -Werror"),
                    Reference(id="REF_HDR", uri="cmcp://spec/header", digest="64B-le", resolved_text="64B header: magic CMCP, version=1, header_crc32c covers 0..59"),
                    Reference(id="REF_BUG", uri="cmcp://trace/crc_err", digest="err-6", resolved_text="CMCP_ERR_PAYLOAD_CRC (-6) caused by truncation: header payload_bytes=128 vs 120 arrived"),
                ],
                constraints=[Constraint(rule="Never advance read_pos on CMCP_ERR_PAYLOAD_CRC (-6)")],
            ),
            needed_ref_id="REF_BUG",
            incremental_ctx=CMCPContext(
                task_id="S1_delta_t4",
                evidences=[Evidence(id="BUG_T4", claim="CMCP_ERR_PAYLOAD_CRC (-6) with valid header_crc32c: payload_bytes=128 vs 120 bytes arrived")],
                constraints=[Constraint(rule="Reject truncated frame; do not advance read_pos")],
            ),
            query="What caused CMCP_ERR_PAYLOAD_CRC (-6) when header_crc32c passed, and what were the expected vs arrived byte counts?",
            expected_items=[
                ["cmcp_err_payload_crc", "-6"],
                ["truncat", "120"],
                ["128"],
            ],
        ),
        MultiTurnScenario(
            scenario_id="S2_latency_budget",
            full_history=(
                "Turn 1 User: We need to keep end-to-end cross-model handoff under 2000 ms on CPU with 4 threads.\n"
                "Turn 1 Assistant: Let's budget each stage: tokenization < 5 ms, prefill < 250 ms, codec < 1 ms.\n"
                "Turn 2 User: In our Qwen2.5-0.5B baseline, tokenize_ms=0.035, prefill_ms=92.731, rss_mib=455.97.\n"
                "Turn 2 Assistant: That leaves plenty of headroom for Llama-3.2-1B receiver inference.\n"
                "Turn 3 User: Remember we also tested compression: raw TLV of 420 bytes compressed to 260 bytes using LZ-lite+RLE.\n"
                "Turn 3 Assistant: Great, 38% byte savings with zero external dependencies.\n"
                "Turn 4 User: New decision for Turn 4: set n_ctx=1024 and max_tokens=96 for all Level-A runs, and reject any run exceeding 2048 MiB RSS."
            ),
            full_ctx=CMCPContext(
                task_id="S2_latency_budget",
                references=[
                    Reference(id="REF_BUDGET", uri="cmcp://perf/budget", digest="2000ms", resolved_text="CPU 4 threads, total < 2000 ms, prefill < 250 ms, codec < 1 ms"),
                    Reference(id="REF_COMP", uri="cmcp://perf/comp", digest="38pct", resolved_text="LZ-lite+RLE compressed 420B TLV to 260B"),
                    Reference(id="REF_T4", uri="cmcp://cfg/turn4", digest="ctx1024", resolved_text="Decision: n_ctx=1024, max_tokens=96, max RSS limit 2048 MiB"),
                ],
            ),
            needed_ref_id="REF_T4",
            incremental_ctx=CMCPContext(
                task_id="S2_delta_t4",
                decisions=[
                    Decision(topic="context_window", choice="n_ctx=1024"),
                    Decision(topic="generation_limit", choice="max_tokens=96"),
                ],
                constraints=[Constraint(rule="Reject any run exceeding 2048 MiB RSS")],
            ),
            query="What are the Turn 4 settings for n_ctx, max_tokens, and the maximum RSS limit?",
            expected_items=[
                ["1024"],
                ["96"],
                ["2048"],
            ],
        ),
        MultiTurnScenario(
            scenario_id="S3_kv_guardrails",
            full_history=(
                "Turn 1 User: Let's document all 4 compatibility tiers in CMCP.\n"
                "Turn 1 Assistant: Tier 0 is Exact KV, Tier 1 is Same Family, Tier 2 is Cross-Model Structured/Latent, Tier 3 is Text Fallback.\n"
                "Turn 2 User: In Tier 0, what fields must match in cmcp_model_profile_t?\n"
                "Turn 2 Assistant: arch_hash, weights_hash, tokenizer_hash, n_layer, n_embd, n_head_kv, rope_type, rope_freq_scale, kv_dtype, n_ctx, and pos_offset.\n"
                "Turn 3 User: What happens if Qwen2.5-0.5B (n_layer=24, n_embd=896) tries to share KV with Llama-3.2-1B (n_layer=16, n_embd=2048)?\n"
                "Turn 3 Assistant: cmcp_check_kv_compat returns kv_compatible=0 with reason CMCP_COMPAT_MISMATCH_ARCH (1) and blocks raw KV injection.\n"
                "Turn 4 User: Confirm the fallback action when strict_kv=0."
            ),
            full_ctx=CMCPContext(
                task_id="S3_kv_guardrails",
                references=[
                    Reference(id="REF_TIERS", uri="cmcp://compat/tiers", digest="4tiers", resolved_text="Tier 0 Exact KV, Tier 1 Same Family, Tier 2 Cross-Model, Tier 3 Text Fallback"),
                    Reference(id="REF_FIELDS", uri="cmcp://compat/fields", digest="11fields", resolved_text="arch_hash, weights_hash, tokenizer_hash, n_layer, n_embd, n_head_kv, rope_type, rope_freq_scale, kv_dtype, n_ctx, pos_offset"),
                    Reference(id="REF_CROSS", uri="cmcp://compat/qwen_llama", digest="arch1", resolved_text="Qwen2.5 vs Llama-3.2: kv_compatible=0, reason CMCP_COMPAT_MISMATCH_ARCH (1); strict_kv=0 falls back to CMCP_TIER_2_CROSS_MODEL"),
                ],
            ),
            needed_ref_id="REF_CROSS",
            incremental_ctx=CMCPContext(
                task_id="S3_delta_t4",
                evidences=[Evidence(id="KV1", claim="Qwen2.5 vs Llama-3.2 -> kv_compatible=0, CMCP_COMPAT_MISMATCH_ARCH (1)")],
                decisions=[Decision(topic="strict_kv_0_fallback", choice="CMCP_TIER_2_CROSS_MODEL", rationale="Block raw KV injection, use structured transfer")],
            ),
            query="What does cmcp_check_kv_compat return for Qwen2.5 vs Llama-3.2, and what tier is selected when strict_kv=0?",
            expected_items=[
                ["0", "incompatible", "mismatch"],
                ["arch", "cmcp_compat_mismatch_arch"],
                ["tier_2", "tier 2", "cross_model", "cross-model"],
            ],
        ),
    ]


def run_experiment_c(reps: int = 3, max_tokens: int = 80) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen = QwenAdapter(n_ctx=1024, n_threads=4)
    llama = LlamaAdapter(n_ctx=1024, n_threads=4)

    _ = qwen.generate("Say OK.", max_tokens=4)
    _ = llama.generate("Say OK.", max_tokens=4)

    scenarios = build_scenarios()
    conditions = ["full_history", "conventional_summary", "cmcp_references", "cmcp_incremental"]
    runs: List[Dict[str, Any]] = []

    for sc in scenarios:
        for cond in conditions:
            for rep in range(1, reps + 1):
                codec_ms = 0.0
                sender_ms = 0.0
                sender_tokens = 0
                wire_bytes = 0

                if cond == "full_history":
                    wire_bytes = len(sc.full_history.encode("utf-8"))
                    recv_prompt = f"History:\n{sc.full_history}\n\nQuestion: {sc.query}\nAnswer concisely."
                elif cond == "conventional_summary":
                    sum_prompt = f"Summarize this multi-turn conversation concisely, keeping all numbers and error codes:\n\n{sc.full_history}"
                    s_res = qwen.generate(sum_prompt, max_tokens=120)
                    sender_tokens = s_res.output_tokens
                    sender_ms = s_res.tokenize_ms + s_res.prefill_ms + s_res.eval_ms
                    wire_bytes = len(s_res.output_text.encode("utf-8"))
                    recv_prompt = f"Summary:\n{s_res.output_text}\n\nQuestion: {sc.query}\nAnswer concisely."
                elif cond == "cmcp_references":
                    t0 = time.perf_counter()
                    # Resolve ONLY needed_ref_id; keep other references as compact URI+digest pointers
                    selective_ctx = sc.full_ctx.model_copy(deep=True)
                    for ref in selective_ctx.references:
                        if ref.id != sc.needed_ref_id:
                            ref.resolved_text = ""
                    frame = encode_frame(selective_ctx.to_tlv_records(), compress=True, sequence_id=rep)
                    wire_bytes = len(frame)
                    _, recs = decode_frame(frame)
                    decoded = CMCPContext.from_tlv_records(recs, task_id=sc.scenario_id)
                    recv_prompt = decoded.to_wire_prompt(sc.query, include_resolved_refs=True)
                    codec_ms = (time.perf_counter() - t0) * 1000.0
                else:  # cmcp_incremental
                    t0 = time.perf_counter()
                    frame = encode_frame(sc.incremental_ctx.to_tlv_records(), compress=True, sequence_id=rep)
                    wire_bytes = len(frame)
                    _, recs = decode_frame(frame)
                    decoded = CMCPContext.from_tlv_records(recs, task_id=sc.incremental_ctx.task_id)
                    recv_prompt = decoded.to_wire_prompt(sc.query)
                    codec_ms = (time.perf_counter() - t0) * 1000.0

                r_res = llama.generate(recv_prompt, max_tokens=max_tokens)
                quality = score_response(r_res.output_text, sc.expected_items)
                total_ms = sender_ms + codec_ms + r_res.tokenize_ms + r_res.prefill_ms + r_res.eval_ms

                runs.append(
                    {
                        "scenario_id": sc.scenario_id,
                        "condition": cond,
                        "rep": rep,
                        "wire_bytes": wire_bytes,
                        "sender_gen_tokens": sender_tokens,
                        "receiver_input_tokens": r_res.input_tokens,
                        "receiver_output_tokens": r_res.output_tokens,
                        "prefill_ms": round(r_res.prefill_ms, 3),
                        "eval_ms": round(r_res.eval_ms, 3),
                        "codec_ms": round(codec_ms, 4),
                        "total_latency_ms": round(total_ms, 3),
                        "quality_score": round(quality, 4),
                        "rss_mib": round(r_res.rss_mib, 2),
                        "receiver_output": r_res.output_text,
                    }
                )

    summary: Dict[str, Any] = {}
    for cond in conditions:
        subset = [r for r in runs if r["condition"] == cond]
        summary[cond] = {
            "n_runs": len(subset),
            "wire_bytes_mean": round(statistics.mean(r["wire_bytes"] for r in subset), 2),
            "receiver_input_tokens_mean": round(statistics.mean(r["receiver_input_tokens"] for r in subset), 2),
            "prefill_ms_mean": round(statistics.mean(r["prefill_ms"] for r in subset), 3),
            "total_latency_ms_mean": round(statistics.mean(r["total_latency_ms"] for r in subset), 3),
            "total_latency_ms_p95": round(float(np.percentile([r["total_latency_ms"] for r in subset], 95)), 3),
            "quality_score_mean": round(statistics.mean(r["quality_score"] for r in subset), 4),
        }

    payload = {
        "experiment": "Exp_C_Token_Reduction",
        "reps": reps,
        "conditions": conditions,
        "summary": summary,
        "runs": runs,
    }

    (out_dir / "exp_c_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_fields = [k for k in runs[0].keys() if k != "receiver_output"]
    with (out_dir / "exp_c_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in runs:
            writer.writerow({k: r[k] for k in csv_fields})

    return payload


if __name__ == "__main__":
    out = run_experiment_c(reps=3, max_tokens=80)
    print(json.dumps(out["summary"], indent=2))
