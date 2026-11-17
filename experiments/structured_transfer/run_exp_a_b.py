"""Experiments A & B: Cross-model structured context transfer (Qwen2.5-0.5B <-> Llama-3.2-1B)."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from adapters.base.adapter import AdapterResponse, ModelAdapter
from adapters.llama import LlamaAdapter
from adapters.qwen import QwenAdapter
from representations.binary.codec import decode_frame, encode_frame
from representations.structured import (
    CMCPContext,
    Constraint,
    Decision,
    Entity,
    Evidence,
    Instruction,
    Reference,
    Relation,
)


@dataclass
class EvalTask:
    task_id: str
    category: str
    title: str
    nl_history: str
    cmcp_ctx: CMCPContext
    query: str
    expected_items: List[List[str]]  # Each inner list = acceptable aliases for 1 required item


def build_tasks() -> List[EvalTask]:
    """Build 8 deterministic multi-step tasks covering research, code, audit, constraints, decisions, contradictions, evidence, and execution."""
    return [
        EvalTask(
            task_id="T1_research_continuation",
            category="research_continuation",
            title="KV-Cache Transfer Feasibility Continuation",
            nl_history=(
                "User: We are continuing our research on cross-model context transfer between Qwen2.5-0.5B and Llama-3.2-1B.\n"
                "Assistant: Sure! Let's review the measurements from Phase 1. Qwen2.5-0.5B uses qwen2 architecture with 24 layers, "
                "896 embedding dimensions, 2 KV heads, and RoPE type 2. Meanwhile Llama-3.2-1B uses llama architecture with 16 layers, "
                "2048 embedding dimensions, 8 KV heads, and RoPE type 0.\n"
                "User: Right, and remember we already ruled out direct raw KV-cache copying across different architectures.\n"
                "Assistant: Agreed. Our working hypothesis H1 is that structured TLV context transfer with CRC32C integrity avoids "
                "re-prefilling raw conversation history while staying architecture-independent."
            ),
            cmcp_ctx=CMCPContext(
                task_id="T1_research_continuation",
                entities=[
                    Entity(name="Qwen2.5-0.5B", kind="model", value="arch=qwen2,layers=24,embd=896,kv_heads=2,rope=2"),
                    Entity(name="Llama-3.2-1B", kind="model", value="arch=llama,layers=16,embd=2048,kv_heads=8,rope=0"),
                ],
                constraints=[Constraint(rule="No direct raw KV-cache copying across different architectures")],
                decisions=[Decision(topic="transfer_mechanism", choice="TLV+CRC32C structured transfer", rationale="Hypothesis H1: architecture-independent")],
                instructions=[Instruction(text="Recommend next validation step for H1")],
            ),
            query="Why is direct KV-cache sharing rejected between Qwen2.5-0.5B and Llama-3.2-1B, and what mechanism (H1) is chosen instead?",
            expected_items=[
                ["qwen2", "24"],
                ["llama", "16"],
                ["incompatible", "different architecture", "ruled out", "rejected", "no direct"],
                ["tlv", "structured"],
                ["crc32c", "h1"],
            ],
        ),
        EvalTask(
            task_id="T2_coding_problem",
            category="coding_problem",
            title="Ring Buffer Overflow Guard",
            nl_history=(
                "User: I am writing a C99 shared-memory ring buffer for CMCP frames.\n"
                "Assistant: Great. What capacity and framing rules do you have?\n"
                "User: The header is fixed at 64 bytes with magic bytes 'CMCP'. The payload limit is 64 MiB. "
                "If a writer tries to enqueue a frame larger than remaining free space, it must return CMCP_ERR_LIMIT_EXCEEDED (-10) "
                "instead of overwriting unread bytes.\n"
                "Assistant: Got it, and remember to validate both header_crc32c and payload_crc32c before advancing read_pos."
            ),
            cmcp_ctx=CMCPContext(
                task_id="T2_coding_problem",
                entities=[
                    Entity(name="CMCPHeader", kind="struct", value="64 bytes, magic='CMCP'"),
                    Entity(name="MaxPayload", kind="limit", value="64 MiB"),
                ],
                constraints=[
                    Constraint(rule="Return CMCP_ERR_LIMIT_EXCEEDED (-10) on insufficient free space; never overwrite unread bytes"),
                    Constraint(rule="Verify header_crc32c and payload_crc32c before advancing read_pos"),
                ],
                instructions=[Instruction(text="List the exact error code and two validation checks required before advancing read_pos")],
            ),
            query="What error code must be returned when free space is insufficient, what is the header size, and which two CRCs must be verified before advancing read_pos?",
            expected_items=[
                ["cmcp_err_limit_exceeded", "-10"],
                ["64"],
                ["header_crc32c"],
                ["payload_crc32c"],
                ["read_pos", "unread"],
            ],
        ),
        EvalTask(
            task_id="T3_code_audit",
            category="code_audit",
            title="Decompression Buffer Safety Audit",
            nl_history=(
                "User: Please audit our cmcp_decompress function in cmcp_tlv.c.\n"
                "Assistant: Let's inspect the bounds checks. Currently, cmcp_decompress receives src, src_len, dst, dst_cap, and out_len. "
                "We noticed that a crafted compressed stream with an RLE run count of 255 could write past dst_cap if the loop only checks pos < dst_cap at the start of the token.\n"
                "User: Good catch! What should it return if pos + run_len > dst_cap?\n"
                "Assistant: It must return CMCP_ERR_BUFFER_TOO_SMALL (-2) immediately, and if a literal or match offset points before the start of dst, return CMCP_ERR_CORRUPT_DATA (-7)."
            ),
            cmcp_ctx=CMCPContext(
                task_id="T3_code_audit",
                entities=[Entity(name="cmcp_decompress", kind="function", value="cmcp_tlv.c")],
                evidences=[Evidence(id="BUG1", claim="RLE run count up to 255 can overflow dst_cap if checked only once")],
                constraints=[
                    Constraint(rule="If pos + run_len > dst_cap return CMCP_ERR_BUFFER_TOO_SMALL (-2)"),
                    Constraint(rule="If offset points before dst start return CMCP_ERR_CORRUPT_DATA (-7)"),
                ],
            ),
            query="Summarize the buffer overflow risk in cmcp_decompress and state the exact return codes for dst_cap overflow and invalid match offset.",
            expected_items=[
                ["255", "rle", "overflow", "dst_cap"],
                ["cmcp_err_buffer_too_small", "-2"],
                ["cmcp_err_corrupt_data", "-7"],
            ],
        ),
        EvalTask(
            task_id="T4_constraint_retention",
            category="constraint_retention",
            title="Strict Runtime & Workspace Boundaries",
            nl_history=(
                "User: Before deploying the CMCP Python SDK, let's lock down all operational constraints.\n"
                "Assistant: Sure, please list them.\n"
                "User: First, never modify anything in Vect-Link. Second, use only local GGUF models with no external API calls. "
                "Third, all binary frames must use CRC32C Castagnoli, not CRC32-IEEE. Fourth, set n_threads=4 for reproducible CPU latency."
            ),
            cmcp_ctx=CMCPContext(
                task_id="T4_constraint_retention",
                constraints=[
                    Constraint(rule="Never modify Vect-Link"),
                    Constraint(rule="Use only local GGUF models; no external API calls"),
                    Constraint(rule="Use CRC32C Castagnoli (not CRC32-IEEE)"),
                    Constraint(rule="Use n_threads=4 for reproducible CPU latency"),
                ],
            ),
            query="List all four operational constraints including the forbidden directory, network rule, checksum algorithm, and thread count.",
            expected_items=[
                ["vect-link"],
                ["local gguf", "no external", "offline"],
                ["crc32c", "castagnoli"],
                ["4", "n_threads=4", "n_threads = 4"],
            ],
        ),
        EvalTask(
            task_id="T5_decision_retention",
            category="decision_retention",
            title="Architecture & Wire Format Decisions",
            nl_history=(
                "User: Let's finalize our architectural decisions for CMCP v1.\n"
                "Assistant: Okay, what did we decide on serialization, compression, and quantization?\n"
                "User: For serialization we chose TLV records with varint lengths instead of JSON on the wire. "
                "For compression we chose our custom LZ-lite + RLE codec so there are zero external dependencies like zlib. "
                "For local inference we chose Q4_K_M quantization to fit within 1 GiB RSS."
            ),
            cmcp_ctx=CMCPContext(
                task_id="T5_decision_retention",
                decisions=[
                    Decision(topic="serialization", choice="TLV with varint lengths", rationale="Replaces JSON on wire"),
                    Decision(topic="compression", choice="LZ-lite + RLE", rationale="Zero external dependencies"),
                    Decision(topic="quantization", choice="Q4_K_M", rationale="Fits within 1 GiB RSS"),
                ],
            ),
            query="State the three finalized decisions (serialization, compression, quantization) and their rationales.",
            expected_items=[
                ["tlv", "varint"],
                ["lz-lite", "rle", "zero external"],
                ["q4_k_m", "1 gib", "rss"],
            ],
        ),
        EvalTask(
            task_id="T6_contradiction_detection",
            category="contradiction_detection",
            title="KV-Cache Compatibility Contradiction",
            nl_history=(
                "User: A teammate claims we can load a KV-cache snapshot from Qwen2.5-0.5B (n_ctx=1024, pos_offset=0) directly into "
                "Qwen2.5-0.5B with n_ctx=512, and also into Llama-3.2-1B without re-prefilling.\n"
                "Assistant: Let's check the evidence. In cmcp_compat.c, cmcp_check_kv_compat requires identical arch_hash, "
                "weights_hash, n_layer, n_embd, n_head_kv, rope_type, kv_dtype, n_ctx, and pos_offset. "
                "Both n_ctx=512 vs 1024 and Qwen2.5 vs Llama-3.2 violate these invariants."
            ),
            cmcp_ctx=CMCPContext(
                task_id="T6_contradiction_detection",
                entities=[
                    Entity(name="ClaimA", kind="claim", value="Load Qwen2.5 n_ctx=1024 KV into Qwen2.5 n_ctx=512"),
                    Entity(name="ClaimB", kind="claim", value="Load Qwen2.5 KV into Llama-3.2-1B"),
                ],
                evidences=[
                    Evidence(id="INV1", claim="cmcp_check_kv_compat requires identical n_ctx, pos_offset, arch_hash, n_layer, n_embd, n_head_kv"),
                ],
                constraints=[Constraint(rule="Flag any claim violating INV1 as INCOMPATIBLE")],
            ),
            query="Identify the two contradictions in the teammate's claims against INV1 and state whether each is compatible or incompatible.",
            expected_items=[
                ["incompatible"],
                ["n_ctx", "512", "1024"],
                ["llama", "qwen", "arch"],
            ],
        ),
        EvalTask(
            task_id="T7_evidence_retrieval",
            category="evidence_retrieval",
            title="Benchmark Telemetry Evidence Lookup",
            nl_history=(
                "User: Let's record the exact baseline metrics from our harness run.\n"
                "Assistant: Sure. For Qwen2.5-0.5B-Instruct on prompt 'Hello', tokenize_ms was 0.035 ms, prefill_ms was 92.731 ms, "
                "eval_ms was 359.304 ms for 16 tokens, and RSS was 455.97 MiB. "
                "The GGUF file size is 491400032 bytes."
            ),
            cmcp_ctx=CMCPContext(
                task_id="T7_evidence_retrieval",
                evidences=[
                    Evidence(id="M1", claim="Qwen2.5-0.5B tokenize_ms=0.035, prefill_ms=92.731, eval_ms=359.304, tokens=16"),
                    Evidence(id="M2", claim="Qwen2.5-0.5B rss_mib=455.97, gguf_bytes=491400032"),
                ],
            ),
            query="Report the exact prefill_ms, eval_ms, rss_mib, and gguf_bytes from the recorded evidence.",
            expected_items=[
                ["92.731"],
                ["359.304"],
                ["455.97"],
                ["491400032"],
            ],
        ),
        EvalTask(
            task_id="T8_multistep_execution",
            category="multistep_execution",
            title="Three-Step Handshake & Transfer Plan",
            nl_history=(
                "User: Outline the exact 3-step execution sequence for a Tier-2 cross-model transfer.\n"
                "Assistant: Step 1: Exchange CMCP_MSG_HELLO with CMCPModelProfile and run cmcp_check_compat(strict_kv=false), "
                "which returns CMCP_TIER_2_CROSS_MODEL. "
                "Step 2: Encode CMCPContext into TLV records, compress with cmcp_compress, and wrap in a 64-byte header via CMCP_MSG_CONTEXT_TRANSFER. "
                "Step 3: Receiver verifies CRC32C, decompresses TLV, renders wire prompt, and sends CMCP_MSG_ACK."
            ),
            cmcp_ctx=CMCPContext(
                task_id="T8_multistep_execution",
                instructions=[
                    Instruction(text="Step 1: Send CMCP_MSG_HELLO and run cmcp_check_compat(strict_kv=false) -> CMCP_TIER_2_CROSS_MODEL"),
                    Instruction(text="Step 2: Encode TLV, cmcp_compress, send CMCP_MSG_CONTEXT_TRANSFER with 64-byte header"),
                    Instruction(text="Step 3: Verify CRC32C, decompress TLV, render wire prompt, reply CMCP_MSG_ACK"),
                ],
            ),
            query="State all 3 steps in order, naming the three message types (HELLO, CONTEXT_TRANSFER, ACK) and the compatibility tier.",
            expected_items=[
                ["cmcp_msg_hello", "hello"],
                ["cmcp_tier_2_cross_model", "tier_2", "tier 2"],
                ["cmcp_msg_context_transfer", "context_transfer"],
                ["crc32c"],
                ["cmcp_msg_ack", "ack"],
            ],
        ),
    ]


def score_response(text: str, expected_items: Sequence[Sequence[str]]) -> float:
    """Deterministic recall score in [0.0, 1.0] over expected_items."""
    if not expected_items:
        return 1.0
    lowered = text.lower()
    hits = 0
    for aliases in expected_items:
        if any(alias.lower() in lowered for alias in aliases):
            hits += 1
    return hits / len(expected_items)


def _summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        "message_bytes",
        "sender_gen_tokens",
        "receiver_input_tokens",
        "tokenize_ms",
        "prefill_ms",
        "eval_ms",
        "codec_ms",
        "total_latency_ms",
        "max_rss_mib",
        "throughput_tok_s",
        "quality_score",
        "info_fidelity_score",
    ]
    summary: Dict[str, Any] = {"n_runs": len(rows)}
    for m in metrics:
        vals = [float(r[m]) for r in rows]
        summary[m] = {
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "p95": round(float(np.percentile(vals, 95)), 4),
        }
    return summary


def run_experiments_ab(reps: int = 3, max_tokens: int = 96) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen = QwenAdapter(n_ctx=1024, n_threads=4)
    llama = LlamaAdapter(n_ctx=1024, n_threads=4)

    # Warm-up runs to stabilize OS page cache
    _ = qwen.generate("Say OK.", max_tokens=4)
    _ = llama.generate("Say OK.", max_tokens=4)

    tasks = build_tasks()
    directions: List[Tuple[str, ModelAdapter, ModelAdapter]] = [
        ("Qwen2.5-0.5B->Llama-3.2-1B", qwen, llama),
        ("Llama-3.2-1B->Qwen2.5-0.5B", llama, qwen),
    ]
    conditions = ["nl_conversation", "json_structured", "cmcp_structured", "cmcp_binary"]

    runs: List[Dict[str, Any]] = []

    for dir_name, sender, receiver in directions:
        for task in tasks:
            for cond in conditions:
                for rep in range(1, reps + 1):
                    sender_gen_tokens = 0
                    sender_ms = 0.0
                    sender_rss = 0.0
                    codec_ms = 0.0
                    wire_bytes = 0
                    info_fidelity = 1.0

                    if cond == "nl_conversation":
                        handoff_prompt = (
                            "Summarize all technical facts, numbers, constraints, decisions, and steps "
                            "from this conversation concisely for another AI model:\n\n"
                            f"{task.nl_history}"
                        )
                        s_res: AdapterResponse = sender.generate(handoff_prompt, max_tokens=128)
                        handoff_text = s_res.output_text
                        sender_gen_tokens = s_res.output_tokens
                        sender_ms = s_res.tokenize_ms + s_res.prefill_ms + s_res.eval_ms
                        sender_rss = s_res.rss_mib
                        wire_bytes = len(handoff_text.encode("utf-8"))
                        info_fidelity = score_response(handoff_text, task.expected_items)
                        recv_prompt = (
                            f"Context from previous model:\n{handoff_text}\n\n"
                            f"Question: {task.query}\nAnswer concisely."
                        )
                    elif cond == "json_structured":
                        t0 = time.perf_counter()
                        wire_str = task.cmcp_ctx.to_json_str(compact=True)
                        wire_bytes = len(wire_str.encode("utf-8"))
                        parsed = CMCPContext.from_json_str(wire_str)
                        recv_prompt = (
                            f"JSON_CTX:\n{parsed.to_json_str(compact=True)}\n\n"
                            f"QUERY: {task.query}\nAnswer concisely."
                        )
                        codec_ms = (time.perf_counter() - t0) * 1000.0
                    elif cond == "cmcp_structured":
                        t0 = time.perf_counter()
                        wire_str = task.cmcp_ctx.to_compact_text()
                        wire_bytes = len(wire_str.encode("utf-8"))
                        recv_prompt = task.cmcp_ctx.to_wire_prompt(task.query)
                        codec_ms = (time.perf_counter() - t0) * 1000.0
                    else:  # cmcp_binary
                        t0 = time.perf_counter()
                        frame = encode_frame(task.cmcp_ctx.to_tlv_records(), compress=True, sequence_id=rep)
                        wire_bytes = len(frame)
                        _, recs = decode_frame(frame)
                        decoded_ctx = CMCPContext.from_tlv_records(recs, task_id=task.task_id)
                        recv_prompt = decoded_ctx.to_wire_prompt(task.query)
                        codec_ms = (time.perf_counter() - t0) * 1000.0

                    r_res: AdapterResponse = receiver.generate(recv_prompt, max_tokens=max_tokens)
                    quality = score_response(r_res.output_text, task.expected_items)
                    total_latency_ms = (
                        sender_ms + codec_ms + r_res.tokenize_ms + r_res.prefill_ms + r_res.eval_ms
                    )
                    max_rss = max(sender_rss, r_res.rss_mib)

                    runs.append(
                        {
                            "direction": dir_name,
                            "task_id": task.task_id,
                            "category": task.category,
                            "condition": cond,
                            "rep": rep,
                            "message_bytes": wire_bytes,
                            "sender_gen_tokens": sender_gen_tokens,
                            "receiver_input_tokens": r_res.input_tokens,
                            "receiver_output_tokens": r_res.output_tokens,
                            "tokenize_ms": round(r_res.tokenize_ms, 3),
                            "prefill_ms": round(r_res.prefill_ms, 3),
                            "eval_ms": round(r_res.eval_ms, 3),
                            "codec_ms": round(codec_ms, 4),
                            "sender_ms": round(sender_ms, 3),
                            "total_latency_ms": round(total_latency_ms, 3),
                            "max_rss_mib": round(max_rss, 2),
                            "throughput_tok_s": round(r_res.throughput_tok_s, 2),
                            "quality_score": round(quality, 4),
                            "info_fidelity_score": round(info_fidelity, 4),
                            "receiver_output": r_res.output_text,
                        }
                    )

    summary: Dict[str, Any] = {}
    for dir_name, _, _ in directions:
        summary[dir_name] = {}
        for cond in conditions:
            subset = [r for r in runs if r["direction"] == dir_name and r["condition"] == cond]
            summary[dir_name][cond] = _summarize_group(subset)

    payload = {
        "experiment": "Exp_A_B_Structured_Transfer",
        "reps_per_task": reps,
        "n_tasks": len(tasks),
        "conditions": conditions,
        "summary": summary,
        "runs": runs,
    }

    json_path = out_dir / "exp_ab_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_dir / "exp_ab_results.csv"
    fieldnames = [k for k in runs[0].keys() if k != "receiver_output"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            writer.writerow({k: r[k] for k in fieldnames})

    return payload


if __name__ == "__main__":
    res = run_experiments_ab(reps=3, max_tokens=96)
    print(json.dumps(res["summary"], indent=2))
