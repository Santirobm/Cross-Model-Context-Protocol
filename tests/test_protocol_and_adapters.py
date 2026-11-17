"""Pytest suite for CMCP binary codec, structured schema, compat tiers, and live adapters."""

from __future__ import annotations

import pytest

from adapters.llama import LlamaAdapter
from adapters.model_registry import to_cmcp_profile
from adapters.qwen import QwenAdapter
from representations.binary.codec import (
    CMCP_ERR_HEADER_CRC,
    CMCP_ERR_PAYLOAD_CRC,
    CMCP_FLAG_COMPRESSED,
    CMCP_HEADER_SIZE,
    CMCP_TIER_0_EXACT_KV,
    CMCP_TIER_2_CROSS_MODEL,
    CMCP_TIER_INCOMPATIBLE,
    CMCPError,
    cmcp_check_compat,
    cmcp_check_kv_compat,
    cmcp_compress,
    cmcp_crc32c,
    cmcp_decompress,
    cmcp_header_decode,
    cmcp_header_encode,
    cmcp_tlv_read,
    cmcp_tlv_write,
    decode_frame,
    encode_frame,
)
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


def _sample_context() -> CMCPContext:
    return CMCPContext(
        task_id="test_ctx_01",
        concepts=["cross-model", "kv-cache", "crc32c"],
        entities=[
            Entity(name="Qwen2.5-0.5B", kind="model", value="qwen2"),
            Entity(name="Llama-3.2-1B", kind="model", value="llama"),
        ],
        relations=[Relation(src="Qwen2.5-0.5B", rel="transfers_to", dst="Llama-3.2-1B")],
        instructions=[Instruction(text="Preserve all constraints verbatim.", priority="high")],
        constraints=[Constraint(rule="No external network calls", scope="runtime")],
        evidences=[Evidence(id="E1", claim="CRC32C detects single-bit errors", source="cmcp_crc32c.c")],
        references=[Reference(id="R1", uri="cmcp://spec/header", digest="64B")],
        decisions=[Decision(topic="wire_format", choice="TLV+CRC32C", rationale="Zero-copy C parsing")],
    )


def test_header_encode_decode_roundtrip() -> None:
    payload = b"hello-cmcp-payload"
    hdr_in, hdr_bytes = cmcp_header_encode(
        payload=payload,
        compat_tier=CMCP_TIER_2_CROSS_MODEL,
        sender_model_hash=0x1122334455667788,
        receiver_model_hash=0x8877665544332211,
        sequence_id=42,
    )
    assert len(hdr_bytes) == CMCP_HEADER_SIZE
    hdr_out = cmcp_header_decode(hdr_bytes, payload=payload)
    assert bytes(hdr_out.magic) == b"CMCP"
    assert hdr_out.version == 1
    assert hdr_out.compat_tier == CMCP_TIER_2_CROSS_MODEL
    assert hdr_out.payload_bytes == len(payload)
    assert hdr_out.sequence_id == 42
    assert hdr_out.payload_crc32c == hdr_in.payload_crc32c
    assert hdr_out.header_crc32c == hdr_in.header_crc32c


def test_tlv_roundtrip_and_context_lossless() -> None:
    ctx = _sample_context()
    records = ctx.to_tlv_records()
    blob = cmcp_tlv_write(records)
    decoded_records = cmcp_tlv_read(blob)
    assert decoded_records == records

    ctx_back = CMCPContext.from_tlv_records(decoded_records, task_id="test_ctx_01")
    assert ctx_back == ctx


def test_lossless_compression_roundtrip() -> None:
    ctx = _sample_context()
    raw = (ctx.to_compact_text() + "\n") * 8
    raw_bytes = raw.encode("utf-8")
    compressed = cmcp_compress(raw_bytes)
    assert len(compressed) < len(raw_bytes)
    decompressed = cmcp_decompress(compressed, len(raw_bytes))
    assert decompressed == raw_bytes


def test_frame_encode_decode_compressed() -> None:
    ctx = _sample_context()
    # Add enough repetitive records to trigger compression
    for i in range(6):
        ctx.evidences.append(Evidence(id=f"E{i+2}", claim="Repeated benchmark telemetry claim", source="harness"))
    frame = encode_frame(ctx.to_tlv_records(), compress=True, sequence_id=7)
    hdr, recs = decode_frame(frame)
    assert hdr.sequence_id == 7
    assert (hdr.flags & CMCP_FLAG_COMPRESSED) != 0
    ctx_out = CMCPContext.from_tlv_records(recs, task_id=ctx.task_id)
    assert ctx_out == ctx


def test_crc32c_corruption_rejection() -> None:
    payload = b"important-context-bytes"
    assert cmcp_crc32c(payload) != 0
    _, hdr_bytes = cmcp_header_encode(payload=payload)

    # Corrupt header byte 10
    bad_hdr = bytearray(hdr_bytes)
    bad_hdr[10] ^= 0x01
    with pytest.raises(CMCPError) as exc_hdr:
        cmcp_header_decode(bytes(bad_hdr), payload=payload)
    assert exc_hdr.value.status == CMCP_ERR_HEADER_CRC

    # Corrupt payload byte 3
    bad_payload = bytearray(payload)
    bad_payload[3] ^= 0x80
    with pytest.raises(CMCPError) as exc_pay:
        cmcp_header_decode(hdr_bytes, payload=bytes(bad_payload))
    assert exc_pay.value.status == CMCP_ERR_PAYLOAD_CRC


def test_compat_tiers_and_kv_validation() -> None:
    qwen_a = to_cmcp_profile("Qwen2.5-0.5B-Instruct", n_ctx=1024, pos_offset=0)
    qwen_b = to_cmcp_profile("Qwen2.5-0.5B-Instruct", n_ctx=1024, pos_offset=0)
    qwen_diff_ctx = to_cmcp_profile("Qwen2.5-0.5B-Instruct", n_ctx=512, pos_offset=0)
    qwen_diff_pos = to_cmcp_profile("Qwen2.5-0.5B-Instruct", n_ctx=1024, pos_offset=16)
    llama = to_cmcp_profile("Llama-3.2-1B-Instruct", n_ctx=1024, pos_offset=0)

    res_exact = cmcp_check_compat(qwen_a, qwen_b, strict_kv=True)
    assert res_exact.tier == CMCP_TIER_0_EXACT_KV
    assert res_exact.kv_compatible == 1

    res_ctx = cmcp_check_kv_compat(qwen_a, qwen_diff_ctx)
    assert res_ctx.kv_compatible == 0

    res_pos = cmcp_check_kv_compat(qwen_a, qwen_diff_pos)
    assert res_pos.kv_compatible == 0

    res_cross_flex = cmcp_check_compat(qwen_a, llama, strict_kv=False)
    assert res_cross_flex.tier == CMCP_TIER_2_CROSS_MODEL
    assert res_cross_flex.kv_compatible == 0

    res_cross_strict = cmcp_check_compat(qwen_a, llama, strict_kv=True)
    assert res_cross_strict.tier == CMCP_TIER_INCOMPATIBLE
    assert res_cross_strict.kv_compatible == 0


def test_live_adapters_smoke() -> None:
    qwen = QwenAdapter(n_ctx=512, n_threads=4)
    q_res = qwen.generate("Reply with only the word READY.", max_tokens=12)
    assert q_res.arch == "qwen2"
    assert q_res.input_tokens > 0
    assert q_res.output_tokens > 0
    assert q_res.prefill_ms > 0.0
    assert q_res.rss_mib > 0.0
    assert len(q_res.output_text) > 0

    llama = LlamaAdapter(n_ctx=512, n_threads=4)
    l_res = llama.generate("Reply with only the word READY.", max_tokens=12)
    assert l_res.arch == "llama"
    assert l_res.input_tokens > 0
    assert l_res.output_tokens > 0
    assert l_res.prefill_ms > 0.0
    assert l_res.rss_mib > 0.0
    assert len(l_res.output_text) > 0
