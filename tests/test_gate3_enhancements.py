"""Unit and integration tests for GATE 3: 1-hop graph expansion & deterministic ingestion hooks."""

from __future__ import annotations

import time

from ingestion import extract_explicit_constraints, ingest_tool_output
from representations import CMCPStateStore, StateOp, select_for_task, serialize


def test_expand_hops_zero_vs_one_and_budget_cap() -> None:
    st = CMCPStateStore()
    st.apply(StateOp("e_gw", "UPSERT", "entity", "Node-10", "primary", seq=1, version=1))
    st.apply(StateOp("e_rel", "UPSERT", "entity", "Relay-2", "active", seq=2, version=1))
    st.apply(StateOp("e_bk", "UPSERT", "entity", "Node-10B", "standby_backup_only", seq=3, version=1))
    st.apply(StateOp("r_path", "UPSERT", "relation", "path", "Node-10 -> Relay-2 -> Sink-3", seq=4, version=1))
    st.apply(StateOp("c_sla", "UPSERT", "constraint", "sla", "SLA<=40ms", seq=5, version=1))
    # Add extra neighbors to test strict budget cap
    for idx in range(8):
        fid = f"f_extra_{idx}"
        st.apply(StateOp(fid, "UPSERT", "fact", f"neighbor_{idx}", f"val_{idx}", seq=10 + idx, version=1))
        st.add_relation("e_gw", "DEPENDS_ON", fid, seq=20 + idx)

    sel_0 = select_for_task("What is the status of Node-10?", st, expand_hops=0)
    assert "e_gw" in sel_0.selected_ids
    assert "c_sla" not in sel_0.selected_ids

    sel_1 = select_for_task("What is the status of Node-10?", st, expand_hops=1, max_expand_items=3)
    assert "e_gw" in sel_1.selected_ids
    # 1-hop added at most max_expand_items=3 neighbors and never standby Node-10B
    assert "e_bk" not in sel_1.selected_ids
    added = set(sel_1.selected_ids) - set(sel_0.selected_ids)
    assert 1 <= len(added) <= 3

    out_default = serialize(st, query="What is the status of Node-10?", mode="compact")
    out_exp0 = serialize(st, query="What is the status of Node-10?", mode="compact", expand_hops=0)
    assert out_default == out_exp0


def test_ingest_tool_output_idempotency_and_updates() -> None:
    st = CMCPStateStore()
    payload = {
        "service": "billing-api",
        "limits": {"timeout_ms": 250, "max_retries": 3},
        "mode": "strict",
        "route_graph": "billing-api -> auth-proxy -> db-main",
        "temp_token": "TK-101",
    }
    ops1 = ingest_tool_output(st, payload, evidence_ref="tool:call1")
    assert len(ops1) == 6
    h1 = st.state_hash()

    # Re-ingesting identical payload must be 100% idempotent
    ops2 = ingest_tool_output(st, payload, evidence_ref="tool:call1")
    assert ops2 == []
    assert st.state_hash() == h1

    # Updating a key & revoking temp_token invalidates obsolete values cleanly
    payload_v2 = {
        "service": "billing-api",
        "limits": {"timeout_ms": 150, "max_retries": 3},
        "mode": "strict",
        "temp_token": "REVOKED",
    }
    ops3 = ingest_tool_output(st, payload_v2, evidence_ref="tool:call2")
    assert len(ops3) == 2
    view = st.active_view()
    assert view["tool:limits.timeout_ms"]["value"] == "150"
    assert "tool:temp_token" not in view


def test_extract_explicit_constraints_rules_and_zero_false_positives() -> None:
    neutral = "The service started at 10:00 UTC, processed 5 items, and latency was 12ms yesterday."
    assert extract_explicit_constraints(neutral) == []

    text = (
        "All workers must use Python 3.11 and SLA <= 40ms. "
        "Never expose raw credentials in logs. "
        "No unauthenticated requests allowed; budget: 120 tokens."
    )
    st = CMCPStateStore()
    ops = extract_explicit_constraints(text, store=st, evidence_ref="policy:doc1")
    assert len(ops) >= 4
    vals = " | ".join(op.value for op in ops)
    assert "SLA<=40ms" in vals
    assert "never expose raw credentials in logs" in vals
    assert "no unauthenticated requests allowed" in vals
    assert "budget<=120tokens" in vals

    # Idempotent on re-run
    assert extract_explicit_constraints(text, store=st, evidence_ref="policy:doc1") == []


def test_deterministic_ingestion_end_to_end_serialize_and_latency() -> None:
    st = CMCPStateStore()
    t0 = time.perf_counter()
    ingest_tool_output(
        st,
        {
            "primary_unit": "Node-7",
            "route_code": "R-88",
            "telemetry_noise": "OK-999",
        },
        evidence_ref="tool:init",
    )
    extract_explicit_constraints(
        "Constraint: latency <= 25ms. Must not route through Node-7B.",
        store=st,
        evidence_ref="policy:sla",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 20.0

    compact_out = serialize(st, query="What is the latency limit and route_code?", mode="compact")
    anchored_out = serialize(st, query="What is the latency limit?", mode="anchored")

    assert "latency<=25ms" in compact_out
    assert "R-88" in compact_out
    assert "OK-999" not in compact_out

    assert "latency<=25ms" in anchored_out
    assert "R-88" in anchored_out
    assert "OK-999" in anchored_out
