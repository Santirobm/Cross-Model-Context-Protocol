"""Tests for the unified serialize(state, query=None, mode='compact') public API."""

from __future__ import annotations

import inspect

import pytest

from representations import (
    CMCPStateStore,
    StateOp,
    render_selected_unified,
    select_for_task,
    serialize,
)
from representations import adapters as adapters_mod


def _build_store() -> CMCPStateStore:
    st = CMCPStateStore()
    st.evidence.register("turn:1", "Node-10 is primary; standby Node-10B is backup only.")
    st.apply(StateOp("e1", "UPSERT", "entity", "Node-10", "primary; role=gateway", seq=1, version=1, evidence_ref="turn:1"))
    st.apply(StateOp("e2", "UPSERT", "entity", "Node-10B", "standby_backup_only", seq=2, version=1, evidence_ref="turn:1"))
    st.apply(StateOp("c1", "UPSERT", "constraint", "sla", "SLA<=40ms", seq=3, version=1))
    st.apply(StateOp("d1", "UPSERT", "decision", "mode", "mode=strict(from:mode=standard)", seq=4, version=2))
    st.apply(StateOp("r1", "UPSERT", "relation", "path", "Node-10 -> Relay-2 -> Sink-3", seq=5, version=1))
    st.apply(StateOp("f1", "UPSERT", "fact", "gateway_role", "gateway", seq=6, version=1))
    st.apply(StateOp("f2", "UPSERT", "fact", "audit_tag", "AUD-77", seq=7, version=1))
    for idx in range(5):
        st.apply(StateOp(f"noise_{idx}", "UPSERT", "fact", f"telemetry_{idx}", f"VAL-{idx}", seq=10 + idx, version=1))
    return st


def test_compact_mode_runs_deterministic_selection_and_is_default() -> None:
    sig = inspect.signature(serialize)
    assert sig.parameters["mode"].default == "compact"
    assert sig.parameters["query"].default is None

    st = _build_store()
    query = "What is the active SLA limit for Node-10 (not Node-10B)?"
    out_default = serialize(st, query=query)
    out_explicit = serialize(st, query=query, mode="compact")
    out_method = st.serialize(query=query)

    assert out_default == out_explicit == out_method
    assert "SLA<=40ms" in out_default
    assert "Node-10" in out_default
    assert "Node-10B" not in out_default
    assert "telemetry_0" not in out_default
    assert "AUD-77" not in out_default


def test_compact_mode_without_query_and_unknown_mode_raise_clear_errors() -> None:
    st = _build_store()

    with pytest.raises(ValueError, match="query"):
        serialize(st, mode="compact")

    with pytest.raises(ValueError, match="query"):
        serialize(st, query="   ", mode="compact")

    with pytest.raises(ValueError, match="query"):
        st.serialize()

    with pytest.raises(ValueError, match="Unknown serialization mode"):
        serialize(st, query="anything", mode="invalid_mode")  # type: ignore[arg-type]


def test_anchored_mode_preserves_full_store_even_with_query() -> None:
    st = _build_store()
    out_no_query = serialize(st, mode="anchored")
    out_with_query = serialize(st, query="What is the SLA limit for Node-10?", mode="anchored")
    out_method = st.serialize(query="What is audit_tag?", mode="anchored")

    assert out_no_query == out_with_query == out_method
    assert out_no_query == st.render_for_llm(style="r3_canon")
    # Canonicalized full state: no history parenthetical, no pruning of telemetry or standby entity
    assert "(from:mode=standard)" not in out_no_query
    assert "mode=strict" in out_no_query
    assert "Node-10B" in out_no_query
    assert "AUD-77" in out_no_query
    for idx in range(5):
        assert f"telemetry_{idx}" in out_no_query


def test_anchored_mode_does_not_invoke_task_selection_or_pruning(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _build_store()

    def _forbidden_select(*args, **kwargs):
        raise AssertionError("select_for_task must not be called in anchored mode")

    monkeypatch.setattr(adapters_mod, "select_for_task", _forbidden_select)
    out = serialize(st, query="Node-10 SLA", mode="anchored")
    assert "telemetry_4" in out
    assert "Node-10B" in out


def test_legacy_apis_remain_backward_compatible() -> None:
    st = _build_store()
    r1 = st.render_for_llm()
    r2 = st.render_for_llm(style="r2")
    r3 = st.render_for_llm(style="r3")
    r3_canon = st.render_for_llm(style="r3_canon")

    assert "CONSTRAINT: SLA<=40ms" in r1
    assert "Active constraints:" in r2
    assert "<constraints>" in r3
    assert "(from:mode=standard)" in r3
    assert "(from:mode=standard)" not in r3_canon

    sel = select_for_task("What is audit_tag?", st)
    unified = render_selected_unified(sel, st)
    assert "AUD-77" in unified


def test_different_queries_produce_different_compact_outputs_without_mutating_state() -> None:
    st = _build_store()
    hash_before = st.state_hash()
    view_before = st.active_view()

    out_sla = serialize(st, query="What is the SLA limit?", mode="compact")
    out_audit = serialize(st, query="What is the audit_tag?", mode="compact")

    assert out_sla != out_audit
    assert "SLA<=40ms" in out_sla
    assert "AUD-77" not in out_sla
    assert "AUD-77" in out_audit
    assert "SLA<=40ms" not in out_audit

    assert st.state_hash() == hash_before
    assert st.active_view() == view_before
