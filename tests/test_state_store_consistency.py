"""Consistency, versioning, tombstone, and conflict tests for CMCPStateStore (Section 10)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from representations.state_store import CMCPStateStore, ReferenceMissingError, StateOp


def test_duplicate_update_is_idempotent() -> None:
    store = CMCPStateStore()
    op = StateOp("f1", "UPSERT", "fact", "port", "7100", seq=1, version=1, agent_id="A")
    assert store.apply(op) == "APPLIED"
    assert store.apply(op) == "DUPLICATE_IGNORED"
    assert store.active_view()["f1"]["value"] == "7100"


def test_out_of_order_stale_update_rejected() -> None:
    store = CMCPStateStore()
    assert store.apply(StateOp("f1", "UPSERT", "fact", "port", "7200", seq=2, version=2)) == "APPLIED"
    assert store.apply(StateOp("f1", "UPSERT", "fact", "port", "7100", seq=1, version=1)) == "STALE_IGNORED"
    assert store.active_view()["f1"]["value"] == "7200"


def test_invalidation_tombstone_prevents_resurrection() -> None:
    store = CMCPStateStore()
    assert store.apply(StateOp("f1", "UPSERT", "entity", "Node-1", "active", seq=1, version=1)) == "APPLIED"
    assert store.apply(StateOp("f1", "INVALIDATE", "entity", "Node-1", "", seq=2, version=2)) == "APPLIED"
    assert "f1" not in store.active_view()
    # Late arrival of older seq must not resurrect
    assert store.apply(StateOp("f1", "UPSERT", "entity", "Node-1", "active", seq=1, version=1)) == "STALE_IGNORED"
    assert "f1" not in store.active_view()


def test_fact_mutation_and_history_checkout() -> None:
    store = CMCPStateStore()
    store.apply(StateOp("c1", "UPSERT", "constraint", "latency", "<=50ms", seq=1, version=1))
    store.apply(StateOp("c1", "UPSERT", "constraint", "latency", "<=25ms", seq=2, version=2))
    assert store.active_view()["c1"]["value"] == "<=25ms"
    past = store.checkout_seq(1)
    assert past["c1"]["value"] == "<=50ms"


def test_missing_reference_raises_explicit_error() -> None:
    store = CMCPStateStore()
    with pytest.raises(ReferenceMissingError):
        store.resolve_reference("nonexistent_ref_999")


def test_concurrent_incompatible_agent_conflict_lww_and_reject() -> None:
    # LWW deterministic tie-break
    s_lww = CMCPStateStore(conflict_policy="lww")
    assert s_lww.apply(StateOp("d1", "UPSERT", "decision", "mode", "fast", seq=5, version=1, agent_id="agent_a")) == "APPLIED"
    assert s_lww.apply(StateOp("d1", "UPSERT", "decision", "mode", "safe", seq=5, version=1, agent_id="agent_b")) == "CONFLICT_RESOLVED"
    assert len(s_lww.conflicts) == 1
    assert s_lww.active_view()["d1"]["value"] == "safe"  # agent_b > agent_a deterministically

    # Reject policy
    s_rej = CMCPStateStore(conflict_policy="reject")
    assert s_rej.apply(StateOp("d1", "UPSERT", "decision", "mode", "fast", seq=5, version=1, agent_id="agent_a")) == "APPLIED"
    assert s_rej.apply(StateOp("d1", "UPSERT", "decision", "mode", "safe", seq=5, version=1, agent_id="agent_b")) == "STALE_IGNORED"
    assert s_rej.active_view()["d1"]["value"] == "fast"
