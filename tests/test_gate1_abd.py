"""Unit tests for Gate 1 Components A (Canonical + Evidence), B (Typed Relations), and D (Selector)."""

from __future__ import annotations

from representations.selector import select_for_task
from representations.state_store import CMCPStateStore, StateOp


def _build_sample_store() -> CMCPStateStore:
    st = CMCPStateStore()
    st.evidence.register("turn:1", "Node-10 is primary; standby Node-10B is backup only.")
    st.evidence.register("turn:2", "Temp token TK-99 was later revoked; active route is R-20.")
    st.apply(StateOp("e1", "UPSERT", "entity", "Node-10", "primary", seq=1, version=1, evidence_ref="turn:1"))
    st.apply(StateOp("e2", "UPSERT", "entity", "Node-10B", "standby_backup_only", seq=2, version=1, evidence_ref="turn:1"))
    st.apply(StateOp("c1", "UPSERT", "constraint", "sla", "SLA<=40ms", seq=3, version=1))
    st.apply(StateOp("d1", "UPSERT", "decision", "mode", "mode=strict(from:mode=standard)", seq=4, version=2))
    st.apply(StateOp("r1", "UPSERT", "relation", "path", "Node-10 -> Relay-2 -> Sink-3", seq=5, version=1))
    st.apply(StateOp("f1", "UPSERT", "fact", "temp", "TK-99", seq=6, version=1, evidence_ref="turn:2"))
    st.apply(StateOp("f1", "INVALIDATE", "fact", "temp", "", seq=7, version=2, evidence_ref="turn:2"))
    for idx in range(10):
        st.apply(StateOp(f"noise_{idx}", "UPSERT", "fact", f"telemetry_{idx}", f"VAL-{idx}", seq=10 + idx, version=1))
    return st


def test_component_a_canonical_and_evidence_store() -> None:
    st = _build_sample_store()
    r3c = st.render_for_llm(style="r3_canon")
    assert "(from:mode=standard)" not in r3c
    assert "mode=strict" in r3c
    assert st.evidence.resolve("turn:1") is not None


def test_component_b_typed_relations_inferred() -> None:
    st = _build_sample_store()
    rels = st.active_relations()
    types = {r.rel_type for r in rels}
    assert "ROUTES_TO" in types
    assert "CONSTRAINS" in types
    assert "CONTRADICTS" in types


def test_component_d_generic_selection_and_tombstone_and_evidence_fallback() -> None:
    st = _build_sample_store()
    # 1. Targeted query should drop noise_0..noise_9 and negated Node-10B
    sel = select_for_task("What is the active SLA limit for Node-10 (not Node-10B)?", st)
    assert "c1" in sel.selected_ids
    assert "e1" in sel.selected_ids
    assert "e2" not in sel.selected_ids
    assert not any(sid.startswith("noise_") for sid in sel.selected_ids)

    # 2. Query asking about invalidated token surfaces <invalidated>
    sel_inv = select_for_task("Is TK-99 valid?", st)
    assert "f1" in sel_inv.invalidated_ids
    assert "<invalidated>" in sel_inv.rendered_text

    # 3. Low-confidence entry triggers deterministic EvidenceStore fallback
    st.apply(StateOp("unc", "UPSERT", "fact", "uncertain_note", "maybe-X", seq=50, version=1, confidence=0.4, evidence_ref="turn:1"))
    sel_fb = select_for_task("Check uncertain_note", st)
    assert sel_fb.used_evidence_fallback is True
    assert "turn:1" in sel_fb.evidence_refs_used
    assert "<evidence>" in sel_fb.rendered_text
