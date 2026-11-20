"""GATE 2 Presentation Renderers (Unified vs. Receiver-Adapted) over SelectedContext.

Strictly domain-agnostic: operates only on generic CMCP categories
(entities, facts, constraints, decisions, relations, invalidated, evidence)
without any benchmark-specific keys or query-turn branching.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Set

from .selector import SelectedContext, select_for_task
from .state_store import CMCPStateStore, FactEntry, canonical_value


def _dedup_entity_value(raw_val: str, active_fact_values: Set[str]) -> str:
    """Strip '; attr=val' clauses from entity values when 'val' already exists in selected facts."""
    parts = [p.strip() for p in raw_val.split(";")]
    kept: List[str] = []
    for p in parts:
        if "=" in p:
            _, v = p.split("=", 1)
            if v.strip().lower() in active_fact_values:
                continue
        if p:
            kept.append(p)
    return "; ".join(kept) if kept else raw_val


def _collect_items(sel: SelectedContext, store: CMCPStateStore, *, dedup: bool = True) -> Dict[str, List[tuple[str, str]]]:
    groups: Dict[str, List[tuple[str, str]]] = {
        "entity": [],
        "fact": [],
        "constraint": [],
        "decision": [],
        "relation": [],
    }
    selected_entries: List[FactEntry] = [store._entries[fid] for fid in sel.selected_ids if fid in store._entries]
    active_fact_vals = {
        canonical_value(e.category, e.key, e.value).lower()
        for e in selected_entries
        if e.category == "fact"
    }
    for e in selected_entries:
        val = canonical_value(e.category, e.key, e.value)
        if e.category == "entity" and dedup:
            val = _dedup_entity_value(val, active_fact_vals)
        groups.setdefault(e.category, []).append((e.key, val))
    return groups


def _append_tail(sections: List[str], sel: SelectedContext, store: CMCPStateStore) -> None:
    inv_lines: List[str] = []
    for fid in sel.invalidated_ids:
        ent = store._entries.get(fid)
        if ent is not None:
            last_v = ""
            for h in ent.history:
                if h.get("value"):
                    last_v = str(h["value"])
            inv_lines.append(f"- {last_v or ent.key} (INVALIDATED)")
    if inv_lines:
        sections.append("<invalidated>\n" + "\n".join(inv_lines) + "\n</invalidated>")

    ev_lines: List[str] = []
    for ref_id in sel.evidence_refs_used:
        ex = store.evidence.resolve(ref_id)
        if ex:
            ev_lines.append(f"- {ex}")
    if ev_lines:
        sections.append("<evidence>\n" + "\n".join(ev_lines) + "\n</evidence>")


def render_selected_ablation(
    sel: SelectedContext,
    store: CMCPStateStore,
    *,
    dedup: bool,
    unified_order: bool,
) -> str:
    """2x2 factorial ablation renderer (benchmark-only): dedup x section order."""
    groups = _collect_items(sel, store, dedup=dedup)
    sections: List[str] = []
    if groups["relation"]:
        sections.append("<relations>\n" + "\n".join(f"- {v}" for _, v in groups["relation"]) + "\n</relations>")
    if groups["entity"]:
        sections.append("<entities>\n" + "\n".join(f"- {k}: {v}" for k, v in groups["entity"]) + "\n</entities>")
    if unified_order:
        if groups["fact"]:
            sections.append("<facts>\n" + "\n".join(f"- {k} = {v}" for k, v in groups["fact"]) + "\n</facts>")
        if groups["constraint"]:
            sections.append("<constraints>\n" + "\n".join(f"- {v}" for _, v in groups["constraint"]) + "\n</constraints>")
        if groups["decision"]:
            sections.append("<decisions>\n" + "\n".join(f"- {k}={v}" for k, v in groups["decision"]) + "\n</decisions>")
    else:
        if groups["constraint"]:
            sections.append("<constraints>\n" + "\n".join(f"- {v}" for _, v in groups["constraint"]) + "\n</constraints>")
        if groups["decision"]:
            sections.append("<decisions>\n" + "\n".join(f"- {k}={v}" for k, v in groups["decision"]) + "\n</decisions>")
        if groups["fact"]:
            sections.append("<facts>\n" + "\n".join(f"- {k} = {v}" for k, v in groups["fact"]) + "\n</facts>")
    _append_tail(sections, sel, store)
    return "\n".join(sections)


def serialize(
    state: CMCPStateStore,
    query: Optional[str] = None,
    mode: Literal["compact", "anchored"] = "compact",
    *,
    expand_hops: int = 0,
) -> str:
    """Serialize CMCPStateStore using either 'compact' (task-selected) or 'anchored' (full canonical) mode."""
    if mode == "compact":
        if query is None or not isinstance(query, str) or not query.strip():
            raise ValueError("`query` must be a non-empty string when mode='compact'.")
        selected = select_for_task(query, state, expand_hops=expand_hops)
        return render_selected_unified(selected, state)
    if mode == "anchored":
        return state.render_for_llm(style="r3_canon")
    raise ValueError(f"Unknown serialization mode {mode!r}; expected 'compact' or 'anchored'.")


def render_selected_unified(sel: SelectedContext, store: CMCPStateStore) -> str:
    """Unified XML-lite renderer: deduplicated entities, value-safe facts, constraints after facts."""
    return render_selected_ablation(sel, store, dedup=True, unified_order=True)


def render_selected_adapted(sel: SelectedContext, store: CMCPStateStore, receiver_family: str) -> str:
    """Receiver-specific XML-lite adapter over the exact same SelectedContext."""
    fam = receiver_family.lower()
    groups = _collect_items(sel, store)
    sections: List[str] = []

    if "llama" in fam:
        # Llama-3.2-1B: preserve canonical XML-lite order (<relations><entities><facts><constraints><decisions>)
        # without extra "Constraint:"/"Decision:" prefixes that trigger echo loops on Llama-3.2-1B.
        if groups["relation"]:
            sections.append("<relations>\n" + "\n".join(f"- {v}" for _, v in groups["relation"]) + "\n</relations>")
        if groups["entity"]:
            sections.append("<entities>\n" + "\n".join(f"- {k}: {v}" for k, v in groups["entity"]) + "\n</entities>")
        if groups["fact"]:
            sections.append("<facts>\n" + "\n".join(f"- {k}: {v}" for k, v in groups["fact"]) + "\n</facts>")
        if groups["constraint"]:
            sections.append("<constraints>\n" + "\n".join(f"- {v}" for _, v in groups["constraint"]) + "\n</constraints>")
        if groups["decision"]:
            sections.append("<decisions>\n" + "\n".join(f"- {k}={v}" for k, v in groups["decision"]) + "\n</decisions>")
    elif "qwen" in fam:
        # Qwen2.5-0.5B: put value before key in facts to prevent key-only truncation
        if groups["relation"]:
            sections.append("<relations>\n" + "\n".join(f"- {v}" for _, v in groups["relation"]) + "\n</relations>")
        if groups["entity"]:
            sections.append("<entities>\n" + "\n".join(f"- {k}: {v}" for k, v in groups["entity"]) + "\n</entities>")
        if groups["fact"]:
            sections.append("<facts>\n" + "\n".join(f"- {v} ({k})" for k, v in groups["fact"]) + "\n</facts>")
        if groups["constraint"]:
            sections.append("<constraints>\n" + "\n".join(f"- {v}" for _, v in groups["constraint"]) + "\n</constraints>")
        if groups["decision"]:
            sections.append("<decisions>\n" + "\n".join(f"- {k}={v}" for k, v in groups["decision"]) + "\n</decisions>")
    else:
        return render_selected_unified(sel, store)

    _append_tail(sections, sel, store)
    return "\n".join(sections)


def serialize(
    state: CMCPStateStore,
    query: Optional[str] = None,
    mode: Literal["compact", "anchored"] = "compact",
    *,
    expand_hops: int = 0,
) -> str:
    """Serialize CMCPStateStore using either 'compact' (task-selected) or 'anchored' (full canonical) mode."""
    if mode == "compact":
        if query is None or not isinstance(query, str) or not query.strip():
            raise ValueError("`query` must be a non-empty string when mode='compact'.")
        selected = select_for_task(query, state, expand_hops=expand_hops)
        return render_selected_unified(selected, state)
    if mode == "anchored":
        return state.render_for_llm(style="r3_canon")
    raise ValueError(f"Unknown serialization mode {mode!r}; expected 'compact' or 'anchored'.")
