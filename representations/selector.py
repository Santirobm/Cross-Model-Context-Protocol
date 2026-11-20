"""Generic, task-conditioned context selector for CMCP (Component D).

Operates strictly over CMCPStateStore entries, TypedRelation edges, and EvidenceStore
using domain-agnostic lexical, category, negation, tombstone, and graph-closure rules.
Zero benchmark-specific entity names, keys, or query-ID branching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .state_store import CMCPStateStore, FactEntry, canonical_value

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "composite", "current",
    "distinguishing", "for", "from", "full", "have", "in", "is", "it", "its",
    "list", "of", "on", "or", "recover", "report", "state", "task", "the", "this",
    "to", "valid", "what", "when", "which", "with",
}

_CATEGORY_CUES: Dict[str, Set[str]] = {
    "constraint": {"limit", "budget", "sla", "cap", "max", "min", "window", "threshold", "constraint", "rate"},
    "decision": {"decision", "mode", "policy", "tier", "chosen", "selected", "strategy"},
    "relation": {"path", "flow", "hop", "entity", "link", "connect", "depends"},
}

_NEGATION_RE = re.compile(r"\b(?:not|ignoring|ignore|except|without|distinguishing\s+from)\s+([A-Za-z0-9_.-]+)", re.I)


def _tokenize(text: str) -> Set[str]:
    raw = re.findall(r"[A-Za-z0-9_-]+", text.lower())
    return {t.strip(".-_") for t in raw if len(t.strip(".-_")) >= 2 and t.strip(".-_") not in _STOPWORDS}


@dataclass
class SelectedContext:
    rendered_text: str
    selected_ids: List[str] = field(default_factory=list)
    invalidated_ids: List[str] = field(default_factory=list)
    used_evidence_fallback: bool = False
    evidence_refs_used: List[str] = field(default_factory=list)


def select_for_task(
    query: str,
    store: CMCPStateStore,
    *,
    use_relations: bool = True,
    use_evidence_fallback: bool = True,
    confidence_threshold: float = 0.7,
    canonicalize: bool = True,
    expand_hops: int = 0,
    max_expand_items: int = 4,
    max_expand_tokens: int = 48,
) -> SelectedContext:
    """Select a minimal task-relevant subset of CMCPStateStore without LLM calls."""
    q_terms = _tokenize(query)
    negated_terms = {m.group(1).lower() for m in _NEGATION_RE.finditer(query)}
    wanted_cats = {cat for cat, cues in _CATEGORY_CUES.items() if q_terms & cues}
    if "->" in query:
        wanted_cats.add("relation")

    selected_ids: Set[str] = set()
    invalidated_lines: List[str] = []
    invalidated_ids: List[str] = []

    for fid, entry in store._entries.items():
        entry_terms = _tokenize(f"{entry.key} {entry.value}")
        key_l = entry.key.lower()

        if entry.tombstone:
            hist_terms: Set[str] = set()
            last_val = ""
            for h in entry.history:
                hv = str(h.get("value", ""))
                if hv:
                    last_val = hv
                hist_terms |= _tokenize(hv)
            if q_terms & (entry_terms | hist_terms):
                invalidated_ids.append(fid)
                label = last_val if last_val else entry.key
                invalidated_lines.append(f"- {label} (INVALIDATED)")
            continue

        # Skip explicitly negated distractor entities unless directly asked as primary target
        if key_l in negated_terms or any(nt == key_l for nt in negated_terms):
            continue

        direct_overlap = len(q_terms & entry_terms)
        cat_match = entry.category in wanted_cats
        # Require either direct lexical overlap or matching a requested semantic category
        if direct_overlap > 0 or cat_match:
            # Avoid pulling unrelated standby entities on broad category match
            if direct_overlap == 0 and "standby" in str(entry.value).lower():
                continue
            selected_ids.add(fid)

    # Also match partial underscore/hyphen sub-tokens (e.g. "audit" in "audit_tag", "route" in "route_code")
    if not selected_ids or wanted_cats or q_terms:
        q_subtokens = {sub for t in q_terms for sub in re.split(r"[_-]+", t) if len(sub) >= 3 and sub not in _STOPWORDS}
        for fid, entry in store._entries.items():
            if entry.tombstone or fid in selected_ids:
                continue
            if entry.key.lower() in negated_terms:
                continue
            if "standby" in str(entry.value).lower() and entry.key.lower() not in q_terms:
                continue
            e_subtokens = {sub for t in _tokenize(f"{entry.key} {entry.value}") for sub in re.split(r"[_-]+", t) if len(sub) >= 3}
            if q_subtokens & e_subtokens:
                selected_ids.add(fid)

    # Component B: 1-hop typed relation closure over active edges
    if use_relations and selected_ids:
        frontier = sorted(selected_ids)
        for fid in frontier:
            for rel in store.relations_for(fid):
                if rel.rel_type in ("ROUTES_TO", "CONSTRAINS", "DEPENDS_ON", "CONDITIONED_BY", "PART_OF", "EXCEPTION_TO"):
                    other = rel.dst_id if rel.src_id == fid else rel.src_id
                    ent = store._entries.get(other)
                    if ent is not None and not ent.tombstone:
                        if ent.key.lower() not in negated_terms and "standby" not in str(ent.value).lower():
                            if ent.category in wanted_cats or (ent.category == "entity" and "relation" in wanted_cats):
                                selected_ids.add(other)

        # Optional deterministic N-hop neighborhood expansion (budget-capped to preserve O(1) scaling)
        if expand_hops > 0 and max_expand_items > 0 and max_expand_tokens > 0:
            added_items = 0
            added_tokens = 0
            hop_frontier = sorted(selected_ids)
            for _ in range(expand_hops):
                next_frontier: List[str] = []
                for fid in hop_frontier:
                    rels = sorted(store.relations_for(fid), key=lambda r: (r.seq, r.rel_type, r.src_id, r.dst_id))
                    for rel in rels:
                        if rel.rel_type not in (
                            "ROUTES_TO",
                            "CONSTRAINS",
                            "DEPENDS_ON",
                            "CONDITIONED_BY",
                            "PART_OF",
                            "EXCEPTION_TO",
                        ):
                            continue
                        other = rel.dst_id if rel.src_id == fid else rel.src_id
                        if other in selected_ids:
                            continue
                        ent = store._entries.get(other)
                        if ent is None or ent.tombstone:
                            continue
                        if ent.key.lower() in negated_terms or "standby" in str(ent.value).lower():
                            continue
                        est_toks = max(2, len(re.findall(r"[A-Za-z0-9_=-]+", f"{ent.key} {ent.value}")) + 2)
                        if added_items + 1 > max_expand_items or added_tokens + est_toks > max_expand_tokens:
                            break
                        selected_ids.add(other)
                        next_frontier.append(other)
                        added_items += 1
                        added_tokens += est_toks
                    if added_items >= max_expand_items or added_tokens >= max_expand_tokens:
                        break
                if not next_frontier or added_items >= max_expand_items or added_tokens >= max_expand_tokens:
                    break
                hop_frontier = sorted(next_frontier)

    # Component A: Deterministic EvidenceStore fallback rule
    evidence_lines: List[str] = []
    evidence_refs_used: List[str] = []
    used_fallback = False
    if use_evidence_fallback:
        for fid in sorted(selected_ids):
            entry = store._entries[fid]
            if entry.confidence < confidence_threshold and entry.evidence_ref:
                excerpt = store.evidence.resolve(entry.evidence_ref)
                if excerpt:
                    evidence_lines.append(f"- {excerpt}")
                    evidence_refs_used.append(entry.evidence_ref)
                    used_fallback = True
        if not selected_ids and not invalidated_ids:
            hit = store.evidence.search_matching(q_terms)
            if hit is not None:
                ref_id, excerpt = hit
                evidence_lines.append(f"- {excerpt}")
                evidence_refs_used.append(ref_id)
                used_fallback = True

    buckets: Dict[str, List[str]] = {
        "relations": [],
        "entities": [],
        "constraints": [],
        "decisions": [],
        "facts": [],
    }
    for fid in sorted(selected_ids):
        e: FactEntry = store._entries[fid]
        val = canonical_value(e.category, e.key, e.value) if canonicalize else str(e.value)
        if e.category == "relation":
            buckets["relations"].append(f"- {val}")
        elif e.category == "entity":
            buckets["entities"].append(f"- {e.key}: {val}")
        elif e.category == "constraint":
            buckets["constraints"].append(f"- {val}")
        elif e.category == "decision":
            buckets["decisions"].append(f"- {e.key}={val}")
        else:
            buckets["facts"].append(f"- {e.key}: {val}")

    sections: List[str] = []
    for tag in ("relations", "entities", "constraints", "decisions", "facts"):
        if buckets[tag]:
            sections.append(f"<{tag}>\n" + "\n".join(buckets[tag]) + f"\n</{tag}>")
    if invalidated_lines:
        sections.append("<invalidated>\n" + "\n".join(invalidated_lines) + "\n</invalidated>")
    if evidence_lines:
        sections.append("<evidence>\n" + "\n".join(evidence_lines) + "\n</evidence>")

    return SelectedContext(
        rendered_text="\n".join(sections),
        selected_ids=sorted(selected_ids),
        invalidated_ids=sorted(invalidated_ids),
        used_evidence_fallback=used_fallback,
        evidence_refs_used=evidence_refs_used,
    )
