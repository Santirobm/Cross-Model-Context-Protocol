"""Versioned, conflict-aware incremental context state store for CMCP 2.2."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


OpKind = Literal["UPSERT", "INVALIDATE"]
ApplyStatus = Literal["APPLIED", "DUPLICATE_IGNORED", "STALE_IGNORED", "CONFLICT_RESOLVED"]
RelationType = Literal[
    "DEPENDS_ON",
    "REPLACES",
    "CONSTRAINS",
    "CONTRADICTS",
    "PRECEDES",
    "EXCEPTION_TO",
    "PART_OF",
    "CONDITIONED_BY",
    "ROUTES_TO",
]


class ReferenceMissingError(KeyError):
    """Raised when a CMCP reference points to a missing or unpersisted state."""


@dataclass(frozen=True)
class StateOp:
    fact_id: str
    kind: OpKind
    category: str  # "entity" | "relation" | "constraint" | "decision" | "fact"
    key: str
    value: Any
    seq: int
    version: int
    agent_id: str = "agent_a"
    timestamp: str = ""
    parent_state_hash: str = ""
    source: str = "system"
    confidence: float = 1.0
    evidence_ref: Optional[str] = None

    def payload_digest(self) -> str:
        raw = json.dumps(
            {"fact_id": self.fact_id, "kind": self.kind, "category": self.category, "key": self.key, "value": self.value},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class FactEntry:
    fact_id: str
    category: str
    key: str
    value: Any
    seq: int
    version: int
    agent_id: str
    timestamp: str
    tombstone: bool
    digest: str
    source: str = "system"
    confidence: float = 1.0
    evidence_ref: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TypedRelation:
    src_id: str
    rel_type: RelationType
    dst_id: str
    seq: int
    active: bool = True


class EvidenceStore:
    """Off-wire store for recoverable source excerpts keyed by evidence_ref."""

    def __init__(self) -> None:
        self._items: Dict[str, str] = {}

    def register(self, ref_id: str, raw_excerpt: str) -> str:
        self._items[ref_id] = raw_excerpt
        return ref_id

    def resolve(self, ref_id: str) -> Optional[str]:
        return self._items.get(ref_id)

    def search_matching(self, query_terms: set[str]) -> Optional[Tuple[str, str]]:
        best: Optional[Tuple[int, str, str]] = None
        for ref_id, text in self._items.items():
            tl = text.lower()
            hits = sum(1 for t in query_terms if t in tl)
            if hits > 0 and (best is None or hits > best[0]):
                best = (hits, ref_id, text)
        return (best[1], best[2]) if best else None


def canonical_value(category: str, key: str, value: Any) -> str:
    """Normalize a state value by stripping inline parenthetical history clutter."""
    s = str(value).strip()
    # Strip inline '(from:...)' annotations; history lives in entry.history / checkout_seq()
    s = re.sub(r"\(from:[^)]+\)", "", s).strip()
    if category == "decision" and s.startswith(f"{key}="):
        s = s[len(key) + 1 :]
    return s


@dataclass
class ConflictRecord:
    fact_id: str
    seq: int
    existing_agent: str
    incoming_agent: str
    resolution: str


class CMCPStateStore:
    """Deterministic versioned store supporting deltas, tombstones, history, and compaction."""

    def __init__(self, conflict_policy: Literal["lww", "reject"] = "lww") -> None:
        self.conflict_policy = conflict_policy
        self._entries: Dict[str, FactEntry] = {}
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._op_log: List[StateOp] = []
        self._conflicts: List[ConflictRecord] = []
        self._relations: List[TypedRelation] = []
        self.evidence = EvidenceStore()
        self.compaction_count: int = 0

    def add_relation(self, src_id: str, rel_type: RelationType, dst_id: str, seq: int) -> None:
        edge = TypedRelation(src_id=src_id, rel_type=rel_type, dst_id=dst_id, seq=seq, active=True)
        if edge not in self._relations:
            self._relations.append(edge)

    def active_relations(self) -> List[TypedRelation]:
        return [r for r in self._relations if r.active]

    def relations_for(self, fact_id: str) -> List[TypedRelation]:
        return [r for r in self._relations if r.active and (r.src_id == fact_id or r.dst_id == fact_id)]

    def _infer_relations_for_op(self, op: StateOp, replaced_prev: bool) -> None:
        if replaced_prev:
            rel_t: RelationType = "CONTRADICTS" if op.kind == "INVALIDATE" else "REPLACES"
            self.add_relation(op.fact_id, rel_t, op.fact_id, op.seq)
        if op.kind == "INVALIDATE":
            return
        v_str = str(op.value)
        if op.category == "relation" and "->" in v_str:
            for fid, ent in self._entries.items():
                if not ent.tombstone and ent.category == "entity" and ent.key in v_str:
                    self.add_relation(op.fact_id, "ROUTES_TO", fid, op.seq)
        elif op.category in ("constraint", "decision"):
            for fid, ent in self._entries.items():
                if not ent.tombstone and ent.category == "entity":
                    if ent.key in v_str or "standby" not in str(ent.value).lower():
                        rel_kind: RelationType = "CONSTRAINS" if op.category == "constraint" else "CONDITIONED_BY"
                        self.add_relation(op.fact_id, rel_kind, fid, op.seq)

    def state_hash(self) -> str:
        active = {
            fid: {
                "cat": e.category,
                "key": e.key,
                "val": e.value,
                "seq": e.seq,
                "ver": e.version,
                "tomb": e.tombstone,
            }
            for fid, e in sorted(self._entries.items())
        }
        return hashlib.sha256(
            json.dumps(active, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    def apply(self, op: StateOp) -> ApplyStatus:
        digest = op.payload_digest()
        cur = self._entries.get(op.fact_id)

        if cur is not None:
            # 1. Duplicate check
            if cur.seq == op.seq and cur.version == op.version and cur.digest == digest:
                return "DUPLICATE_IGNORED"

            # 2. Out-of-order / stale check
            if op.seq < cur.seq or (op.seq == cur.seq and op.version < cur.version):
                return "STALE_IGNORED"

            # 3. Concurrent conflict: same seq & version, different payload/agent
            if op.seq == cur.seq and op.version == cur.version and cur.digest != digest:
                if self.conflict_policy == "reject":
                    self._conflicts.append(
                        ConflictRecord(op.fact_id, op.seq, cur.agent_id, op.agent_id, "rejected_incoming")
                    )
                    return "STALE_IGNORED"
                # Deterministic tie-break by lexicographical agent_id + digest
                winner_incoming = (op.agent_id, digest) > (cur.agent_id, cur.digest)
                res_msg = "accepted_incoming_lww" if winner_incoming else "kept_existing_lww"
                self._conflicts.append(
                    ConflictRecord(op.fact_id, op.seq, cur.agent_id, op.agent_id, res_msg)
                )
                if not winner_incoming:
                    return "CONFLICT_RESOLVED"
                status: ApplyStatus = "CONFLICT_RESOLVED"
            else:
                status = "APPLIED"

            cur.history.append({
                "seq": cur.seq,
                "version": cur.version,
                "value": cur.value,
                "tombstone": cur.tombstone,
                "agent_id": cur.agent_id,
            })
            cur.category = op.category
            cur.key = op.key
            cur.value = op.value
            cur.seq = op.seq
            cur.version = op.version
            cur.agent_id = op.agent_id
            cur.timestamp = op.timestamp
            cur.tombstone = op.kind == "INVALIDATE"
            cur.digest = digest
            cur.source = op.source
            cur.confidence = op.confidence
            if op.evidence_ref is not None:
                cur.evidence_ref = op.evidence_ref
            self._op_log.append(op)
            self._infer_relations_for_op(op, replaced_prev=True)
            return status

        self._entries[op.fact_id] = FactEntry(
            fact_id=op.fact_id,
            category=op.category,
            key=op.key,
            value=op.value,
            seq=op.seq,
            version=op.version,
            agent_id=op.agent_id,
            timestamp=op.timestamp,
            tombstone=(op.kind == "INVALIDATE"),
            digest=digest,
            source=op.source,
            confidence=op.confidence,
            evidence_ref=op.evidence_ref,
        )
        self._op_log.append(op)
        self._infer_relations_for_op(op, replaced_prev=False)
        return "APPLIED"

    def save_snapshot(self, ref_id: str) -> str:
        snap = {
            fid: {
                "fact_id": e.fact_id,
                "category": e.category,
                "key": e.key,
                "value": e.value,
                "seq": e.seq,
                "version": e.version,
                "tombstone": e.tombstone,
            }
            for fid, e in sorted(self._entries.items())
        }
        self._snapshots[ref_id] = snap
        return ref_id

    def resolve_reference(self, ref_id: str) -> Dict[str, Dict[str, Any]]:
        if ref_id not in self._snapshots:
            raise ReferenceMissingError(f"Reference {ref_id!r} does not exist in store")
        return self._snapshots[ref_id]

    def checkout_seq(self, target_seq: int) -> Dict[str, Dict[str, Any]]:
        """Reconstruct active state as of target_seq from the operation log."""
        temp = CMCPStateStore(conflict_policy=self.conflict_policy)
        for op in self._op_log:
            if op.seq <= target_seq:
                temp.apply(op)
        return temp.active_view()

    def active_view(self) -> Dict[str, Dict[str, Any]]:
        return {
            fid: {
                "category": e.category,
                "key": e.key,
                "value": e.value,
                "seq": e.seq,
                "version": e.version,
            }
            for fid, e in sorted(self._entries.items())
            if not e.tombstone
        }

    def render_for_llm(self, style: str = "r1") -> str:
        """Render active state preserving entities, ternary relations, constraints, and decisions."""
        active = self.active_view()
        if style == "r2":
            ents, rels, cons, decs, facts = [], [], [], [], []
            for _, item in active.items():
                cat, k, v = item["category"], item["key"], item["value"]
                if cat == "entity":
                    ents.append(f"{k} ({v})")
                elif cat == "relation":
                    rels.append(v)
                elif cat == "constraint":
                    cons.append(v)
                elif cat == "decision":
                    decs.append(v if v.startswith(f"{k}=") else f"{k}={v}")
                else:
                    facts.append(f"{k} is {v}")
            parts: List[str] = []
            if rels:
                parts.append("Active path: " + "; ".join(rels) + ".")
            if ents:
                parts.append("Entities: " + "; ".join(ents) + ".")
            if cons:
                parts.append("Active constraints: " + "; ".join(cons) + ".")
            if decs:
                parts.append("Active decisions: " + "; ".join(decs) + ".")
            if facts:
                parts.append("Current facts: " + "; ".join(facts) + ".")
            return "\n".join(parts)

        if style == "r3_canon":
            buckets_c: Dict[str, List[str]] = {
                "relations": [],
                "entities": [],
                "constraints": [],
                "decisions": [],
                "facts": [],
            }
            for _, item in active.items():
                cat, k = item["category"], item["key"]
                v = canonical_value(cat, k, item["value"])
                if cat == "relation":
                    buckets_c["relations"].append(f"- {v}")
                elif cat == "entity":
                    buckets_c["entities"].append(f"- {k}: {v}")
                elif cat == "constraint":
                    buckets_c["constraints"].append(f"- {v}")
                elif cat == "decision":
                    buckets_c["decisions"].append(f"- {k}={v}")
                else:
                    buckets_c["facts"].append(f"- {k}: {v}")
            sections_c: List[str] = []
            for tag in ("relations", "entities", "constraints", "decisions", "facts"):
                if buckets_c[tag]:
                    sections_c.append(f"<{tag}>\n" + "\n".join(buckets_c[tag]) + f"\n</{tag}>")
            return "\n".join(sections_c)

        if style == "r3":
            buckets: Dict[str, List[str]] = {
                "relations": [],
                "entities": [],
                "constraints": [],
                "decisions": [],
                "facts": [],
            }
            for _, item in active.items():
                cat, k, v = item["category"], item["key"], item["value"]
                if cat == "relation":
                    buckets["relations"].append(f"- {v}")
                elif cat == "entity":
                    buckets["entities"].append(f"- {k}: {v}")
                elif cat == "constraint":
                    buckets["constraints"].append(f"- {v}")
                elif cat == "decision":
                    buckets["decisions"].append(f"- {v if v.startswith(f'{k}=') else f'{k}={v}'}")
                else:
                    buckets["facts"].append(f"- {k}: {v}")
            sections: List[str] = []
            for tag in ("relations", "entities", "constraints", "decisions", "facts"):
                if buckets[tag]:
                    sections.append(f"<{tag}>\n" + "\n".join(buckets[tag]) + f"\n</{tag}>")
            return "\n".join(sections)

        lines: List[str] = []
        for _, item in active.items():
            cat = item["category"]
            k = item["key"]
            v = item["value"]
            if cat == "relation":
                lines.append(f"REL: {v}")
            elif cat == "constraint":
                lines.append(f"CONSTRAINT: {v}")
            elif cat == "decision":
                if v.startswith(f"{k}="):
                    lines.append(f"DECISION: {v}")
                else:
                    lines.append(f"DECISION: {k} = {v}")
            elif cat == "entity":
                lines.append(f"ENTITY {k}: {v}")
            else:
                lines.append(f"FACT {k}: {v}")
        return "\n".join(lines)

    def serialize(
        self,
        query: Optional[str] = None,
        mode: Literal["compact", "anchored"] = "compact",
        *,
        expand_hops: int = 0,
    ) -> str:
        """Serialize this store in 'compact' (task-selected) or 'anchored' (full canonical) mode."""
        from .adapters import serialize as _serialize

        return _serialize(self, query=query, mode=mode, expand_hops=expand_hops)

    def compact(self, prune_tombstones_before_seq: Optional[int] = None) -> int:
        """Trim per-entry history and optionally prune old tombstones; returns bytes freed estimate."""
        removed = 0
        for e in list(self._entries.values()):
            removed += len(e.history)
            e.history.clear()
        if prune_tombstones_before_seq is not None:
            to_del = [
                fid for fid, e in self._entries.items()
                if e.tombstone and e.seq < prune_tombstones_before_seq
            ]
            for fid in to_del:
                del self._entries[fid]
                removed += 1
        self.compaction_count += 1
        return removed

    def serialized_size_bytes(self) -> int:
        return len(json.dumps(self.active_view(), separators=(",", ":"), sort_keys=True).encode("utf-8"))

    @property
    def conflicts(self) -> List[ConflictRecord]:
        return list(self._conflicts)
