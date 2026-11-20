"""Deterministic structured ingestion hooks for CMCP (zero LLM calls).

Provides two 100% deterministic, auditable ingestion paths into CMCPStateStore:
1. `ingest_tool_output`: declarative schema/key mapping for JSON/dict tool outputs
   with idempotent deduplication and monotonic state updates/invalidations.
2. `extract_explicit_constraints`: auditable rule/pattern extractor for explicit
   operational limits, prohibitions, and mandatory versions/formats.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from representations.state_store import CMCPStateStore, StateOp

FactCategory = Literal["entity", "fact", "constraint", "decision", "relation"]

_CONSTRAINT_KEY_RE = re.compile(r"(?:^|[._-])(limit|budget|sla|cap|max|min|constraint|timeout|rate|threshold)(?:$|[._-])", re.I)
_DECISION_KEY_RE = re.compile(r"(?:^|[._-])(mode|policy|tier|decision|strategy)(?:$|[._-])", re.I)
_ENTITY_KEY_RE = re.compile(r"(?:^|[._-])(entity|node|service|host|unit|primary|gateway)(?:$|[._-])", re.I)
_RELATION_KEY_RE = re.compile(r"(?:^|[._-])(path|route_graph|relation|topology)(?:$|[._-])", re.I)

_CONSTRAINT_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("prohibition", re.compile(r"\b(must\s+not|never|do\s+not|don't|prohibited\s+(?:from|to)|forbidden\s+to)\s+([^.;\n]{3,80})", re.I)),
    ("no_allowed", re.compile(r"\bno\s+([A-Za-z0-9_ -]{2,40}?)\s+allowed\b", re.I)),
    ("cmp_limit", re.compile(r"\b([A-Za-z][A-Za-z0-9_-]{1,31})\s*(<=|>=|<|>)\s*([0-9]+(?:\.[0-9]+)?(?:\s*(?:ms|s|sec|mb|gb|kb|tokens|req/s|%))?)\b")),
    ("named_limit", re.compile(r"\b(max(?:imum)?|min(?:imum)?|limit|budget|cap|sla|timeout)\s*(?:of|:|=|is)?\s*(<=|>=|<|>)?\s*([0-9]+(?:\.[0-9]+)?(?:\s*(?:ms|s|sec|mb|gb|kb|tokens|req/s|%))?)\b", re.I)),
    ("mandatory", re.compile(r"\b(?:must\s+use|required\s+(?:version|format)|strictly\s+(?:use|require)|requires)\s+([^.;\n]{2,60})", re.I)),
)


def _normalize_key(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip()).strip("_.")
    return cleaned.lower() or "item"


def _next_store_seq(store: Optional[CMCPStateStore], seq_base: Optional[int]) -> int:
    max_existing = max((e.seq for e in store._entries.values()), default=0) if store is not None else 0
    return max(seq_base or 0, max_existing) + 1


def _flatten_payload(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if isinstance(obj, Mapping):
        for k in sorted(obj.keys(), key=str):
            sub_key = f"{prefix}.{k}" if prefix else str(k)
            items.extend(_flatten_payload(obj[k], sub_key))
    elif isinstance(obj, list):
        if all(not isinstance(x, (Mapping, list)) for x in obj):
            items.append((prefix, ", ".join(str(x) for x in obj)))
        else:
            for idx, val in enumerate(obj):
                items.extend(_flatten_payload(val, f"{prefix}[{idx}]"))
    else:
        items.append((prefix, obj))
    return items


def _infer_category(full_key: str, leaf_key: str, value: Any, category_map: Optional[Mapping[str, FactCategory]]) -> FactCategory:
    if category_map:
        if full_key in category_map:
            return category_map[full_key]
        if leaf_key in category_map:
            return category_map[leaf_key]
    if isinstance(value, str) and "->" in value:
        return "relation"
    if _RELATION_KEY_RE.search(full_key):
        return "relation"
    if _CONSTRAINT_KEY_RE.search(full_key):
        return "constraint"
    if _DECISION_KEY_RE.search(full_key):
        return "decision"
    if _ENTITY_KEY_RE.search(full_key):
        return "entity"
    return "fact"


def ingest_tool_output(
    store: CMCPStateStore,
    payload: Mapping[str, Any] | str,
    *,
    namespace: str = "tool",
    source_id: str = "tool",
    seq_base: Optional[int] = None,
    evidence_ref: Optional[str] = None,
    category_map: Optional[Mapping[str, FactCategory]] = None,
) -> List[StateOp]:
    """Deterministically ingest a JSON/dict tool output into CMCPStateStore.

    Returns the list of StateOps actually applied (empty when re-ingesting an
    identical payload, guaranteeing idempotency).
    """
    data: Mapping[str, Any]
    if isinstance(payload, str):
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise ValueError("JSON tool output payload must decode to an object/mapping.")
        data = parsed
    elif isinstance(payload, Mapping):
        data = payload
    else:
        raise TypeError("payload must be a Mapping or JSON object string.")

    if evidence_ref and store.evidence.resolve(evidence_ref) is None:
        store.evidence.register(evidence_ref, json.dumps(dict(data), sort_keys=True, ensure_ascii=False))

    seq = _next_store_seq(store, seq_base)
    applied_ops: List[StateOp] = []

    for full_key, raw_val in _flatten_payload(data):
        if not full_key:
            continue
        leaf_key = full_key.split(".")[-1]
        norm_fid = f"{namespace}:{_normalize_key(full_key)}"
        cat = _infer_category(full_key, leaf_key, raw_val, category_map)
        existing = store._entries.get(norm_fid)

        is_invalidation = raw_val is None or (isinstance(raw_val, str) and raw_val.strip().upper() in ("REVOKED", "INVALIDATED"))
        if is_invalidation:
            if existing is None or existing.tombstone:
                continue
            op = StateOp(
                fact_id=norm_fid,
                kind="INVALIDATE",
                category=existing.category,
                key=existing.key,
                value="",
                seq=seq,
                version=existing.version + 1,
                agent_id=source_id,
                evidence_ref=evidence_ref,
                confidence=1.0,
            )
            if store.apply(op) == "APPLIED":
                applied_ops.append(op)
                seq += 1
            continue

        val_str = str(raw_val).strip()
        if existing is not None and not existing.tombstone and str(existing.value) == val_str and existing.category == cat:
            # Idempotent: identical active state already stored
            continue

        next_ver = (existing.version + 1) if existing is not None else 1
        op = StateOp(
            fact_id=norm_fid,
            kind="UPSERT",
            category=cat,
            key=leaf_key,
            value=val_str,
            seq=seq,
            version=next_ver,
            agent_id=source_id,
            evidence_ref=evidence_ref,
            confidence=1.0,
        )
        if store.apply(op) == "APPLIED":
            applied_ops.append(op)
            seq += 1

    return applied_ops


def extract_explicit_constraints(
    text: str,
    *,
    store: Optional[CMCPStateStore] = None,
    source_id: str = "rule_extractor",
    seq_base: Optional[int] = None,
    evidence_ref: Optional[str] = None,
) -> List[StateOp]:
    """Deterministically extract explicit operational/negative constraints from text.

    Zero LLM calls, zero false positives on neutral descriptive prose.
    When `store` is provided, applies non-duplicate updates to `store` and returns
    the applied ops; otherwise returns the candidate ops.
    """
    if not text or not text.strip():
        return []

    if store is not None and evidence_ref and store.evidence.resolve(evidence_ref) is None:
        store.evidence.register(evidence_ref, text.strip())

    candidates: List[Tuple[str, str]] = []
    seen_keys: set[str] = set()

    for kind, pattern in _CONSTRAINT_PATTERNS:
        for m in pattern.finditer(text):
            if kind == "prohibition":
                phrase = f"{m.group(1).strip().lower()} {m.group(2).strip()}"
                key = _normalize_key(f"prohibit_{m.group(2)[:32]}")
            elif kind == "no_allowed":
                phrase = f"no {m.group(1).strip()} allowed"
                key = _normalize_key(f"no_{m.group(1)}")
            elif kind == "cmp_limit":
                lhs = m.group(1).strip()
                op_sym = m.group(2).strip()
                rhs = re.sub(r"\s+", "", m.group(3).strip())
                phrase = f"{lhs}{op_sym}{rhs}"
                key = _normalize_key(lhs)
            elif kind == "named_limit":
                lhs = m.group(1).strip().lower()
                op_sym = (m.group(2) or "<=").strip()
                rhs = re.sub(r"\s+", "", m.group(3).strip())
                phrase = f"{lhs}{op_sym}{rhs}"
                key = _normalize_key(lhs)
            else:  # mandatory
                target = m.group(1).strip()
                phrase = f"require {target}"
                key = _normalize_key(f"require_{target[:32]}")

            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append((key, phrase))

    seq = _next_store_seq(store, seq_base)
    ops: List[StateOp] = []
    for key, phrase in candidates:
        fid = f"con:{key}"
        if store is not None:
            existing = store._entries.get(fid)
            if existing is not None and not existing.tombstone and str(existing.value) == phrase:
                continue
            ver = (existing.version + 1) if existing is not None else 1
        else:
            ver = 1

        op = StateOp(
            fact_id=fid,
            kind="UPSERT",
            category="constraint",
            key=key,
            value=phrase,
            seq=seq,
            version=ver,
            agent_id=source_id,
            evidence_ref=evidence_ref,
            confidence=0.95,
        )
        if store is not None:
            if store.apply(op) == "APPLIED":
                ops.append(op)
                seq += 1
        else:
            ops.append(op)
            seq += 1

    return ops
