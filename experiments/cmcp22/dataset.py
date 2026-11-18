"""CMCP 2.2 Multi-Turn Realistic Agent Dataset (66 scenarios: 8 dev, 8 tune, 50 eval).

Strict split separation:
- dev  (8 scenarios): telecom_core, iot_fleet
- tune (8 scenarios): telecom_core, iot_fleet (distinct IDs/entities)
- eval (50 scenarios): fin_ledger, med_triage, supply_chain, ci_release, energy_grid
  (completely unseen domains, templates, and entities)

Every 8-turn scenario exercises all 10 hard cases from Section 6.1:
  1. Late-added fact
  2. Later-corrected fact
  3. Invalidated fact (tombstone)
  4. Two entities with similar names (e.g. Vault-101 vs Vault-101X)
  5. Ternary relation across 3 entities
  6. Changing constraint
  7. Version-dependent decision
  8. Conflict between old and new information
  9. Early-turn fact absent from recent turns (must be preserved)
 10. Composite multi-turn query combining facts across turns
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class TurnEvent:
    turn_id: int
    speaker: str  # "agent_a" | "agent_b_query"
    utterance: str
    ops: List[Dict[str, Any]]  # ground-truth semantic ops (available only to deterministic extractor / evaluator)


@dataclass(frozen=True)
class MultiTurnScenario:
    scenario_id: str
    split: str
    domain: str
    turns: List[TurnEvent]
    # Ground truth for final composite query (Turn 8) + intermediate checks (Turns 4, 6, 7)
    eval_queries: List[Dict[str, Any]]
    true_active_facts: Dict[str, str]
    superseded_values: List[str]
    invalidated_values: List[str]
    active_relations: List[str]
    active_constraints: List[str]
    active_decisions: List[str]


def _make_scenario(idx: int, split: str, dom_spec: tuple) -> MultiTurnScenario:
    dom, p_noun, s_noun, t_noun, c_name = dom_spec
    base = 200 + idx * 13

    # Similar-named twin entities (#4)
    e_primary = f"{p_noun}-{base}"
    e_twin = f"{p_noun}-{base}B"
    e_mid = f"{s_noun}-{base + 4}"
    e_dst = f"{t_noun}-{base + 9}"

    early_anchor = f"ANCHOR-{chr(65 + (idx % 20))}{base}"  # #9: only in Turn 1
    old_route_code = f"R-{1000 + idx}"
    new_route_code = f"R-{2000 + idx}"
    temp_bad_token = f"TEMP-{9000 + idx}"  # #3: invalidated in Turn 3
    late_audit_tag = f"AUD-{3000 + idx}"  # #1: added late in Turn 5

    c_v1 = f"{c_name}<={80 + (idx % 5) * 10}ms"
    c_v2 = f"{c_name}<={35 + (idx % 4) * 5}ms"  # #6: tightened constraint
    dec_v1 = f"tier_mode=standard"
    dec_v2 = f"tier_mode=strict_v2(from:{dec_v1})"  # #7: depends on v1

    turns = [
        TurnEvent(
            turn_id=1,
            speaker="agent_a",
            utterance=(
                f"[{dom}] Turn 1: Primary unit {e_primary} initialized with anchor {early_anchor} "
                f"and twin backup {e_twin} (standby-only). Initial route code is {old_route_code}. "
                f"Constraint set to {c_v1}. Initial decision: {dec_v1}."
            ),
            ops=[
                {"fact_id": "ent:primary", "kind": "UPSERT", "cat": "entity", "key": e_primary, "val": f"primary; anchor={early_anchor}", "seq": 1, "ver": 1},
                {"fact_id": "ent:twin", "kind": "UPSERT", "cat": "entity", "key": e_twin, "val": "standby_backup_only", "seq": 2, "ver": 1},
                {"fact_id": "fact:anchor", "kind": "UPSERT", "cat": "fact", "key": "early_anchor", "val": early_anchor, "seq": 3, "ver": 1},
                {"fact_id": "fact:route", "kind": "UPSERT", "cat": "fact", "key": "route_code", "val": old_route_code, "seq": 4, "ver": 1},
                {"fact_id": "con:budget", "kind": "UPSERT", "cat": "constraint", "key": c_name, "val": c_v1, "seq": 5, "ver": 1},
                {"fact_id": "dec:mode", "kind": "UPSERT", "cat": "decision", "key": "tier_mode", "val": dec_v1, "seq": 6, "ver": 1},
            ],
        ),
        TurnEvent(
            turn_id=2,
            speaker="agent_a",
            utterance=(
                f"[{dom}] Turn 2: Discovered relay {e_mid} and sink {e_dst}. "
                f"Established 3-entity path: {e_primary} -> {e_mid} -> {e_dst}. "
                f"Provisional scratch token {temp_bad_token} opened for diagnostics."
            ),
            ops=[
                {"fact_id": "ent:mid", "kind": "UPSERT", "cat": "entity", "key": e_mid, "val": "relay", "seq": 7, "ver": 1},
                {"fact_id": "ent:dst", "kind": "UPSERT", "cat": "entity", "key": e_dst, "val": "sink", "seq": 8, "ver": 1},
                {"fact_id": "rel:path3", "kind": "UPSERT", "cat": "relation", "key": "ternary_path", "val": f"{e_primary} -> {e_mid} -> {e_dst}", "seq": 9, "ver": 1},
                {"fact_id": "fact:scratch", "kind": "UPSERT", "cat": "fact", "key": "scratch_token", "val": temp_bad_token, "seq": 10, "ver": 1},
            ],
        ),
        TurnEvent(
            turn_id=3,
            speaker="agent_a",
            utterance=(
                f"[{dom}] Turn 3: Correction: route code for {e_primary} changed from {old_route_code} to {new_route_code}. "
                f"Invalidated scratch token {temp_bad_token} (revoked; do not use)."
            ),
            ops=[
                {"fact_id": "fact:route", "kind": "UPSERT", "cat": "fact", "key": "route_code", "val": new_route_code, "seq": 11, "ver": 2},
                {"fact_id": "fact:scratch", "kind": "INVALIDATE", "cat": "fact", "key": "scratch_token", "val": "", "seq": 12, "ver": 2},
            ],
        ),
        TurnEvent(
            turn_id=4,
            speaker="agent_b_query",
            utterance=f"What is the current valid route code for {e_primary} (not {e_twin}), and is {temp_bad_token} valid?",
            ops=[],
        ),
        TurnEvent(
            turn_id=5,
            speaker="agent_a",
            utterance=(
                f"[{dom}] Turn 5: Policy update: {c_name} tightened from {c_v1} to {c_v2}. "
                f"Decision upgraded to {dec_v2}. Late audit tag {late_audit_tag} registered."
            ),
            ops=[
                {"fact_id": "con:budget", "kind": "UPSERT", "cat": "constraint", "key": c_name, "val": c_v2, "seq": 13, "ver": 2},
                {"fact_id": "dec:mode", "kind": "UPSERT", "cat": "decision", "key": "tier_mode", "val": dec_v2, "seq": 14, "ver": 2},
                {"fact_id": "fact:audit", "kind": "UPSERT", "cat": "fact", "key": "audit_tag", "val": late_audit_tag, "seq": 15, "ver": 1},
            ],
        ),
        TurnEvent(
            turn_id=6,
            speaker="agent_b_query",
            utterance=f"State the active {c_name} constraint, active decision, and audit tag.",
            ops=[],
        ),
        TurnEvent(
            turn_id=7,
            speaker="agent_b_query",
            utterance=f"Recover the original anchor assigned to {e_primary} at initialization.",
            ops=[],
        ),
        TurnEvent(
            turn_id=8,
            speaker="agent_b_query",
            utterance=(
                f"Composite task: For {e_primary} (distinguishing from {e_twin}), report its full 3-entity path "
                f"({e_primary} -> ... -> ...), active route code, initial anchor, active {c_name} limit, and active decision."
            ),
            ops=[],
        ),
    ]

    eval_queries = [
        {
            "turn": 4,
            "query": turns[3].utterance,
            "expected_keywords": [new_route_code],
            "forbidden_keywords": [old_route_code],
            "expected_entities": [e_primary],
            "expected_relations": [],
            "expected_constraints": [],
            "expected_decisions": [],
        },
        {
            "turn": 6,
            "query": turns[5].utterance,
            "expected_keywords": [c_v2.split("<=")[1], "strict_v2", late_audit_tag],
            "forbidden_keywords": [c_v1.split("<=")[1]],
            "expected_entities": [],
            "expected_relations": [],
            "expected_constraints": [c_v2.split("<=")[1]],
            "expected_decisions": ["strict_v2"],
        },
        {
            "turn": 7,
            "query": turns[6].utterance,
            "expected_keywords": [early_anchor],
            "forbidden_keywords": [],
            "expected_entities": [e_primary],
            "expected_relations": [],
            "expected_constraints": [],
            "expected_decisions": [],
        },
        {
            "turn": 8,
            "query": turns[7].utterance,
            "expected_keywords": [e_primary, e_mid, e_dst, new_route_code, early_anchor, c_v2.split("<=")[1], "strict_v2"],
            "forbidden_keywords": [old_route_code, temp_bad_token],
            "expected_entities": [e_primary, e_mid, e_dst],
            "expected_relations": [f"{e_primary}", f"{e_mid}", f"{e_dst}"],
            "expected_constraints": [c_v2.split("<=")[1]],
            "expected_decisions": ["strict_v2"],
        },
    ]

    return MultiTurnScenario(
        scenario_id=f"mt_{split}_{idx:02d}",
        split=split,
        domain=dom,
        turns=turns,
        eval_queries=eval_queries,
        true_active_facts={
            "primary": e_primary,
            "twin": e_twin,
            "mid": e_mid,
            "dst": e_dst,
            "early_anchor": early_anchor,
            "route_code": new_route_code,
            "audit_tag": late_audit_tag,
        },
        superseded_values=[old_route_code, c_v1],
        invalidated_values=[temp_bad_token],
        active_relations=[f"{e_primary} -> {e_mid} -> {e_dst}"],
        active_constraints=[c_v2],
        active_decisions=[dec_v2],
    )


def _build_all() -> List[MultiTurnScenario]:
    dev_tune_domains = [
        ("telecom_core", "Router", "Switch", "Peer", "JitterCap"),
        ("iot_fleet", "Sensor", "Gateway", "Broker", "PollBudget"),
    ]
    eval_domains = [
        ("fin_ledger", "Vault", "Clearing", "Settlement", "AuditSLA"),
        ("med_triage", "Ward", "RouterPod", "LabUnit", "ResponseCap"),
        ("supply_chain", "Depot", "Carrier", "PortHub", "LeadWindow"),
        ("ci_release", "Artifact", "Builder", "Registry", "SignTimeout"),
        ("energy_grid", "Substation", "Feeder", "Transformer", "LoadMargin"),
    ]

    scenarios: List[MultiTurnScenario] = []
    for i in range(8):
        scenarios.append(_make_scenario(i, "dev", dev_tune_domains[i % 2]))
    for i in range(8, 16):
        scenarios.append(_make_scenario(i, "tune", dev_tune_domains[i % 2]))
    for i in range(16, 66):
        scenarios.append(_make_scenario(i, "eval", eval_domains[(i - 16) % len(eval_domains)]))
    return scenarios


SCENARIOS: List[MultiTurnScenario] = _build_all()


def get_scenarios(split: str | None = None) -> List[MultiTurnScenario]:
    if split is None:
        return list(SCENARIOS)
    return [s for s in SCENARIOS if s.split == split]


def dataset_sha256() -> str:
    payload = json.dumps([asdict(s) for s in SCENARIOS], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
