"""Controlled synthetic 30-task dataset for CMCP 2.1 Context Transfer Benchmark.

Splits (disjoint by scenario_id and entity names):
- dev  (6 tasks): format sanity & smoke checks
- tune (6 tasks): prompt/schema tuning
- eval (18 tasks): held-out evaluation only

Each task contains entities, attributes, relations, constraints, decisions,
instructions, temporal markers, and multi-hop dependencies, plus a deterministic
ground-truth answer and expected keywords/relations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ContextTask:
    task_id: str
    split: str
    domain: str
    history_turns: List[str]
    active_facts: Dict[str, str]
    distractor_facts: Dict[str, str]
    relations: List[List[str]]
    constraints: List[str]
    decisions: List[str]
    instructions: List[str]
    temporal_note: str
    question: str
    expected_answer: str
    expected_keywords: List[str]
    expected_relations: List[str]


def _build_tasks() -> List[ContextTask]:
    tasks: List[ContextTask] = []

    domains = [
        ("cluster_ops", "Node", "Cluster", "Region", "LatencyBudget"),
        ("storage_eng", "Shard", "Volume", "Zone", "ThroughputCap"),
        ("sec_policy", "Gateway", "Tenant", "Realm", "TokenTTL"),
        ("ml_pipeline", "Worker", "Queue", "Pool", "BatchLimit"),
        ("build_sys", "Target", "Runner", "Cache", "TimeoutSec"),
        ("net_mesh", "Ingress", "Service", "Subnet", "RetryCap"),
    ]

    # 30 deterministic tasks: 6 dev, 6 tune, 18 eval (disjoint IDs, values, timestamps)
    for idx in range(30):
        if idx < 6:
            split = "dev"
        elif idx < 12:
            split = "tune"
        else:
            split = "eval"

        dom, ent_a, ent_b, ent_c, metric_k = domains[idx % len(domains)]
        code = 100 + idx * 7
        a_val = f"{ent_a}-{code}"
        b_val = f"{ent_b}-{code + 3}"
        c_val = f"{ent_c}-{(idx % 5) + 1}"
        budget = f"{25 + (idx * 5)}ms"
        port = 7100 + idx * 11
        day = f"2026-11-{(idx % 20) + 10:02d}"
        owner = f"Team-{chr(65 + (idx % 6))}{idx}"
        old_port = port - 1000

        # Multi-turn history includes prior superseded state + current active state
        history_turns = [
            f"[Turn 1 | {day}T08:00Z] Initial draft assigned {a_val} to port {old_port} in {c_val}.",
            f"[Turn 2 | {day}T09:15Z] Distractor telemetry: archive mirror M-{code} runs on port {old_port + 55} with owner Legacy-{idx}.",
            f"[Turn 3 | {day}T10:30Z] Decision update: {a_val} is bound to {b_val} in {c_val}; active port changed from {old_port} to {port}.",
            f"[Turn 4 | {day}T11:45Z] Constraint: {metric_k} must stay <= {budget} and owner is {owner}.",
            f"[Turn 5 | {day}T12:00Z] Instruction: Report the active port and owner of {b_val} via {a_val} respecting {metric_k}.",
        ]

        active_facts = {
            "primary_entity": a_val,
            "bound_target": b_val,
            "zone": c_val,
            "active_port": str(port),
            "metric_limit": f"{metric_k}<={budget}",
            "owner": owner,
        }
        distractor_facts = {
            "deprecated_port": str(old_port),
            "archive_mirror": f"M-{code}",
            "archive_port": str(old_port + 55),
            "legacy_owner": f"Legacy-{idx}",
        }
        relations = [
            [a_val, "binds_to", b_val],
            [b_val, "hosted_in", c_val],
            [a_val, "uses_port", str(port)],
            [b_val, "owned_by", owner],
        ]
        constraints = [
            f"{metric_k} <= {budget}",
            f"Ignore deprecated port {old_port}",
        ]
        decisions = [
            f"Active port for {a_val}->{b_val} is {port} (supersedes {old_port})",
            f"Assigned owner is {owner}",
        ]
        instructions = [
            f"Report the active port, owner, and {metric_k} limit for {b_val} using active facts only."
        ]
        question = (
            f"Which active port and owner govern {b_val} via {a_val} in {c_val}, "
            f"and what is the {metric_k} limit?"
        )
        expected_answer = f"port {port}, owner {owner}, {metric_k} <= {budget}"
        expected_keywords = [str(port), owner, budget]
        expected_relations = [f"{a_val}->{b_val}", f"{b_val}->{owner}"]

        tasks.append(
            ContextTask(
                task_id=f"cmcp21_{split}_{idx:02d}",
                split=split,
                domain=dom,
                history_turns=history_turns,
                active_facts=active_facts,
                distractor_facts=distractor_facts,
                relations=relations,
                constraints=constraints,
                decisions=decisions,
                instructions=instructions,
                temporal_note=f"active_since={day}T11:45Z",
                question=question,
                expected_answer=expected_answer,
                expected_keywords=expected_keywords,
                expected_relations=expected_relations,
            )
        )

    return tasks


TASKS: List[ContextTask] = _build_tasks()


def get_tasks(split: str | None = None) -> List[ContextTask]:
    if split is None:
        return list(TASKS)
    return [t for t in TASKS if t.split == split]


def dataset_sha256() -> str:
    payload = json.dumps([asdict(t) for t in TASKS], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
