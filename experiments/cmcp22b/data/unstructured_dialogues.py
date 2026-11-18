"""CMCP 2.2b Gate 3: Realistic Unstructured Natural-Language Dialogues (n=30, 8 turns each).

Domains: cloud_infra, ml_ops, sec_ops (strictly disjoint from dev/tune/eval).
Contains implicit opinion changes, informal invalidations, cross-turn dependencies,
and conversational noise (greetings, hesitations, clarifications).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATA_PATH = Path(__file__).resolve().parent / "unstructured_dialogues.json"

_DOMAINS = [
    ("cloud_infra", "svc", ["PostgreSQL", "SQLite", "DuckDB", "MariaDB"], ["Redis", "Memcached"], ["us-east", "eu-west", "ap-south"]),
    ("ml_ops", "pipe", ["PyTorch", "ONNX", "JAX", "TensorRT"], ["Ray", "Celery"], ["gpu-a10", "cpu-avx", "tpu-v4"]),
    ("sec_ops", "vault", ["OAuth2", "mTLS", "OIDC", "SAML"], ["Vault", "KMS"], ["prod-eu", "prod-us", "gov-core"]),
]


def build_unstructured_dialogues(n: int = 30) -> List[Dict[str, Any]]:
    dialogues: List[Dict[str, Any]] = []
    for i in range(n):
        dom, prefix, dbs, caches, regions = _DOMAINS[i % len(_DOMAINS)]
        unit = f"{prefix}-{100 + i}"
        alt_unit = f"{unit}-shadow"
        db_old = dbs[i % len(dbs)]
        db_new = dbs[(i + 1) % len(dbs)]
        cache = caches[i % len(caches)]
        reg_old = regions[i % len(regions)]
        reg_new = regions[(i + 1) % len(regions)]
        limit_old = 400 + (i % 5) * 100
        limit_new = 1200 + (i % 7) * 50
        tok_old = f"TK-{10 + i}"
        tok_new = f"TK-{50 + i}"

        variant = i % 3
        if variant == 0:
            u3 = (
                f"Wait, scratch {db_old}—it's overkill for this node. Let's switch from {db_old} to {db_new} instead. "
                f"Also forget the previous limit, we just got bumped and now the cap is {limit_new} req/s."
            )
            u5 = (
                f"Sorry for the noise, coffee just kicked in! One more thing: we're retiring {tok_old} completely, "
                f"no longer valid. Use {tok_new} going forward, and reroute {unit} -> {cache} -> {reg_new}."
            )
        elif variant == 1:
            u3 = (
                f"Actually, drop {db_old} and move to {db_new} for this service. "
                f"Ignore the earlier cap—bump it up to {limit_new} req/s."
            )
            u5 = (
                f"Quick heads up: {tok_old} is dead now, please replace it with {tok_new}. "
                f"Also send {unit} traffic via {cache} into {reg_new} from now on."
            )
        else:
            u3 = (
                f"Let's swap {db_old} out for {db_new}, team agreed this morning. "
                f"Oh, and the new ceiling is {limit_new} req/s instead of the old one."
            )
            u5 = (
                f"Almost forgot—cancel {tok_old}, we switched to {tok_new}. "
                f"Update the path so {unit} goes through {cache} to {reg_new}."
            )

        turns = [
            {
                "turn_id": 1,
                "speaker": "agent_a",
                "utterance": (
                    f"Hey team, quick kickoff for {unit}. Right now we're leaning toward {db_old} "
                    f"in {reg_old}, and let's keep the rate cap under {limit_old} req/s for safety. "
                    f"Ignore {alt_unit} for now, that's just a staging sandbox."
                ),
                "gold_ops": [
                    {"fact_id": "primary_unit", "kind": "UPSERT", "category": "entity", "key": unit, "value": f"primary; region={reg_old}", "seq": 1},
                    {"fact_id": "alt_unit", "kind": "UPSERT", "category": "entity", "key": alt_unit, "value": "staging_sandbox_ignore", "seq": 2},
                    {"fact_id": "backend_db", "kind": "UPSERT", "category": "decision", "key": "backend", "value": db_old, "seq": 3},
                    {"fact_id": "rate_limit", "kind": "UPSERT", "category": "constraint", "key": "rate_limit", "value": f"rate_limit<={limit_old}", "seq": 4},
                ],
            },
            {
                "turn_id": 2,
                "speaker": "agent_a",
                "utterance": (
                    f"Oh, before I forget, traffic from {unit} flows through {cache} and lands on {reg_old}. "
                    f"Also tagged this batch as {tok_old} in case anyone checks."
                ),
                "gold_ops": [
                    {"fact_id": "rel_flow", "kind": "UPSERT", "category": "relation", "key": "flow", "value": f"{unit} -> {cache} -> {reg_old}", "seq": 5},
                    {"fact_id": "batch_tag", "kind": "UPSERT", "category": "fact", "key": "batch_tag", "value": tok_old, "seq": 6},
                ],
            },
            {
                "turn_id": 3,
                "speaker": "agent_a",
                "utterance": u3,
                "gold_ops": [
                    {"fact_id": "backend_db", "kind": "UPSERT", "category": "decision", "key": "backend", "value": db_new, "seq": 7},
                    {"fact_id": "rate_limit", "kind": "UPSERT", "category": "constraint", "key": "rate_limit", "value": f"rate_limit<={limit_new}", "seq": 8},
                ],
            },
            {
                "turn_id": 4,
                "speaker": "agent_b",
                "utterance": "Got it, thanks for the heads up. Checking the handoff state now.",
                "gold_ops": [],
            },
            {
                "turn_id": 5,
                "speaker": "agent_a",
                "utterance": u5,
                "gold_ops": [
                    {"fact_id": "batch_tag", "kind": "INVALIDATE", "category": "fact", "key": "batch_tag", "value": "", "seq": 9},
                    {"fact_id": "active_tag", "kind": "UPSERT", "category": "fact", "key": "active_tag", "value": tok_new, "seq": 10},
                    {"fact_id": "rel_flow", "kind": "UPSERT", "category": "relation", "key": "flow", "value": f"{unit} -> {cache} -> {reg_new}", "seq": 11},
                ],
            },
            {
                "turn_id": 6,
                "speaker": "agent_b",
                "utterance": "Understood, noted the reroute and new tag.",
                "gold_ops": [],
            },
            {
                "turn_id": 7,
                "speaker": "agent_a",
                "utterance": "Looks good on my end too. No other changes before final sign-off.",
                "gold_ops": [],
            },
            {
                "turn_id": 8,
                "speaker": "agent_b",
                "utterance": "Ready for the final composite summary.",
                "gold_ops": [],
            },
        ]

        eval_query = {
            "turn": 8,
            "query": (
                f"For {unit} (ignoring {alt_unit}), what is the final flow path ({unit} -> ... -> ...), "
                f"chosen backend, active rate limit, and active batch tag?"
            ),
            "expected_keywords": [unit, cache, reg_new, db_new, str(limit_new), tok_new],
            "expected_entities": [unit, cache, reg_new],
            "forbidden_keywords": [db_old, reg_old, str(limit_old), tok_old],
            "expected_relations": [f"{unit} -> {cache} -> {reg_new}"],
            "expected_constraints": [str(limit_new)],
            "expected_decisions": [db_new],
        }

        dialogues.append({
            "dialogue_id": f"unstruct_{i:02d}",
            "domain": dom,
            "turns": turns,
            "eval_query": eval_query,
        })
    return dialogues


def save_dialogues() -> Path:
    data = build_unstructured_dialogues(30)
    DATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return DATA_PATH


if __name__ == "__main__":
    p = save_dialogues()
    print(f"Saved {len(json.loads(p.read_text(encoding='utf-8')))} dialogues to {p}")
