"""Context extractors for CMCP 2.2 (Modalidad B: Deterministic, Modalidad A: LLM-assisted).

Both extractors operate ONLY on TurnEvent.utterance (natural language text),
never touching ground-truth TurnEvent.ops.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple

from representations.state_store import StateOp


class DeterministicExtractor:
    """Rule-based non-generative extractor operating strictly on utterance text."""

    _RE_PRIMARY = re.compile(r"Primary unit ([A-Za-z0-9_-]+) initialized with anchor ([A-Za-z0-9_-]+)")
    _RE_TWIN = re.compile(r"twin backup ([A-Za-z0-9_-]+) \(standby-only\)")
    _RE_INIT_ROUTE = re.compile(r"Initial route code is ([A-Za-z0-9_-]+)")
    _RE_INIT_CON = re.compile(r"Constraint set to ([A-Za-z0-9_<=]+)")
    _RE_INIT_DEC = re.compile(r"Initial decision: ([A-Za-z0-9_=:()-]+)")
    _RE_PATH3 = re.compile(r"3-entity path: ([A-Za-z0-9_-]+ -> [A-Za-z0-9_-]+ -> [A-Za-z0-9_-]+)")
    _RE_SCRATCH = re.compile(r"scratch token ([A-Za-z0-9_-]+) opened")
    _RE_CORR_ROUTE = re.compile(r"route code for [A-Za-z0-9_-]+ changed from [A-Za-z0-9_-]+ to ([A-Za-z0-9_-]+)")
    _RE_INV_SCRATCH = re.compile(r"Invalidated scratch token ([A-Za-z0-9_-]+)")
    _RE_TIGHT_CON = re.compile(r"Policy update: ([A-Za-z0-9_-]+) tightened from [A-Za-z0-9_<=]+ to ([A-Za-z0-9_<=]+)")
    _RE_UPG_DEC = re.compile(r"Decision upgraded to ([A-Za-z0-9_=:()-]+)")
    _RE_AUDIT = re.compile(r"Late audit tag ([A-Za-z0-9_-]+) registered")

    def extract_turn(self, utterance: str, seq_base: int) -> Tuple[List[StateOp], float, int]:
        """Extract StateOps from natural language utterance. Returns (ops, ms, errors)."""
        t0 = time.perf_counter()
        ops: List[StateOp] = []
        seq = seq_base

        m = self._RE_PRIMARY.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("ent:primary", "UPSERT", "entity", m.group(1), f"primary; anchor={m.group(2)}", seq=seq, version=1))
            seq += 1
            ops.append(StateOp("fact:anchor", "UPSERT", "fact", "early_anchor", m.group(2), seq=seq, version=1))

        m = self._RE_TWIN.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("ent:twin", "UPSERT", "entity", m.group(1), "standby_backup_only", seq=seq, version=1))

        m = self._RE_INIT_ROUTE.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("fact:route", "UPSERT", "fact", "route_code", m.group(1), seq=seq, version=1))

        m = self._RE_INIT_CON.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("con:budget", "UPSERT", "constraint", "budget", m.group(1), seq=seq, version=1))

        m = self._RE_INIT_DEC.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("dec:mode", "UPSERT", "decision", "tier_mode", m.group(1), seq=seq, version=1))

        m = self._RE_PATH3.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("rel:path3", "UPSERT", "relation", "ternary_path", m.group(1), seq=seq, version=1))

        m = self._RE_SCRATCH.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("fact:scratch", "UPSERT", "fact", "scratch_token", m.group(1), seq=seq, version=1))

        m = self._RE_CORR_ROUTE.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("fact:route", "UPSERT", "fact", "route_code", m.group(1), seq=seq, version=2))

        m = self._RE_INV_SCRATCH.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("fact:scratch", "INVALIDATE", "fact", "scratch_token", "", seq=seq, version=2))

        m = self._RE_TIGHT_CON.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("con:budget", "UPSERT", "constraint", m.group(1), m.group(2), seq=seq, version=2))

        m = self._RE_UPG_DEC.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("dec:mode", "UPSERT", "decision", "tier_mode", m.group(1), seq=seq, version=2))

        m = self._RE_AUDIT.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("fact:audit", "UPSERT", "fact", "audit_tag", m.group(1), seq=seq, version=1))

        ms = (time.perf_counter() - t0) * 1000.0
        errors = 1 if ("Turn" in utterance and len(ops) == 0) else 0
        return ops, ms, errors


class LLMExtractor:
    """Modalidad A: LLM-assisted extraction using Agent A's model via persistent harness."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    def extract_turn(self, utterance: str, seq_base: int) -> Dict[str, Any]:
        prompt = (
            "Extract active facts, relations, constraints, decisions, and invalidated items as concise KEY=VALUE lines.\n"
            f"Text: {utterance}\n"
            "Output:"
        )
        resp = self.runner.query(prompt, max_tokens=48)
        text = resp.get("output_text", "").strip()
        return {
            "extracted_text": text,
            "in_tokens": int(resp.get("n_input_tokens", 0)),
            "out_tokens": int(resp.get("n_gen_tokens", 0)),
            "extract_ms": float(resp.get("tokenize_ms", 0.0)) + float(resp.get("prefill_ms", 0.0)) + float(resp.get("eval_ms", 0.0)),
            "error": 1 if ("error" in resp or len(text) < 5) else 0,
        }
