"""CMCP 2.2b Gate 3 Extractors for Unstructured Natural-Language Dialogues:
  E1: OracleStructuredExtractor (upper bound / tool_use schema capture)
  E2: DeterministicNLPExtractor (CPU regex + morphosyntactic patterns, no LLM)
  E3: HybridGatedExtractor (E2 + selective gated LLM JSON-schema fallback on uncertain turns)
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from representations.state_store import StateOp


class OracleStructuredExtractor:
    """E1: Upper-bound extractor simulating native structured tool_use events."""

    def extract_turn(self, turn: Dict[str, Any], seq_base: int) -> Tuple[List[StateOp], float, int]:
        t0 = time.perf_counter()
        ops: List[StateOp] = []
        for idx, g in enumerate(turn.get("gold_ops", []), start=1):
            ops.append(
                StateOp(
                    fact_id=g["fact_id"],
                    kind=g["kind"],
                    category=g["category"],
                    key=g["key"],
                    value=g["value"],
                    seq=seq_base + idx,
                    version=1,
                    agent_id="agent_a",
                )
            )
        return ops, (time.perf_counter() - t0) * 1000.0, 0


class DeterministicNLPExtractor:
    """E2: Lightweight CPU regex + morphosyntactic extractor over raw utterances."""

    _RE_PRIMARY = re.compile(r"kickoff for\s+([A-Za-z]+-\d+).*?in\s+([a-z]+-[a-z0-9]+)", re.I)
    _RE_IGNORE = re.compile(r"Ignore\s+([A-Za-z]+-\d+-shadow)", re.I)
    _RE_LEAN = re.compile(r"leaning toward\s+([A-Za-z0-9_.-]+)\s+in", re.I)
    _RE_SWITCH = re.compile(r"switch from\s+([A-Za-z0-9_.-]+)\s+to\s+([A-Za-z0-9_.-]+)", re.I)
    _RE_CAP = re.compile(r"(?:cap|limit)\s+(?:under|is)\s+(\d+)\s*req/s", re.I)
    _RE_FLOW_PROSE = re.compile(r"traffic from\s+([A-Za-z0-9_-]+)\s+flows through\s+([A-Za-z0-9_-]+)\s+and lands on\s+([A-Za-z0-9_-]+)", re.I)
    _RE_FLOW_ARROW = re.compile(r"reroute\s+([A-Za-z0-9_-]+)\s*->\s*([A-Za-z0-9_-]+)\s*->\s*([A-Za-z0-9_-]+)", re.I)
    _RE_TAG_ADD = re.compile(r"tagged this batch as\s+(TK-\d+)", re.I)
    _RE_TAG_RET = re.compile(r"retiring\s+(TK-\d+)", re.I)
    _RE_TAG_USE = re.compile(r"Use\s+(TK-\d+)\s+going forward", re.I)

    def extract_turn(self, utterance: str, seq_base: int) -> Tuple[List[StateOp], float, int]:
        t0 = time.perf_counter()
        ops: List[StateOp] = []
        seq = seq_base

        m = self._RE_PRIMARY.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("primary_unit", "UPSERT", "entity", m.group(1), f"primary; region={m.group(2)}", seq, 1, "agent_a"))

        m = self._RE_IGNORE.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("alt_unit", "UPSERT", "entity", m.group(1), "staging_sandbox_ignore", seq, 1, "agent_a"))

        m = self._RE_SWITCH.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("backend_db", "UPSERT", "decision", "backend", m.group(2), seq, 1, "agent_a"))
        else:
            m = self._RE_LEAN.search(utterance)
            if m:
                seq += 1
                ops.append(StateOp("backend_db", "UPSERT", "decision", "backend", m.group(1), seq, 1, "agent_a"))

        m = self._RE_CAP.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("rate_limit", "UPSERT", "constraint", "rate_limit", f"rate_limit<={m.group(1)}", seq, 1, "agent_a"))

        m = self._RE_FLOW_ARROW.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("rel_flow", "UPSERT", "relation", "flow", f"{m.group(1)} -> {m.group(2)} -> {m.group(3)}", seq, 1, "agent_a"))
        else:
            m = self._RE_FLOW_PROSE.search(utterance)
            if m:
                seq += 1
                ops.append(StateOp("rel_flow", "UPSERT", "relation", "flow", f"{m.group(1)} -> {m.group(2)} -> {m.group(3)}", seq, 1, "agent_a"))

        m = self._RE_TAG_ADD.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("batch_tag", "UPSERT", "fact", "batch_tag", m.group(1), seq, 1, "agent_a"))

        m = self._RE_TAG_RET.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("batch_tag", "INVALIDATE", "fact", "batch_tag", "", seq, 1, "agent_a"))

        m = self._RE_TAG_USE.search(utterance)
        if m:
            seq += 1
            ops.append(StateOp("active_tag", "UPSERT", "fact", "active_tag", m.group(1), seq, 1, "agent_a"))

        return ops, (time.perf_counter() - t0) * 1000.0, 0


class HybridGatedExtractor:
    """E3: Deterministic NLP rules + selective gated LLM JSON-schema call on uncertain turns."""

    _UNCERTAINTY_MARKERS = re.compile(r"\b(wait|scratch|forget|instead|retiring|no longer|actually|switch)\b", re.I)

    def __init__(self, runner: Any) -> None:
        self.det = DeterministicNLPExtractor()
        self.runner = runner
        self.gated_calls = 0

    def extract_turn(self, utterance: str, seq_base: int) -> Tuple[List[StateOp], float, int]:
        ops, det_ms, errs = self.det.extract_turn(utterance, seq_base)
        # Gate: trigger LLM structured verification ONLY if implicit mutation markers appear
        if not self._UNCERTAINTY_MARKERS.search(utterance):
            return ops, det_ms, errs

        self.gated_calls += 1
        prompt = (
            "Extract active state updates from the utterance as a JSON array of objects "
            "with keys op(UPSERT|INVALIDATE), cat(decision|constraint|relation|fact), k, v.\n"
            f"Utterance: {utterance}\n"
            "JSON:"
        )
        t0 = time.perf_counter()
        resp = self.runner.query(prompt, max_tokens=64)
        llm_ms = float(resp.get("tokenize_ms", 0.0)) + float(resp.get("prefill_ms", 0.0)) + float(resp.get("eval_ms", 0.0))
        if llm_ms <= 0.0:
            llm_ms = (time.perf_counter() - t0) * 1000.0

        txt = resp.get("output_text", "").strip()
        parse_err = 0
        try:
            start = txt.find("[")
            end = txt.rfind("]")
            if start != -1 and end > start:
                arr = json.loads(txt[start : end + 1])
                if isinstance(arr, list):
                    existing = {(op.category, op.key) for op in ops}
                    seq = max((op.seq for op in ops), default=seq_base)
                    for item in arr:
                        if not isinstance(item, dict):
                            continue
                        cat = str(item.get("cat", "fact"))
                        key = str(item.get("k", ""))
                        val = str(item.get("v", ""))
                        kind = str(item.get("op", "UPSERT")).upper()
                        if kind not in ("UPSERT", "INVALIDATE") or not key:
                            continue
                        if (cat, key) not in existing:
                            seq += 1
                            fid = f"{cat}_{key}"
                            ops.append(StateOp(fid, kind, cat, key, val, seq, 1, "agent_a"))
                            existing.add((cat, key))
            else:
                parse_err = 1
        except Exception:
            parse_err = 1

        return ops, det_ms + llm_ms, errs + parse_err
