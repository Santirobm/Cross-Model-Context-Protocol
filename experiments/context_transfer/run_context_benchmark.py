"""CMCP 2.1 Priority Benchmark: Context & Reference Transfer (Phase 4).

Compares 5 semantically equivalent context representations on real llama.cpp
CPU inference (Llama-3.2-1B-Instruct-Q4_K_M.gguf as receiver):
  A. full_history       - Complete multi-turn conversation transcript
  B. json_structured    - Compact JSON with equivalent active/deprecated facts
  C. cmcp_textual       - Human-readable CMCPContext wire prompt
  D. cmcp_binary_tlv    - Typed binary TLV + C99 CMCP frame + decode -> prompt
  E. cmcp_incremental   - Delta reference frame + persisted store lookup

Separates preparation, serialization, transfer, reconstruction, tokenization,
prefill, and generation times, and logs raw CSV, config JSON, summary JSON,
and execution logs under experiments/context_transfer/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.llama import LlamaAdapter
from experiments.context_transfer.dataset import ContextTask, dataset_sha256, get_tasks
from representations.binary import (
    CMCP_MSG_CONTEXT_TRANSFER,
    CMCP_TIER_2_CROSS_MODEL,
    CMCPHeader,
    decode_frame,
    encode_frame,
)
from representations.structured.schema import (
    CMCPContext,
    Constraint,
    Decision,
    Entity,
    Evidence,
    Instruction,
    Reference,
    Relation,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.\-<= ]+", " ", text.lower())).strip()


def _token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = 0
    g_counts: Dict[str, int] = {}
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p_toks:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec = common / len(g_toks)
    return 2.0 * prec * rec / (prec + rec)


def _score_output(pred: str, task: ContextTask) -> Dict[str, float]:
    norm_pred = pred.lower()
    kw_hits = sum(1 for kw in task.expected_keywords if kw.lower() in norm_pred)
    kw_recall = kw_hits / max(1, len(task.expected_keywords))
    em = 1.0 if kw_hits == len(task.expected_keywords) else 0.0
    f1 = _token_f1(pred, task.expected_answer)
    # Relation recovery checks if primary entity/target/port/owner co-occur coherently
    rel_hits = 0
    for rel in task.expected_relations:
        parts = [p.lower() for p in rel.split("->")]
        if all(p in norm_pred for p in parts):
            rel_hits += 1
    rel_rec = rel_hits / max(1, len(task.expected_relations))
    return {
        "exact_match": em,
        "f1": f1,
        "keyword_recall": kw_recall,
        "relation_recovery": rel_rec,
    }


def _build_equivalent_cmcp_context(task: ContextTask) -> CMCPContext:
    entities = [Entity(name=k, kind="active", value=v) for k, v in task.active_facts.items()]
    entities.extend(Entity(name=k, kind="deprecated", value=v) for k, v in task.distractor_facts.items())
    relations = [Relation(src=s, rel=r, dst=d) for s, r, d in task.relations]
    constraints = [Constraint(rule=c) for c in task.constraints]
    decisions = [Decision(topic="routing", choice=d) for d in task.decisions]
    evidences = [Evidence(id="temporal", claim=task.temporal_note, source="audit_log")]
    instructions = [Instruction(text=ins) for ins in task.instructions]
    return CMCPContext(
        task_id=task.task_id,
        concepts=[task.domain],
        entities=entities,
        relations=relations,
        instructions=instructions,
        constraints=constraints,
        evidences=evidences,
        decisions=decisions,
    )


def _prepare_method_input(
    method: str,
    task: ContextTask,
    persisted_store: Dict[str, bytes],
) -> Dict[str, Any]:
    """Prepare wire payload and receiver prompt, timing every step honestly."""
    t_ext0 = time.perf_counter()
    if method == "full_history":
        prompt = (
            "Conversation history:\n"
            + "\n".join(task.history_turns)
            + f"\n\nQuestion: {task.question}\nAnswer concisely:"
        )
        t_ext1 = time.perf_counter()
        raw_bytes = prompt.encode("utf-8")
        return {
            "prompt": prompt,
            "wire_bytes": len(raw_bytes),
            "wire_text": prompt,
            "prep_extract_ms": (t_ext1 - t_ext0) * 1000.0,
            "prep_build_ms": 0.0,
            "serialize_ms": 0.0,
            "transfer_ms": 0.0,
            "reconstruct_ms": 0.0,
            "persisted_state_bytes": 0,
        }

    # Methods B, C, D, E all extract the same structured facts
    ctx = _build_equivalent_cmcp_context(task)
    t_ext1 = time.perf_counter()
    prep_extract_ms = (t_ext1 - t_ext0) * 1000.0

    if method == "json_structured":
        t_b0 = time.perf_counter()
        json_str = ctx.to_json_str(compact=True)
        wire_bytes = len(json_str.encode("utf-8"))
        t_b1 = time.perf_counter()
        prompt = (
            f"JSON_CTX:\n{json_str}\n\n"
            f"Question: {task.question}\n"
            "Answer concisely using active facts only:"
        )
        t_r1 = time.perf_counter()
        return {
            "prompt": prompt,
            "wire_bytes": wire_bytes,
            "wire_text": json_str,
            "prep_extract_ms": prep_extract_ms,
            "prep_build_ms": (t_b1 - t_b0) * 1000.0,
            "serialize_ms": (t_b1 - t_b0) * 1000.0,
            "transfer_ms": 0.0,
            "reconstruct_ms": (t_r1 - t_b1) * 1000.0,
            "persisted_state_bytes": 0,
        }

    if method == "cmcp_textual":
        t_b0 = time.perf_counter()
        wire_text = ctx.to_compact_text()
        wire_bytes = len(wire_text.encode("utf-8"))
        t_b1 = time.perf_counter()
        prompt = ctx.to_wire_prompt(task_query=task.question)
        t_r1 = time.perf_counter()
        return {
            "prompt": prompt,
            "wire_bytes": wire_bytes,
            "wire_text": wire_text,
            "prep_extract_ms": prep_extract_ms,
            "prep_build_ms": (t_b1 - t_b0) * 1000.0,
            "serialize_ms": (t_b1 - t_b0) * 1000.0,
            "transfer_ms": 0.0,
            "reconstruct_ms": (t_r1 - t_b1) * 1000.0,
            "persisted_state_bytes": 0,
        }

    if method == "cmcp_binary_tlv":
        t_s0 = time.perf_counter()
        records = ctx.to_typed_tlv_records()
        frame = encode_frame(
            records,
            compress=True,
            msg_type=CMCP_MSG_CONTEXT_TRANSFER,
            compat_tier=CMCP_TIER_2_CROSS_MODEL,
            sender_model_hash=0x1111,
            receiver_model_hash=0x2222,
            tokenizer_hash=0x3333,
            sequence_id=1,
        )
        t_s1 = time.perf_counter()

        # Simulate byte transfer over buffer copy
        t_x0 = time.perf_counter()
        wire_copy = bytes(frame)
        t_x1 = time.perf_counter()

        t_r0 = time.perf_counter()
        _, dec_recs = decode_frame(wire_copy)
        dec_ctx = CMCPContext.from_typed_tlv_records(dec_recs, task_id=task.task_id)
        prompt = dec_ctx.to_wire_prompt(task_query=task.question)
        t_r1 = time.perf_counter()

        return {
            "prompt": prompt,
            "wire_bytes": len(frame),
            "wire_text": dec_ctx.to_compact_text(),
            "prep_extract_ms": prep_extract_ms,
            "prep_build_ms": 0.0,
            "serialize_ms": (t_s1 - t_s0) * 1000.0,
            "transfer_ms": (t_x1 - t_x0) * 1000.0,
            "reconstruct_ms": (t_r1 - t_r0) * 1000.0,
            "persisted_state_bytes": 0,
        }

    if method == "cmcp_incremental":
        # Persist active facts blob in store (representing prior turn state) and send
        # only compact reference ID + active decision/constraint delta over binary frame.
        t_p0 = time.perf_counter()
        ref_key = f"ref:{task.task_id}:facts"
        facts_blob = json.dumps(task.active_facts, separators=(",", ":"), sort_keys=True).encode("utf-8")
        persisted_store[ref_key] = facts_blob
        digest = hashlib.sha256(facts_blob).hexdigest()[:12]
        t_p1 = time.perf_counter()

        t_s0 = time.perf_counter()
        delta_ctx = CMCPContext(
            task_id=task.task_id,
            references=[Reference(id="R1", uri=ref_key, digest=digest)],
            constraints=[Constraint(rule=task.constraints[0])],
            instructions=[Instruction(text=task.instructions[0])],
        )
        records = delta_ctx.to_typed_tlv_records()
        frame = encode_frame(
            records,
            compress=True,
            msg_type=CMCP_MSG_CONTEXT_TRANSFER,
            compat_tier=CMCP_TIER_2_CROSS_MODEL,
            sender_model_hash=0x1111,
            receiver_model_hash=0x2222,
            tokenizer_hash=0x3333,
            sequence_id=2,
        )
        t_s1 = time.perf_counter()

        t_x0 = time.perf_counter()
        wire_copy = bytes(frame)
        t_x1 = time.perf_counter()

        t_r0 = time.perf_counter()
        _, dec_recs = decode_frame(wire_copy)
        dec_delta = CMCPContext.from_typed_tlv_records(dec_recs, task_id=task.task_id)
        for ref in dec_delta.references:
            resolved = persisted_store[ref.uri].decode("utf-8")
            ref.resolved_text = resolved
        prompt = dec_delta.to_wire_prompt(task_query=task.question, include_resolved_refs=True)
        t_r1 = time.perf_counter()

        return {
            "prompt": prompt,
            "wire_bytes": len(frame),
            "wire_text": dec_delta.to_compact_text(include_resolved_refs=True),
            "prep_extract_ms": prep_extract_ms,
            "prep_build_ms": (t_p1 - t_p0) * 1000.0,
            "serialize_ms": (t_s1 - t_s0) * 1000.0,
            "transfer_ms": (t_x1 - t_x0) * 1000.0,
            "reconstruct_ms": (t_r1 - t_r0) * 1000.0,
            "persisted_state_bytes": len(facts_blob),
        }

    raise ValueError(f"Unknown method: {method}")


class PersistentLlamaRunner:
    def __init__(self, model_path: str, n_ctx: int = 1024, threads: int = 4) -> None:
        exe = ROOT / "build" / "cmcp_llama_harness.exe"
        env = os.environ.copy()
        env["PATH"] = (
            r"C:\msys64\clang64\bin;C:\Users\sgi.lm\Desktop\llama.cpp\build\bin;"
            + env.get("PATH", "")
        )
        self.proc = subprocess.Popen(
            [str(exe), "--mode", "persistent", "--model", model_path,
             "--n-ctx", str(n_ctx), "--threads", str(threads)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=env,
        )
        assert self.proc.stdout is not None
        ready_line = self.proc.stdout.readline()
        if not ready_line or '"ready":true' not in ready_line:
            raise RuntimeError(f"Persistent harness failed to start: {ready_line!r}")

    def query(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        escaped = (
            prompt.replace("\\", "\\\\")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        self.proc.stdin.write(f"{max_tokens}\t{escaped}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line)

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="eval", choices=["dev", "tune", "eval", "all"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=36)
    parser.add_argument("--model", default="C:/Users/sgi.lm/Desktop/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    res_dir = out_dir / "results"
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = get_tasks(None if args.split == "all" else args.split)
    methods = [
        "full_history",
        "json_structured",
        "cmcp_textual",
        "cmcp_binary_tlv",
        "cmcp_incremental",
    ]

    cfg = {
        "split": args.split,
        "num_tasks": len(tasks),
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "receiver_model": Path(args.model).name,
        "dataset_sha256": dataset_sha256(),
        "methods": methods,
    }
    (cfg_dir / "benchmark_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    runner = PersistentLlamaRunner(args.model, n_ctx=1024, threads=4)
    persisted_store: Dict[str, bytes] = {}

    # Warm-up
    _ = runner.query("Warm-up prompt: say OK.", max_tokens=4)

    csv_path = res_dir / "context_benchmark_runs.csv"
    log_path = log_dir / "run.log"
    fieldnames = [
        "task_id", "split", "method", "rep",
        "serialized_bytes", "wire_tokens", "receiver_input_tokens",
        "prep_extract_ms", "prep_build_ms", "serialize_ms", "transfer_ms", "reconstruct_ms",
        "tokenize_ms", "prefill_ms", "generation_ms", "end_to_end_ms",
        "persisted_state_bytes", "exact_match", "f1", "keyword_recall", "relation_recovery",
        "error", "max_rss_mib", "output_preview",
    ]

    rows: List[Dict[str, Any]] = []
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, open(log_path, "w", encoding="utf-8") as flog:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()

            total_jobs = len(tasks) * len(methods) * args.reps
            done = 0
            for rep in range(1, args.reps + 1):
                for task in tasks:
                    for method in methods:
                        prep = _prepare_method_input(method, task, persisted_store)
                        wire_tok_resp = runner.query(prep["wire_text"], max_tokens=0)
                        resp = runner.query(prep["prompt"], max_tokens=args.max_tokens)

                        err = 1.0 if "error" in resp else 0.0
                        out_text = resp.get("output_text", "")
                        scores = _score_output(out_text, task)
                        tok_ms = float(resp.get("tokenize_ms", 0.0))
                        pre_ms = float(resp.get("prefill_ms", 0.0))
                        ev_ms = float(resp.get("eval_ms", 0.0))
                        in_tok = int(resp.get("n_input_tokens", 0))
                        rss_mib = float(resp.get("rss_mib", 0.0))

                        e2e_ms = (
                            prep["prep_extract_ms"]
                            + prep["prep_build_ms"]
                            + prep["serialize_ms"]
                            + prep["transfer_ms"]
                            + prep["reconstruct_ms"]
                            + tok_ms
                            + pre_ms
                            + ev_ms
                        )

                        row = {
                            "task_id": task.task_id,
                            "split": task.split,
                            "method": method,
                            "rep": rep,
                            "serialized_bytes": prep["wire_bytes"],
                            "wire_tokens": int(wire_tok_resp.get("n_input_tokens", 0)),
                            "receiver_input_tokens": in_tok,
                            "prep_extract_ms": round(prep["prep_extract_ms"], 4),
                            "prep_build_ms": round(prep["prep_build_ms"], 4),
                            "serialize_ms": round(prep["serialize_ms"], 4),
                            "transfer_ms": round(prep["transfer_ms"], 4),
                            "reconstruct_ms": round(prep["reconstruct_ms"], 4),
                            "tokenize_ms": round(tok_ms, 4),
                            "prefill_ms": round(pre_ms, 4),
                            "generation_ms": round(ev_ms, 4),
                            "end_to_end_ms": round(e2e_ms, 4),
                            "persisted_state_bytes": prep["persisted_state_bytes"],
                            "exact_match": scores["exact_match"],
                            "f1": round(scores["f1"], 4),
                            "keyword_recall": round(scores["keyword_recall"], 4),
                            "relation_recovery": round(scores["relation_recovery"], 4),
                            "error": err,
                            "max_rss_mib": round(rss_mib, 2),
                            "output_preview": out_text.replace("\n", " ")[:120],
                        }
                        writer.writerow(row)
                        fcsv.flush()
                        rows.append(row)
                        done += 1
                        flog.write(f"[{done}/{total_jobs}] {task.task_id} {method} rep={rep} em={scores['exact_match']} tok={in_tok} prefill={pre_ms:.1f}ms\n")
                        flog.flush()
    finally:
        runner.close()

    # Aggregate summary
    summary: Dict[str, Any] = {"config": cfg, "methods": {}}
    num_keys = [
        "serialized_bytes", "wire_tokens", "receiver_input_tokens",
        "prep_extract_ms", "prep_build_ms", "serialize_ms", "transfer_ms", "reconstruct_ms",
        "tokenize_ms", "prefill_ms", "generation_ms", "end_to_end_ms",
        "persisted_state_bytes", "exact_match", "f1", "keyword_recall", "relation_recovery",
        "error", "max_rss_mib",
    ]
    for m in methods:
        mrows = [r for r in rows if r["method"] == m]
        stats: Dict[str, Any] = {"n": len(mrows)}
        for k in num_keys:
            vals = [float(r[k]) for r in mrows]
            stats[k] = {
                "mean": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            }
        summary["methods"][m] = stats

    (res_dir / "context_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
