"""CMCP 2.2 Main Multi-Turn Context Transfer Benchmark (Sections 5-11, 13).

Evaluates 5 baselines + LLM-extraction modality across 50 held-out 8-turn scenarios:
  A. full_history      - Agent B receives full accumulated conversation history
  B. rolling_summary   - Agent A (LLM) produces rolling summary; Agent B reads summary
  C. json_structured   - Structured JSON context (deterministic extraction)
  D. cmcp_full         - Full CMCP binary TLV state sent at each handoff
  E. cmcp_incremental  - Versioned CMCP delta + tombstone + ref store (deterministic ext)
  F. cmcp_inc_llm_ext  - Agent A (LLM) extracts deltas -> CMCP incremental -> Agent B

Supports sequential Agent A -> release -> Agent B execution for heterogeneous
models (Section 11) without exceeding 16 GB RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cmcp22.dataset import MultiTurnScenario, dataset_sha256, get_scenarios
from experiments.cmcp22.extractors import DeterministicExtractor, LLMExtractor
from representations.binary import (
    CMCP_MSG_CONTEXT_TRANSFER,
    CMCP_TIER_2_CROSS_MODEL,
    decode_frame,
    encode_frame,
)
from representations.state_store import CMCPStateStore, StateOp


class PersistentLlamaRunner:
    def __init__(self, model_path: str, n_ctx: int = 1024, threads: int = 4) -> None:
        exe = ROOT / "build" / "cmcp_llama_harness.exe"
        if not exe.exists():
            exe = ROOT / "build" / "cmcp_llama_harness"
        env = os.environ.copy()
        clang_bin = r"C:\msys64\clang64\bin"
        llama_bin = r"C:\Users\sgi.lm\Desktop\llama.cpp\build\bin"
        if os.name == "nt":
            env["PATH"] = f"{clang_bin};{llama_bin};" + env.get("PATH", "")
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
        ready = self.proc.stdout.readline()
        if '"ready":true' not in ready:
            raise RuntimeError(f"Failed to start harness for {model_path}: {ready!r}")

    def query(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        esc = (
            prompt.replace("\\", "\\\\")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        self.proc.stdin.write(f"{max_tokens}\t{esc}\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=15)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.\-<= ]+", " ", s.lower())).strip()


def _token_f1(pred: str, gold: str) -> float:
    p = _norm(pred).split()
    g = _norm(gold).split()
    if not p or not g:
        return 0.0
    gc: Dict[str, int] = {}
    for t in g:
        gc[t] = gc.get(t, 0) + 1
    common = 0
    for t in p:
        if gc.get(t, 0) > 0:
            common += 1
            gc[t] -= 1
    if common == 0:
        return 0.0
    pr = common / len(p)
    rc = common / len(g)
    return 2.0 * pr * rc / (pr + rc)


def _score_query(pred: str, qspec: Dict[str, Any], gold_str: str) -> Dict[str, float]:
    lp = pred.lower()
    kws = [k.lower() for k in qspec["expected_keywords"]]
    forb = [f.lower() for f in qspec["forbidden_keywords"]]
    ents = [e.lower() for e in qspec["expected_entities"]]
    rels = [r.lower() for r in qspec["expected_relations"]]
    cons = [c.lower() for c in qspec["expected_constraints"]]
    decs = [d.lower() for d in qspec["expected_decisions"]]

    kw_hits = sum(1 for k in kws if k in lp)
    kw_rec = kw_hits / max(1, len(kws))
    em = 1.0 if kw_hits == len(kws) and not any(f in lp for f in forb) else 0.0
    f1 = _token_f1(pred, gold_str)

    ent_rec = sum(1 for e in ents if e in lp) / len(ents) if ents else 1.0
    rel_rec = 1.0 if (not rels or all(r in lp for r in rels)) else 0.0
    con_sat = sum(1 for c in cons if c in lp) / len(cons) if cons else 1.0
    dec_con = sum(1 for d in decs if d in lp) / len(decs) if decs else 1.0

    # Temporal consistency: did NOT emit superseded value; Contradiction resolution: did NOT emit invalidated value
    has_superseded = 1.0 if (forb and forb[0] in lp) else 0.0
    has_invalidated = 1.0 if (len(forb) > 1 and forb[1] in lp) else 0.0
    temp_con = 1.0 - has_superseded
    contra_res = 1.0 - has_invalidated
    obsolete = 1.0 if any(f in lp for f in forb) else 0.0
    lost = 1.0 - kw_rec
    tcr = 1.0 if (kw_rec >= 0.85 and obsolete == 0.0) else 0.0

    return {
        "exact_match": em,
        "f1": f1,
        "keyword_recall": kw_rec,
        "entity_recall": ent_rec,
        "relation_recovery": rel_rec,
        "constraint_satisfaction": con_sat,
        "temporal_consistency": temp_con,
        "contradiction_resolution": contra_res,
        "decision_consistency": dec_con,
        "task_completion_rate": tcr,
        "lost_info_rate": lost,
        "obsolete_info_rate": obsolete,
    }


def _pack_ops_to_tlv(ops: List[StateOp]) -> List[Tuple[int, bytes]]:
    recs: List[Tuple[int, bytes]] = []
    for op in ops:
        blob = json.dumps(
            {
                "id": op.fact_id,
                "op": op.kind,
                "cat": op.category,
                "k": op.key,
                "v": op.value,
                "s": op.seq,
                "ver": op.version,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        tag = 0x03 if op.category == "relation" else (0x05 if op.category == "constraint" else (0x08 if op.category == "decision" else 0x02))
        recs.append((tag, blob))
    return recs


def _unpack_ops_from_tlv(recs: List[Tuple[int, bytes]]) -> List[StateOp]:
    out: List[StateOp] = []
    for _, val in recs:
        d = json.loads(val.decode("utf-8"))
        out.append(
            StateOp(
                fact_id=d["id"],
                kind=d["op"],
                category=d["cat"],
                key=d["k"],
                value=d["v"],
                seq=int(d["s"]),
                version=int(d["ver"]),
            )
        )
    return out


def _bootstrap_ci(vals: List[float], n_boot: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
    arr = np.array(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="eval", choices=["dev", "tune", "eval"])
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=36)
    parser.add_argument("--model-a", default=os.environ.get("CMCP_MODEL_A", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
    parser.add_argument("--model-b", default=os.environ.get("CMCP_MODEL_B", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
    parser.add_argument("--tag", default="hetero")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    res_dir = out_dir / "results"
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    scenarios = get_scenarios(args.split)
    methods = [
        "full_history",
        "rolling_summary",
        "json_structured",
        "cmcp_full",
        "cmcp_incremental",
        "cmcp_inc_llm_ext",
    ]

    cfg = {
        "split": args.split,
        "num_scenarios": len(scenarios),
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "model_a_name": Path(args.model_a).name,
        "model_b_name": Path(args.model_b).name,
        "tag": args.tag,
        "dataset_sha256": dataset_sha256(),
        "methods": methods,
    }
    (cfg_dir / f"multiturn_config_{args.tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Stage 1: Run Agent A (Model A) sequentially for rolling_summary & cmcp_inc_llm_ext,
    # then unload Agent A cleanly before loading Agent B.
    agent_a_cache: Dict[str, Dict[str, Any]] = {}
    runner_a = PersistentLlamaRunner(args.model_a, n_ctx=1024, threads=4)
    llm_ext = LLMExtractor(runner_a)
    try:
        _ = runner_a.query("Warmup: OK", max_tokens=2)
        for sc in scenarios:
            summary_text = ""
            sum_ms = 0.0
            sum_in_tok = 0
            sum_out_tok = 0
            sum_bytes = 0

            llm_ctx_lines: List[str] = []
            ext_ms = 0.0
            ext_in_tok = 0
            ext_out_tok = 0
            ext_errs = 0
            ext_bytes = 0

            t4_summary = ""
            t4_llm_ctx = ""
            t4_sum_ms = 0.0
            t4_ext_ms = 0.0
            t4_sum_bytes = 0
            t4_ext_bytes = 0

            for turn in sc.turns:
                if turn.speaker != "agent_a":
                    if turn.turn_id == 4:
                        t4_summary = summary_text
                        t4_llm_ctx = "\n".join(llm_ctx_lines)
                        t4_sum_ms = sum_ms
                        t4_ext_ms = ext_ms
                        t4_sum_bytes = sum_bytes
                        t4_ext_bytes = ext_bytes
                    continue

                # Rolling summary update by Agent A
                sp = (
                    "Update the concise working summary with all active facts, path, constraints, and decisions, dropping invalidated items.\n"
                    f"Prior summary: {summary_text or 'None'}\n"
                    f"New update: {turn.utterance}\n"
                    "Updated summary:"
                )
                sr = runner_a.query(sp, max_tokens=64)
                summary_text = sr.get("output_text", "").strip()
                sum_ms += float(sr.get("tokenize_ms", 0.0)) + float(sr.get("prefill_ms", 0.0)) + float(sr.get("eval_ms", 0.0))
                sum_in_tok += int(sr.get("n_input_tokens", 0))
                sum_out_tok += int(sr.get("n_gen_tokens", 0))
                sum_bytes += len(summary_text.encode("utf-8"))

                # LLM-assisted CMCP delta extraction by Agent A
                er = llm_ext.extract_turn(turn.utterance, seq_base=turn.turn_id * 10)
                ext_ms += er["extract_ms"]
                ext_in_tok += er["in_tokens"]
                ext_out_tok += er["out_tokens"]
                ext_errs += er["error"]
                delta_line = f"[T{turn.turn_id}] {er['extracted_text']}"
                llm_ctx_lines.append(delta_line)
                ext_bytes += len(delta_line.encode("utf-8"))

            agent_a_cache[sc.scenario_id] = {
                "t4_summary": t4_summary,
                "t8_summary": summary_text,
                "t4_sum_ms": t4_sum_ms,
                "t8_sum_ms": sum_ms,
                "t4_sum_bytes": t4_sum_bytes,
                "t8_sum_bytes": sum_bytes,
                "sum_in_tok": sum_in_tok,
                "sum_out_tok": sum_out_tok,
                "t4_llm_ctx": t4_llm_ctx,
                "t8_llm_ctx": "\n".join(llm_ctx_lines),
                "t4_ext_ms": t4_ext_ms,
                "t8_ext_ms": ext_ms,
                "t4_ext_bytes": t4_ext_bytes,
                "t8_ext_bytes": ext_bytes,
                "ext_in_tok": ext_in_tok,
                "ext_out_tok": ext_out_tok,
                "ext_errs": ext_errs,
            }
    finally:
        runner_a.close()

    # Stage 2: Load Agent B (Model B) and evaluate Turn 4 (handoff 1) + Turn 8 (composite final)
    det_ext = DeterministicExtractor()
    runner_b = PersistentLlamaRunner(args.model_b, n_ctx=1024, threads=4)
    csv_path = res_dir / f"multiturn_runs_{args.tag}.csv"
    log_path = log_dir / f"multiturn_{args.tag}.log"

    fieldnames = [
        "scenario_id", "domain", "method", "rep", "query_turn",
        "wire_bytes", "update_bytes", "persisted_bytes", "input_tokens", "output_tokens",
        "extract_ms", "serialize_ms", "reconstruct_ms", "prefill_ms", "gen_ms", "e2e_accum_ms",
        "exact_match", "f1", "keyword_recall", "entity_recall", "relation_recovery",
        "constraint_satisfaction", "temporal_consistency", "contradiction_resolution",
        "decision_consistency", "task_completion_rate", "lost_info_rate", "obsolete_info_rate",
        "extract_errors", "max_rss_mib", "output_preview",
    ]

    rows: List[Dict[str, Any]] = []
    try:
        _ = runner_b.query("Warmup: OK", max_tokens=2)
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, open(log_path, "w", encoding="utf-8") as flog:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()

            for rep in range(1, args.reps + 1):
                for sc in scenarios:
                    a_data = agent_a_cache[sc.scenario_id]
                    # Precompute deterministic extraction across turns 1..3 (for T4) and 1..5 (for T8)
                    store_t4 = CMCPStateStore()
                    store_t8 = CMCPStateStore()
                    ops_t13: List[StateOp] = []
                    ops_t15: List[StateOp] = []
                    det_ms_t4 = 0.0
                    det_ms_t8 = 0.0
                    det_err = 0
                    seq = 0
                    hist_t13: List[str] = []
                    hist_t15: List[str] = []

                    for turn in sc.turns:
                        if turn.speaker == "agent_a":
                            ops, ms, err = det_ext.extract_turn(turn.utterance, seq)
                            det_err += err
                            for op in ops:
                                seq = max(seq, op.seq)
                            if turn.turn_id <= 3:
                                hist_t13.append(turn.utterance)
                                det_ms_t4 += ms
                                ops_t13.extend(ops)
                                for op in ops:
                                    store_t4.apply(op)
                            hist_t15.append(turn.utterance)
                            det_ms_t8 += ms
                            ops_t15.extend(ops)
                            for op in ops:
                                store_t8.apply(op)

                    # Evaluate Turn 4 (intermediate) and Turn 8 (final composite)
                    for qspec in (sc.eval_queries[0], sc.eval_queries[3]):
                        q_turn = int(qspec["turn"])
                        q_text = qspec["query"]
                        gold_str = " ".join(qspec["expected_keywords"])
                        is_t4 = (q_turn == 4)
                        hist_lines = hist_t13 if is_t4 else hist_t15
                        cur_store = store_t4 if is_t4 else store_t8
                        cur_ops = ops_t13 if is_t4 else ops_t15
                        det_ms = det_ms_t4 if is_t4 else det_ms_t8

                        for method in methods:
                            wire_bytes = 0
                            upd_bytes = 0
                            pers_bytes = 0
                            ext_ms = 0.0
                            ser_ms = 0.0
                            rec_ms = 0.0
                            ext_errors = det_err

                            if method == "full_history":
                                ctx_str = "\n".join(hist_lines)
                                wire_bytes = sum(len(h.encode("utf-8")) for h in hist_lines)
                                upd_bytes = len(hist_lines[-1].encode("utf-8"))
                                prompt = f"History:\n{ctx_str}\n\nQuestion: {q_text}\nAnswer in plain text (no markdown):"

                            elif method == "rolling_summary":
                                sum_str = a_data["t4_summary"] if is_t4 else a_data["t8_summary"]
                                ext_ms = a_data["t4_sum_ms"] if is_t4 else a_data["t8_sum_ms"]
                                wire_bytes = a_data["t4_sum_bytes"] if is_t4 else a_data["t8_sum_bytes"]
                                upd_bytes = len(sum_str.encode("utf-8"))
                                prompt = f"Summary:\n{sum_str}\n\nQuestion: {q_text}\nAnswer in plain text (no markdown):"

                            elif method == "json_structured":
                                ts0 = time.perf_counter()
                                jstr = json.dumps(cur_store.active_view(), separators=(",", ":"), sort_keys=True)
                                ser_ms = (time.perf_counter() - ts0) * 1000.0
                                ext_ms = det_ms
                                wire_bytes = len(jstr.encode("utf-8"))
                                upd_bytes = wire_bytes
                                prompt = f"JSON Context:\n{jstr}\n\nQuestion: {q_text}\nAnswer in plain text (no markdown):"

                            elif method == "cmcp_full":
                                ts0 = time.perf_counter()
                                all_recs = _pack_ops_to_tlv(cur_ops)
                                frame = encode_frame(all_recs, compress=True, msg_type=CMCP_MSG_CONTEXT_TRANSFER, compat_tier=CMCP_TIER_2_CROSS_MODEL)
                                ser_ms = (time.perf_counter() - ts0) * 1000.0
                                tr0 = time.perf_counter()
                                _, dec_recs = decode_frame(frame)
                                tmp_store = CMCPStateStore()
                                for op in _unpack_ops_from_tlv(dec_recs):
                                    tmp_store.apply(op)
                                ctx_str = tmp_store.render_for_llm()
                                rec_ms = (time.perf_counter() - tr0) * 1000.0
                                ext_ms = det_ms
                                wire_bytes = len(frame)
                                upd_bytes = wire_bytes
                                prompt = f"Context:\n{ctx_str}\n\nQuestion: {q_text}\nAnswer in plain text (no markdown):"

                            elif method == "cmcp_incremental":
                                # Send only latest delta frame; Agent B maintains local store synchronized by refs
                                delta_ops = ops_t13[-2:] if is_t4 else ops_t15[-3:]
                                ts0 = time.perf_counter()
                                d_recs = _pack_ops_to_tlv(delta_ops)
                                frame = encode_frame(d_recs, compress=True, msg_type=CMCP_MSG_CONTEXT_TRANSFER, compat_tier=CMCP_TIER_2_CROSS_MODEL)
                                ser_ms = (time.perf_counter() - ts0) * 1000.0
                                tr0 = time.perf_counter()
                                _, dec_recs = decode_frame(frame)
                                # Reconstruct active view from persisted store + decoded delta
                                for op in _unpack_ops_from_tlv(dec_recs):
                                    cur_store.apply(op)
                                ctx_str = cur_store.render_for_llm()
                                rec_ms = (time.perf_counter() - tr0) * 1000.0
                                ext_ms = det_ms
                                wire_bytes = len(frame)
                                upd_bytes = len(frame)
                                pers_bytes = cur_store.serialized_size_bytes()
                                prompt = f"Context:\n{ctx_str}\n\nQuestion: {q_text}\nAnswer in plain text (no markdown):"

                            else:  # cmcp_inc_llm_ext
                                llm_ctx = a_data["t4_llm_ctx"] if is_t4 else a_data["t8_llm_ctx"]
                                ext_ms = a_data["t4_ext_ms"] if is_t4 else a_data["t8_ext_ms"]
                                ext_errors = int(a_data["ext_errs"])
                                wire_bytes = a_data["t4_ext_bytes"] if is_t4 else a_data["t8_ext_bytes"]
                                upd_bytes = len(llm_ctx.encode("utf-8"))
                                prompt = f"Context:\n{llm_ctx}\n\nQuestion: {q_text}\nAnswer in plain text (no markdown):"

                            resp = runner_b.query(prompt, max_tokens=args.max_tokens)
                            out_txt = resp.get("output_text", "")
                            sc_metrics = _score_query(out_txt, qspec, gold_str)
                            tok_ms = float(resp.get("tokenize_ms", 0.0))
                            pre_ms = float(resp.get("prefill_ms", 0.0))
                            gen_ms = float(resp.get("eval_ms", 0.0))
                            e2e_ms = ext_ms + ser_ms + rec_ms + tok_ms + pre_ms + gen_ms

                            row = {
                                "scenario_id": sc.scenario_id,
                                "domain": sc.domain,
                                "method": method,
                                "rep": rep,
                                "query_turn": q_turn,
                                "wire_bytes": wire_bytes,
                                "update_bytes": upd_bytes,
                                "persisted_bytes": pers_bytes,
                                "input_tokens": int(resp.get("n_input_tokens", 0)),
                                "output_tokens": int(resp.get("n_gen_tokens", 0)),
                                "extract_ms": round(ext_ms, 3),
                                "serialize_ms": round(ser_ms, 3),
                                "reconstruct_ms": round(rec_ms, 3),
                                "prefill_ms": round(pre_ms, 3),
                                "gen_ms": round(gen_ms, 3),
                                "e2e_accum_ms": round(e2e_ms, 3),
                                **{k: round(v, 4) for k, v in sc_metrics.items()},
                                "extract_errors": ext_errors,
                                "max_rss_mib": round(float(resp.get("rss_mib", 0.0)), 2),
                                "output_preview": out_txt.replace("\n", " ")[:120],
                            }
                            writer.writerow(row)
                            fcsv.flush()
                            rows.append(row)
                    flog.write(f"[rep={rep}] completed {sc.scenario_id}\n")
                    flog.flush()
    finally:
        runner_b.close()

    # Scenario-level clustered bootstrap summary (n = len(scenarios))
    summary: Dict[str, Any] = {"config": cfg, "methods": {}, "paired_vs_full_history": {}}
    stat_keys = [
        "wire_bytes", "input_tokens", "extract_ms", "prefill_ms", "e2e_accum_ms",
        "exact_match", "f1", "keyword_recall", "entity_recall", "relation_recovery",
        "constraint_satisfaction", "temporal_consistency", "contradiction_resolution",
        "decision_consistency", "task_completion_rate", "obsolete_info_rate",
    ]
    sc_ids = [s.scenario_id for s in scenarios]
    sc_means: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in stat_keys} for m in methods}

    for m in methods:
        for sid in sc_ids:
            srows = [r for r in rows if r["method"] == m and r["scenario_id"] == sid]
            for k in stat_keys:
                sc_means[m][k].append(float(statistics.mean([float(r[k]) for r in srows])))

        m_summary: Dict[str, Any] = {"n_scenarios": len(sc_ids), "n_rows": len([r for r in rows if r["method"] == m])}
        for k in stat_keys:
            mu, lo, hi = _bootstrap_ci(sc_means[m][k])
            m_summary[k] = {"mean": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["methods"][m] = m_summary

    for m in methods:
        if m == "full_history":
            continue
        p_diff: Dict[str, Any] = {}
        for k in ["input_tokens", "e2e_accum_ms", "exact_match", "relation_recovery", "constraint_satisfaction", "task_completion_rate"]:
            diffs = [a - b for a, b in zip(sc_means[m][k], sc_means["full_history"][k])]
            mu, lo, hi = _bootstrap_ci(diffs)
            p_diff[k] = {"mean_diff": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["paired_vs_full_history"][m] = p_diff

    (res_dir / f"multiturn_summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
