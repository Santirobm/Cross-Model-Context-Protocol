"""CMCP 2.2b Gate 2: Heterogeneous Semantic Portability Benchmark.

Separates the model-agnostic binary TLV wire format from the semantic rendering
strategy (R1, R2, R3) in CMCPStateStore.render_for_llm(style=...).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cmcp22.dataset import dataset_sha256, get_scenarios
from experiments.cmcp22.extractors import DeterministicExtractor
from experiments.cmcp22.run_multiturn_benchmark import (
    PersistentLlamaRunner,
    _bootstrap_ci,
    _pack_ops_to_tlv,
    _score_query,
    _unpack_ops_from_tlv,
)
from representations.binary import (
    CMCP_MSG_CONTEXT_TRANSFER,
    CMCP_TIER_2_CROSS_MODEL,
    decode_frame,
    encode_frame,
)
from representations.state_store import CMCPStateStore, StateOp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tune", choices=["dev", "tune", "eval"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model-a", default=os.environ.get("CMCP_MODEL_A", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
    ap.add_argument("--model-b", default=os.environ.get("CMCP_MODEL_B", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--styles", nargs="+", default=["r1", "r2", "r3"])
    ap.add_argument("--tag", default="port_tune")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    res_dir = out_dir / "results"
    cfg_dir = out_dir / "configs"
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    scenarios = get_scenarios(args.split)
    methods = ["full_history"] + [f"cmcp_{s}" for s in args.styles]

    cfg = {
        "split": args.split,
        "num_scenarios": len(scenarios),
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "model_a_name": Path(args.model_a).name,
        "model_b_name": Path(args.model_b).name,
        "styles": args.styles,
        "tag": args.tag,
        "dataset_sha256": dataset_sha256(),
        "bootstrap_seed": 42,
        "decoding": "greedy",
        "methods": methods,
    }
    (cfg_dir / f"portability_config_{args.tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    det_ext = DeterministicExtractor()
    runner_b = PersistentLlamaRunner(args.model_b, n_ctx=1024, threads=4)
    csv_path = res_dir / f"portability_runs_{args.tag}.csv"

    fieldnames = [
        "scenario_id", "domain", "method", "rep", "query_turn",
        "wire_bytes", "input_tokens", "output_tokens",
        "extract_ms", "serialize_ms", "reconstruct_ms", "prefill_ms", "decode_ms", "e2e_accum_ms",
        "exact_match", "relation_recovery", "constraint_satisfaction", "obsolete_info_rate",
        "output_preview",
    ]
    rows: List[Dict[str, Any]] = []
    try:
        _ = runner_b.query("Warmup: OK", max_tokens=2)
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for rep in range(1, args.reps + 1):
                for sc in scenarios:
                    store_t4, store_t8 = CMCPStateStore(), CMCPStateStore()
                    ops_t13, ops_t15 = [], []
                    det_ms_t4, det_ms_t8 = 0.0, 0.0
                    hist_t13, hist_t15 = [], []
                    seq = 0
                    for turn in sc.turns:
                        if turn.speaker == "agent_a":
                            ops, ms, _ = det_ext.extract_turn(turn.utterance, seq)
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

                    for qspec in (sc.eval_queries[0], sc.eval_queries[3]):
                        q_turn = int(qspec["turn"])
                        q_text = qspec["query"]
                        gold_str = " ".join(qspec["expected_keywords"])
                        is_t4 = (q_turn == 4)
                        hist_lines = hist_t13 if is_t4 else hist_t15
                        base_store = store_t4 if is_t4 else store_t8
                        delta_ops = ops_t13[-2:] if is_t4 else ops_t15[-3:]
                        det_ms = det_ms_t4 if is_t4 else det_ms_t8

                        for method in methods:
                            if method == "full_history":
                                ctx_str = "\n".join(hist_lines)
                                wire_bytes = sum(len(h.encode("utf-8")) for h in hist_lines)
                                ext_ms, ser_ms, rec_ms = 0.0, 0.0, 0.0
                                prompt = f"History:\n{ctx_str}\n\nQuestion: {q_text}\nAnswer in plain text (no markdown):"
                            else:
                                style = method.replace("cmcp_", "")
                                ts0 = time.perf_counter()
                                frame = encode_frame(
                                    _pack_ops_to_tlv(delta_ops),
                                    compress=True,
                                    msg_type=CMCP_MSG_CONTEXT_TRANSFER,
                                    compat_tier=CMCP_TIER_2_CROSS_MODEL,
                                )
                                ser_ms = (time.perf_counter() - ts0) * 1000.0
                                tr0 = time.perf_counter()
                                _, dec_recs = decode_frame(frame)
                                for op in _unpack_ops_from_tlv(dec_recs):
                                    base_store.apply(op)
                                ctx_str = base_store.render_for_llm(style=style)
                                rec_ms = (time.perf_counter() - tr0) * 1000.0
                                ext_ms = det_ms
                                wire_bytes = len(frame)
                                prompt = f"Context:\n{ctx_str}\n\nQuestion: {q_text}\nAnswer in plain text, one concise sentence:"

                            resp = runner_b.query(prompt, max_tokens=args.max_tokens)
                            out_txt = resp.get("output_text", "")
                            sc_m = _score_query(out_txt, qspec, gold_str)
                            tok_ms = float(resp.get("tokenize_ms", 0.0))
                            pre_ms = float(resp.get("prefill_ms", 0.0))
                            dec_ms = float(resp.get("eval_ms", 0.0))
                            e2e_ms = ext_ms + ser_ms + rec_ms + tok_ms + pre_ms + dec_ms

                            row = {
                                "scenario_id": sc.scenario_id,
                                "domain": sc.domain,
                                "method": method,
                                "rep": rep,
                                "query_turn": q_turn,
                                "wire_bytes": wire_bytes,
                                "input_tokens": int(resp.get("n_input_tokens", 0)),
                                "output_tokens": int(resp.get("n_gen_tokens", 0)),
                                "extract_ms": round(ext_ms, 3),
                                "serialize_ms": round(ser_ms, 3),
                                "reconstruct_ms": round(rec_ms, 3),
                                "prefill_ms": round(pre_ms, 3),
                                "decode_ms": round(dec_ms, 3),
                                "e2e_accum_ms": round(e2e_ms, 3),
                                "exact_match": round(sc_m["exact_match"], 4),
                                "relation_recovery": round(sc_m["relation_recovery"], 4),
                                "constraint_satisfaction": round(sc_m["constraint_satisfaction"], 4),
                                "obsolete_info_rate": round(sc_m["obsolete_info_rate"], 4),
                                "output_preview": out_txt.replace("\n", " ")[:120],
                            }
                            writer.writerow(row)
                            fcsv.flush()
                            rows.append(row)
    finally:
        runner_b.close()

    stat_keys = [
        "wire_bytes", "input_tokens", "output_tokens",
        "prefill_ms", "decode_ms", "e2e_accum_ms",
        "exact_match", "relation_recovery", "constraint_satisfaction", "obsolete_info_rate",
    ]
    sc_ids = [s.scenario_id for s in scenarios]
    sc_means: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in stat_keys} for m in methods}
    summary: Dict[str, Any] = {"config": cfg, "methods": {}, "paired_vs_full_history": {}}

    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        for sid in sc_ids:
            srows = [r for r in m_rows if r["scenario_id"] == sid]
            for k in stat_keys:
                sc_means[m][k].append(float(statistics.mean([float(r[k]) for r in srows])))
        m_sum: Dict[str, Any] = {"n_scenarios": len(sc_ids), "n_rows": len(m_rows)}
        for k in stat_keys:
            mu, lo, hi = _bootstrap_ci(sc_means[m][k])
            entry: Dict[str, Any] = {"mean": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            if k in ("prefill_ms", "decode_ms", "e2e_accum_ms", "output_tokens"):
                raw_vals = [float(r[k]) for r in m_rows]
                entry["p50"] = round(float(np.percentile(raw_vals, 50)), 4)
                entry["p95"] = round(float(np.percentile(raw_vals, 95)), 4)
                entry["std"] = round(float(np.std(raw_vals, ddof=1)) if len(raw_vals) > 1 else 0.0, 4)
            m_sum[k] = entry
        summary["methods"][m] = m_sum

    for m in methods:
        if m == "full_history":
            continue
        p_diff: Dict[str, Any] = {}
        for k in ["input_tokens", "output_tokens", "prefill_ms", "e2e_accum_ms", "exact_match", "relation_recovery", "constraint_satisfaction"]:
            diffs = [a - b for a, b in zip(sc_means[m][k], sc_means["full_history"][k])]
            mu, lo, hi = _bootstrap_ci(diffs)
            p_diff[k] = {"mean_diff": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["paired_vs_full_history"][m] = p_diff

    (res_dir / f"portability_summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
