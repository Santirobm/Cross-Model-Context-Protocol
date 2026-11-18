"""CMCP 2.2b Gate 3: Extraction Benchmark on Unstructured Natural-Language Dialogues.

Compares 4 approaches across 30 8-turn unstructured dialogues:
  E1: oracle_structured (tool_use / native schema upper bound)
  E2: deterministic_nlp_rules (CPU regex/morphosyntactic rules, no LLM)
  E3: hybrid_gated (E2 + selective gated LLM JSON extraction on uncertain turns)
  E4: full_history_baseline (raw dialogue history, no extraction)
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
from typing import Any, Dict, List, Set

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cmcp22.run_multiturn_benchmark import (
    PersistentLlamaRunner,
    _bootstrap_ci,
    _pack_ops_to_tlv,
    _score_query,
    _unpack_ops_from_tlv,
)
from experiments.cmcp22b.extractors_nlp import (
    DeterministicNLPExtractor,
    HybridGatedExtractor,
    OracleStructuredExtractor,
)
from representations.binary import (
    CMCP_MSG_CONTEXT_TRANSFER,
    CMCP_TIER_2_CROSS_MODEL,
    decode_frame,
    encode_frame,
)
from representations.state_store import CMCPStateStore, StateOp


def _active_values(store: CMCPStateStore) -> Set[str]:
    return {f"{v['category']}:{v['key']}={v['value']}".lower() for v in store.active_view().values()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/cmcp22b/data/unstructured_dialogues.json")
    ap.add_argument("--model-a", default=os.environ.get("CMCP_MODEL_A", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
    ap.add_argument("--model-b", default=os.environ.get("CMCP_MODEL_B", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--style", default="r2")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tag", default="unstruct_v1")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    res_dir = out_dir / "results"
    cfg_dir = out_dir / "configs"
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    dialogues = json.loads((ROOT / args.data).read_text(encoding="utf-8"))
    methods = ["oracle_structured", "deterministic_nlp_rules", "hybrid_gated", "full_history_baseline"]

    cfg = {
        "num_dialogues": len(dialogues),
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "style": args.style,
        "model_a_name": Path(args.model_a).name,
        "model_b_name": Path(args.model_b).name,
        "tag": args.tag,
        "bootstrap_seed": 42,
        "methods": methods,
    }
    (cfg_dir / f"unstructured_config_{args.tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Stage 1: Run E3 (hybrid_gated) extraction with Agent A, then unload Agent A cleanly
    runner_a = PersistentLlamaRunner(args.model_a, n_ctx=1024, threads=4)
    e3_cache: Dict[str, Dict[str, Any]] = {}
    try:
        _ = runner_a.query("Warmup: OK", max_tokens=2)
        hyb = HybridGatedExtractor(runner_a)
        for dlg in dialogues:
            st = CMCPStateStore()
            ops_all: List[StateOp] = []
            tot_ms = 0.0
            tot_errs = 0
            seq = 0
            calls_before = hyb.gated_calls
            for t in dlg["turns"]:
                if t["speaker"] == "agent_a":
                    ops, ms, err = hyb.extract_turn(t["utterance"], seq)
                    tot_ms += ms
                    tot_errs += err
                    for op in ops:
                        seq = max(seq, op.seq)
                        st.apply(op)
                    ops_all.extend(ops)
            e3_cache[dlg["dialogue_id"]] = {
                "ops": ops_all,
                "extract_ms": tot_ms,
                "extract_errors": tot_errs,
                "gated_calls": hyb.gated_calls - calls_before,
                "rendered": st.render_for_llm(style=args.style),
                "active_vals": _active_values(st),
            }
    finally:
        runner_a.close()

    # Stage 2: Load Agent B and evaluate all 4 approaches
    e1_ext = OracleStructuredExtractor()
    e2_ext = DeterministicNLPExtractor()
    runner_b = PersistentLlamaRunner(args.model_b, n_ctx=1024, threads=4)
    csv_path = res_dir / f"unstructured_runs_{args.tag}.csv"
    fieldnames = [
        "dialogue_id", "domain", "method", "rep",
        "wire_bytes", "input_tokens", "output_tokens",
        "extract_ms", "prefill_ms", "decode_ms", "e2e_accum_ms",
        "state_precision", "state_recall", "obsolete_pruning_accuracy",
        "exact_match", "relation_recovery", "constraint_satisfaction", "obsolete_info_rate",
        "gated_calls", "output_preview",
    ]
    rows: List[Dict[str, Any]] = []
    try:
        _ = runner_b.query("Warmup: OK", max_tokens=2)
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for rep in range(1, args.reps + 1):
                for dlg in dialogues:
                    did = dlg["dialogue_id"]
                    qspec = dlg["eval_query"]
                    gold_str = " ".join(qspec["expected_keywords"])
                    forbidden = [f.lower() for f in qspec.get("forbidden_keywords", [])]

                    # Oracle E1 state (reference for state precision/recall)
                    st_e1 = CMCPStateStore()
                    ops_e1: List[StateOp] = []
                    ms_e1 = 0.0
                    seq1 = 0
                    hist_lines: List[str] = []
                    for t in dlg["turns"]:
                        if t["speaker"] == "agent_a":
                            hist_lines.append(t["utterance"])
                            ops, ms, _ = e1_ext.extract_turn(t, seq1)
                            ms_e1 += ms
                            for op in ops:
                                seq1 = max(seq1, op.seq)
                                st_e1.apply(op)
                            ops_e1.extend(ops)
                    oracle_vals = _active_values(st_e1)

                    # E2 deterministic NLP state
                    st_e2 = CMCPStateStore()
                    ops_e2: List[StateOp] = []
                    ms_e2 = 0.0
                    seq2 = 0
                    for t in dlg["turns"]:
                        if t["speaker"] == "agent_a":
                            ops, ms, _ = e2_ext.extract_turn(t["utterance"], seq2)
                            ms_e2 += ms
                            for op in ops:
                                seq2 = max(seq2, op.seq)
                                st_e2.apply(op)
                            ops_e2.extend(ops)

                    for method in methods:
                        gated_calls = 0
                        if method == "full_history_baseline":
                            ctx_str = "\n".join(hist_lines)
                            wire_bytes = len(ctx_str.encode("utf-8"))
                            ext_ms, ser_ms, rec_ms = 0.0, 0.0, 0.0
                            s_prec, s_rec, obs_acc = 1.0, 1.0, 0.0
                            prompt = f"History:\n{ctx_str}\n\nQuestion: {qspec['query']}\nAnswer in plain text, one concise sentence:"
                        else:
                            if method == "oracle_structured":
                                ops_use, ext_ms, cur_st = ops_e1, ms_e1, st_e1
                                pred_vals = oracle_vals
                            elif method == "deterministic_nlp_rules":
                                ops_use, ext_ms, cur_st = ops_e2, ms_e2, st_e2
                                pred_vals = _active_values(st_e2)
                            else:
                                c3 = e3_cache[did]
                                ops_use, ext_ms = c3["ops"], c3["extract_ms"]
                                gated_calls = int(c3["gated_calls"])
                                pred_vals = c3["active_vals"]
                                cur_st = None

                            tp = len(pred_vals & oracle_vals)
                            s_prec = tp / len(pred_vals) if pred_vals else 0.0
                            s_rec = tp / len(oracle_vals) if oracle_vals else 1.0
                            rendered = cur_st.render_for_llm(style=args.style) if cur_st is not None else e3_cache[did]["rendered"]
                            obs_hits = sum(1 for fk in forbidden if fk in rendered.lower())
                            obs_acc = 1.0 - (obs_hits / len(forbidden)) if forbidden else 1.0

                            ts0 = time.perf_counter()
                            frame = encode_frame(_pack_ops_to_tlv(ops_use[-3:] if len(ops_use) >= 3 else ops_use), compress=True, msg_type=CMCP_MSG_CONTEXT_TRANSFER, compat_tier=CMCP_TIER_2_CROSS_MODEL)
                            ser_ms = (time.perf_counter() - ts0) * 1000.0
                            tr0 = time.perf_counter()
                            _ = decode_frame(frame)
                            rec_ms = (time.perf_counter() - tr0) * 1000.0
                            wire_bytes = len(frame)
                            prompt = f"Context:\n{rendered}\n\nQuestion: {qspec['query']}\nAnswer in plain text, one concise sentence:"

                        resp = runner_b.query(prompt, max_tokens=args.max_tokens)
                        out_txt = resp.get("output_text", "")
                        sc_m = _score_query(out_txt, qspec, gold_str)
                        tok_ms = float(resp.get("tokenize_ms", 0.0))
                        pre_ms = float(resp.get("prefill_ms", 0.0))
                        dec_ms = float(resp.get("eval_ms", 0.0))
                        e2e_ms = ext_ms + ser_ms + rec_ms + tok_ms + pre_ms + dec_ms

                        row = {
                            "dialogue_id": did,
                            "domain": dlg["domain"],
                            "method": method,
                            "rep": rep,
                            "wire_bytes": wire_bytes,
                            "input_tokens": int(resp.get("n_input_tokens", 0)),
                            "output_tokens": int(resp.get("n_gen_tokens", 0)),
                            "extract_ms": round(ext_ms, 3),
                            "prefill_ms": round(pre_ms, 3),
                            "decode_ms": round(dec_ms, 3),
                            "e2e_accum_ms": round(e2e_ms, 3),
                            "state_precision": round(s_prec, 4),
                            "state_recall": round(s_rec, 4),
                            "obsolete_pruning_accuracy": round(obs_acc, 4),
                            "exact_match": round(sc_m["exact_match"], 4),
                            "relation_recovery": round(sc_m["relation_recovery"], 4),
                            "constraint_satisfaction": round(sc_m["constraint_satisfaction"], 4),
                            "obsolete_info_rate": round(sc_m["obsolete_info_rate"], 4),
                            "gated_calls": gated_calls,
                            "output_preview": out_txt.replace("\n", " ")[:120],
                        }
                        writer.writerow(row)
                        fcsv.flush()
                        rows.append(row)
    finally:
        runner_b.close()

    stat_keys = [
        "wire_bytes", "input_tokens", "output_tokens", "extract_ms", "prefill_ms", "decode_ms", "e2e_accum_ms",
        "state_precision", "state_recall", "obsolete_pruning_accuracy",
        "exact_match", "relation_recovery", "constraint_satisfaction", "obsolete_info_rate",
    ]
    dids = [d["dialogue_id"] for d in dialogues]
    summary: Dict[str, Any] = {"config": cfg, "methods": {}, "paired_vs_full_history": {}}
    sc_means: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in stat_keys} for m in methods}

    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        for did in dids:
            drows = [r for r in m_rows if r["dialogue_id"] == did]
            for k in stat_keys:
                sc_means[m][k].append(float(statistics.mean([float(r[k]) for r in drows])))
        m_sum: Dict[str, Any] = {"n_dialogues": len(dids), "n_rows": len(m_rows)}
        for k in stat_keys:
            mu, lo, hi = _bootstrap_ci(sc_means[m][k])
            m_sum[k] = {"mean": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["methods"][m] = m_sum

    for m in methods:
        if m == "full_history_baseline":
            continue
        p_diff: Dict[str, Any] = {}
        for k in ["input_tokens", "e2e_accum_ms", "exact_match", "relation_recovery", "constraint_satisfaction"]:
            diffs = [a - b for a, b in zip(sc_means[m][k], sc_means["full_history_baseline"][k])]
            mu, lo, hi = _bootstrap_ci(diffs)
            p_diff[k] = {"mean_diff": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["paired_vs_full_history"][m] = p_diff

    (res_dir / f"unstructured_summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
