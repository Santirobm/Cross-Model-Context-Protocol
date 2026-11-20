"""GATE 1 Experimental Benchmark: Baseline R3 vs Variants A, B (A+B), D (A+B+D), and Ablation D-only.

Evaluates all 4 query types (T4, T6, T7, T8) with real llama_tokenize() token counts,
zero extra LLM calls, explicit EvidenceStore fallback tracking, and scenario-clustered
bootstrap 95% CIs.
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

from experiments.cmcp22.dataset import dataset_sha256, get_scenarios  # noqa: E402
from experiments.cmcp22.extractors import DeterministicExtractor  # noqa: E402
from experiments.cmcp22.run_multiturn_benchmark import (  # noqa: E402
    PersistentLlamaRunner,
    _bootstrap_ci,
    _pack_ops_to_tlv,
    _score_query,
    _unpack_ops_from_tlv,
)
from representations.binary import (  # noqa: E402
    CMCP_MSG_CONTEXT_TRANSFER,
    CMCP_TIER_2_CROSS_MODEL,
    decode_frame,
    encode_frame,
)
from representations.selector import select_for_task  # noqa: E402
from representations.state_store import CMCPStateStore, StateOp  # noqa: E402


def _build_store_up_to(turns: List[Any], max_turn: int, det_ext: DeterministicExtractor) -> tuple[CMCPStateStore, List[StateOp], float]:
    st = CMCPStateStore()
    ops_all: List[StateOp] = []
    tot_ms = 0.0
    seq = 0
    for turn in turns:
        if turn.speaker == "agent_a" and turn.turn_id <= max_turn:
            st.evidence.register(f"turn:{turn.turn_id}", turn.utterance)
            ops, ms, _ = det_ext.extract_turn(turn.utterance, seq)
            tot_ms += ms
            for op in ops:
                seq = max(seq, op.seq)
                # Attach evidence_ref to StateOp (Component A)
                op_ev = StateOp(
                    fact_id=op.fact_id,
                    kind=op.kind,
                    category=op.category,
                    key=op.key,
                    value=op.value,
                    seq=op.seq,
                    version=op.version,
                    agent_id=op.agent_id,
                    timestamp=op.timestamp,
                    parent_state_hash=op.parent_state_hash,
                    source="deterministic_extractor",
                    confidence=1.0,
                    evidence_ref=f"turn:{turn.turn_id}",
                )
                st.apply(op_ev)
                ops_all.append(op_ev)
    return st, ops_all, tot_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["dev", "tune", "eval"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model-b", default=os.environ.get("CMCP_MODEL_B", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--variants", nargs="+", default=["baseline_r3", "var_a", "var_b", "var_d", "ablation_d_only"])
    ap.add_argument("--tag", default="gate1_pair1")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    res_dir = out_dir / "results"
    cfg_dir = out_dir / "configs"
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    scenarios = get_scenarios(args.split)
    variants = args.variants
    cfg = {
        "split": args.split,
        "num_scenarios": len(scenarios),
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "model_b_name": Path(args.model_b).name,
        "variants": variants,
        "tag": args.tag,
        "dataset_sha256": dataset_sha256(),
        "bootstrap_seed": 42,
        "decoding": "greedy",
        "extra_llm_calls": 0,
    }
    (cfg_dir / f"gate1_config_{args.tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    det_ext = DeterministicExtractor()
    runner_b = PersistentLlamaRunner(args.model_b, n_ctx=1024, threads=4)
    csv_path = res_dir / f"gate1_runs_{args.tag}.csv"
    fieldnames = [
        "scenario_id", "domain", "variant", "rep", "query_turn",
        "wire_bytes", "input_tokens", "output_tokens", "net_tokens",
        "extract_ms", "select_ms", "serialize_ms", "prefill_ms", "decode_ms", "e2e_accum_ms",
        "exact_match", "entity_recall", "relation_recovery", "constraint_satisfaction",
        "obsolete_info_rate", "evidence_fallback_used", "selected_count",
    ]
    rows: List[Dict[str, Any]] = []
    try:
        _ = runner_b.query("Warmup: OK", max_tokens=2)
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for rep in range(1, args.reps + 1):
                for sc in scenarios:
                    for qspec in sc.eval_queries:
                        q_turn = int(qspec["turn"])
                        q_text = qspec["query"]
                        gold_str = " ".join(qspec["expected_keywords"])
                        # Build store from agent_a turns preceding query turn
                        st, ops, ext_ms = _build_store_up_to(sc.turns, q_turn - 1, det_ext)
                        delta_ops = ops[-3:] if len(ops) >= 3 else ops

                        for var in variants:
                            ts0 = time.perf_counter()
                            frame = encode_frame(
                                _pack_ops_to_tlv(delta_ops),
                                compress=True,
                                msg_type=CMCP_MSG_CONTEXT_TRANSFER,
                                compat_tier=CMCP_TIER_2_CROSS_MODEL,
                            )
                            _, dec_recs = decode_frame(frame)
                            _ = _unpack_ops_from_tlv(dec_recs)
                            ser_ms = (time.perf_counter() - ts0) * 1000.0

                            tsel0 = time.perf_counter()
                            ev_used = 0
                            sel_cnt = len(st.active_view())
                            if var == "baseline_r3":
                                ctx_str = st.render_for_llm(style="r3")
                            elif var == "var_a":
                                ctx_str = st.render_for_llm(style="r3_canon")
                            elif var == "var_b":
                                # A + explicit typed relation edges appended compactly
                                base_txt = st.render_for_llm(style="r3_canon")
                                rel_edges = [
                                    f"{r.rel_type}:{st._entries[r.src_id].key}->{st._entries[r.dst_id].key}"
                                    for r in st.active_relations()
                                    if r.rel_type == "ROUTES_TO" and r.src_id in st._entries and r.dst_id in st._entries
                                ]
                                ctx_str = base_txt + ("\n<edges>" + "; ".join(rel_edges) + "</edges>" if rel_edges else "")
                            elif var == "var_d":
                                sel = select_for_task(q_text, st, use_relations=True, use_evidence_fallback=True, canonicalize=True)
                                ctx_str = sel.rendered_text
                                ev_used = 1 if sel.used_evidence_fallback else 0
                                sel_cnt = len(sel.selected_ids)
                            else:  # ablation_d_only
                                sel = select_for_task(q_text, st, use_relations=False, use_evidence_fallback=False, canonicalize=False)
                                ctx_str = sel.rendered_text
                                sel_cnt = len(sel.selected_ids)
                            select_ms = (time.perf_counter() - tsel0) * 1000.0

                            prompt = f"Context:\n{ctx_str}\n\nQuestion: {q_text}\nAnswer in plain text, one concise sentence:"
                            resp = runner_b.query(prompt, max_tokens=args.max_tokens)
                            out_txt = resp.get("output_text", "")
                            sc_m = _score_query(out_txt, qspec, gold_str)
                            in_tok = int(resp.get("n_input_tokens", 0))
                            out_tok = int(resp.get("n_gen_tokens", 0))
                            tok_ms = float(resp.get("tokenize_ms", 0.0))
                            pre_ms = float(resp.get("prefill_ms", 0.0))
                            dec_ms = float(resp.get("eval_ms", 0.0))
                            e2e_ms = ext_ms + ser_ms + select_ms + tok_ms + pre_ms + dec_ms

                            row = {
                                "scenario_id": sc.scenario_id,
                                "domain": sc.domain,
                                "variant": var,
                                "rep": rep,
                                "query_turn": q_turn,
                                "wire_bytes": len(frame),
                                "input_tokens": in_tok,
                                "output_tokens": out_tok,
                                "net_tokens": in_tok + out_tok,
                                "extract_ms": round(ext_ms, 3),
                                "select_ms": round(select_ms, 4),
                                "serialize_ms": round(ser_ms, 4),
                                "prefill_ms": round(pre_ms, 3),
                                "decode_ms": round(dec_ms, 3),
                                "e2e_accum_ms": round(e2e_ms, 3),
                                "exact_match": round(sc_m["exact_match"], 4),
                                "entity_recall": round(sc_m["entity_recall"], 4),
                                "relation_recovery": round(sc_m["relation_recovery"], 4),
                                "constraint_satisfaction": round(sc_m["constraint_satisfaction"], 4),
                                "obsolete_info_rate": round(sc_m["obsolete_info_rate"], 4),
                                "evidence_fallback_used": ev_used,
                                "selected_count": sel_cnt,
                            }
                            writer.writerow(row)
                            fcsv.flush()
                            rows.append(row)
    finally:
        runner_b.close()

    stat_keys = [
        "input_tokens", "output_tokens", "net_tokens",
        "select_ms", "prefill_ms", "decode_ms", "e2e_accum_ms",
        "exact_match", "entity_recall", "relation_recovery",
        "constraint_satisfaction", "obsolete_info_rate", "evidence_fallback_used",
    ]
    sc_ids = [s.scenario_id for s in scenarios]
    sc_means: Dict[str, Dict[str, List[float]]] = {v: {k: [] for k in stat_keys} for v in variants}
    summary: Dict[str, Any] = {"config": cfg, "variants": {}, "paired_vs_baseline_r3": {}}

    for v in variants:
        v_rows = [r for r in rows if r["variant"] == v]
        for sid in sc_ids:
            srows = [r for r in v_rows if r["scenario_id"] == sid]
            for k in stat_keys:
                sc_means[v][k].append(float(statistics.mean([float(r[k]) for r in srows])))
        v_sum: Dict[str, Any] = {"n_scenarios": len(sc_ids), "n_rows": len(v_rows)}
        for k in stat_keys:
            mu, lo, hi = _bootstrap_ci(sc_means[v][k])
            v_sum[k] = {"mean": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["variants"][v] = v_sum

    for v in variants:
        if v == "baseline_r3":
            continue
        p_diff: Dict[str, Any] = {}
        for k in ["input_tokens", "net_tokens", "e2e_accum_ms", "exact_match", "entity_recall", "relation_recovery", "constraint_satisfaction", "obsolete_info_rate"]:
            diffs = [a - b for a, b in zip(sc_means[v][k], sc_means["baseline_r3"][k])]
            mu, lo, hi = _bootstrap_ci(diffs)
            p_diff[k] = {"mean_diff": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["paired_vs_baseline_r3"][v] = p_diff

    (res_dir / f"gate1_summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
