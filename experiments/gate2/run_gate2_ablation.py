"""GATE 2.5 Factorial Ablation Benchmark (2x2: Dedup x Order + ref_var_a + baseline_r3)."""

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
from experiments.gate1.run_gate1_benchmark import _build_store_up_to  # noqa: E402
from representations.adapters import render_selected_ablation  # noqa: E402
from representations.binary import (  # noqa: E402
    CMCP_MSG_CONTEXT_TRANSFER,
    CMCP_TIER_2_CROSS_MODEL,
    decode_frame,
    encode_frame,
)
from representations.selector import select_for_task  # noqa: E402

ALL_VARIANTS = ["baseline_r3", "ref_var_a", "abl_A", "abl_B", "abl_C", "abl_D"]


def _render_variant(var: str, q_text: str, st: Any) -> tuple[str, int]:
    if var == "baseline_r3":
        return st.render_for_llm(style="r3"), 0
    if var == "ref_var_a":
        return st.render_for_llm(style="r3_canon"), 0
    sel = select_for_task(q_text, st, use_relations=True, use_evidence_fallback=True, canonicalize=True)
    ev = 1 if sel.used_evidence_fallback else 0
    if var == "abl_A":
        return render_selected_ablation(sel, st, dedup=False, unified_order=False), ev
    if var == "abl_B":
        return render_selected_ablation(sel, st, dedup=True, unified_order=False), ev
    if var == "abl_C":
        return render_selected_ablation(sel, st, dedup=False, unified_order=True), ev
    return render_selected_ablation(sel, st, dedup=True, unified_order=True), ev


def summarize_csv(csv_path: Path, summary_path: Path, split: str, reps: int, max_tokens: int, model_b_name: str, tag: str) -> Dict[str, Any]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    variants = [v for v in ALL_VARIANTS if any(r["variant"] == v for r in rows)]
    sc_ids = sorted({r["scenario_id"] for r in rows})
    stat_keys = [
        "input_tokens", "output_tokens", "net_tokens",
        "select_ms", "prefill_ms", "decode_ms", "e2e_accum_ms",
        "exact_match", "entity_recall", "relation_recovery",
        "constraint_satisfaction", "obsolete_info_rate", "evidence_fallback_used",
    ]
    sc_means: Dict[str, Dict[str, List[float]]] = {v: {k: [] for k in stat_keys} for v in variants}
    summary: Dict[str, Any] = {
        "config": {
            "split": split,
            "num_scenarios": len(sc_ids),
            "reps": reps,
            "max_tokens": max_tokens,
            "model_b_name": model_b_name,
            "variants": variants,
            "tag": tag,
            "dataset_sha256": dataset_sha256(),
            "bootstrap_seed": 42,
            "decoding": "greedy",
            "extra_llm_calls": 0,
        },
        "variants": {},
        "paired_vs_baseline_r3": {},
    }
    for v in variants:
        vr = [r for r in rows if r["variant"] == v]
        for sid in sc_ids:
            sr = [r for r in vr if r["scenario_id"] == sid]
            for k in stat_keys:
                sc_means[v][k].append(float(statistics.mean([float(r[k]) for r in sr])))
        v_sum: Dict[str, Any] = {"n_scenarios": len(sc_ids), "n_rows": len(vr)}
        for k in stat_keys:
            mu, lo, hi = _bootstrap_ci(sc_means[v][k])
            v_sum[k] = {"mean": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["variants"][v] = v_sum

    if "baseline_r3" in variants:
        for v in variants:
            if v == "baseline_r3":
                continue
            p_diff: Dict[str, Any] = {}
            for k in ["input_tokens", "net_tokens", "e2e_accum_ms", "exact_match", "entity_recall", "relation_recovery", "constraint_satisfaction", "obsolete_info_rate"]:
                diffs = [a - b for a, b in zip(sc_means[v][k], sc_means["baseline_r3"][k])]
                mu, lo, hi = _bootstrap_ci(diffs)
                p_diff[k] = {"mean_diff": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            summary["paired_vs_baseline_r3"][v] = p_diff

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["dev", "tune", "eval"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model-b", default=os.environ.get("CMCP_MODEL_B", r"C:\Users\sgi.lm\Desktop\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS)
    ap.add_argument("--tag", default="gate2_llama_ablation")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    res_dir = out_dir / "results"
    cfg_dir = out_dir / "configs"
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    csv_path = res_dir / f"gate2_runs_{args.tag}.csv"
    sum_path = res_dir / f"gate2_summary_{args.tag}.json"
    model_b_name = Path(args.model_b).name

    if args.summarize_only:
        s = summarize_csv(csv_path, sum_path, args.split, args.reps, args.max_tokens, model_b_name, args.tag)
        print(json.dumps(s, indent=2))
        return 0

    scenarios = get_scenarios(args.split)
    cfg = {
        "split": args.split,
        "num_scenarios": len(scenarios),
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "model_b_name": model_b_name,
        "variants": args.variants,
        "tag": args.tag,
        "dataset_sha256": dataset_sha256(),
        "bootstrap_seed": 42,
        "decoding": "greedy",
        "extra_llm_calls": 0,
    }
    (cfg_dir / f"gate2_config_{args.tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    det_ext = DeterministicExtractor()
    runner_b = PersistentLlamaRunner(args.model_b, n_ctx=1024, threads=4)
    fieldnames = [
        "scenario_id", "domain", "variant", "rep", "query_turn",
        "wire_bytes", "input_tokens", "output_tokens", "net_tokens",
        "extract_ms", "select_ms", "serialize_ms", "prefill_ms", "decode_ms", "e2e_accum_ms",
        "exact_match", "entity_recall", "relation_recovery", "constraint_satisfaction",
        "obsolete_info_rate", "evidence_fallback_used",
    ]
    mode = "a" if (args.append and csv_path.exists()) else "w"
    try:
        _ = runner_b.query("Warmup: OK", max_tokens=2)
        with open(csv_path, mode, newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            for rep in range(1, args.reps + 1):
                for sc in scenarios:
                    for qspec in sc.eval_queries:
                        q_turn = int(qspec["turn"])
                        q_text = qspec["query"]
                        gold_str = " ".join(qspec["expected_keywords"])
                        st, ops, ext_ms = _build_store_up_to(sc.turns, q_turn - 1, det_ext)
                        delta_ops = ops[-3:] if len(ops) >= 3 else ops

                        for var in args.variants:
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
                            ctx_str, ev_used = _render_variant(var, q_text, st)
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

                            writer.writerow({
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
                            })
                            fcsv.flush()
    finally:
        runner_b.close()

    if not args.append:
        s = summarize_csv(csv_path, sum_path, args.split, args.reps, args.max_tokens, model_b_name, args.tag)
        print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
