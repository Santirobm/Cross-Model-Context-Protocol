"""CMCP 2.2 Long-Horizon Scaling Benchmark (Section 12).

Evaluates scaling across 5, 10, 20, and 40 turns for:
  - full_history (accumulated raw turns)
  - json_structured (full active JSON state at each handoff)
  - cmcp_full (full TLV binary frame at each handoff)
  - cmcp_incremental (delta TLV frames + versioned store + compact() + checkout_seq())

Measures cumulative wire bytes, update bytes, persisted store growth,
compact() savings, checkout_seq() historical query latency, input tokens,
and real LLM prefill latency (with n_ctx=2048).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cmcp22.run_multiturn_benchmark import (
    PersistentLlamaRunner,
    _pack_ops_to_tlv,
    _unpack_ops_from_tlv,
)
from representations.binary import (
    CMCP_MSG_CONTEXT_TRANSFER,
    CMCP_TIER_2_CROSS_MODEL,
    decode_frame,
    encode_frame,
)
from representations.state_store import CMCPStateStore, StateOp


def _generate_turn_ops(turn_idx: int, seq_base: int) -> tuple[str, List[StateOp]]:
    """Generate a realistic turn utterance + deterministic ops with updates and invalidations."""
    ops: List[StateOp] = []
    if turn_idx == 1:
        utt = "Primary node Node-A0 is active; anchor=ANC-0; budget<=100ms; route=R-01; decision: mode=standard."
        ops = [
            StateOp(fact_id="node_a0", kind="UPSERT", category="entity", key="Node-A0", value="primary; anchor=ANC-0", seq=seq_base + 1, version=1),
            StateOp(fact_id="c_budget", kind="UPSERT", category="constraint", key="budget", value="budget<=100ms", seq=seq_base + 2, version=1),
            StateOp(fact_id="f_route", kind="UPSERT", category="fact", key="route", value="R-01", seq=seq_base + 3, version=1),
            StateOp(fact_id="d_mode", kind="UPSERT", category="decision", key="mode", value="mode=standard", seq=seq_base + 4, version=1),
        ]
    elif turn_idx % 5 == 0:
        # Invalidate old temporary ticket and tighten budget / update route
        prev_tmp = f"tmp_{turn_idx - 1}"
        new_budget = max(20, 100 - turn_idx * 2)
        utt = (
            f"Turn {turn_idx}: invalidate {prev_tmp}; tighten budget<={new_budget}ms; "
            f"update route=R-{turn_idx:02d}; path Node-A0 -> Relay-{turn_idx} -> Sink-Z."
        )
        ops = [
            StateOp(fact_id=prev_tmp, kind="INVALIDATE", category="fact", key=prev_tmp, value="", seq=seq_base + 1, version=1),
            StateOp(fact_id="c_budget", kind="UPSERT", category="constraint", key="budget", value=f"budget<={new_budget}ms", seq=seq_base + 2, version=1),
            StateOp(fact_id="f_route", kind="UPSERT", category="fact", key="route", value=f"R-{turn_idx:02d}", seq=seq_base + 3, version=1),
            StateOp(fact_id="rel_path", kind="UPSERT", category="relation", key="path", value=f"Node-A0 -> Relay-{turn_idx} -> Sink-Z", seq=seq_base + 4, version=1),
        ]
    else:
        utt = (
            f"Turn {turn_idx}: register temporary check tmp_{turn_idx}=CHK-{turn_idx:03d}; "
            f"log telemetry shard S-{turn_idx:03d} on Node-A0 (note: ignore standby Node-A0B)."
        )
        ops = [
            StateOp(fact_id=f"tmp_{turn_idx}", kind="UPSERT", category="fact", key=f"tmp_{turn_idx}", value=f"CHK-{turn_idx:03d}", seq=seq_base + 1, version=1),
            StateOp(fact_id=f"shard_{turn_idx}", kind="UPSERT", category="fact", key=f"shard_{turn_idx}", value=f"S-{turn_idx:03d}", seq=seq_base + 2, version=1),
        ]
    return utt, ops


def run_scaling(model_path: str, tag: str, turns_list: List[int]) -> Dict[str, Any]:
    out_dir = ROOT / "experiments" / "cmcp22"
    res_dir = out_dir / "results"
    cfg_dir = out_dir / "configs"
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "turns_list": turns_list,
        "model_name": Path(model_path).name,
        "n_ctx": 2048,
        "tag": tag,
    }
    (cfg_dir / f"scaling_config_{tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    runner = PersistentLlamaRunner(model_path, n_ctx=2048, threads=4)
    rows: List[Dict[str, Any]] = []
    try:
        _ = runner.query("Warmup: OK", max_tokens=2)

        for horizon in turns_list:
            store_inc = CMCPStateStore()
            hist_lines: List[str] = []
            all_ops: List[StateOp] = []

            cum_bytes = {"full_history": 0, "json_structured": 0, "cmcp_full": 0, "cmcp_incremental": 0}
            last_bytes = {"full_history": 0, "json_structured": 0, "cmcp_full": 0, "cmcp_incremental": 0}
            last_ser_ms = {"full_history": 0.0, "json_structured": 0.0, "cmcp_full": 0.0, "cmcp_incremental": 0.0}
            last_rec_ms = {"full_history": 0.0, "json_structured": 0.0, "cmcp_full": 0.0, "cmcp_incremental": 0.0}

            for t in range(1, horizon + 1):
                utt, ops = _generate_turn_ops(t, seq_base=t * 10)
                hist_lines.append(utt)
                all_ops.extend(ops)
                for op in ops:
                    store_inc.apply(op)

                # full_history
                fh_str = "\n".join(hist_lines)
                fh_b = len(fh_str.encode("utf-8"))
                cum_bytes["full_history"] += fh_b
                last_bytes["full_history"] = fh_b

                # json_structured
                t0 = time.perf_counter()
                js_str = json.dumps(store_inc.active_view(), separators=(",", ":"), sort_keys=True)
                last_ser_ms["json_structured"] = (time.perf_counter() - t0) * 1000.0
                js_b = len(js_str.encode("utf-8"))
                cum_bytes["json_structured"] += js_b
                last_bytes["json_structured"] = js_b

                # cmcp_full
                t0 = time.perf_counter()
                f_full = encode_frame(_pack_ops_to_tlv(all_ops), compress=True, msg_type=CMCP_MSG_CONTEXT_TRANSFER, compat_tier=CMCP_TIER_2_CROSS_MODEL)
                last_ser_ms["cmcp_full"] = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                _, dec_full = decode_frame(f_full)
                tmp_s = CMCPStateStore()
                for op in _unpack_ops_from_tlv(dec_full):
                    tmp_s.apply(op)
                _ = tmp_s.render_for_llm()
                last_rec_ms["cmcp_full"] = (time.perf_counter() - t1) * 1000.0
                cf_b = len(f_full)
                cum_bytes["cmcp_full"] += cf_b
                last_bytes["cmcp_full"] = cf_b

                # cmcp_incremental
                t0 = time.perf_counter()
                f_inc = encode_frame(_pack_ops_to_tlv(ops), compress=True, msg_type=CMCP_MSG_CONTEXT_TRANSFER, compat_tier=CMCP_TIER_2_CROSS_MODEL)
                last_ser_ms["cmcp_incremental"] = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                _, dec_inc = decode_frame(f_inc)
                _ = _unpack_ops_from_tlv(dec_inc)
                _ = store_inc.render_for_llm()
                last_rec_ms["cmcp_incremental"] = (time.perf_counter() - t1) * 1000.0
                ci_b = len(f_inc)
                cum_bytes["cmcp_incremental"] += ci_b
                last_bytes["cmcp_incremental"] = ci_b

            store_before = store_inc.serialized_size_bytes()
            t_chk0 = time.perf_counter()
            hist_view = store_inc.checkout_seq(target_seq=30)
            checkout_ms = (time.perf_counter() - t_chk0) * 1000.0

            compact_store = CMCPStateStore()
            for op in all_ops:
                compact_store.apply(op)
            compact_removed = compact_store.compact(prune_tombstones_before_seq=horizon * 10)
            store_after = compact_store.serialized_size_bytes()

            q = "Report active budget, active route, and current path."
            prompts = {
                "full_history": f"History:\n{'\n'.join(hist_lines)}\n\nQuestion: {q}\nAnswer in plain text (no markdown):",
                "json_structured": f"JSON Context:\n{json.dumps(store_inc.active_view(), separators=(',', ':'), sort_keys=True)}\n\nQuestion: {q}\nAnswer in plain text (no markdown):",
                "cmcp_full": f"Context:\n{store_inc.render_for_llm()}\n\nQuestion: {q}\nAnswer in plain text (no markdown):",
                "cmcp_incremental": f"Context:\n{store_inc.render_for_llm()}\n\nQuestion: {q}\nAnswer in plain text (no markdown):",
            }

            for method in ("full_history", "json_structured", "cmcp_full", "cmcp_incremental"):
                resp = runner.query(prompts[method], max_tokens=32)
                rows.append({
                    "horizon_turns": horizon,
                    "method": method,
                    "cum_wire_bytes": cum_bytes[method],
                    "last_wire_bytes": last_bytes[method],
                    "store_bytes_before_compact": store_before if method == "cmcp_incremental" else 0,
                    "store_bytes_after_compact": store_after if method == "cmcp_incremental" else 0,
                    "compact_ops_removed": compact_removed if method == "cmcp_incremental" else 0,
                    "checkout_seq_ms": round(checkout_ms, 4) if method == "cmcp_incremental" else 0.0,
                    "checkout_active_count": len(hist_view) if method == "cmcp_incremental" else 0,
                    "serialize_ms": round(last_ser_ms[method], 4),
                    "reconstruct_ms": round(last_rec_ms[method], 4),
                    "input_tokens": int(resp.get("n_input_tokens", 0)),
                    "prefill_ms": round(float(resp.get("prefill_ms", 0.0)), 3),
                    "gen_ms": round(float(resp.get("eval_ms", 0.0)), 3),
                })
    finally:
        runner.close()

    csv_path = res_dir / f"scaling_runs_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"tag": tag, "turns_list": turns_list, "rows": rows}
    sum_path = res_dir / f"scaling_summary_{tag}.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--turns", nargs="+", type=int, default=[5, 10, 20, 40])
    args = ap.parse_args()
    summary = run_scaling(args.model, args.tag, args.turns)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
