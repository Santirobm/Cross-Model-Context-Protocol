"""GATE 3 Long-Horizon Scaling Benchmark (5, 10, 20, 40 turns): baseline_r3 vs compact_h0 vs compact_h1 vs anchored."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cmcp22.run_multiturn_benchmark import PersistentLlamaRunner  # noqa: E402
from experiments.cmcp22.run_scaling_benchmark import _generate_turn_ops  # noqa: E402
from representations import serialize  # noqa: E402
from representations.state_store import CMCPStateStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CMCP_MODEL_B", r"C:\Users\sgi.lm\Desktop\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--turns", nargs="+", type=int, default=[5, 10, 20, 40])
    ap.add_argument("--variants", nargs="+", default=["baseline_r3", "compact_h0", "compact_h1", "anchored"])
    ap.add_argument("--tag", default="gate3_scaling")
    args = ap.parse_args()

    res_dir = Path(__file__).resolve().parent / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    runner = PersistentLlamaRunner(args.model, n_ctx=2048, threads=4)
    rows: List[Dict[str, Any]] = []
    q = "Report active budget, active route, and current path."
    try:
        _ = runner.query("Warmup: OK", max_tokens=2)
        for horizon in args.turns:
            st = CMCPStateStore()
            for t in range(1, horizon + 1):
                utt, ops = _generate_turn_ops(t, seq_base=t * 10)
                st.evidence.register(f"turn:{t}", utt)
                for op in ops:
                    st.apply(op)

            for var in args.variants:
                t0 = time.perf_counter()
                if var == "baseline_r3":
                    ctx_str = st.render_for_llm(style="r3")
                elif var == "compact_h0":
                    ctx_str = serialize(st, query=q, mode="compact", expand_hops=0)
                elif var == "compact_h1":
                    ctx_str = serialize(st, query=q, mode="compact", expand_hops=1)
                elif var == "anchored":
                    ctx_str = serialize(st, query=q, mode="anchored")
                else:
                    raise ValueError(f"Unknown variant: {var}")
                sel_ms = (time.perf_counter() - t0) * 1000.0
                prompt = f"Context:\n{ctx_str}\n\nQuestion: {q}\nAnswer in plain text, one concise sentence:"
                resp = runner.query(prompt, max_tokens=32)
                rows.append({
                    "horizon_turns": horizon,
                    "variant": var,
                    "select_ms": round(sel_ms, 4),
                    "input_tokens": int(resp.get("n_input_tokens", 0)),
                    "output_tokens": int(resp.get("n_gen_tokens", 0)),
                    "prefill_ms": round(float(resp.get("prefill_ms", 0.0)), 3),
                    "decode_ms": round(float(resp.get("eval_ms", 0.0)), 3),
                })
    finally:
        runner.close()

    csv_path = res_dir / f"gate3_scaling_{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"model": Path(args.model).name, "turns": args.turns, "variants": args.variants, "rows": rows}
    (res_dir / f"gate3_scaling_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved gate3_scaling_{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
