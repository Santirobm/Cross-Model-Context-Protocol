"""GATE 1 Long-Horizon Scaling Benchmark (5, 10, 20, 40 turns): baseline_r3 vs var_d."""

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
from representations.selector import select_for_task  # noqa: E402
from representations.state_store import CMCPStateStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CMCP_MODEL_B", r"C:\Users\sgi.lm\Desktop\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--turns", nargs="+", type=int, default=[5, 10, 20, 40])
    ap.add_argument("--tag", default="gate1_scaling")
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

            for var in ("baseline_r3", "var_d"):
                t0 = time.perf_counter()
                if var == "baseline_r3":
                    ctx_str = st.render_for_llm(style="r3")
                else:
                    sel = select_for_task(q, st, use_relations=True, use_evidence_fallback=True, canonicalize=True)
                    ctx_str = sel.rendered_text
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

    csv_path = res_dir / f"gate1_scaling_{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"model": Path(args.model).name, "turns": args.turns, "rows": rows}
    (res_dir / f"gate1_scaling_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
