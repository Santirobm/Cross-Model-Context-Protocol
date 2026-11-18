"""CMCP 2.2 Independent Audit & Validation Script (Section 15).

Verifies:
  1. Dataset split disjointness (scenario IDs and domains across dev/tune/eval).
  2. No ground-truth leakage in DeterministicExtractor (AST check + 0 errors).
  3. No machine-specific absolute paths in configs, results, or reports.
  4. Raw CSV integrity and exact agreement between summary JSONs and raw CSVs.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cmcp22.dataset import get_scenarios
from experiments.cmcp22.extractors import DeterministicExtractor


def check_dataset_disjointness() -> Dict[str, Any]:
    splits = {s: get_scenarios(s) for s in ("dev", "tune", "eval")}
    ids = {s: {sc.scenario_id for sc in scs} for s, scs in splits.items()}
    domains = {s: {sc.domain for sc in scs} for s, scs in splits.items()}

    id_overlap = (ids["dev"] & ids["tune"]) | (ids["dev"] & ids["eval"]) | (ids["tune"] & ids["eval"])
    dom_overlap = (domains["dev"] & domains["eval"]) | (domains["tune"] & domains["eval"])
    return {
        "passed": len(id_overlap) == 0 and len(dom_overlap) == 0 and len(splits["eval"]) == 50,
        "counts": {s: len(v) for s, v in splits.items()},
        "id_overlap": sorted(id_overlap),
        "eval_domain_overlap": sorted(dom_overlap),
    }


def check_no_leakage() -> Dict[str, Any]:
    ext_src = (ROOT / "experiments" / "cmcp22" / "extractors.py").read_text(encoding="utf-8")
    tree = ast.parse(ext_src)
    ops_attr_hits: List[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "ops":
            ops_attr_hits.append(node.lineno)

    ext = DeterministicExtractor()
    total_errs = 0
    total_ops = 0
    for split in ("dev", "tune", "eval"):
        for sc in get_scenarios(split):
            seq = 0
            for t in sc.turns:
                if t.speaker == "agent_a":
                    ops, _, err = ext.extract_turn(t.utterance, seq)
                    total_errs += err
                    total_ops += len(ops)
                    for op in ops:
                        seq = max(seq, op.seq)

    return {
        "passed": len(ops_attr_hits) == 0 and total_errs == 0 and total_ops > 0,
        "ops_attr_hits_in_extractors": ops_attr_hits,
        "total_extract_errors": total_errs,
        "total_extracted_ops": total_ops,
    }


def check_no_machine_paths() -> Dict[str, Any]:
    cmcp22_dir = ROOT / "experiments" / "cmcp22"
    patterns = [re.compile(r"[A-Za-z]:\\\\Users\\\\"), re.compile(r"[A-Za-z]:/Users/"), re.compile(r"/Users/[^/]+/")]
    hits: List[str] = []
    targets: List[Path] = []
    for sub in ("configs", "results"):
        d = cmcp22_dir / sub
        if d.exists():
            targets.extend(d.glob("*.json"))
            targets.extend(d.glob("*.csv"))
    report_p = ROOT / "FINAL_RESEARCH_REPORT_V3.md"
    if report_p.exists():
        targets.append(report_p)

    for p in targets:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for pat in patterns:
            if pat.search(txt):
                hits.append(str(p.relative_to(ROOT)))
                break
    return {"passed": len(hits) == 0, "files_with_machine_paths": hits, "files_checked": len(targets)}


def check_csv_summary_consistency() -> Dict[str, Any]:
    res_dir = ROOT / "experiments" / "cmcp22" / "results"
    checked: List[Dict[str, Any]] = []
    all_ok = True
    for sum_path in sorted(res_dir.glob("multiturn_summary_*.json")):
        tag = sum_path.stem.replace("multiturn_summary_", "")
        csv_path = res_dir / f"multiturn_runs_{tag}.csv"
        if not csv_path.exists():
            all_ok = False
            checked.append({"tag": tag, "passed": False, "reason": "missing_csv"})
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        summary = json.loads(sum_path.read_text(encoding="utf-8"))
        max_diff = 0.0
        for method, mstats in summary.get("methods", {}).items():
            mrows = [r for r in rows if r["method"] == method]
            if not mrows:
                all_ok = False
                continue
            csv_em = statistics.mean(float(r["exact_match"]) for r in mrows)
            sum_em = float(mstats["exact_match"]["mean"])
            max_diff = max(max_diff, abs(csv_em - sum_em))
            csv_wb = statistics.mean(float(r["wire_bytes"]) for r in mrows)
            sum_wb = float(mstats["wire_bytes"]["mean"])
            max_diff = max(max_diff, abs(csv_wb - sum_wb))
        ok = len(rows) > 0 and max_diff < 1e-3
        if not ok:
            all_ok = False
        checked.append({"tag": tag, "rows": len(rows), "max_diff": round(max_diff, 6), "passed": ok})
    return {"passed": all_ok and len(checked) > 0, "summaries_checked": checked}


def main() -> int:
    report = {
        "dataset_disjointness": check_dataset_disjointness(),
        "no_ground_truth_leakage": check_no_leakage(),
        "no_machine_paths": check_no_machine_paths(),
        "csv_summary_consistency": check_csv_summary_consistency(),
    }
    overall = all(v["passed"] for v in report.values())
    report["overall_passed"] = overall

    out_path = ROOT / "experiments" / "cmcp22" / "audit" / "validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
