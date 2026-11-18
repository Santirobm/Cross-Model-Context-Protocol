"""CMCP 2.2b Independent Audit Script (Gate 4).

Verifies:
  1. Strict disjointness (IDs and domains) across tune, eval, and unstructured_dialogues.json.
  2. No machine-specific local paths in configs, results, or FINAL_RESEARCH_REPORT_V4.md.
  3. Exact mathematical consistency (max_diff == 0.0) between raw CSVs and summary JSONs.
"""

from __future__ import annotations

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


def check_disjointness() -> Dict[str, Any]:
    tune = get_scenarios("tune")
    ev = get_scenarios("eval")
    unstruct_path = ROOT / "experiments" / "cmcp22b" / "data" / "unstructured_dialogues.json"
    unstruct = json.loads(unstruct_path.read_text(encoding="utf-8"))

    ids_t = {s.scenario_id for s in tune}
    ids_e = {s.scenario_id for s in ev}
    ids_u = {d["dialogue_id"] for d in unstruct}

    dom_t = {s.domain for s in tune}
    dom_e = {s.domain for s in ev}
    dom_u = {d["domain"] for d in unstruct}

    id_overlap = (ids_t & ids_e) | (ids_t & ids_u) | (ids_e & ids_u)
    dom_overlap = (dom_t & dom_e) | (dom_t & dom_u) | (dom_e & dom_u)
    return {
        "passed": len(id_overlap) == 0 and len(dom_overlap) == 0 and len(unstruct) >= 30,
        "counts": {"tune": len(tune), "eval": len(ev), "unstructured": len(unstruct)},
        "id_overlap": sorted(id_overlap),
        "domain_overlap": sorted(dom_overlap),
    }


def check_no_machine_paths() -> Dict[str, Any]:
    patterns = [
        re.compile(r"[A-Za-z]:\\\\Users\\\\"),
        re.compile(r"[A-Za-z]:/Users/"),
        re.compile(r"/Users/[^/]+/"),
        re.compile(r"/home/[^/]+/"),
    ]
    targets: List[Path] = []
    for base in (ROOT / "experiments" / "cmcp22b", ROOT / "experiments" / "cmcp22" / "results" / "cmcp22b_repro"):
        if base.exists():
            targets.extend(base.rglob("*.json"))
            targets.extend(base.rglob("*.csv"))
    rep4 = ROOT / "FINAL_RESEARCH_REPORT_V4.md"
    if rep4.exists():
        targets.append(rep4)

    hits: List[str] = []
    for p in targets:
        if p.name == "validation_report_cmcp22b.json":
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for pat in patterns:
            if pat.search(txt):
                hits.append(str(p.relative_to(ROOT)))
                break
    return {"passed": len(hits) == 0, "files_with_machine_paths": hits, "files_checked": len(targets)}


def _check_pair(csv_path: Path, sum_path: Path, group_col: str) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"file": sum_path.name, "passed": False, "reason": "missing_csv"}
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    max_diff = 0.0
    for method, mstats in summary.get("methods", {}).items():
        mrows = [r for r in rows if r["method"] == method]
        if not mrows:
            return {"file": sum_path.name, "passed": False, "reason": f"empty_{method}"}
        gids = sorted({r[group_col] for r in mrows})
        for metric in ("exact_match", "relation_recovery", "e2e_accum_ms", "input_tokens"):
            if metric not in mstats:
                continue
            g_means = [
                statistics.mean(float(r[metric]) for r in mrows if r[group_col] == gid)
                for gid in gids
            ]
            csv_mu = round(statistics.mean(g_means), 4)
            sum_mu = float(mstats[metric]["mean"])
            max_diff = max(max_diff, abs(csv_mu - sum_mu))
    return {"file": sum_path.name, "rows": len(rows), "max_diff": round(max_diff, 6), "passed": max_diff == 0.0}


def check_csv_summary_consistency() -> Dict[str, Any]:
    checked: List[Dict[str, Any]] = []
    res22b = ROOT / "experiments" / "cmcp22b" / "results"
    if res22b.exists():
        for sp in sorted(res22b.glob("portability_summary_*.json")):
            tag = sp.stem.replace("portability_summary_", "")
            checked.append(_check_pair(res22b / f"portability_runs_{tag}.csv", sp, "scenario_id"))
        for sp in sorted(res22b.glob("unstructured_summary_*.json")):
            tag = sp.stem.replace("unstructured_summary_", "")
            checked.append(_check_pair(res22b / f"unstructured_runs_{tag}.csv", sp, "dialogue_id"))
    repro_dir = ROOT / "experiments" / "cmcp22" / "results" / "cmcp22b_repro"
    if repro_dir.exists():
        for sp in sorted(repro_dir.glob("multiturn_summary_*.json")):
            tag = sp.stem.replace("multiturn_summary_", "")
            checked.append(_check_pair(repro_dir / f"multiturn_runs_{tag}.csv", sp, "scenario_id"))
    all_ok = len(checked) > 0 and all(c["passed"] for c in checked)
    return {"passed": all_ok, "summaries_checked": checked}


def main() -> int:
    report = {
        "disjointness": check_disjointness(),
        "no_machine_paths": check_no_machine_paths(),
        "csv_summary_consistency": check_csv_summary_consistency(),
    }
    report["overall_passed"] = all(v["passed"] for v in report.values())
    out_path = ROOT / "experiments" / "cmcp22b" / "audit" / "validation_report_cmcp22b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
