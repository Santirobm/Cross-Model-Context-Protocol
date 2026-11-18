"""Deterministic Data Leakage & Artifact Integrity Audit for CMCP 2.1.

Exits with non-zero status if any critical violation is found.
Writes experiments/audit/validation_report.json.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.context_transfer.dataset import dataset_sha256, get_tasks
from experiments.latent_alignment.run_cmcp21_latent import _build_domain_split_items


def main() -> int:
    violations: List[str] = []
    checks: Dict[str, Any] = {}

    # 1. Context transfer dataset split disjointness & no answer leakage in instructions
    dev = get_tasks("dev")
    tune = get_tasks("tune")
    ev = get_tasks("eval")
    all_t = get_tasks(None)

    ids_dev = {t.task_id for t in dev}
    ids_tune = {t.task_id for t in tune}
    ids_eval = {t.task_id for t in ev}
    if (ids_dev & ids_tune) or (ids_dev & ids_eval) or (ids_tune & ids_eval):
        violations.append("Overlapping task_ids across dev/tune/eval splits.")

    q_dev = {t.question for t in dev}
    q_tune = {t.question for t in tune}
    q_eval = {t.question for t in ev}
    if (q_dev & q_tune) or (q_dev & q_eval) or (q_tune & q_eval):
        violations.append("Duplicate questions across dev/tune/eval splits.")

    # Ensure instructions don't leak exact expected_keywords (port/owner/budget)
    leaked_instr = []
    for t in all_t:
        instr_text = " ".join(t.instructions).lower()
        for kw in t.expected_keywords:
            if kw.lower() in instr_text:
                leaked_instr.append((t.task_id, kw))
    if leaked_instr:
        violations.append(f"Answer keywords leaked in instructions: {leaked_instr[:3]}")

    checks["context_dataset"] = {
        "total": len(all_t),
        "dev": len(dev),
        "tune": len(tune),
        "eval": len(ev),
        "sha256": dataset_sha256(),
        "leaked_instructions": len(leaked_instr),
    }

    # 2. Latent & KV domain-disjoint split check
    tr_items, ev_items = _build_domain_split_items()
    tr_doms = {i["domain"] for i in tr_items}
    ev_doms = {i["domain"] for i in ev_items}
    tr_texts = {i["text"] for i in tr_items}
    ev_texts = {i["text"] for i in ev_items}
    if tr_doms & ev_doms:
        violations.append(f"Overlapping domains in latent/KV split: {tr_doms & ev_doms}")
    if tr_texts & ev_texts:
        violations.append("Duplicate texts between latent/KV train and eval splits.")
    checks["latent_kv_split"] = {
        "train_n": len(tr_items),
        "eval_n": len(ev_items),
        "train_domains": sorted(tr_doms),
        "eval_domains": sorted(ev_doms),
    }

    # 3. Raw CSV/JSON artifact integrity checks
    expected_csvs = [
        (ROOT / "experiments/context_transfer/results/context_benchmark_runs.csv", 270),
        (ROOT / "experiments/latent_alignment/results/cmcp21_latent_results.csv", 24),
        (ROOT / "experiments/kv_compatibility/results/cmcp21_kv_transfer_runs.csv", 48),
    ]
    csv_status = {}
    for path, min_rows in expected_csvs:
        if not path.exists():
            violations.append(f"Missing raw CSV: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < min_rows:
            violations.append(f"CSV {path.name} has {len(rows)} rows, expected >= {min_rows}")
        bad_num = False
        for r in rows:
            for v in r.values():
                if v in ("NaN", "nan", "Inf", "inf", "-inf"):
                    bad_num = True
        if bad_num:
            violations.append(f"CSV {path.name} contains NaN/Inf values.")
        csv_status[path.name] = {"rows": len(rows), "nan_inf": bad_num}
    checks["csv_artifacts"] = csv_status

    # 4. Config machine-path check
    cfg_path = ROOT / "experiments/context_transfer/configs/benchmark_config.json"
    if cfg_path.exists():
        cfg_txt = cfg_path.read_text(encoding="utf-8")
        if "C:\\" in cfg_txt or "C:/Users" in cfg_txt:
            violations.append("Machine-specific absolute path found in benchmark_config.json")
        checks["config_clean"] = True
    else:
        violations.append("Missing benchmark_config.json")

    report = {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "checks": checks,
    }
    out_path = Path(__file__).resolve().parent / "validation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
