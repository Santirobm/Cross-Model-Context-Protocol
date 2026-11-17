"""Aggregate raw experiment results and generate comparison charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(repo_root: Path) -> Dict[str, str]:
    raw_dir = repo_root / "results" / "raw"
    plot_dir = repo_root / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}

    ab_path = raw_dir / "exp_ab_results.json"
    if ab_path.exists():
        ab = json.loads(ab_path.read_text(encoding="utf-8"))
        summary = ab["summary"]
        directions = list(summary.keys())
        conditions = ab["conditions"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        x = np.arange(len(conditions))
        width = 0.36

        metrics = [
            ("message_bytes", "Wire Size (bytes)", "Wire Bytes by Condition"),
            ("receiver_input_tokens", "Receiver Input Tokens", "Receiver Tokens by Condition"),
            ("quality_score", "Deterministic Recall (0-1)", "Quality Score by Condition"),
        ]

        for ax, (m_key, ylabel, title) in zip(axes, metrics):
            for idx, d in enumerate(directions):
                means = [summary[d][c][m_key]["mean"] for c in conditions]
                offset = (idx - 0.5) * width
                ax.bar(x + offset, means, width, label=d)
            ax.set_xticks(x)
            ax.set_xticklabels(["NL Conv", "JSON", "CMCP Struct", "CMCP Bin"], rotation=18)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if m_key == "quality_score":
                ax.set_ylim(0.0, 1.08)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)

        fig.suptitle("CMCP Level-A Experiments A & B: Cross-Model Context Transfer", fontsize=12)
        fig.tight_layout()
        ab_plot = plot_dir / "exp_ab_comparison.png"
        fig.savefig(ab_plot, dpi=150)
        plt.close(fig)
        outputs["exp_ab_comparison"] = str(ab_plot)

    c_path = raw_dir / "exp_c_results.json"
    if c_path.exists():
        exp_c = json.loads(c_path.read_text(encoding="utf-8"))
        c_sum = exp_c["summary"]
        conds = exp_c["conditions"]
        labels = ["Full History", "Conv Summary", "CMCP Refs", "CMCP Incr"]

        tokens = [c_sum[c]["receiver_input_tokens_mean"] for c in conds]
        latency = [c_sum[c]["total_latency_ms_mean"] for c in conds]
        quality = [c_sum[c]["quality_score_mean"] for c in conds]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        colors = ["#6c757d", "#e07a5f", "#3d405b", "#2a9d8f"]

        axes[0].bar(labels, tokens, color=colors)
        axes[0].set_ylabel("Mean Receiver Input Tokens")
        axes[0].set_title("Token Reduction (Exp C)")
        axes[0].tick_params(axis="x", rotation=18)
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)

        axes[1].bar(labels, latency, color=colors)
        axes[1].set_ylabel("Mean Total Latency (ms)")
        axes[1].set_title("End-to-End Latency (Exp C)")
        axes[1].tick_params(axis="x", rotation=18)
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)

        axes[2].bar(labels, quality, color=colors)
        axes[2].set_ylabel("Mean Quality Recall (0-1)")
        axes[2].set_title("Quality Preservation (Exp C)")
        axes[2].set_ylim(0.0, 1.08)
        axes[2].tick_params(axis="x", rotation=18)
        axes[2].grid(axis="y", linestyle="--", alpha=0.4)

        fig.suptitle("CMCP Experiment C: Qwen2.5-0.5B -> Llama-3.2-1B Multi-Turn Context", fontsize=12)
        fig.tight_layout()
        c_plot = plot_dir / "exp_c_tokens_latency.png"
        fig.savefig(c_plot, dpi=150)
        plt.close(fig)
        outputs["exp_c_tokens_latency"] = str(c_plot)

    return outputs


def aggregate_all() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "results" / "raw"
    plots = generate_plots(repo_root)

    report: Dict[str, Any] = {"plots": plots, "experiments": {}}
    for name, fname in [
        ("exp_ab", "exp_ab_results.json"),
        ("exp_c", "exp_c_results.json"),
        ("exp_f", "exp_f_kv_compat.json"),
    ]:
        fpath = raw_dir / fname
        if fpath.exists():
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if "summary" in data:
                report["experiments"][name] = data["summary"]
            elif "harness_kv_compat" in data:
                report["experiments"][name] = {
                    "harness_cases": data["harness_kv_compat"].get("cases", {}),
                    "python_sdk_checks": data.get("python_sdk_checks", {}),
                }

    agg_path = raw_dir / "benchmark_summary.json"
    agg_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    summary = aggregate_all()
    print(json.dumps(summary, indent=2))
