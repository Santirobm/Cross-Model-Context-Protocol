"""Experiment F: Real llama.cpp KV-state transfer & CMCP protocol compatibility checks."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from adapters.model_registry import get_model_spec, to_cmcp_profile
from representations.binary.codec import cmcp_check_compat, cmcp_check_kv_compat


def run_experiment_f() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    harness = repo_root / "build" / "cmcp_llama_harness.exe"
    out_dir = repo_root / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen_spec = get_model_spec("Qwen2.5-0.5B-Instruct")
    llama_spec = get_model_spec("Llama-3.2-1B-Instruct")

    env = os.environ.copy()
    env["PATH"] = r"C:\msys64\clang64\bin;C:\Users\sgi.lm\Desktop\llama.cpp\build\bin;" + env.get("PATH", "")

    proc = subprocess.run(
        [
            str(harness),
            "--model",
            qwen_spec.gguf_path,
            "--kv-test",
            llama_spec.gguf_path,
            "--mode",
            "kv-compat",
            "--prompt",
            "CMCP KV-cache compatibility verification prompt.",
            "--n-ctx",
            "512",
            "--threads",
            "4",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    harness_res = json.loads(proc.stdout)

    # Also verify via Python ctypes bindings to cmcp.dll
    p_qwen = to_cmcp_profile("Qwen2.5-0.5B-Instruct", n_ctx=512, pos_offset=0)
    p_qwen_same = to_cmcp_profile("Qwen2.5-0.5B-Instruct", n_ctx=512, pos_offset=0)
    p_qwen_ctx = to_cmcp_profile("Qwen2.5-0.5B-Instruct", n_ctx=256, pos_offset=0)
    p_qwen_pos = to_cmcp_profile("Qwen2.5-0.5B-Instruct", n_ctx=512, pos_offset=16)
    p_llama = to_cmcp_profile("Llama-3.2-1B-Instruct", n_ctx=512, pos_offset=0)

    c_same = cmcp_check_kv_compat(p_qwen, p_qwen_same)
    c_ctx = cmcp_check_kv_compat(p_qwen, p_qwen_ctx)
    c_pos = cmcp_check_kv_compat(p_qwen, p_qwen_pos)
    c_cross = cmcp_check_kv_compat(p_qwen, p_llama)
    c_cross_tier = cmcp_check_compat(p_qwen, p_llama, strict_kv=False)

    payload: Dict[str, Any] = {
        "experiment": "Exp_F_KV_Compatibility",
        "harness_kv_compat": harness_res,
        "python_sdk_checks": {
            "same_model_same_cfg": {
                "kv_compatible": bool(c_same.kv_compatible),
                "reason_code": c_same.reason_code,
                "reason": c_same.reason.decode("utf-8", errors="replace"),
            },
            "same_model_diff_n_ctx": {
                "kv_compatible": bool(c_ctx.kv_compatible),
                "reason_code": c_ctx.reason_code,
                "reason": c_ctx.reason.decode("utf-8", errors="replace"),
            },
            "same_model_diff_pos_offset": {
                "kv_compatible": bool(c_pos.kv_compatible),
                "reason_code": c_pos.reason_code,
                "reason": c_pos.reason.decode("utf-8", errors="replace"),
            },
            "cross_arch_qwen_llama": {
                "kv_compatible": bool(c_cross.kv_compatible),
                "reason_code": c_cross.reason_code,
                "reason": c_cross.reason.decode("utf-8", errors="replace"),
                "fallback_tier_non_strict": c_cross_tier.tier,
            },
        },
    }

    (out_dir / "exp_f_kv_compat.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    res = run_experiment_f()
    print(json.dumps(res, indent=2))
