"""CMCP 2.1 Phase 6: Cross-Architecture KV-Cache Extraction, Transformation & Injection.

Loads ONE model at a time on CPU (16 GB RAM budget):
  Sender:   HuggingFaceTB/SmolLM2-135M-Instruct (LlamaForCausalLM, 30L, 3 KV heads, d_head=64, rope_theta=100k)
  Receiver: Qwen/Qwen2.5-0.5B-Instruct (Qwen2ForCausalLM, 24L, 2 KV heads, d_head=64, rope_theta=1M)

Evaluates Hypotheses H1-H5:
  - H1: Same-model self KV injection control (Qwen -> Qwen)
  - H2/H3: Cross-architecture K/V transformation (layer mapping + head projection via
           separate Ridge regressors for K and V per target layer) + real DynamicCache injection
  - H4: Partial-layer KV initialization (top-12 layers only vs all 24 layers)
  - H5: Practical infeasibility of functional cross-architecture KV transfer without E2E tuning

Negative controls: zero_context, random_kv_init (norm-matched Gaussian KV).
Baselines: full_prefill (Baseline A), self_kv_restore (Baseline B).
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.latent_alignment.run_cmcp21_latent import _build_domain_split_items


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.\-<= ]+", " ", text.lower())).strip()


def _token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = 0
    g_counts: Dict[str, int] = {}
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p_toks:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec = common / len(g_toks)
    return 2.0 * prec * rec / (prec + rec)


def _score_qa(pred: str, item: Dict[str, str]) -> Dict[str, float]:
    lp = pred.lower()
    kws = [item["port"].lower(), item["owner"].lower()]
    hits = sum(1 for k in kws if k in lp)
    em = 1.0 if hits == len(kws) else 0.0
    kw_rec = hits / len(kws)
    f1 = _token_f1(pred, item["answer"])
    return {"exact_match": em, "keyword_recall": kw_rec, "f1": f1}


def _pool_kv_to_slots(kv_tensor: torch.Tensor, n_slots: int) -> np.ndarray:
    """Pool (n_kv_heads, seq_len, head_dim) along seq_len into n_slots fixed slots."""
    # kv_tensor: (H_kv, T, D_h)
    h_kv, t_len, d_h = kv_tensor.shape
    arr = kv_tensor.numpy()
    out = np.zeros((h_kv, n_slots, d_h), dtype=np.float32)
    for s in range(n_slots):
        lo = int(math.floor(s * t_len / n_slots))
        hi = int(math.floor((s + 1) * t_len / n_slots))
        if hi <= lo:
            hi = min(t_len, lo + 1)
        out[:, s, :] = arr[:, lo:hi, :].mean(axis=1)
    return out


def _extract_kv_dataset(
    model_id: str,
    texts: List[str],
    n_slots: int,
    out_path: Path,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    cfg = model.config
    arch_info = {
        "model_id": model_id,
        "architectures": getattr(cfg, "architectures", []),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        "head_dim": int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        "rope_theta": float(getattr(cfg, "rope_theta", 10000.0)),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
        "rms_norm_eps": float(getattr(cfg, "rms_norm_eps", 1e-6)),
        "vocab_size": int(cfg.vocab_size),
        "attention_bias": bool(getattr(cfg, "attention_bias", False)),
    }

    n_layers = arch_info["num_hidden_layers"]
    n_kv = arch_info["num_key_value_heads"]
    d_h = arch_info["head_dim"]
    n_items = len(texts)

    k_all = np.zeros((n_items, n_layers, n_kv, n_slots, d_h), dtype=np.float32)
    v_all = np.zeros((n_items, n_layers, n_kv, n_slots, d_h), dtype=np.float32)
    seq_lens: List[int] = []

    with torch.no_grad():
        for idx, txt in enumerate(texts):
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=128)
            seq_lens.append(int(enc["input_ids"].shape[1]))
            out = model(**enc, use_cache=True)
            pkv = out.past_key_values
            for l in range(n_layers):
                # Support both DynamicCache (layers[l].keys/values or key_cache[l]) and legacy tuple
                if hasattr(pkv, "layers"):
                    k_l = pkv.layers[l].keys[0].detach().cpu()
                    v_l = pkv.layers[l].values[0].detach().cpu()
                elif hasattr(pkv, "key_cache"):
                    k_l = pkv.key_cache[l][0].detach().cpu()
                    v_l = pkv.value_cache[l][0].detach().cpu()
                else:
                    k_l = pkv[l][0][0].detach().cpu()
                    v_l = pkv[l][1][0].detach().cpu()
                k_all[idx, l] = _pool_kv_to_slots(k_l, n_slots)
                v_all[idx, l] = _pool_kv_to_slots(v_l, n_slots)

    np.savez(out_path, k=k_all, v=v_all, seq_lens=np.array(seq_lens, dtype=np.int32))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    arch_info["extract_total_ms"] = round(elapsed_ms, 2)
    arch_info["extract_per_item_ms"] = round(elapsed_ms / max(1, n_items), 2)

    del model, tok
    gc.collect()
    return arch_info


def _fit_ridge_kv(X: np.ndarray, Y: np.ndarray, alpha: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """X: (N, D_in), Y: (N, D_out) -> W: (D_in, D_out), b: (D_out,)."""
    xm = X.mean(axis=0, keepdims=True)
    ym = Y.mean(axis=0, keepdims=True)
    xc = X - xm
    yc = Y - ym
    d_in = xc.shape[1]
    W = np.linalg.solve(xc.T @ xc + alpha * np.eye(d_in, dtype=np.float32), xc.T @ yc)
    b = (ym - xm @ W).squeeze(0)
    return W.astype(np.float32), b.astype(np.float32)


def _eval_with_cache(
    model: Any,
    tok: Any,
    prompt_text: str,
    answer_text: str,
    cache: DynamicCache | None,
    cache_len: int,
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    """Run receiver forward/generation optionally conditioned on an injected DynamicCache."""
    q_ids = tok(prompt_text, return_tensors="pt")["input_ids"]
    a_ids = tok(" " + answer_text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    t0 = time.perf_counter()
    with torch.no_grad():
        # Compute NLL on gold answer tokens
        full_ids = torch.cat([q_ids, a_ids], dim=1)
        pos_ids = torch.arange(cache_len, cache_len + full_ids.shape[1], dtype=torch.long).unsqueeze(0)

        # Make a fresh copy of cache if provided because forward mutates DynamicCache in-place
        if cache is not None:
            dc_nll = DynamicCache()
            for l in range(len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)):
                if hasattr(cache, "layers"):
                    kl, vl = cache.layers[l].keys.clone(), cache.layers[l].values.clone()
                else:
                    kl, vl = cache.key_cache[l].clone(), cache.value_cache[l].clone()
                dc_nll.update(kl, vl, l)
            out_nll = model(input_ids=full_ids, attention_mask=torch.ones((1, cache_len + full_ids.shape[1]), dtype=torch.long),
                            position_ids=pos_ids, past_key_values=dc_nll, use_cache=True)
        else:
            out_nll = model(input_ids=full_ids, use_cache=False)

        # Logits predicting a_ids
        q_len = q_ids.shape[1]
        logits_ans = out_nll.logits[:, q_len - 1 : q_len - 1 + a_ids.shape[1], :]
        nll = float(torch.nn.functional.cross_entropy(
            logits_ans.reshape(-1, logits_ans.shape[-1]),
            a_ids.reshape(-1),
            reduction="mean",
        ).item())

        # Greedy generation (16 tokens) from question
        if cache is not None:
            dc_gen = DynamicCache()
            for l in range(len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)):
                if hasattr(cache, "layers"):
                    kl, vl = cache.layers[l].keys.clone(), cache.layers[l].values.clone()
                else:
                    kl, vl = cache.key_cache[l].clone(), cache.value_cache[l].clone()
                dc_gen.update(kl, vl, l)
            q_pos = torch.arange(cache_len, cache_len + q_len, dtype=torch.long).unsqueeze(0)
            q_mask = torch.ones((1, cache_len + q_len), dtype=torch.long)
            out_g = model(input_ids=q_ids, attention_mask=q_mask, position_ids=q_pos,
                          past_key_values=dc_gen, use_cache=True)
        else:
            out_g = model(input_ids=q_ids, use_cache=True)
            dc_gen = out_g.past_key_values

        cur_pos = cache_len + q_len
        gen_tokens: List[int] = []
        next_tok = int(torch.argmax(out_g.logits[:, -1, :], dim=-1).item())
        for _ in range(max_new_tokens):
            gen_tokens.append(next_tok)
            if next_tok == tok.eos_token_id:
                break
            step_in = torch.tensor([[next_tok]], dtype=torch.long)
            step_pos = torch.tensor([[cur_pos]], dtype=torch.long)
            step_mask = torch.ones((1, cur_pos + 1), dtype=torch.long)
            step_out = model(input_ids=step_in, attention_mask=step_mask, position_ids=step_pos,
                             past_key_values=dc_gen, use_cache=True)
            next_tok = int(torch.argmax(step_out.logits[:, -1, :], dim=-1).item())
            cur_pos += 1

        gen_text = tok.decode(gen_tokens, skip_special_tokens=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {"nll": nll, "ppl": math.exp(min(20.0, nll)), "gen_text": gen_text, "ms": elapsed_ms}


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "offline_kv"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_items, eval_items = _build_domain_split_items()
    # Use 24 train items and 8 disjoint eval items for fast CPU turnaround
    train_sub = train_items[:24]
    eval_sub = eval_items[:8]
    all_sub = train_sub + eval_sub
    all_texts = [it["text"] for it in all_sub]
    n_train = len(train_sub)
    n_slots = 16

    src_npz = cache_dir / "smollm2_kv.npz"
    dst_npz = cache_dir / "qwen25_kv.npz"
    meta_file = cache_dir / "kv_arch_meta.json"

    if src_npz.exists() and dst_npz.exists() and meta_file.exists():
        arch_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        src_arch = arch_meta["sender"]
        dst_arch = arch_meta["receiver"]
    else:
        src_arch = _extract_kv_dataset("HuggingFaceTB/SmolLM2-135M-Instruct", all_texts, n_slots, src_npz)
        dst_arch = _extract_kv_dataset("Qwen/Qwen2.5-0.5B-Instruct", all_texts, n_slots, dst_npz)
        meta_file.write_text(json.dumps({"sender": src_arch, "receiver": dst_arch}, indent=2), encoding="utf-8")

    src_kv = np.load(src_npz)
    dst_kv = np.load(dst_npz)

    # Fit separate Ridge regressors for K and V per target layer l=0..23
    # Map target layer l -> source layer round(l * (L_src - 1) / (L_dst - 1))
    L_src = src_arch["num_hidden_layers"]
    L_dst = dst_arch["num_hidden_layers"]
    H_src = src_arch["num_key_value_heads"]
    H_dst = dst_arch["num_key_value_heads"]
    D_h = dst_arch["head_dim"]

    t_fit0 = time.perf_counter()
    k_weights: List[Tuple[int, np.ndarray, np.ndarray]] = []
    v_weights: List[Tuple[int, np.ndarray, np.ndarray]] = []

    for l_dst in range(L_dst):
        l_src = int(round(l_dst * (L_src - 1) / max(1, L_dst - 1)))
        # Flatten (N, S, H_src*D_h) -> (N*S, H_src*D_h) to learn head+channel mapping per slot
        Xk_tr = src_kv["k"][:n_train, l_src].transpose(0, 2, 1, 3).reshape(-1, H_src * D_h)
        Yk_tr = dst_kv["k"][:n_train, l_dst].transpose(0, 2, 1, 3).reshape(-1, H_dst * D_h)
        Wk, bk = _fit_ridge_kv(Xk_tr, Yk_tr, alpha=50.0)
        k_weights.append((l_src, Wk, bk))

        Xv_tr = src_kv["v"][:n_train, l_src].transpose(0, 2, 1, 3).reshape(-1, H_src * D_h)
        Yv_tr = dst_kv["v"][:n_train, l_dst].transpose(0, 2, 1, 3).reshape(-1, H_dst * D_h)
        Wv, bv = _fit_ridge_kv(Xv_tr, Yv_tr, alpha=50.0)
        v_weights.append((l_src, Wv, bv))
    fit_ms = (time.perf_counter() - t_fit0) * 1000.0

    # Load receiver model ONCE for all evaluations
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
    model.eval()

    rng = np.random.default_rng(123)
    methods = [
        "baseline_a_full_prefill",
        "h1_self_kv_restore",
        "neg_zero_context",
        "neg_random_kv",
        "h2_h3_cross_kv_ridge",
        "h4_partial_top12_kv",
    ]
    rows: List[Dict[str, Any]] = []

    for ev_i, item in enumerate(eval_sub):
        global_idx = n_train + ev_i
        q_prompt = f"Question: {item['question']}\nAnswer concisely:"
        full_prompt = f"Context: {item['text']}\n\nQuestion: {item['question']}\nAnswer concisely:"

        # 1. Baseline A: Full prefill on receiver
        r_full = _eval_with_cache(model, tok, full_prompt, item["answer"], None, 0)
        sc_full = _score_qa(r_full["gen_text"], item)
        rows.append({"id": item["id"], "method": "baseline_a_full_prefill", **r_full, **sc_full, "transform_ms": 0.0, "kv_bytes": 0})

        # 2. H1: Self KV restore (exact Qwen -> Qwen unpooled past_key_values on context)
        with torch.no_grad():
            ctx_ids = tok(f"Context: {item['text']}\n\n", return_tensors="pt")["input_ids"]
            self_out = model(input_ids=ctx_ids, use_cache=True)
            self_cache = self_out.past_key_values
            self_len = int(ctx_ids.shape[1])
        r_self = _eval_with_cache(model, tok, q_prompt, item["answer"], self_cache, self_len)
        sc_self = _score_qa(r_self["gen_text"], item)
        rows.append({"id": item["id"], "method": "h1_self_kv_restore", **r_self, **sc_self, "transform_ms": 0.0, "kv_bytes": self_len * L_dst * H_dst * D_h * 2 * 4})

        # 3. Negative control: zero context
        r_zero = _eval_with_cache(model, tok, q_prompt, item["answer"], None, 0)
        sc_zero = _score_qa(r_zero["gen_text"], item)
        rows.append({"id": item["id"], "method": "neg_zero_context", **r_zero, **sc_zero, "transform_ms": 0.0, "kv_bytes": 0})

        # 4. Negative control: norm-matched random Gaussian KV
        rand_cache = DynamicCache()
        for l in range(L_dst):
            k_std = float(np.std(dst_kv["k"][global_idx, l])) + 1e-4
            v_std = float(np.std(dst_kv["v"][global_idx, l])) + 1e-4
            kr = torch.tensor(rng.normal(0.0, k_std, (1, H_dst, n_slots, D_h)).astype(np.float32))
            vr = torch.tensor(rng.normal(0.0, v_std, (1, H_dst, n_slots, D_h)).astype(np.float32))
            rand_cache.update(kr, vr, l)
        r_rand = _eval_with_cache(model, tok, q_prompt, item["answer"], rand_cache, n_slots)
        sc_rand = _score_qa(r_rand["gen_text"], item)
        rows.append({"id": item["id"], "method": "neg_random_kv", **r_rand, **sc_rand, "transform_ms": 0.0, "kv_bytes": n_slots * L_dst * H_dst * D_h * 2 * 4})

        # 5. H2/H3: Cross-architecture Ridge-transformed KV (all 24 layers)
        t_tr0 = time.perf_counter()
        proj_cache = DynamicCache()
        for l_dst in range(L_dst):
            l_src, Wk, bk = k_weights[l_dst]
            _, Wv, bv = v_weights[l_dst]
            xk = src_kv["k"][global_idx, l_src].transpose(1, 0, 2).reshape(n_slots, H_src * D_h)
            xv = src_kv["v"][global_idx, l_src].transpose(1, 0, 2).reshape(n_slots, H_src * D_h)
            yk = (xk @ Wk + bk).reshape(n_slots, H_dst, D_h).transpose(1, 0, 2)[None, ...]
            yv = (xv @ Wv + bv).reshape(n_slots, H_dst, D_h).transpose(1, 0, 2)[None, ...]
            proj_cache.update(torch.from_numpy(yk.copy()), torch.from_numpy(yv.copy()), l_dst)
        tr_ms = (time.perf_counter() - t_tr0) * 1000.0
        r_cross = _eval_with_cache(model, tok, q_prompt, item["answer"], proj_cache, n_slots)
        sc_cross = _score_qa(r_cross["gen_text"], item)
        rows.append({"id": item["id"], "method": "h2_h3_cross_kv_ridge", **r_cross, **sc_cross, "transform_ms": round(tr_ms, 3), "kv_bytes": n_slots * L_dst * H_dst * D_h * 2 * 4})

        # 6. H4: Partial-layer KV initialization (zeroed early layers 0..11, projected top layers 12..23)
        part_cache = DynamicCache()
        for l_dst in range(L_dst):
            if l_dst < L_dst // 2:
                kz = torch.zeros((1, H_dst, n_slots, D_h), dtype=torch.float32)
                vz = torch.zeros((1, H_dst, n_slots, D_h), dtype=torch.float32)
                part_cache.update(kz, vz, l_dst)
            else:
                l_src, Wk, bk = k_weights[l_dst]
                _, Wv, bv = v_weights[l_dst]
                xk = src_kv["k"][global_idx, l_src].transpose(1, 0, 2).reshape(n_slots, H_src * D_h)
                xv = src_kv["v"][global_idx, l_src].transpose(1, 0, 2).reshape(n_slots, H_src * D_h)
                yk = (xk @ Wk + bk).reshape(n_slots, H_dst, D_h).transpose(1, 0, 2)[None, ...]
                yv = (xv @ Wv + bv).reshape(n_slots, H_dst, D_h).transpose(1, 0, 2)[None, ...]
                part_cache.update(torch.from_numpy(yk.copy()), torch.from_numpy(yv.copy()), l_dst)
        r_part = _eval_with_cache(model, tok, q_prompt, item["answer"], part_cache, n_slots)
        sc_part = _score_qa(r_part["gen_text"], item)
        rows.append({"id": item["id"], "method": "h4_partial_top12_kv", **r_part, **sc_part, "transform_ms": round(tr_ms * 0.5, 3), "kv_bytes": (n_slots * (L_dst // 2) * H_dst * D_h * 2 * 4)})

    del model, tok
    gc.collect()

    csv_path = out_dir / "cmcp21_kv_transfer_runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_methods: Dict[str, Any] = {}
    for m in methods:
        mr = [r for r in rows if r["method"] == m]
        summary_methods[m] = {
            "n": len(mr),
            "mean_nll": round(float(np.mean([r["nll"] for r in mr])), 4),
            "mean_ppl": round(float(np.mean([r["ppl"] for r in mr])), 4),
            "exact_match": round(float(np.mean([r["exact_match"] for r in mr])), 4),
            "keyword_recall": round(float(np.mean([r["keyword_recall"] for r in mr])), 4),
            "f1": round(float(np.mean([r["f1"] for r in mr])), 4),
            "mean_eval_ms": round(float(np.mean([r["ms"] for r in mr])), 2),
            "mean_transform_ms": round(float(np.mean([r["transform_ms"] for r in mr])), 3),
            "sample_output": mr[0]["gen_text"][:100],
        }

    summary = {
        "architectures": {"sender": src_arch, "receiver": dst_arch},
        "ridge_fit_ms": round(fit_ms, 2),
        "n_slots": n_slots,
        "methods": summary_methods,
    }
    (out_dir / "cmcp21_kv_transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
