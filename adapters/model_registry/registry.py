"""Registry of local GGUF model profiles for CMCP adapters and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from representations.binary.codec import CMCPModelProfile, cmcp_fnv1a_64, make_model_profile


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    arch: str
    gguf_path: str
    n_layer: int
    n_embd: int
    n_head_kv: int
    rope_type: int
    rope_freq_scale: float
    n_vocab: int
    kv_dtype: int = 1
    default_n_ctx: int = 1024


MODEL_PROFILES: Dict[str, ModelSpec] = {
    "Qwen2.5-0.5B-Instruct": ModelSpec(
        model_id="Qwen2.5-0.5B-Instruct",
        arch="qwen2",
        gguf_path=r"C:\Users\sgi.lm\Desktop\models\qwen2.5-0.5b-instruct-q4_k_m.gguf",
        n_layer=24,
        n_embd=896,
        n_head_kv=2,
        rope_type=2,
        rope_freq_scale=1.0,
        n_vocab=151936,
    ),
    "Llama-3.2-1B-Instruct": ModelSpec(
        model_id="Llama-3.2-1B-Instruct",
        arch="llama",
        gguf_path=r"C:\Users\sgi.lm\Desktop\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        n_layer=16,
        n_embd=2048,
        n_head_kv=8,
        rope_type=0,
        rope_freq_scale=1.0,
        n_vocab=128256,
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    if model_id not in MODEL_PROFILES:
        raise KeyError(f"Unknown model_id '{model_id}'. Available: {list(MODEL_PROFILES.keys())}")
    return MODEL_PROFILES[model_id]


def to_cmcp_profile(
    model_id: str,
    n_ctx: int = 1024,
    pos_offset: int = 0,
) -> CMCPModelProfile:
    spec = get_model_spec(model_id)
    return make_model_profile(
        arch_name=spec.arch,
        weights_hash=cmcp_fnv1a_64(spec.model_id.encode("utf-8")),
        tokenizer_hash=cmcp_fnv1a_64(f"{spec.arch}:{spec.n_vocab}".encode("utf-8")),
        n_layer=spec.n_layer,
        n_embd=spec.n_embd,
        n_head_kv=spec.n_head_kv,
        rope_type=spec.rope_type,
        rope_freq_scale=spec.rope_freq_scale,
        kv_dtype=spec.kv_dtype,
        n_ctx=n_ctx,
        pos_offset=pos_offset,
    )
