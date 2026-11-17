"""Model registry for supported CMCP local GGUF models."""

from .registry import MODEL_PROFILES, ModelSpec, get_model_spec, to_cmcp_profile

__all__ = ["MODEL_PROFILES", "ModelSpec", "get_model_spec", "to_cmcp_profile"]
