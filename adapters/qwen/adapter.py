"""Qwen2.5-0.5B-Instruct subprocess adapter using ChatML template."""

from __future__ import annotations

from typing import Optional

from adapters.base.adapter import LlamaCppSubprocessAdapter
from adapters.model_registry.registry import get_model_spec


class QwenAdapter(LlamaCppSubprocessAdapter):
    """Adapter for Qwen2.5-0.5B-Instruct via build/cmcp_llama_harness.exe."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 1024,
        n_threads: int = 4,
        harness_path: Optional[str] = None,
    ) -> None:
        spec = get_model_spec("Qwen2.5-0.5B-Instruct")
        super().__init__(
            model_name=spec.model_id,
            model_path=model_path or spec.gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            harness_path=harness_path,
        )

    def format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
