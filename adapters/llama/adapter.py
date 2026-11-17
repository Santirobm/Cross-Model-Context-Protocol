"""Llama-3.2-1B-Instruct subprocess adapter using Llama-3.2 instruct template."""

from __future__ import annotations

from typing import Optional

from adapters.base.adapter import LlamaCppSubprocessAdapter
from adapters.model_registry.registry import get_model_spec


class LlamaAdapter(LlamaCppSubprocessAdapter):
    """Adapter for Llama-3.2-1B-Instruct via build/cmcp_llama_harness.exe."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 1024,
        n_threads: int = 4,
        harness_path: Optional[str] = None,
    ) -> None:
        spec = get_model_spec("Llama-3.2-1B-Instruct")
        super().__init__(
            model_name=spec.model_id,
            model_path=model_path or spec.gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            harness_path=harness_path,
        )

    def format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
