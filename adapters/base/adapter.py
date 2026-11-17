"""Base ModelAdapter interface and LlamaCppSubprocessAdapter implementation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AdapterResponse:
    """Standardized response and telemetry from a ModelAdapter invocation."""

    model_name: str
    arch: str
    output_text: str
    input_tokens: int
    output_tokens: int
    tokenize_ms: float
    prefill_ms: float
    eval_ms: float
    wall_ms: float
    rss_mib: float
    info_loss_notes: List[str] = field(default_factory=list)
    raw_meta: Dict[str, object] = field(default_factory=dict)

    @property
    def throughput_tok_s(self) -> float:
        if self.eval_ms <= 0.0:
            return 0.0
        return (self.output_tokens / self.eval_ms) * 1000.0


class ModelAdapter(ABC):
    """Abstract base class for CMCP model adapters."""

    def __init__(self, model_name: str, model_path: str, n_ctx: int = 1024, n_threads: int = 4) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads

    @abstractmethod
    def format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format system and user prompts into the model's native chat template."""

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a precise technical assistant.",
        max_tokens: int = 128,
        info_loss_notes: Optional[List[str]] = None,
    ) -> AdapterResponse:
        """Run inference and return AdapterResponse with token and timing telemetry."""


class LlamaCppSubprocessAdapter(ModelAdapter):
    """Real adapter invoking build/cmcp_llama_harness.exe against a local GGUF model."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        n_ctx: int = 1024,
        n_threads: int = 4,
        harness_path: Optional[str] = None,
    ) -> None:
        super().__init__(model_name=model_name, model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        repo_root = Path(__file__).resolve().parents[2]
        self.harness_path = str(
            Path(harness_path) if harness_path else repo_root / "build" / "cmcp_llama_harness.exe"
        )

    @staticmethod
    def _build_env() -> Dict[str, str]:
        env = os.environ.copy()
        extra = r"C:\msys64\clang64\bin;C:\Users\sgi.lm\Desktop\llama.cpp\build\bin;"
        env["PATH"] = extra + env.get("PATH", "")
        return env

    def _run_harness(self, args: List[str]) -> Dict[str, object]:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [self.harness_path, *args],
            env=self._build_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            raise RuntimeError(
                f"cmcp_llama_harness.exe failed (rc={proc.returncode}): {proc.stderr.strip()}"
            )
        data = json.loads(proc.stdout)
        data["_wall_ms"] = wall_ms
        return data

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a precise technical assistant.",
        max_tokens: int = 128,
        info_loss_notes: Optional[List[str]] = None,
    ) -> AdapterResponse:
        formatted = self.format_chat_prompt(system_prompt, user_prompt)
        data = self._run_harness(
            [
                "--model",
                self.model_path,
                "--mode",
                "bench",
                "--prompt",
                formatted,
                "--max-tokens",
                str(max_tokens),
                "--n-ctx",
                str(self.n_ctx),
                "--threads",
                str(self.n_threads),
            ]
        )
        notes = list(info_loss_notes) if info_loss_notes else []
        raw_out = str(data.get("output_text", "")).strip()
        return AdapterResponse(
            model_name=self.model_name,
            arch=str(data.get("arch", "unknown")),
            output_text=raw_out,
            input_tokens=int(data.get("n_input_tokens", 0)),
            output_tokens=int(data.get("n_gen_tokens", 0)),
            tokenize_ms=float(data.get("tokenize_ms", 0.0)),
            prefill_ms=float(data.get("prefill_ms", 0.0)),
            eval_ms=float(data.get("eval_ms", 0.0)),
            wall_ms=float(data.get("_wall_ms", 0.0)),
            rss_mib=float(data.get("rss_mib", 0.0)),
            info_loss_notes=notes,
            raw_meta=data,
        )

    def embed(self, text: str) -> Dict[str, object]:
        """Extract sequence embedding metadata via --mode embed."""
        return self._run_harness(
            [
                "--model",
                self.model_path,
                "--mode",
                "embed",
                "--prompt",
                text,
                "--n-ctx",
                str(self.n_ctx),
                "--threads",
                str(self.n_threads),
            ]
        )
