"""Deterministic structured ingestion hooks for CMCP (zero LLM calls)."""

from .structured_hooks import extract_explicit_constraints, ingest_tool_output

__all__ = [
    "extract_explicit_constraints",
    "ingest_tool_output",
]
