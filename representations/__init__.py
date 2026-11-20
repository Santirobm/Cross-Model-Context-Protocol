"""CMCP context representations (structured, binary, latent)."""

from .adapters import render_selected_unified, serialize
from .selector import SelectedContext, select_for_task
from .state_store import CMCPStateStore, StateOp

__all__ = [
    "CMCPStateStore",
    "SelectedContext",
    "StateOp",
    "render_selected_unified",
    "select_for_task",
    "serialize",
]
