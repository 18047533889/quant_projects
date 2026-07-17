"""Public daily-factor DSL allowlist.

The registry contains additional compatibility and research runtimes, but new
factor formulas only see the causal daily-factor surface.
"""
from __future__ import annotations

from typing import Any, Callable

from api.columns import col
from backend.cleaned_bridge import build_cleaned_dsl_allowlist

STUB_IR_OPS: frozenset[str] = frozenset()


def build_dsl_allowlist(*, surface: str = "daily") -> dict[str, Callable[..., Any]]:
    """Return a surface-scoped DSL mapping.

    ``compat`` is reserved for parsing already-published formulas.  It adds
    extended convenience dispatchers without exposing them to new daily
    authoring or production admission.
    """
    allow: dict[str, Callable[..., Any]] = {"col": col}
    surfaces = ("daily", "extended") if surface == "compat" else (surface,)
    for selected in surfaces:
        allow.update(build_cleaned_dsl_allowlist(set(), surface=selected))
    return allow
