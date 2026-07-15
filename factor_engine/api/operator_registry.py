"""Public daily-factor DSL allowlist.

The registry contains additional compatibility and research runtimes, but new
factor formulas only see the causal daily-factor surface.
"""
from __future__ import annotations

from typing import Any, Callable

from api.columns import col
from backend.cleaned_bridge import build_cleaned_dsl_allowlist

STUB_IR_OPS: frozenset[str] = frozenset()


def build_dsl_allowlist() -> dict[str, Callable[..., Any]]:
    """Return the public causal daily-factor DSL function mapping."""
    allow: dict[str, Callable[..., Any]] = {"col": col}
    allow.update(build_cleaned_dsl_allowlist(set(), surface="daily"))
    return allow
