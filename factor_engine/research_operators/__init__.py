"""Explicit research-only operator access.

This package is intentionally separate from ``api``.  Importing it is an
explicit acknowledgement that the selected operators are not part of the
normal daily-factor DSL and are not production-certified.
"""
from __future__ import annotations

from typing import Any

from backend.cleaned_bridge import build_cleaned_dsl_allowlist, ensure_cleaned_loaded
from cleaned_operators.operator_surface import (
    RESEARCH_ONLY_CANONICALS,
    UNSAFE_CANONICALS,
)


def build_research_dsl_allowlist(*, include_unsafe: bool = False) -> dict[str, Any]:
    """Build the research utility DSL surface, optionally including unsafe tools."""
    out = build_cleaned_dsl_allowlist(surface="research")
    if include_unsafe:
        out.update(build_cleaned_dsl_allowlist(surface="unsafe"))
    return out


def build_unsafe_dsl_allowlist() -> dict[str, Any]:
    """Build the explicit non-causal/random utility surface."""
    return build_cleaned_dsl_allowlist(surface="unsafe")


def get_research_operator(name: str, *, backend: str = "pandas_numpy"):
    """Return a research operator runtime after validating its surface."""
    ensure_cleaned_loaded()
    from cleaned_operators.registry import OperatorRegistry

    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in RESEARCH_ONLY_CANONICALS:
        raise KeyError(f"{name!r} is not a research-only operator")
    operator = OperatorRegistry.get(canonical, backend=backend)
    if operator is None:
        raise KeyError(f"research operator {name!r} has no {backend!r} runtime")
    return operator


def get_unsafe_operator(name: str, *, backend: str = "pandas_numpy"):
    """Return an unsafe operator only through explicit opt-in."""
    ensure_cleaned_loaded()
    from cleaned_operators.registry import OperatorRegistry

    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in UNSAFE_CANONICALS:
        raise KeyError(f"{name!r} is not an unsafe operator")
    operator = OperatorRegistry.get(canonical, backend=backend)
    if operator is None:
        raise KeyError(f"unsafe operator {name!r} has no {backend!r} runtime")
    return operator


__all__ = [
    "build_research_dsl_allowlist",
    "build_unsafe_dsl_allowlist",
    "get_research_operator",
    "get_unsafe_operator",
]
