"""Explicit research-only operator access.

Research implementations are physically separated from the production
``OperatorRegistry``.  Importing this package is an explicit acknowledgement
that the selected utility is not part of the daily Factor DSL.
"""
from __future__ import annotations

from typing import Any

from factor_engine.backend.cleaned_bridge import build_cleaned_dsl_allowlist, ensure_cleaned_loaded
from factor_engine.cleaned_operators.operator_surface import UNSAFE_CANONICALS
from research_tools.registry import ResearchToolRegistry


def build_research_dsl_allowlist(*, include_unsafe: bool = False) -> dict[str, Any]:
    """Build the explicit research-tool surface outside production runtime."""
    ensure_cleaned_loaded()
    out: dict[str, Any] = {}
    for canonical, implementations in ResearchToolRegistry._tools.items():
        operator = implementations.get("pandas_numpy") or next(iter(implementations.values()), None)
        if operator is None:
            continue
        out[canonical] = operator
        for alias in ResearchToolRegistry._catalog.get(canonical, {}).get("aliases", []):
            out.setdefault(alias, operator)
    if include_unsafe:
        out.update(build_unsafe_dsl_allowlist())
    return out


def build_unsafe_dsl_allowlist() -> dict[str, Any]:
    """Build the explicit non-causal/random utility surface."""
    return build_cleaned_dsl_allowlist(surface="unsafe")


def get_research_operator(name: str, *, backend: str = "pandas_numpy"):
    """Return a research implementation from the separate research registry."""
    ensure_cleaned_loaded()
    canonical = name
    if canonical not in ResearchToolRegistry._tools:
        for candidate, catalog in ResearchToolRegistry._catalog.items():
            if name in catalog.get("aliases", []):
                canonical = candidate
                break
    operator = ResearchToolRegistry.get(canonical, backend=backend)
    if operator is None:
        raise KeyError(f"research tool {name!r} has no {backend!r} runtime")
    return operator


def get_unsafe_operator(name: str, *, backend: str = "pandas_numpy"):
    """Return an unsafe operator only through explicit opt-in."""
    ensure_cleaned_loaded()
    from factor_engine.cleaned_operators.registry import OperatorRegistry

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
