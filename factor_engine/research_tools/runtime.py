# -*- coding: utf-8 -*-
"""Executable runtime for diagnostics and research-only operators.

Research-only means "not admitted to production factor publication"; it must
not mean "unusable".  This runtime gives every implementation retained in
``ResearchToolRegistry`` a supported call path without polluting the production
Factor DSL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_tools.registry import ResearchToolRegistry


class ResearchToolError(RuntimeError):
    """Research tool lookup or execution failed."""


@dataclass(frozen=True)
class ResearchToolSelection:
    canonical: str
    backend: str
    available_backends: tuple[str, ...]


class ResearchToolRuntime:
    """Run diagnostics/research utilities with explicit backend selection."""

    @staticmethod
    def select(name: str, *, backend: str = "auto") -> ResearchToolSelection:
        canonical = str(name)
        implementations = ResearchToolRegistry._tools.get(canonical)
        if implementations is None:
            # Research tool aliases are metadata only after migration; resolve
            # them deterministically here.
            for candidate, meta in ResearchToolRegistry._catalog.items():
                if canonical in set(meta.get("aliases") or []):
                    canonical = candidate
                    implementations = ResearchToolRegistry._tools.get(candidate)
                    break
        if not implementations:
            raise ResearchToolError(f"unknown research tool: {name!r}")

        available = tuple(sorted(implementations))
        requested = str(backend or "auto").lower()
        if requested == "auto":
            if "polars" in implementations:
                selected = "polars"
            elif "pandas_numpy" in implementations:
                selected = "pandas_numpy"
            else:
                selected = available[0]
        else:
            selected = requested
            if selected not in implementations:
                raise ResearchToolError(
                    f"research tool {canonical!r} has no backend {selected!r}; "
                    f"available={available}"
                )
        return ResearchToolSelection(canonical, selected, available)

    @classmethod
    def run(
        cls,
        name: str,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        selection = cls.select(name, backend=backend)
        operator = ResearchToolRegistry.get(selection.canonical, selection.backend)
        if operator is None:
            raise ResearchToolError(
                f"research tool backend disappeared: {selection.canonical}/{selection.backend}"
            )
        try:
            return operator.calculate(*args, **kwargs)
        except Exception as exc:
            raise ResearchToolError(
                f"research tool {selection.canonical!r} failed on {selection.backend}: {exc}"
            ) from exc

    @staticmethod
    def catalog() -> dict[str, dict[str, Any]]:
        return ResearchToolRegistry.catalog()

    @staticmethod
    def list_tools() -> list[str]:
        return ResearchToolRegistry.list_canonical()


def run_research_tool(
    name: str,
    *args: Any,
    backend: str = "auto",
    **kwargs: Any,
) -> Any:
    """Functional convenience wrapper around :class:`ResearchToolRuntime`."""
    return ResearchToolRuntime.run(name, *args, backend=backend, **kwargs)
