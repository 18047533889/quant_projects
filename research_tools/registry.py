# -*- coding: utf-8 -*-
"""Registry for diagnostics and research utilities excluded from factor authoring."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class ResearchToolRegistry:
    """Store research implementations without exposing them to the Factor DSL."""

    _tools: dict[str, dict[str, Any]] = {}
    _catalog: dict[str, dict[str, Any]] = {}

    @classmethod
    def register_moved(
        cls,
        canonical: str,
        implementations: dict[str, Any],
        catalog: dict[str, Any] | None,
        *,
        reason: str,
    ) -> None:
        cls._tools[canonical] = dict(implementations)
        metadata = deepcopy(catalog or {})
        metadata.update(
            {
                "canonical": canonical,
                "surface": "research_tools",
                "migration_reason": reason,
                "backends": sorted(implementations),
            }
        )
        cls._catalog[canonical] = metadata

    @classmethod
    def get(cls, name: str, backend: str = "pandas_numpy") -> Any | None:
        return cls._tools.get(name, {}).get(backend)

    @classmethod
    def list_canonical(cls) -> list[str]:
        return sorted(cls._tools)

    @classmethod
    def catalog(cls) -> dict[str, dict[str, Any]]:
        return deepcopy(cls._catalog)
