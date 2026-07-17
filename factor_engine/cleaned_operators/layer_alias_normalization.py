# -*- coding: utf-8 -*-
"""Flatten historical operator aliases to exact runtime canonicals."""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry


def normalize_historical_aliases() -> None:
    aliases = OperatorRegistry._aliases

    # Historical modules occasionally register ``canonical -> canonical``.
    # These are terminal markers, not meaningful aliases.
    for alias, target in list(aliases.items()):
        if alias == target:
            aliases.pop(alias, None)

    for alias in list(aliases):
        current = aliases.get(alias)
        visited = {alias}
        while current in aliases:
            if current in visited:
                raise RuntimeError(f"cyclic operator alias involving {alias!r}")
            visited.add(str(current))
            next_target = aliases[current]
            if next_target == current:
                aliases.pop(current, None)
                break
            current = next_target
        if current not in OperatorRegistry._operators:
            raise RuntimeError(f"operator alias target unavailable: {alias} -> {current}")
        aliases[alias] = current

    for canonical, catalog in OperatorRegistry._catalog.items():
        catalog["aliases"] = sorted(
            alias for alias, target in aliases.items() if target == canonical
        )


__all__ = ["normalize_historical_aliases"]
