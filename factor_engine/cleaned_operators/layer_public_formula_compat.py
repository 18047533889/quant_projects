# -*- coding: utf-8 -*-
"""Compatibility adjustments grounded in validated public factor formulas."""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry


def apply_public_formula_compatibility() -> None:
    from cleaned_operators import operator_surface

    if "tanh" not in OperatorRegistry._operators:
        return
    operator_surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(getattr(operator_surface, "EXTENDED_ONLY_CANONICALS", ())) - {"tanh"}
    )
    operator_surface.DAILY_CANONICALS = frozenset(
        set(operator_surface.DAILY_CANONICALS) | {"tanh"}
    )
    catalog = OperatorRegistry._catalog.get("tanh")
    if catalog is not None:
        catalog["surface"] = "daily"
        catalog["authoring_default"] = True
        catalog.pop("extended_reason", None)


__all__ = ["apply_public_formula_compatibility"]
