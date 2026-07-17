# -*- coding: utf-8 -*-
"""Small post-finalization adjustments for renamed cross-sectional canonicals."""
from cleaned_operators.registry import OperatorRegistry

_APPLIED = False


def apply_post_governance() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from cleaned_operators import operator_surface

    renamed = {"cs_count", "cs_mean", "cs_std", "cs_sum"} & set(OperatorRegistry._operators)
    operator_surface.DAILY_CANONICALS = frozenset(set(operator_surface.DAILY_CANONICALS) | renamed)
    for canonical in renamed:
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is not None:
            catalog["surface"] = "daily"
            catalog["scope"] = "cross_sectional"
            catalog["pit_safe"] = True
    _APPLIED = True
