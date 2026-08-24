# -*- coding: utf-8 -*-
"""Keep reviewed factor-shaped Extended operators in FactorEngine."""
from __future__ import annotations

_APPLIED = False


def enable_research_factor_runtime() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from factor_engine.cleaned_operators.production_tiers import FACTOR_LIKE_RESEARCH_CANONICALS
    from factor_engine.cleaned_operators.operator_surface import EXTENDED_ONLY_CANONICALS
    from factor_engine.cleaned_operators import layer_governance

    keep = set(FACTOR_LIKE_RESEARCH_CANONICALS) | set(EXTENDED_ONLY_CANONICALS)
    layer_governance.RESEARCH_CANONICALS = frozenset(
        set(layer_governance.RESEARCH_CANONICALS) - keep
    )
    _APPLIED = True
