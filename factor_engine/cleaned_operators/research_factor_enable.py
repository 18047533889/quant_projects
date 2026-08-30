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
    # R55 platform-audit P0: the LQTP batch re-landing relies on three FE
    # canonicals that layer_governance moved to RESEARCH_CANONICALS because
    # they were historically classified as "unlimited repair" utilities:
    #   - fillna(x, v)        — 81 platform formulas (nan_to_num rewrite target)
    #   - avg2(a, b)          — 11 platform formulas
    #   - ts_positive_streak  — 1 platform formula (consecutive-positive run)
    # They are elementwise PIT-safe helpers, not repair escapes; keep them in
    # the FactorEngine runtime surface so the LQTP formulas parse and land.
    keep |= {"fillna", "avg2", "ts_positive_streak"}
    layer_governance.RESEARCH_CANONICALS = frozenset(
        set(layer_governance.RESEARCH_CANONICALS) - keep
    )
    _APPLIED = True
