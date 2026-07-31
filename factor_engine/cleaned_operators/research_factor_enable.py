# -*- coding: utf-8 -*-
"""Keep factor-shaped research operators in the executable FactorEngine registry.

``layer_governance`` correctly moves diagnostics, tests, matrix decompositions
and other non-factor utilities into ``ResearchToolRegistry``.  A small subset of
research operators, however, still has the normal factor contract
(panel -> same-shape panel) and is useful for experimentation.  Those operators
should remain executable under ``surface='research'`` instead of becoming dead
code.

This hook runs immediately before ``finalize_layer_governance`` and only adjusts
which research entries are moved; it does not grant production admission.
"""
from __future__ import annotations

_APPLIED = False


def enable_research_factor_runtime() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from cleaned_operators.production_tiers import FACTOR_LIKE_RESEARCH_CANONICALS
    from cleaned_operators import layer_governance

    layer_governance.RESEARCH_CANONICALS = frozenset(
        set(layer_governance.RESEARCH_CANONICALS)
        - set(FACTOR_LIKE_RESEARCH_CANONICALS)
    )
    _APPLIED = True
