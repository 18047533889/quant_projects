# -*- coding: utf-8 -*-
"""Approved hyperparameter search spaces and per-parameter search governance
(Model Layer Major Redesign taskbook §38 / §69).

``APPROVED_SEARCH_SPACES`` are the grids the miner may search.  ``param_search_policy``
resolves the §69 recommendation table so a parameter is only searchable when its
role and ``searchable_by_miner`` flag allow it.
"""
from __future__ import annotations

from modeling.contracts import ParameterSearchPolicy, ParamRole
from modeling.presets import recommendation_map

__all__ = ["APPROVED_SEARCH_SPACES", "param_search_policy"]


APPROVED_SEARCH_SPACES: dict[str, list[dict]] = {
    "pcr": [
        {"n_components": 2},
        {"n_components": 3},
        {"n_components": 5},
        {"n_components": 8},
    ],
    "pls": [
        {"n_components": 2},
        {"n_components": 3},
        {"n_components": 5},
        {"n_components": 8},
    ],
    "elastic_net": [
        {"alpha": alpha, "l1_ratio": l1_ratio}
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1)
        for l1_ratio in (0.0, 0.25, 0.5, 0.75, 1.0)
    ],
    "regime": [{"n_regimes": 2}, {"n_regimes": 3}, {"n_regimes": 4}],
    "moe": [{"n_experts": 2}, {"n_experts": 3}],
}


def param_search_policy(family: str, param: str) -> ParameterSearchPolicy | None:
    """Resolve the §69 recommendation for ``(family, param)``.

    Returns ``None`` for unknown parameters.  NUMERICAL_POLICY parameters are
    never searchable (§15) regardless of the table flag.
    """
    rec = recommendation_map().get((family, param))
    if rec is None:
        return None
    role = rec["role"]
    searchable = bool(rec["searchable_by_miner"])
    if role == ParamRole.NUMERICAL_POLICY:
        searchable = False
    return ParameterSearchPolicy(role=role, searchable=searchable)
