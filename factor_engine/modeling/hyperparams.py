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

__all__ = ["APPROVED_SEARCH_SPACES", "param_search_policy", "validate_search_grid"]


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


def validate_search_grid(family: str, grid: list[dict]) -> list[dict]:
    """Require a non-empty, duplicate-free subset of the reviewed search space.

    Unknown families, parameters, and values are boundary errors.  They must not
    be converted into candidate-level failures because that would let a miner
    bypass governance by submitting arbitrary values and relying on a skip.
    """
    import json

    approved = APPROVED_SEARCH_SPACES.get(family)
    if approved is None:
        raise ValueError(f"no approved search space for model family {family!r}")
    approved_ids = {
        json.dumps(candidate, sort_keys=True, separators=(",", ":"), default=str)
        for candidate in approved
    }
    result: list[dict] = []
    seen: set[str] = set()
    for candidate in grid:
        if not isinstance(candidate, dict) or not candidate:
            raise ValueError(f"invalid empty/non-mapping candidate: {candidate!r}")
        identity = json.dumps(candidate, sort_keys=True, separators=(",", ":"), default=str)
        if identity not in approved_ids:
            raise ValueError(
                f"candidate {candidate!r} is outside approved {family!r} search space"
            )
        if identity in seen:
            raise ValueError(f"duplicate search candidate {candidate!r}")
        seen.add(identity)
        result.append(dict(candidate))
    if not result:
        raise ValueError(f"empty approved search grid for family {family!r}")
    return result
