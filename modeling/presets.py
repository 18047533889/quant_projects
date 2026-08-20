# -*- coding: utf-8 -*-
"""A-share model-training presets and parameter recommendation table
(Model Layer Major Redesign taskbook §3.3 / §67 / §68 / §69).

Enterprise defaults — explicit, audited starting points, never hardcoded
truth.  The mining search space may not widen these without a reviewed
:class:`modeling.contracts.ParameterSearchPolicy`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modeling.contracts import ParamRole

__all__ = [
    "WalkForwardPreset",
    "PREDICTIVE_LINEAR_DEFAULT",
    "PREDICTIVE_REGIME_DEFAULT",
    "PREDICTIVE_MOE_DEFAULT",
    "LOCAL_LEGACY_FLOORS",
    "MODEL_PARAM_RECOMMENDATIONS",
]


@dataclass(frozen=True)
class WalkForwardPreset:
    """A validated walk-forward preset (§3.3 / §68)."""

    name: str
    train_mode: str                       # rolling / expanding / decay_weighted_expanding
    train_lookback_bars: int
    validation_bars: int
    test_bars: int
    step_bars: int
    retrain_every_bars: int
    purge_by_label_interval: bool = True
    embargo_bars: int = 5
    min_train_dates: int = 252
    min_train_stocks: int = 30
    min_train_obs: int = 10_000
    decay_half_life_bars: int | None = None   # decay_weighted_expanding only


PREDICTIVE_LINEAR_DEFAULT = WalkForwardPreset(
    name="predictive_linear_default",
    train_mode="rolling",
    train_lookback_bars=1000,     # ~4 trading years
    validation_bars=252,          # ~12 months
    test_bars=126,                # ~6 months
    step_bars=21,
    retrain_every_bars=21,        # monthly
    purge_by_label_interval=True,
    embargo_bars=5,
    min_train_dates=252,
    min_train_stocks=30,
    min_train_obs=10_000,
)

PREDICTIVE_REGIME_DEFAULT = WalkForwardPreset(
    name="predictive_regime_default",
    train_mode="decay_weighted_expanding",
    train_lookback_bars=1000,
    validation_bars=252,
    test_bars=126,
    step_bars=21,
    retrain_every_bars=21,
    purge_by_label_interval=True,
    embargo_bars=5,
    min_train_dates=252,
    min_train_stocks=30,
    min_train_obs=50_000,
    decay_half_life_bars=1260,
)

PREDICTIVE_MOE_DEFAULT = WalkForwardPreset(
    name="predictive_moe_default",
    train_mode="decay_weighted_expanding",
    train_lookback_bars=1000,
    validation_bars=252,
    test_bars=126,
    step_bars=21,
    retrain_every_bars=21,
    purge_by_label_interval=True,
    embargo_bars=5,
    min_train_dates=252,
    min_train_stocks=30,
    min_train_obs=100_000,
    decay_half_life_bars=1260,
)


# §67 — research-safety floors for the retained legacy local variants.
LOCAL_LEGACY_FLOORS: dict[str, dict[str, Any]] = {
    "local_pcr_pls_enet": {
        "min_effective_obs": None,        # computed as max(60, 10 * free_params)
        "min_obs_per_parameter": 10.0,
        "min_effective_obs_floor": 60,
    },
    "local_regime": {
        "min_regime_obs": None,           # max(30, 10 * free_params) per regime
        "min_obs_per_parameter": 10.0,
        "min_regime_obs_floor": 30,
    },
    "local_moe": {
        "min_expert_obs": None,           # max(30, 10 * free_params) per expert
        "min_obs_per_parameter": 10.0,
        "min_expert_obs_floor": 30,
    },
}


# §69 — per-model parameter recommendation table.
# (canonical_family, parameter, role, searchable_by_miner)
MODEL_PARAM_RECOMMENDATIONS: list[tuple[str, str, ParamRole, bool]] = [
    ("pcr", "train_lookback", ParamRole.ECONOMIC_HORIZON, True),
    ("pcr", "n_components", ParamRole.MODEL_COMPLEXITY, True),
    ("pcr", "label_horizon", ParamRole.ECONOMIC_HORIZON, True),
    ("pls", "n_components", ParamRole.MODEL_COMPLEXITY, True),
    ("elastic_net", "alpha", ParamRole.REGULARIZATION, False),
    ("elastic_net", "l1_ratio", ParamRole.REGULARIZATION, False),
    ("regime", "n_regimes", ParamRole.MODEL_COMPLEXITY, True),
    ("regime", "market_state", ParamRole.ECONOMIC_HORIZON, True),
    ("moe", "n_experts", ParamRole.MODEL_COMPLEXITY, True),
    ("kalman", "q", ParamRole.REGULARIZATION, False),
    ("kalman", "r", ParamRole.REGULARIZATION, False),
    ("garch", "window", ParamRole.ECONOMIC_HORIZON, True),
    ("ar", "order", ParamRole.MODEL_COMPLEXITY, True),
    ("ar", "window", ParamRole.ECONOMIC_HORIZON, True),
    ("dmd", "rank", ParamRole.ESTIMATOR_RESOLUTION, False),
    ("dmd", "dim", ParamRole.ESTIMATOR_RESOLUTION, False),
    ("dmd", "delay", ParamRole.ESTIMATOR_RESOLUTION, False),
    ("rqa", "dim", ParamRole.ESTIMATOR_RESOLUTION, False),
    ("rqa", "delay", ParamRole.ESTIMATOR_RESOLUTION, False),
    ("rqa", "eps", ParamRole.ESTIMATOR_RESOLUTION, False),
    ("te", "bins", ParamRole.ESTIMATOR_RESOLUTION, False),
    ("knn", "k", ParamRole.ESTIMATOR_RESOLUTION, True),
    ("ssa", "components", ParamRole.ESTIMATOR_RESOLUTION, False),
]


def recommendation_map() -> dict[tuple[str, str], dict[str, Any]]:
    """Index the §69 table by ``(family, parameter)``."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for family, param, role, searchable in MODEL_PARAM_RECOMMENDATIONS:
        out[(family, param)] = {"role": role, "searchable_by_miner": searchable}
    return out
