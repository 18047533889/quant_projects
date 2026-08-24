# -*- coding: utf-8 -*-
"""R35 §92 / §104: model-operator contracts — FeatureLabelTiming + ModelOperatorContract.

``model_timing.ModelTimingContract`` stays the single PIT-timing authority (R28
gate).  This module ADDS the two things R35-P0-M04 / §104 require on top of it
without touching that file:

1. :class:`FeatureLabelTiming` — the precise feature/label visibility of a
   forward-label model that a single ``fit_cutoff_offset`` cannot express
   ("features through t, labels only through t-1, output at t+1").
2. :class:`ModelOperatorContract` — the full per-model role / inputs / fit /
   convergence / cost contract used by the model lane assignment (§104) and the
   model test obligations (§61).

Consumes ``ModelTimingContract`` (from :mod:`cleaned_operators.model_timing`);
does not redefine or duplicate its facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from factor_engine.cleaned_operators.model_timing import (
    ModelTimingContract,
    get_model_timing_contract,
    is_model_like_name,
)

__all__ = [
    "FeatureLabelTiming",
    "ModelOperatorContract",
    "MODEL_OPERATOR_CONTRACTS",
    "get_model_operator_contract",
    "feature_label_timing_of",
]


@dataclass(frozen=True)
class FeatureLabelTiming:
    """Precise feature/label visibility of a forward-label model (R35-P0-M04).

    A single ``fit_cutoff_offset`` cannot express "features through t, labels
    only through t-1, output at t+1" — the shape of HAR / panel forecasters.
    This structure pins each side of the training information independently::

        decision_row = t
        feature_origin_offset    : most recent FEATURE row that may enter a
                                   training sample (0 => feature row t usable)
        label_origin_offset      : most recent LABEL anchor s that may enter a
                                   training sample (1 => labels anchored at t-1
                                   and earlier only)
        label_maturity_offset    : H in ``label origin s matures at s + H``
        fit_latest_mature_label_offset : rows before decision_row at which the
                                   newest MATURE label may sit (0 => a label
                                   matured at t may be fitted)
        score_feature_offset     : feature rows usable for the OUTPUT score
                                   (0 => score from feature row t)
    """

    feature_origin_offset: int
    label_origin_offset: int
    label_maturity_offset: int = 1
    fit_latest_mature_label_offset: int = 0
    score_feature_offset: int = 0

    def implies_fit_cutoff(self) -> int:
        """Aggregate fit cutoff consistent with these precise offsets.

        The fit may use feature rows through ``t - feature_origin_offset`` and
        labels matured at ``t - fit_latest_mature_label_offset``; the effective
        prior-cutoff is the stricter of the two.
        """
        return max(
            0,
            self.feature_origin_offset,
            self.fit_latest_mature_label_offset,
        )


@dataclass(frozen=True)
class ModelOperatorContract:
    """Full per-model-operator contract (R35 §104).

    ``timing`` delegates to :class:`cleaned_operators.model_timing.ModelTimingContract`
    (the single PIT authority).  The rest covers role / feature inputs / label /
    fit / convergence / cost for model-lane assignment and test obligations.
    """

    canonical: str
    role: str                            # model_feature / model_score / diagnostic
    feature_params: tuple[str, ...] = ()
    required_feature_count_min: int = 1
    required_feature_count_max: int = 4
    label_param: str | None = None
    label_horizon_param: str | None = "label_horizon"
    fit_cutoff_rule: str = "fit_cutoff_offset"     # contract field used for fit cutoff
    label_maturity_rule: str = "label_horizon"     # how maturity is enforced
    scaler_cutoff_rule: str = "scaler_fit_cutoff_offset"
    hyperparam_cutoff_rule: str = "hyperparam_fit_cutoff_offset"
    refit_policy: str = "each_window"              # each_window / step_every_k
    missing_policy: str = "finite_only"
    convergence_policy: str = "fail_closed"
    min_train_obs: int = 10
    stateful: bool = False
    checkpoint_supported: bool = False
    cost_class: str = "medium"                     # low / medium / high / very_high

    @property
    def timing(self) -> ModelTimingContract:
        return get_model_timing_contract(self.canonical)


#: Explicit R35 model-operator contracts.  Kept deliberately small — the set of
#: models that need full per-operator modelling (vs a family default) is the
#: supervised / forward-label / stateful subset.  Descriptive in-sample stats
#: without a label structure resolve to the timing contract alone.
MODEL_OPERATOR_CONTRACTS: dict[str, ModelOperatorContract] = {
    "panel_rolling_pcr_forecast": ModelOperatorContract(
        "panel_rolling_pcr_forecast", "model_score",
        feature_params=("x1", "x2", "x3", "x4"),
        required_feature_count_min=1, required_feature_count_max=4,
        label_param="y", label_horizon_param="label_horizon",
        min_train_obs=10, cost_class="high",
    ),
    "panel_rolling_pls_forecast": ModelOperatorContract(
        "panel_rolling_pls_forecast", "model_score",
        feature_params=("x1", "x2", "x3", "x4"),
        required_feature_count_min=1, required_feature_count_max=4,
        label_param="y", label_horizon_param="label_horizon",
        min_train_obs=10, cost_class="high",
    ),
    "panel_rolling_elastic_net_forecast": ModelOperatorContract(
        "panel_rolling_elastic_net_forecast", "model_score",
        feature_params=("x1", "x2", "x3", "x4"),
        required_feature_count_min=1, required_feature_count_max=4,
        label_param="y", label_horizon_param="label_horizon",
        convergence_policy="fail_closed", min_train_obs=10, cost_class="high",
    ),
    "panel_regime_conditioned_forecast": ModelOperatorContract(
        "panel_regime_conditioned_forecast", "model_score",
        feature_params=("x1", "x2", "x3", "x4", "market_state"),
        required_feature_count_min=2, required_feature_count_max=5,
        label_param="y", label_horizon_param="label_horizon",
        min_train_obs=8, cost_class="high",
    ),
    "panel_mixture_of_experts_score": ModelOperatorContract(
        "panel_mixture_of_experts_score", "model_score",
        feature_params=("x1", "x2", "x3", "x4", "market_state"),
        required_feature_count_min=2, required_feature_count_max=5,
        label_param="y", label_horizon_param="label_horizon",
        min_train_obs=15, cost_class="very_high",
    ),
    "ts_garch_next_vol_forecast": ModelOperatorContract(
        "ts_garch_next_vol_forecast", "model_score",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        label_param=None, label_horizon_param=None,
        convergence_policy="fail_closed", cost_class="high",
    ),
    "ts_garch_standardized_shock": ModelOperatorContract(
        "ts_garch_standardized_shock", "model_feature",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        cost_class="high",
    ),
    "ts_har_rv_next_vol_forecast": ModelOperatorContract(
        "ts_har_rv_next_vol_forecast", "model_score",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        label_param=None, label_horizon_param=None,
        convergence_policy="fail_closed", cost_class="medium",
    ),
    "ts_ar_prior_forecast": ModelOperatorContract(
        "ts_ar_prior_forecast", "model_score",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        cost_class="low",
    ),
    "ts_kalman_level": ModelOperatorContract(
        "ts_kalman_level", "model_feature",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        stateful=True, checkpoint_supported=False, cost_class="low",
    ),
    # M-070: all six Kalman canonicals are causal one-pass filters and therefore
    # STATEFUL (time-shard requires state handoff), but NOT checkpointable — the
    # runtime has no Kalman checkpoint authority (StatefulCheckpointRegistry);
    # the honest execution model is full-history replay
    # (restore_strategy=full_replay).  checkpoint_supported=False matches the
    # kernel contract in ts_model/state_space.py (KALMAN_STATEFUL_CANONICALS +
    # kalman_stateful_contract()).
    "ts_kalman_trend": ModelOperatorContract(
        "ts_kalman_trend", "model_feature",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        stateful=True, checkpoint_supported=False, cost_class="low",
    ),
    "ts_kalman_beta": ModelOperatorContract(
        "ts_kalman_beta", "model_feature",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        stateful=True, checkpoint_supported=False, cost_class="low",
    ),
    "ts_kalman_beta_change": ModelOperatorContract(
        "ts_kalman_beta_change", "model_feature",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        stateful=True, checkpoint_supported=False, cost_class="low",
    ),
    "ts_kalman_beta_uncertainty": ModelOperatorContract(
        "ts_kalman_beta_uncertainty", "model_feature",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        stateful=True, checkpoint_supported=False, cost_class="low",
    ),
    "ts_kalman_innovation_z": ModelOperatorContract(
        "ts_kalman_innovation_z", "model_feature",
        feature_params=(), required_feature_count_min=0, required_feature_count_max=0,
        stateful=True, checkpoint_supported=False, cost_class="low",
    ),
}


def get_model_operator_contract(canonical: str) -> ModelOperatorContract | None:
    """Return the explicit R35 model-operator contract, or None for model-like
    canonicals without one (descriptive in-sample stats use timing alone)."""
    return MODEL_OPERATOR_CONTRACTS.get(canonical)


def feature_label_timing_of(canonical: str) -> FeatureLabelTiming | None:
    """FeatureLabelTiming for the model operators that need one (R35-P0-M04).

    Returns the precise feature/label visibility for forward-label / next-period
    forecast models whose single ``fit_cutoff_offset`` is insufficiently precise.
    ``None`` means the canonical is descriptive in-sample state or otherwise has
    no label structure — the timing contract alone is authoritative.

    The values here are derived from the actual kernels:
    - ``panel_*_forecast`` (``_forecast_loop``): trains on feature rows strictly
      before the current row and excludes the last ``label_horizon`` label rows
      (maturity), scores the current feature row.  Fit cutoff = 1, label
      maturity = label_horizon.
    - ``ts_har_*`` next-period forecasts (``_har_rv`` ``forecast``): trains on
      label ``RV_{s+1}`` anchored at s (matures at s+1); at decision row t the
      newest mature label is RV_t, scored from feature row t.  The innovation /
      error-z variants are strictly prior (fit through t-1, score from t-1).
    """
    if canonical.startswith("panel_") and canonical.endswith("_forecast"):
        return FeatureLabelTiming(
            feature_origin_offset=1,   # train on rows < t
            label_origin_offset=1,
            label_maturity_offset=1,   # default label_horizon; kernel enforces h
            fit_latest_mature_label_offset=1,
            score_feature_offset=0,    # score current feature row
        )
    if canonical in (
        "ts_har_rv_next_vol_forecast",
        "ts_har_rv_next_var_forecast",
        "ts_har_rv_forecast",
        "ts_har_from_return_next_vol",
    ):
        return FeatureLabelTiming(
            feature_origin_offset=0,   # feature row t usable (RV_t + comps)
            label_origin_offset=1,     # newest label anchored at t-1
            label_maturity_offset=1,   # RV_{s+1} matures at s+1
            fit_latest_mature_label_offset=0,  # RV_t matured at t is fitted
            score_feature_offset=0,    # score from feature row t
        )
    if canonical in (
        "ts_har_rv_forecast_error_z",
        "ts_har_rv_innovation_z",
        "ts_har_from_return_forecast_error_z",
    ):
        return FeatureLabelTiming(
            feature_origin_offset=1,   # train through t-1
            label_origin_offset=2,     # newest label anchored at t-2
            label_maturity_offset=1,
            fit_latest_mature_label_offset=1,
            score_feature_offset=1,    # score from feature row t-1
        )
    return None
