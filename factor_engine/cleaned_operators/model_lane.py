# -*- coding: utf-8 -*-
"""R35 §179 / §185: per-model readiness lane assignment.

The six-lane model taxonomy (taskbook §1 / §179) sits ON TOP of the surface
classification (:func:`cleaned_operators.operator_surface.classify_canonical`).
Every model-like canonical gets exactly ONE lane; zero unclassified.

Lanes:
- FAST_NATIVE_ALPHA          : cheap, PIT-correct, default-searchable (rolling
                               regression prior, AR prior, Kalman causal state)
- EXPENSIVE_CERTIFIED_ALPHA  : correct but high-cost (GARCH, DMD, robust/quantile)
- STATE_CONDITION_EVENT      : state/condition/event intermediates
- MODEL_FEATURE_SCORE        : supervised / model score (needs timing contract)
- DIAGNOSTIC_RESEARCH        : in-sample diagnostics / research-only
- DELETE_TOMBSTONE           : tombstoned / forbidden

Consumes:
- ``classify_canonical`` (surface) for the surface layer
- ``is_model_like_name`` / ``get_model_timing_contract`` (timing)
- ``get_model_operator_contract`` (explicit model contract)
- ``MODEL_OPERATOR_CONTRACTS`` role (model_feature / model_score / diagnostic)
"""
from __future__ import annotations

from typing import Any

from cleaned_operators.model_contract import (
    MODEL_OPERATOR_CONTRACTS,
    get_model_operator_contract,
)
from cleaned_operators.model_timing import (
    is_model_like_name,
    get_model_timing_contract,
)
from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.tombstones import is_tombstoned

__all__ = [
    "MODEL_LANES",
    "MODEL_LANE_LABELS",
    "assign_model_lane",
    "model_lane_of",
    "model_lane_inventory",
    "model_lane_errors",
    "MODEL_LANE_EXPLICIT",
]

#: Lane labels (taskbook §1).
MODEL_LANES: tuple[str, ...] = (
    "FAST_NATIVE_ALPHA",
    "EXPENSIVE_CERTIFIED_ALPHA",
    "STATE_CONDITION_EVENT",
    "MODEL_FEATURE_SCORE",
    "DIAGNOSTIC_RESEARCH",
    "DELETE_TOMBSTONE",
)
MODEL_LANE_LABELS = MODEL_LANES

#: Explicit per-canonical lane overrides (R35-reviewed).  Canonicals NOT listed
#: here resolve deterministically from surface + role + timing (see
#: :func:`assign_model_lane`), and every model-like canonical must land in a
#: lane — the inventory test asserts zero unclassified.
MODEL_LANE_EXPLICIT: dict[str, str] = {
    # --- EXPENSIVE_CERTIFIED_ALPHA ---
    "ts_garch_standardized_shock": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_garch_next_vol_forecast": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_garch_persistence": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_garch_vol_surprise": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_gjr_garch_vol_forecast": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_gjr_leverage": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_dmd_dominant_growth_rate": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_dmd_dominant_frequency": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_dmd_level_dominant_growth_rate": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_dmd_level_dominant_frequency": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_dmd_level_mode_concentration": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_dmd_mode_concentration": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_dmd_return_dominant_growth_rate": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_dmd_return_dominant_frequency": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_dmd_return_mode_concentration": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_hankel_singular_gap": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_hankel_effective_rank": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_ssa_reconstruction_residual": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_huber_regression_coeff": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_quantile_regression_coeff": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_quantile_regression_slope": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_expectile_regression_coeff": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_matrix_profile_discord_score": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_matrix_profile_motif_age": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_matrix_profile_motif_frequency": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_matrix_profile_neighbor_dispersion": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_matrix_profile_novelty": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_motif_recurrence_count": "EXPENSIVE_CERTIFIED_ALPHA",
    "ts_signature_mahalanobis_anomaly": "EXPENSIVE_CERTIFIED_ALPHA",
    # --- FAST_NATIVE_ALPHA ---
    "ts_ar_prior_forecast": "FAST_NATIVE_ALPHA",
    "ts_ar_prior_innovation": "FAST_NATIVE_ALPHA",
    "ts_ar_prior_innovation_z": "FAST_NATIVE_ALPHA",
    "ts_ar_prior_coeff": "FAST_NATIVE_ALPHA",
    "ts_ar_coefficient": "FAST_NATIVE_ALPHA",
    "ts_ar_coeff_stability": "FAST_NATIVE_ALPHA",
    "ts_har_rv_next_vol_forecast": "FAST_NATIVE_ALPHA",
    "ts_har_rv_next_var_forecast": "FAST_NATIVE_ALPHA",
    "ts_har_from_return_next_vol": "FAST_NATIVE_ALPHA",
    "ts_kalman_level": "FAST_NATIVE_ALPHA",
    "ts_kalman_trend": "FAST_NATIVE_ALPHA",
    "ts_kalman_beta": "FAST_NATIVE_ALPHA",
    "ts_kalman_beta_change": "FAST_NATIVE_ALPHA",
    "ts_kalman_beta_uncertainty": "FAST_NATIVE_ALPHA",
    "ts_kalman_innovation_z": "FAST_NATIVE_ALPHA",
    "ts_mean_reversion_half_life": "FAST_NATIVE_ALPHA",
    "ts_mean_reversion_ou_approx_half_life": "FAST_NATIVE_ALPHA",
    "ts_ridge_regression_coeff_prior": "FAST_NATIVE_ALPHA",
    "ts_multi_regression_coeff_prior": "FAST_NATIVE_ALPHA",
    "ts_multi_regression_r2_prior": "FAST_NATIVE_ALPHA",
    "ts_regression_slope": "FAST_NATIVE_ALPHA",
    # --- STATE_CONDITION_EVENT ---
    "ts_dynamic_knn_state": "STATE_CONDITION_EVENT",
    "ts_markov_regime_state": "STATE_CONDITION_EVENT",
    "ts_markov_transition_probability": "STATE_CONDITION_EVENT",
    "ts_regime_conditional_moment": "STATE_CONDITION_EVENT",
    "ts_state_space_regime_probability": "STATE_CONDITION_EVENT",
    "ts_hmm_*": "STATE_CONDITION_EVENT",  # placeholder, removed below if absent
    # --- MODEL_FEATURE_SCORE (supervised) ---
    "panel_rolling_pcr_forecast": "MODEL_FEATURE_SCORE",
    "panel_rolling_pls_forecast": "MODEL_FEATURE_SCORE",
    "panel_rolling_elastic_net_forecast": "MODEL_FEATURE_SCORE",
    "panel_regime_conditioned_forecast": "MODEL_FEATURE_SCORE",
    "panel_mixture_of_experts_score": "MODEL_FEATURE_SCORE",
    # --- DIAGNOSTIC_RESEARCH (in-sample) ---
    "ts_ar_fitted_value": "DIAGNOSTIC_RESEARCH",
    "ts_ar_in_sample_resid": "DIAGNOSTIC_RESEARCH",
    "ts_ar_forecast": "DIAGNOSTIC_RESEARCH",
    "ts_ar_innovation": "DIAGNOSTIC_RESEARCH",
    "ts_ar_innovation_z": "DIAGNOSTIC_RESEARCH",
    "ts_regression_resid": "DIAGNOSTIC_RESEARCH",
    "ts_regression_r2": "DIAGNOSTIC_RESEARCH",
    "cs_knn_local_linear_residual": "DIAGNOSTIC_RESEARCH",
    "ts_dynamic_knn_residual": "DIAGNOSTIC_RESEARCH",
    "ts_kernel_granger_causality": "DIAGNOSTIC_RESEARCH",
    "ts_kernel_granger_oos": "DIAGNOSTIC_RESEARCH",
    "ts_residualized_hsic": "DIAGNOSTIC_RESEARCH",
    "ts_hsic_dependence": "DIAGNOSTIC_RESEARCH",
    "ts_lyapunov_exponent": "DIAGNOSTIC_RESEARCH",
    "ts_rqa_*": "DIAGNOSTIC_RESEARCH",  # placeholder, removed below if absent
}
# Drop wildcard placeholders that don't correspond to real canonicals (they are
# family markers in the dict above; the deterministic fallback covers the rest).
_MODEL_LANE_EXPLICIT = {k: v for k, v in MODEL_LANE_EXPLICIT.items() if "*" not in k}


def _resolved_role(name: str) -> str | None:
    mc = get_model_operator_contract(name)
    if mc is not None:
        return mc.role
    return None


def assign_model_lane(name: str) -> str | None:
    """Deterministic lane for a model-like canonical, or None if not model-like.

    Priority:
    1. tombstoned / forbidden   -> DELETE_TOMBSTONE
    2. explicit MODEL_LANE_EXPLICIT
    3. model-operator contract role:
         model_score   -> MODEL_FEATURE_SCORE
         model_feature -> STATE_CONDITION_EVENT if stateful else FAST_NATIVE_ALPHA
         diagnostic    -> DIAGNOSTIC_RESEARCH
    4. surface-based fallback:
         research/unsafe surface or non-production -> DIAGNOSTIC_RESEARCH
         else -> FAST_NATIVE_ALPHA
    """
    if is_tombstoned(name):
        return "DELETE_TOMBSTONE"
    if name in _MODEL_LANE_EXPLICIT:
        return _MODEL_LANE_EXPLICIT[name]
    role = _resolved_role(name)
    if role == "model_score":
        return "MODEL_FEATURE_SCORE"
    if role == "diagnostic":
        return "DIAGNOSTIC_RESEARCH"
    if role == "model_feature":
        mc = get_model_operator_contract(name)
        if mc is not None and mc.stateful:
            return "STATE_CONDITION_EVENT"
        return "FAST_NATIVE_ALPHA"
    # surface fallback
    try:
        surface = classify_canonical(name)
    except Exception:
        surface = "research"
    if surface in ("research", "unsafe", "legacy", "internal"):
        return "DIAGNOSTIC_RESEARCH"
    # descriptive in-sample with fit_cutoff=0 defaults to diagnostic
    c = get_model_timing_contract(name)
    if c is not None and c.descriptive and c.forecast_horizon == 0:
        return "DIAGNOSTIC_RESEARCH"
    return "FAST_NATIVE_ALPHA"


def model_lane_of(name: str) -> str | None:
    return assign_model_lane(name)


def _category_of(canonical: str) -> str:
    """Category from the registry (for model-like classification that needs it)."""
    try:
        from cleaned_operators.registry import OperatorRegistry

        ops = OperatorRegistry._operators.get(canonical, {}) or {}
        if ops:
            return str(getattr(next(iter(ops.values())).metadata, "category", "") or "")
    except Exception:
        pass
    return ""


def model_lane_inventory(canonicals) -> dict[str, str]:
    """Lane per model-like canonical (for the R35 inventory).  Category-aware so
    model-like classification matches the R28 model causality audit."""
    out: dict[str, str] = {}
    for c in sorted(canonicals):
        if not is_model_like_name(c, _category_of(c)):
            continue
        out[c] = assign_model_lane(c)
    return out


def model_lane_errors(canonicals) -> list[str]:
    """Production gate: every model-like canonical must resolve to a lane."""
    errors: list[str] = []
    for c in sorted(canonicals):
        if not is_model_like_name(c, _category_of(c)):
            continue
        lane = assign_model_lane(c)
        if lane is None:
            errors.append(f"{c}: no model lane assigned")
    return errors
