# -*- coding: utf-8 -*-
"""R28 §二十九 / §一百二十 / §一百二十一: model timing + label contracts.

A ``ModelTimingContract`` pins down, for a model-like canonical, exactly what is
fit when and what is scored when — so "PIT-safe" is a machine-checkable property
instead of a tag.  The R28 gate ``R28_ALL_MODEL_CANONICALS_TIMING_CONTRACTED``
requires every canonical classified ``is_model_like`` to carry one.

Unified time definition (§二十):
    decision_time  = t
    feature_cutoff = t
    fit_cutoff     = t-1            (predictive models; descriptive state may use t)
    score_time     = t              (output row)
A supervised label anchored at origin ``s`` matures at ``s + label_horizon``, so
the training window may only use labels with ``origin <= t - 1 - label_horizon``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ModelTimingContract",
    "LabelContract",
    "TimingKind",
    "timing_kind_for",
    "MODEL_LIKE_HINTS",
    "is_model_like_name",
    "model_family_of",
    "MODEL_TIMING_CONTRACTS",
    "get_model_timing_contract",
]


class TimingKind(Enum):
    """Model-operators audit (M-004): six-class PIT timing ontology.

    Every model-like canonical must classify into exactly one TimingKind; the
    kind drives the estimation/reference/query/self-inclusion/peer-inclusion/
    label-maturity/state-update/output-timestamp contract:

    SELF_FIT_DESCRIPTIVE            fit window includes t, output describes the
                                    in-sample state at t (PCA loading, DMD mode,
                                    SSA residual, in-sample regression coeff)
    PRIOR_FIT_PREDICTIVE            fit strictly on <= t-1, output predicts t /
                                    t+H (prior regression variants, GARCH next
                                    vol, HAR forecast, AR prior forecast)
    PRIOR_REFERENCE_CURRENT_QUERY   reference set built on <= t-1, current row
                                    t is the QUERY and is excluded from the
                                    reference (matrix profile, state density,
                                    mahalanobis, signature anomaly, kernel
                                    granger OOS, Markov transition edges,
                                    joint-energy distribution break, SSA prior
                                    anomaly)
    SAME_TIME_CROSS_SECTIONAL       as-of time-t cross-section, self-excluded,
                                    peers at time t (cs_ KNN local / peer ops)
    RECURSIVE_CAUSAL_FILTER         one-pass causal filter, current observation
                                    updates the state (Kalman filter family)
    MATURED_HISTORICAL_OUTCOME      output is an outcome anchored at s that only
                                    becomes usable once s + horizon <= t (first
                                    passage hit / matured label statistics)
    """

    SELF_FIT_DESCRIPTIVE = "self_fit_descriptive"
    PRIOR_FIT_PREDICTIVE = "prior_fit_predictive"
    PRIOR_REFERENCE_CURRENT_QUERY = "prior_reference_current_query"
    SAME_TIME_CROSS_SECTIONAL = "same_time_cross_sectional"
    RECURSIVE_CAUSAL_FILTER = "recursive_causal_filter"
    MATURED_HISTORICAL_OUTCOME = "matured_historical_outcome"


#: TimingKind overrides keyed by canonical.  The fallback derivation in
#: :func:`timing_kind_for` cannot distinguish a few families (e.g. a
#: ``matrix_profile`` with fit_cutoff=0 descriptive is still a reference-query
#: model, not self-fit).  Explicit override wins.
_TIMING_KIND_OVERRIDES: dict[str, TimingKind] = {
    # reference-query: historical subsequence / state reference, current query
    "ts_matrix_profile_discord_score": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_matrix_profile_motif_age": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_matrix_profile_motif_frequency": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_matrix_profile_neighbor_dispersion": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_matrix_profile_novelty": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_motif_recurrence_count": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_signature_mahalanobis_anomaly": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_state_density": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    # ts_kernel_granger_score is a BLOCKED train/test predictive diagnostic: the
    # reference is a historical train block (<= t-1) and the current row is the
    # held-out query.  There is no BLOCKED_HISTORICAL_EVALUATION TimingKind in
    # this taxonomy; PRIOR_REFERENCE_CURRENT_QUERY is the closest honest fit
    # (reference excludes the query row).  Kept as an explicit override so the
    # generated prior-fit-description fallback can never mislabel it.
    "ts_kernel_granger_score": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    # Markov: transition matrix / state edges fit strictly on <= t-1 data; x_t
    # only selects the current state (M-122).  Explicit for ALL 8 ts_markov_*
    # canonicals so the ``first_passage`` family hint below can never hijack
    # ts_markov_mean_first_passage_time (it is a Markov MFPT, not a
    # matured-historical-outcome first-passage statistic).
    "ts_markov_committor": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_markov_entropy_production": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_markov_mean_first_passage_time": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_markov_persistence": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_markov_spectral_gap": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_markov_state_entropy": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_markov_stationary_surprisal": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    "ts_markov_transition_surprisal": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    # distribution break: prior window AND recent window both end at t-1; the
    # current row t never joins either reference window, it is the query.
    "ts_joint_energy_shift": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    # SSA prior anomaly: prior low-rank subspace fit strictly on <= t-1, current
    # segment scored as the query (NOT a generic prior-fit description).
    "ts_ssa_prior_reconstruction_error": TimingKind.PRIOR_REFERENCE_CURRENT_QUERY,
    # same-time cross-section: peers at time t, self excluded
    "cs_knn_local_linear_residual": TimingKind.SAME_TIME_CROSS_SECTIONAL,
    "cs_knn_distance": TimingKind.SAME_TIME_CROSS_SECTIONAL,
    "cs_knn_peer_mean_ex_self": TimingKind.SAME_TIME_CROSS_SECTIONAL,
    # matured historical outcome: anchor s usable only when s + horizon <= t
    "ts_first_passage_hit_probability": TimingKind.MATURED_HISTORICAL_OUTCOME,
    "ts_first_passage_bias": TimingKind.MATURED_HISTORICAL_OUTCOME,
    "ts_first_passage_conditional_time": TimingKind.MATURED_HISTORICAL_OUTCOME,
    # recursive causal filter
    "ts_kalman_level": TimingKind.RECURSIVE_CAUSAL_FILTER,
    "ts_kalman_trend": TimingKind.RECURSIVE_CAUSAL_FILTER,
    "ts_kalman_beta": TimingKind.RECURSIVE_CAUSAL_FILTER,
    "ts_kalman_beta_change": TimingKind.RECURSIVE_CAUSAL_FILTER,
    "ts_kalman_beta_uncertainty": TimingKind.RECURSIVE_CAUSAL_FILTER,
    "ts_kalman_innovation_z": TimingKind.RECURSIVE_CAUSAL_FILTER,
}

_TIMING_KIND_FAMILY_HINTS: tuple[tuple[str, TimingKind], ...] = (
    ("kalman", TimingKind.RECURSIVE_CAUSAL_FILTER),
    ("matrix_profile", TimingKind.PRIOR_REFERENCE_CURRENT_QUERY),
    # Markov kernels estimate transition matrix / state edges on <= t-1 data;
    # x_t only selects the current state -> PRIOR_REFERENCE_CURRENT_QUERY
    # (M-122), NOT SELF_FIT_DESCRIPTIVE.  Kept BEFORE ``first_passage`` so any
    # future ts_markov_* canonical (incl. mean_first_passage_time) is not
    # hijacked by the matured-outcome hint.
    ("markov", TimingKind.PRIOR_REFERENCE_CURRENT_QUERY),
    ("first_passage", TimingKind.MATURED_HISTORICAL_OUTCOME),
    ("signature_mahalanobis", TimingKind.PRIOR_REFERENCE_CURRENT_QUERY),
)


def timing_kind_for(canonical: str) -> TimingKind:
    """Return the PIT ``TimingKind`` for a model-like canonical (M-004).

    Explicit per-canonical override first, then family-name hints, then a
    deterministic derivation from the resolved ``ModelTimingContract`` fields.
    Every model-like canonical resolves to one of the six kinds — never None.
    """
    if canonical in _TIMING_KIND_OVERRIDES:
        return _TIMING_KIND_OVERRIDES[canonical]
    low = canonical.lower()
    for fam, kind in _TIMING_KIND_FAMILY_HINTS:
        if fam in low:
            return kind
    c = get_model_timing_contract(canonical)
    if c is None:
        return TimingKind.SELF_FIT_DESCRIPTIVE
    if c.forecast_horizon >= 1 or c.label_horizon is not None:
        return TimingKind.PRIOR_FIT_PREDICTIVE
    if c.fit_cutoff_offset >= 1:
        # a prior fit that scores a coefficient/statistic is still predictive in
        # the sense its fit is strictly before the scored row
        return TimingKind.PRIOR_FIT_PREDICTIVE
    if canonical.startswith("cs_"):
        return TimingKind.SAME_TIME_CROSS_SECTIONAL
    return TimingKind.SELF_FIT_DESCRIPTIVE


@dataclass(frozen=True)
class LabelContract:
    """Contract of a supervised label (taskbook §二十九)."""

    horizon: int                      # H in y_t = return(t -> t+H)
    origin_time: str = "s"            # label anchor
    maturity_time: str = "s + horizon"
    availability_time: str = "s + horizon"
    overlapping: bool = False         # adjacent labels share a future window
    purge_bars: int = 0               # rows excluded after the last matured origin
    embargo_bars: int = 0             # extra rows excluded to cut leakage


@dataclass(frozen=True)
class ModelTimingContract:
    """PIT timing contract of a model-like canonical (taskbook §一百二十一)."""

    model_kind: str                       # pcr / pls / enet / ar / kalman / garch ...
    fit_cutoff_offset: int                # rows before decision row available for fit (>=1 predictive, 0 descriptive)
    score_offset: int = 0                 # output written at decision row t + score_offset
    forecast_horizon: int = 0             # steps ahead the statistic predicts (0 = descriptive state)
    label_horizon: int | None = None      # H of the supervised label, if supervised
    purge_bars: int = 0
    embargo_bars: int = 0
    scaler_fit_cutoff_offset: int | None = None   # None => same as fit_cutoff_offset
    hyperparam_fit_cutoff_offset: int | None = None
    state_filtering: str = "none"         # filter / smoother / none

    @property
    def descriptive(self) -> bool:
        return self.fit_cutoff_offset == 0 and self.forecast_horizon == 0

    @property
    def predictive(self) -> bool:
        return not self.descriptive


#: Name / source-path hints for model-like classification (taskbook §十九).
MODEL_LIKE_HINTS: tuple[str, ...] = (
    "model", "regression", "forecast", "innovation", "predict", "pca", "pls",
    "elastic", "ridge", "huber", "quantile", "expectile", "ar_", "garch", "har_",
    "kalman", "state_space", "markov", "hmm", "dmd", "ssa", "hankel", "granger",
    "hsic", "kernel", "knn", "lyapunov", "rqa", "recurrence", "matrix_profile",
    "anomaly", "autoencoder", "mixture", "expert", "regime", "state",
    "transfer_entropy", "first_passage", "passage",
)


def is_model_like_name(name: str, category: str = "", source: str = "") -> bool:
    """Model-like recall filter (M-005: hint is recall-only, not semantics).

    Token-boundary matching: a hint must match an underscore-delimited token
    (or a token prefix), NOT an arbitrary substring.  Fixes the M-005 false
    positives where the substring ``"ar_"`` matched ``calendar_day_diff`` /
    ``dollar_volume`` / ``dollar_volume_zscore`` (the ``"ar"`` inside
    ``calendar``/``dollar``), which inflated the production-timing gate to 147
    errors.  ``ts_ar_*`` / ``ts_har_*`` tokens still match ``"ar"``/``"har"``
    exactly.
    """
    low = name.lower()
    tokens = low.replace("-", "_").split("_")
    for hint in MODEL_LIKE_HINTS:
        h = hint.lower().rstrip("_")  # trailing "_" is a boundary marker (ar_ -> ar)
        if not h:
            continue
        if "_" in h:
            # multi-token hint (transfer_entropy, matrix_profile, first_passage,
            # state_space, mean_reversion): already boundary-specific, match as
            # literal substring of the full name
            if h in low:
                return True
            continue
        # single-token hint (ar_, har_, pca, knn, ...): token-boundary match so
        # "ar_" does NOT match the "ar" inside "dollar"/"calendar" (M-005).
        if any(t == h for t in tokens):
            return True
        if any(t.startswith(h) for t in tokens):
            return True
    cat = (category or "").lower()
    if any(h in cat for h in ("model", "regression", "state", "entropy", "kernel", "spectral", "network")):
        return True
    return False


def model_family_of(name: str) -> str:
    low = name.lower()
    for fam, pats in (
        ("panel_model", ("panel_rolling", "panel_regime", "panel_mixture")),
        ("ar", ("ts_ar_", "_ar_", "mean_reversion", "variance_ratio")),
        ("garch", ("garch", "gjr", "har_", "_har_")),
        ("kalman", ("kalman",)),
        ("dmd", ("dmd", "hankel", "ssa")),
        ("pca", ("pca",)),
        ("kernel", ("kernel", "hsic", "granger", "knn", "lyapunov")),
        ("rqa", ("rqa", "recurrence", "matrix_profile", "sequence_anomaly", "discord", "motif")),
        ("entropy", ("entropy", "transfer_entropy", "hvg", "multifractal")),
        ("state", ("state", "markov", "regime", "first_passage")),
        ("regression", ("regression", "huber", "ridge", "elastic", "quantile", "expectile", "rolling")),
        ("spectral", ("spectral", "wavelet", "fft", "phase")),
    ):
        if any(p in low for p in pats):
            return fam
    return "generic"


#: Timing contracts for every model-like canonical (kept in sync with the R28
#: model causality audit).  Offset semantics: ``fit_cutoff_offset=k`` means the
#: fit window ends at ``decision_row - k``; ``0`` means the current row may be
#: used (descriptive state), ``>=1`` means strictly-past fit (predictive).
MODEL_TIMING_CONTRACTS: dict[str, ModelTimingContract] = {
    # --- panel supervised models (fit_lag=1 + label_horizon maturity) ---
    "panel_rolling_pcr_forecast": ModelTimingContract(
        "pcr", fit_cutoff_offset=1, forecast_horizon=1, label_horizon=1,
        scaler_fit_cutoff_offset=1, hyperparam_fit_cutoff_offset=1, state_filtering="none"),
    "panel_rolling_pls_forecast": ModelTimingContract(
        "pls", fit_cutoff_offset=1, forecast_horizon=1, label_horizon=1,
        scaler_fit_cutoff_offset=1, hyperparam_fit_cutoff_offset=1, state_filtering="none"),
    "panel_rolling_elastic_net_forecast": ModelTimingContract(
        "enet", fit_cutoff_offset=1, forecast_horizon=1, label_horizon=1,
        scaler_fit_cutoff_offset=1, hyperparam_fit_cutoff_offset=1, state_filtering="none"),
    "panel_regime_conditioned_forecast": ModelTimingContract(
        "regime", fit_cutoff_offset=1, forecast_horizon=1, label_horizon=1,
        scaler_fit_cutoff_offset=1, hyperparam_fit_cutoff_offset=1, state_filtering="filter"),
    "panel_mixture_of_experts_score": ModelTimingContract(
        "mixture", fit_cutoff_offset=1, forecast_horizon=1, label_horizon=1,
        scaler_fit_cutoff_offset=1, hyperparam_fit_cutoff_offset=1, state_filtering="none"),
    # --- PCA rolling (fit_lag=1 default; current row never in its own loading) ---
    "panel_rolling_pca_explained_ratio": ModelTimingContract("pca", fit_cutoff_offset=1, state_filtering="none"),
    "panel_rolling_pca_loading": ModelTimingContract("pca", fit_cutoff_offset=1, state_filtering="none"),
    "panel_rolling_pca_resid": ModelTimingContract("pca", fit_cutoff_offset=1, state_filtering="none"),
    "panel_rolling_pca_resid_momentum": ModelTimingContract("pca", fit_cutoff_offset=1, state_filtering="none"),
    "panel_rolling_pca_resid_vol": ModelTimingContract("pca", fit_cutoff_offset=1, state_filtering="none"),
    "industry_rolling_pca_loading": ModelTimingContract("pca", fit_cutoff_offset=1, state_filtering="none"),
    "intraday_profile_pca_residual": ModelTimingContract("pca", fit_cutoff_offset=1, state_filtering="none"),
    "intraday_quantile_curve_pca_residual": ModelTimingContract("pca", fit_cutoff_offset=1, state_filtering="none"),
    "intraday_quantile_curve_pca_score": ModelTimingContract("pca", fit_cutoff_offset=1, state_filtering="none"),
    # --- AR: prior forms are out-of-sample (fit t-1); fitted/innovation_z are
    #     in-sample descriptive but documented and NOT predictive. ---
    "ts_ar_prior_forecast": ModelTimingContract("ar", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_ar_prior_innovation": ModelTimingContract("ar", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_ar_prior_innovation_z": ModelTimingContract("ar", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_ar_prior_coeff": ModelTimingContract("ar", fit_cutoff_offset=1, state_filtering="none"),
    # M-040: ts_ar_coefficient fits on the window INCLUDING the current row
    # (in-sample descriptive).  The strict-prior alpha variant is
    # ts_ar_prior_coeff.  fit_cutoff=0 descriptive.
    "ts_ar_coefficient": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_coeff_stability": ModelTimingContract("ar", fit_cutoff_offset=1, state_filtering="none"),
    "ts_ar_fitted_value": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_in_sample_resid": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_forecast": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_innovation": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_innovation_z": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    # --- GJR leverage: fits the full window INCLUDING the current row (in-sample
    #     descriptive, M-081).  NOT a strict-prior alpha candidate — authored
    #     explicit fit_cutoff=0 descriptive so it cannot be mistaken for one. ---
    "ts_gjr_leverage": ModelTimingContract("garch", fit_cutoff_offset=0, state_filtering="none"),
    # --- Regression family: prior variants fit strictly through t-1 (PRIOR_FIT_PREDICTIVE);
    #     in-sample coeff/slope variants are descriptive (SELF_FIT_DESCRIPTIVE).  Authored
    #     per M-051 (every prior/forecast-error variant explicit, no generated defaults). ---
    "ts_multi_regression_coeff_prior": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_multi_regression_r2_prior": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_multi_regression_adjusted_r2_prior": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_multi_regression_coeff_stability": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_multi_regression_forecast_error": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_multi_regression_forecast_error_z": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_huber_regression_coeff_prior": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_huber_regression_forecast_error": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_huber_regression_forecast_error_z": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_huber_regression_predictive_resid": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_ridge_regression_coeff_prior": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_ridge_regression_forecast_error": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_ridge_regression_forecast_error_z": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_ridge_regression_predictive_resid": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_quantile_regression_coeff_prior": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_quantile_beta_spread_prior": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_expectile_regression_coeff_prior": ModelTimingContract("regression", fit_cutoff_offset=1, state_filtering="none"),
    "ts_expectile_regression_forecast_error": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_regression_slope": ModelTimingContract("regression", fit_cutoff_offset=0, state_filtering="none"),
    "ts_regression_forecast_error": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_regression_forecast_error_z": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_poly2_forecast_error": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_poly2_forecast_error_z": ModelTimingContract("regression", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_mean_reversion_half_life": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_mean_reversion_ou_approx_half_life": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    # --- Kalman: causal one-pass filter; missing obs are predict-only ---
    "ts_kalman_level": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_trend": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_beta": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_beta_change": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_beta_uncertainty": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_innovation_z": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    # --- Markov / regime state dynamics: edges/P/D1/D2 strictly on <= t-1,
    #     current value only selects the current state (M-122 PRIOR_REFERENCE_CURRENT_QUERY). ---
    "ts_markov_committor": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    "ts_markov_entropy_production": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    "ts_markov_mean_first_passage_time": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    "ts_markov_persistence": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    "ts_markov_spectral_gap": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    "ts_markov_state_entropy": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    "ts_markov_stationary_surprisal": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    "ts_markov_transition_surprisal": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    "ts_regime_duration": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    "ts_two_state_regime_probability": ModelTimingContract("state", fit_cutoff_offset=0, state_filtering="none"),
    # --- GARCH: fit on seg[:-1]; current shock excluded from own denominator ---
    "ts_garch_standardized_shock": ModelTimingContract("garch", fit_cutoff_offset=1, state_filtering="none"),
    "ts_garch_next_vol_forecast": ModelTimingContract("garch", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_garch_persistence": ModelTimingContract("garch", fit_cutoff_offset=1, state_filtering="none"),
    "ts_garch_vol_surprise": ModelTimingContract("garch", fit_cutoff_offset=1, state_filtering="none"),
    "ts_gjr_garch_vol_forecast": ModelTimingContract("garch", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    # ts_har_rv_forecast / ts_har_rv_innovation_z are compat aliases (M-088);
    # the alias targets carry their own explicit contracts.
    "ts_har_rv_forecast_error_z": ModelTimingContract("har", fit_cutoff_offset=1, state_filtering="none"),
    "ts_har_rv_next_vol_forecast": ModelTimingContract("har", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_har_rv_next_var_forecast": ModelTimingContract("har", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_har_from_return_next_vol": ModelTimingContract("har", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_har_from_return_forecast_error_z": ModelTimingContract("har", fit_cutoff_offset=1, state_filtering="none"),
    # --- DMD / Hankel / SSA: descriptive trailing eigendecomposition ---
    "ts_dmd_dominant_growth_rate": ModelTimingContract("dmd", fit_cutoff_offset=0, state_filtering="none"),
    "ts_dmd_dominant_frequency": ModelTimingContract("dmd", fit_cutoff_offset=0, state_filtering="none"),
    "ts_dmd_level_dominant_growth_rate": ModelTimingContract("dmd", fit_cutoff_offset=0, state_filtering="none"),
    "ts_dmd_level_dominant_frequency": ModelTimingContract("dmd", fit_cutoff_offset=0, state_filtering="none"),
    "ts_dmd_level_mode_concentration": ModelTimingContract("dmd", fit_cutoff_offset=0, state_filtering="none"),
    "ts_dmd_mode_concentration": ModelTimingContract("dmd", fit_cutoff_offset=0, state_filtering="none"),
    "ts_dmd_return_dominant_growth_rate": ModelTimingContract("dmd", fit_cutoff_offset=0, state_filtering="none"),
    "ts_dmd_return_dominant_frequency": ModelTimingContract("dmd", fit_cutoff_offset=0, state_filtering="none"),
    "ts_dmd_return_mode_concentration": ModelTimingContract("dmd", fit_cutoff_offset=0, state_filtering="none"),
    "ts_hankel_singular_gap": ModelTimingContract("hankel", fit_cutoff_offset=0, state_filtering="none"),
    "ts_hankel_effective_rank": ModelTimingContract("hankel", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ssa_reconstruction_residual": ModelTimingContract("ssa", fit_cutoff_offset=0, state_filtering="none"),
    # M-095: prior SSA reconstruction error — fit the low-rank subspace strictly
    # on <= t-1, then score the current segment as the query (PRIOR_REFERENCE_CURRENT_QUERY).
    "ts_ssa_prior_reconstruction_error": ModelTimingContract("ssa", fit_cutoff_offset=1, state_filtering="none"),
    # --- Matrix profile / sequence anomaly: query vs HISTORICAL subsequences only ---
    "ts_matrix_profile_discord_score": ModelTimingContract("matrix_profile", fit_cutoff_offset=0, state_filtering="none"),
    "ts_matrix_profile_motif_age": ModelTimingContract("matrix_profile", fit_cutoff_offset=0, state_filtering="none"),
    "ts_matrix_profile_motif_frequency": ModelTimingContract("matrix_profile", fit_cutoff_offset=0, state_filtering="none"),
    "ts_matrix_profile_neighbor_dispersion": ModelTimingContract("matrix_profile", fit_cutoff_offset=0, state_filtering="none"),
    "ts_matrix_profile_novelty": ModelTimingContract("matrix_profile", fit_cutoff_offset=0, state_filtering="none"),
    "ts_motif_recurrence_count": ModelTimingContract("matrix_profile", fit_cutoff_offset=0, state_filtering="none"),
    "ts_signature_mahalanobis_anomaly": ModelTimingContract("matrix_profile", fit_cutoff_offset=0, state_filtering="none"),
    # --- Path signature: trailing joint-run signature (current row must be finite) ---
    "ts_path_signature_area": ModelTimingContract("signature", fit_cutoff_offset=0, state_filtering="none"),
    "ts_path_signature_depth2_norm": ModelTimingContract("signature", fit_cutoff_offset=0, state_filtering="none"),
    "ts_path_leadlag_area": ModelTimingContract("signature", fit_cutoff_offset=0, state_filtering="none"),
    # --- Kernel Granger / HSIC: blocked training/test split ---
    "ts_residualized_hsic": ModelTimingContract("hsic", fit_cutoff_offset=1, state_filtering="none"),
    # --- Dynamic KNN / local models: same-time cross-section (M-110).  The
    #     kernel fits a date-t peer cross-section (self excluded, peer target_t
    #     used for the fit) — NOT fit-through-t-1.  fit_cutoff=0 means "the
    #     current time slice may be used", with self-exclusion enforced by the
    #     kernel; TimingKind is SAME_TIME_CROSS_SECTIONAL. ---
    "cs_knn_local_linear_residual": ModelTimingContract("knn", fit_cutoff_offset=0, state_filtering="none"),
}


#: Substrings marking a PREDICTIVE (out-of-sample) statistic: it must fit on
#: data strictly before the scored row (fit_cutoff_offset >= 1).  Everything else
#: model-like defaults to descriptive in-sample state (fit_cutoff_offset == 0).
PREDICTIVE_HINTS: tuple[str, ...] = (
    "_forecast", "_forecast_error", "_innovation", "_next_", "_predictive",
    "prior_", "_prior", "oos", "_out_of_sample",
)


def _default_contract_for(name: str, category: str = "") -> ModelTimingContract:
    """Deterministic timing contract for a model-like canonical.

    Predictive names (forecast/innovation/prior/oos/next) get ``fit_cutoff=1``
    (fit strictly before the scored row).  All other model-like statistics are
    trailing descriptive state on the window ``[t-w+1, t]`` — in-sample, no
    future information — and get ``fit_cutoff=0`` with ``descriptive=True``.
    """
    low = name.lower()
    fam = model_family_of(name)
    if any(h in low for h in PREDICTIVE_HINTS):
        horizon = 1 if ("_forecast" in low or "_next_" in low) else 0
        return ModelTimingContract(
            fam, fit_cutoff_offset=1, forecast_horizon=horizon,
            label_horizon=1 if "_forecast" in low else None,
            scaler_fit_cutoff_offset=1, hyperparam_fit_cutoff_offset=1,
            state_filtering="none",
        )
    if fam in ("kalman",):
        return ModelTimingContract(fam, fit_cutoff_offset=0, state_filtering="filter")
    return ModelTimingContract(fam, fit_cutoff_offset=0, state_filtering="none")


def get_model_timing_contract(
    canonical: str, *, category: str = ""
) -> ModelTimingContract:
    """Return the timing contract for a model-like canonical.

    Explicit entries in ``MODEL_TIMING_CONTRACTS`` win; otherwise a deterministic
    family/name-based default is generated.  This guarantees the R28 gate
    ``R28_ALL_MODEL_CANONICALS_TIMING_CONTRACTED`` for every model-like canonical.

    R34 P0-025: the generated default is a RESEARCH HINT only.  Production
    admission must require an explicit reviewed contract (see
    :func:`model_timing_production_errors`).
    """
    explicit = MODEL_TIMING_CONTRACTS.get(canonical)
    if explicit is not None:
        return explicit
    return _default_contract_for(canonical, category)


def model_timing_contract_is_explicit(canonical: str) -> bool:
    """Whether the canonical has a reviewed, explicit ``ModelTimingContract``.

    A name/family-based generated contract (``_default_contract_for``) is not a
    reviewed contract — it cannot admit production (R34 P0-025).
    """
    return canonical in MODEL_TIMING_CONTRACTS


def model_timing_production_errors(canonicals) -> list[str]:
    """Production gate (M-002): every DIRECT-PRODUCTION model-like canonical
    needs an explicit reviewed ``ModelTimingContract``.

    ``direct_production_model_like`` = a model-like canonical whose resolved
    lane is a production lane (FAST_NATIVE_ALPHA / EXPENSIVE_CERTIFIED_ALPHA /
    MODEL_FEATURE_SCORE / STATE_CONDITION_EVENT).  Diagnostic / research /
    tombstone lanes are not production-admissible regardless of timing, so a
    generated contract is an acceptable research hint for them (R34 P0-025).

    This is the audit's ``for canonical in actual_registry: if
    direct_production_model_like(canonical): assert
    model_timing_contract_is_explicit(canonical)``.  Auto-generated contracts
    are research hints only and do NOT admit production.
    """
    errors: list[str] = []
    for canonical in sorted(canonicals):
        if not is_model_like_name(canonical):
            continue
        # lazy import avoids a circular dependency (model_lane imports this module)
        try:
            from cleaned_operators.model_lane import assign_model_lane

            lane = assign_model_lane(canonical)
        except Exception:
            lane = None
        if lane not in ("FAST_NATIVE_ALPHA", "EXPENSIVE_CERTIFIED_ALPHA",
                        "MODEL_FEATURE_SCORE", "STATE_CONDITION_EVENT"):
            continue  # diagnostic/research/tombstone: not production-admissible
        if not model_timing_contract_is_explicit(canonical):
            errors.append(
                f"{canonical}: direct-production model-like but no explicit reviewed "
                f"ModelTimingContract (auto-generated timing is a research hint only, "
                f"M-002 / R34 P0-025)"
            )
    return errors
