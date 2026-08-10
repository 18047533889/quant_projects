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
from typing import Any

__all__ = [
    "ModelTimingContract",
    "LabelContract",
    "MODEL_LIKE_HINTS",
    "is_model_like_name",
    "model_family_of",
    "MODEL_TIMING_CONTRACTS",
    "get_model_timing_contract",
]


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
    "anomaly", "autoencoder", "mixture", "expert", "regime", "state", "transfer_entropy",
)


def is_model_like_name(name: str, category: str = "", source: str = "") -> bool:
    low = name.lower()
    if any(h in low for h in MODEL_LIKE_HINTS):
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
    "ts_ar_coefficient": ModelTimingContract("ar", fit_cutoff_offset=1, state_filtering="none"),
    "ts_ar_coeff_stability": ModelTimingContract("ar", fit_cutoff_offset=1, state_filtering="none"),
    "ts_ar_fitted_value": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_in_sample_resid": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_forecast": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_innovation": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    # --- Kalman: causal one-pass filter; missing obs are predict-only ---
    "ts_kalman_level": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_trend": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_beta": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_beta_change": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_beta_uncertainty": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_innovation_z": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    # --- GARCH: fit on seg[:-1]; current shock excluded from own denominator ---
    "ts_garch_standardized_shock": ModelTimingContract("garch", fit_cutoff_offset=1, state_filtering="none"),
    "ts_garch_next_vol_forecast": ModelTimingContract("garch", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_garch_persistence": ModelTimingContract("garch", fit_cutoff_offset=1, state_filtering="none"),
    "ts_garch_vol_surprise": ModelTimingContract("garch", fit_cutoff_offset=1, state_filtering="none"),
    "ts_gjr_garch_vol_forecast": ModelTimingContract("garch", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_har_rv_forecast": ModelTimingContract("har", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_har_rv_forecast_error_z": ModelTimingContract("har", fit_cutoff_offset=1, state_filtering="none"),
    "ts_har_rv_innovation_z": ModelTimingContract("har", fit_cutoff_offset=1, state_filtering="none"),
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
    "ts_kernel_granger_causality": ModelTimingContract("kernel", fit_cutoff_offset=1, state_filtering="none"),
    "ts_kernel_granger_oos": ModelTimingContract("kernel", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_residualized_hsic": ModelTimingContract("hsic", fit_cutoff_offset=1, state_filtering="none"),
    "ts_hsic_dependence": ModelTimingContract("hsic", fit_cutoff_offset=1, state_filtering="none"),
    # --- Dynamic KNN / local models: as-of membership + historical scaler ---
    "cs_knn_local_linear_residual": ModelTimingContract("knn", fit_cutoff_offset=1, state_filtering="none"),
    "ts_dynamic_knn_state": ModelTimingContract("knn", fit_cutoff_offset=1, state_filtering="none"),
    # --- Supervised panel models (fit_lag=1 + label_horizon maturity) ---
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
    # --- Kalman: causal one-pass filter; missing obs are predict-only ---
    "ts_kalman_level": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_trend": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_beta": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_beta_change": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_beta_uncertainty": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    "ts_kalman_innovation_z": ModelTimingContract("kalman", fit_cutoff_offset=0, state_filtering="filter"),
    # --- GARCH: fit on seg[:-1]; current shock excluded from own denominator ---
    "ts_garch_standardized_shock": ModelTimingContract("garch", fit_cutoff_offset=1, state_filtering="none"),
    "ts_garch_next_vol_forecast": ModelTimingContract("garch", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_garch_persistence": ModelTimingContract("garch", fit_cutoff_offset=1, state_filtering="none"),
    "ts_garch_vol_surprise": ModelTimingContract("garch", fit_cutoff_offset=1, state_filtering="none"),
    "ts_gjr_garch_vol_forecast": ModelTimingContract("garch", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    # --- HAR (volatility) ---
    "ts_har_rv_forecast": ModelTimingContract("har", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_har_rv_forecast_error_z": ModelTimingContract("har", fit_cutoff_offset=1, state_filtering="none"),
    "ts_har_rv_innovation_z": ModelTimingContract("har", fit_cutoff_offset=1, state_filtering="none"),
    "ts_har_rv_next_vol_forecast": ModelTimingContract("har", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_har_rv_next_var_forecast": ModelTimingContract("har", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_har_from_return_next_vol": ModelTimingContract("har", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_har_from_return_forecast_error_z": ModelTimingContract("har", fit_cutoff_offset=1, state_filtering="none"),
    # --- AR: prior forms are out-of-sample (fit t-1); fitted/legacy are in-sample ---
    "ts_ar_prior_forecast": ModelTimingContract("ar", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_ar_prior_innovation": ModelTimingContract("ar", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_ar_prior_innovation_z": ModelTimingContract("ar", fit_cutoff_offset=1, forecast_horizon=1, state_filtering="none"),
    "ts_ar_prior_coeff": ModelTimingContract("ar", fit_cutoff_offset=1, state_filtering="none"),
    "ts_ar_coefficient": ModelTimingContract("ar", fit_cutoff_offset=1, state_filtering="none"),
    "ts_ar_coeff_stability": ModelTimingContract("ar", fit_cutoff_offset=1, state_filtering="none"),
    "ts_ar_fitted_value": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_in_sample_resid": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_forecast": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
    "ts_ar_innovation": ModelTimingContract("ar", fit_cutoff_offset=0, state_filtering="none"),
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
    """
    explicit = MODEL_TIMING_CONTRACTS.get(canonical)
    if explicit is not None:
        return explicit
    return _default_contract_for(canonical, category)
