# -*- coding: utf-8
"""算子语义版本：定义变更须 bump version 并在 factor metadata 记录。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSemanticVersion:
    canonical: str
    version: int
    note: str = ""


# 已知语义变更历史（新因子应使用最新 version）
OPERATOR_SEMANTIC_VERSIONS: dict[str, int] = {
    # v7: stable original-unit Ridge kernel; old ill-conditioned outputs differ.
    "ts_ridge_regression_coeff": 2,
    "ts_ridge_regression_resid_z": 2,
    "ts_ridge_regression_coeff_prior": 2,
    "ts_ridge_regression_forecast_error": 2,
    "ts_ridge_regression_forecast_error_z": 2,
    "ts_ridge_regression_in_sample_resid": 2,
    "ts_ridge_regression_predictive_resid": 2,
    # v7: one finite cohort for PCA fit/projection and missing current masks.
    "panel_rolling_pca_loading": 2,
    "panel_rolling_pca_resid": 2,
    "panel_rolling_pca_explained_ratio": 2,
    "panel_rolling_pca_resid_vol": 2,
    "panel_rolling_pca_resid_momentum": 2,
    "industry_rolling_pca_loading": 2,
    "panel_mixture_of_experts_score": 2,  # stable gate over usable experts only
    # v7: GJR shared recursion and prefix-local return-domain validation.
    "ts_garch_next_vol_forecast": 2,
    "ts_garch_vol_surprise": 2,
    "ts_garch_persistence": 2,
    "ts_garch_standardized_shock": 2,
    "ts_gjr_garch_vol_forecast": 2,
    "ts_gjr_leverage": 2,
    # v7: HAR rows preserve original feature/label clock and common cohort.
    "ts_har_rv_next_vol_forecast": 2,
    "ts_har_rv_next_var_forecast": 2,
    "ts_har_rv_forecast_error_z": 2,
    "ts_har_from_return_next_vol": 2,
    "ts_har_from_return_forecast_error_z": 2,
    "group_percentile": 3,  # v3: backend emitters preserve current quantile/null semantics
    "signed_log": 1,  # sign(x)*log(abs(x)+1e-10)
    "compare": 2,  # v2: NULL/NaN propagate
    "maximum": 2,
    "minimum": 2,
    "protected_log": 2,  # NULL preserved
    "protected_div": 2,
    "rank": 1,
    "rank_pct": 1,
    "ts_beta": 3,  # v3: paired finite cohort + explicit ddof semantics
    "ts_zscore": 2,  # v2: all declared window/numeric parameters affect execution
    # m_beta / rolling_beta are runtime aliases of ts_beta and inherit v3.
    "rolling_beta_to_market": 2,  # v2: paired cohort + benchmark alignment
    "downside_beta": 2,  # v2: downside mask after paired finite alignment
    "tail_beta": 2,  # v2: tail mask after paired finite alignment
    "intra_entropy": 2,  # v2: normalized histogram probability entropy
    "intra_limit_first_hit_time": 2,  # v2: limits bind by date+instrument identity
    "intra_limit_duration": 2,  # v2: limits bind by date+instrument identity
    "intra_limit_reopen_count": 2,  # v2: labelled limits + strict transition domain
    "fin_component_score": 2,  # v2: finite FALSE contributes 0; missing remains NaN
    "ts_kurt": 2,  # v2: backend emitter parity for finite/Inf handling
    "cs_quantile": 2,  # v2: backend emitter parity for finite/Inf handling
    "ts_quantile": 2,  # v2: backend emitter parity for finite/Inf handling
    "true_range": 2,  # v2: backend emitter parity for finite/Inf handling
    "ts_ema": 2,  # v2: canonical EMA emitter parity; aliases inherit this version
    "MACD_line": 2,  # v2: backend EMA/finite semantics aligned
    "MACD_signal": 2,  # v2: backend EMA/finite semantics aligned
    "MACD_hist": 2,  # v2: backend EMA/finite semantics aligned
    "ts_max_drawdown": 2,  # v2: generic delegate binds positional scalar parameters
    "overnight_return": 2,  # v2: shared concrete price basis + paired finite positive prices
    "open_close_return": 2,  # v2: shared concrete price basis + paired finite positive prices
    "open_to_vwap_return": 2,  # v2: shared concrete price basis + paired finite positive prices
    "vwap_to_close_return": 2,  # v2: shared concrete price basis + paired finite positive prices
    "ts_corr": 2,
    "ts_cov": 2,
}


def semantic_version(canon: str) -> int:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return OPERATOR_SEMANTIC_VERSIONS.get(name, 1)


def versioned_name(canon: str) -> str:
    return f"{canon}@v{semantic_version(canon)}"


def record_version_in_factor_metadata() -> bool:
    """Factor metadata 应记录 ``operator_semantic_versions`` 映射。"""
    return True
