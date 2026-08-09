# -*- coding: utf-8
"""SQL 下推三层准入：Implemented / Parity Verified / Production Safe。

``SQL_CAPABLE_CANONICALS`` 仅表示 emitter 有实现，不等于 production-safe。
"""
from __future__ import annotations

from backend.production_fastpath_tiers import (
    P0_PRODUCTION_FASTPATH_CANONICALS,
    P1_DUCKDB_PARITY_PENDING,
    P1_DUCKDB_PRODUCTION_SAFE,
    P2_RESEARCH_ONLY,
    FASTPATH_DEFERRED_CANONICALS,
)

# 向后兼容：``SQL_CAPABLE_CANONICALS`` == ``SQL_IMPLEMENTED_CANONICALS``

# emitter 有实现、可尝试编译（原 SQL_CAPABLE 大白名单）
SQL_IMPLEMENTED_CANONICALS: frozenset[str] = frozenset(
    {
        "column",
        "literal",
        "add",
        "subtract",
        "multiply",
        "divide",
        "fin_ratio",
        "sqrt_abs",
        "book_to_price",
        "earnings_yield",
        "float_share_ratio",
        "free_float_share_ratio",
        "benchmark_relative_price",
        "benchmark_excess_return",
        "holder_concentration",
        "ashare_limit_distance",
        "row_sum_skipna",
        "neg",
        "abs",
        "sign",
        "log",
        "exp",
        "sqrt",
        "clip",
        "ts_mean",
        "ts_delay",
        "ts_delta",
        "ts_std",
        "ts_sum",
        "ts_max",
        "ts_min",
        "ts_pct",
        "ts_zscore",
        "ts_corr",
        "rank",
        "zscore",
        "scale",
        "cs_demean",
        "cs_resid",
        "cs_regression",
        "size_neutralize",
        "industry_size_neutralize",
        "index_weight",
        "group_neutralize",
        "where",
        "if_else",
        "ts_median",
        "ts_var",
        "group_rank",
        "group_mean",
        "group_sum",
        "group_min",
        "group_max",
        "group_count",
        "group_zscore",
        "winsorize",
        "group_winsorize",
        "ts_beta",
        "ts_mad",
        "ts_ema",
        "ewm_mean",
        "standardize",
        "ts_rank",
        "ts_ema",
        "ffill",
        "fillna_const",
        "ts_decay_linear",
        "coalesce",
        "protected_div",
        "safe_div_null",
        "protected_log",
        "protected_sqrt",
        "nan_to_num",
        "div_or_default",
        "log_fill_invalid",
        "fillna",
        "is_nan",
        "is_null",
        "is_not_null",
        "is_finite",
        "is_infinite",
        "normalize",
        "group_normalize",
        "group_percentile",
        "group_decay_linear",
        "power",
        "gt",
        "lt",
        "eq",
        "ge",
        "le",
        "ne",
        "and_",
        "or_",
        "not_",
        "group_std",
        "ts_cov",
        "ts_quantile",
        "ts_product",
        "ts_regression_slope",
        "ts_skew",
        "cum_sum",
        "cum_max",
        "cum_min",
        "WMA",
        "ts_ewm_std",
        "ts_ewm_var",
        "cum_std",
        "expanding_std",
        "floor",
        "ceil",
        "inverse",
        "count",
        "ts_time_slope",
        "ts_argmax",
        "ts_argmin",
        "ts_ewm_corr",
        "ts_ewm_cov",
        "ts_sharpe",
        "ts_autocorr",
        "rolling_beta",
        "rank_pct",
        "cs_pct_rank",
        "cs_quantile",
        "c_percentile",
        "ts_log_return",
        "volatility",
        "vwap",
        "maximum",
        "minimum",
        "cum_prod",
        "cum_delta",
        "expanding_mean",
        "expanding_sum",
        "log_abs",
        "signed_log",
        "signed_sqrt",
        "tanh",
        "cbrt",
        "truncate",
        "cs_mean",
        "cs_std",
        "cs_sum",
        "cs_count",
        "cs_mad",
        "cs_mad_zscore",
        "RSI_WILDER",
        "ATR_WILDER",
        "NATR",
        "MACD_line",
        "MACD_signal",
        "MACD_hist",
        "ts_count_if",
        "ts_sum_if",
        "ts_mean_if",
        "ts_std_if",
        "ts_last_if",
        "ts_days_since",
        "ts_true_streak",
        "cs_bucket",
        "cs_multi_resid",
        "cs_wls_resid",
        "period_lag",
        "period_change",
        "period_average",
        "period_cagr",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
        "ts_regression_tstat",
        "ts_trend_tstat",
        "ts_max_drawdown",
        "ts_partial_corr",
        "ts_nth_value",
        # elementwise math
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "log2",
        "log10",
        "csc",
        "sec",
        "cot",
        "square",
        "cube",
        "sigmoid",
        "exp_neg",
        "saturate",
        "round",
        "fix",
        "signed_power",
        "lerp",
        # technical indicators (EMA / ATR / rolling)
        "DEMA",
        "TEMA",
        "PPO",
        "PPO_signal",
        "PPO_hist",
        "PVO",
        "PVO_signal",
        "PVO_hist",
        "TSI",
        "TSI_signal",
        "CMO",
        "VortexPlus",
        "VortexMinus",
        "KeltnerMid",
        "KeltnerUpper",
        "KeltnerLower",
        "KeltnerPosition",
        "DMI_plus",
        "DMI_minus",
        "DX",
        "ADX",
        "UltimateOscillator",
        "ADL",
        "ChaikinOscillator",
        "ForceIndex",
        "EaseOfMovement",
        "CMF",
        "MFI",
        # candle geometry / return decomposition
        "candle_body",
        "candle_abs_body",
        "candle_range",
        "candle_body_ratio",
        "candle_upper_shadow_ratio",
        "candle_lower_shadow_ratio",
        "candle_upper_shadow",
        "candle_lower_shadow",
        "candle_close_location",
        "candle_body_position",
        "candle_close_strength",
        "candle_rejection_upper",
        "candle_rejection_lower",
        "candle_direction",
        "candle_gap",
        "candle_gap_pct",
        "candle_range_atr",
        "candle_gap_atr",
        "candle_overlap_ratio",
        "candle_inside_ratio",
        "open_close_return",
        "overnight_return",
        "open_to_vwap_return",
        "vwap_to_close_return",
        "limit_up_close",
        "limit_down_close",
        "true_range",
        # OHLC volatility estimators
        "parkinson_vol",
        "garman_klass_vol",
        "rogers_satchell_vol",
        "yang_zhang_vol",
        "overnight_volatility",
        "intraday_volatility",
        "range_volatility",
        "ulcer_index",
        "high_low_spread_proxy",
        # volume / turnover rolling
        "average_volume",
        "ts_average_volume",
        "average_turnover",
        "volume_to_range",
        "abnormal_volume",
        "abnormal_turnover",
        "relative_volume",
        "volume_zscore",
        "turnover_zscore",
        "volume_shock",
        "turnover_shock",
        "volume_momentum",
        "turnover_momentum",
        "volume_volatility",
        "turnover_volatility",
        "volume_autocorr",
        "turnover_autocorr",
        "volume_acceleration",
        "turnover_acceleration",
        "up_volume_ratio",
        "down_volume_ratio",
        "signed_volume_imbalance",
        "up_down_volume_ratio",
        "volume_weighted_return",
        "volume_weighted_momentum",
        "rolling_vwap",
        "vwap_deviation",
        "rolling_obv",
        "rolling_pvt",
        "signed_volume",
        "signed_dollar_volume",
        "dollar_volume",
        "dollar_volume_zscore",
        "adv",
        "amihud_illiquidity",
        "price_impact",
        "return_per_turnover",
        "return_volume_corr",
        "return_volume_beta",
        "return_turnover_beta",
        "price_volume_divergence",
        "price_turnover_divergence",
        "turnover_adjusted_volatility",
        "zero_return_ratio",
        "roll_spread_proxy",
        "corwin_schultz_spread",
        "bounded_nvi",
        "bounded_pvi",
        # band / channel ops
        "bollinger_pct_b",
        "bollinger_width",
        "donchian_upper",
        "donchian_lower",
        "donchian_mid",
        "donchian_position",
        # misc technical: efficiency ratio / choppiness / market coskewness
        "efficiency_ratio",
        "choppiness_index",
        "coskewness_to_market",
        # ts_* rolling window stats (validity / concentration / deviation / impulse / prior-extreme)
        "ts_valid_count",
        "ts_coverage_ratio",
        "ts_abs_concentration",
        "ts_abs_entropy",
        "ts_downside_deviation",
        "ts_upside_deviation",
        "ts_impulse_return",
        "ts_impulse_strength",
        "ts_impulse_volume",
        # Alpha-language SQL subset (2026-08): window-function-natural ops.
        "event_frequency",
        "ts_semivariance_balance",
        "ts_realized_quarticity",
        "ts_vol_of_vol",
        "ts_vol_acceleration",
        "ts_vol_term_structure",
        "ts_prev_high",
        "ts_prev_low",
        "ts_distance_to_high",
        "ts_distance_to_low",
        "ts_breakout_high",
        "ts_breakdown_low",
        "ts_channel_position",
        "ts_new_high",
        "ts_new_low",
        "ts_argmax_age",
        "ts_argmin_age",
        "ts_argmax_index_from_oldest",
        "ts_argmin_index_from_oldest",
        "ts_staleness",
        "ts_days_since_high",
        "ts_days_since_low",
        # cross-sectional fill / count / weighted / residual-percentile
        "cs_valid_count",
        "cs_coverage_ratio",
        "cs_fill_mean",
        "cs_fill_median",
        "cs_impute_mean",
        "cs_impute_median",
        "cs_residual_percentile",
        "cs_weighted_mean",
        "cs_weighted_demean",
        "cs_weighted_zscore",
        # ichimoku family
        "ichimoku_tenkan",
        "ichimoku_kijun",
        "ichimoku_senkou_a",
        "ichimoku_senkou_b",
        "ichimoku_cloud_width",
        "ichimoku_cloud_position",
        # candlestick patterns (cdl_*)
        "cdl_doji",
        "cdl_hammer",
        "cdl_inverted_hammer",
        "cdl_shooting_star",
        "cdl_marubozu",
        "cdl_spinning_top",
        "cdl_engulfing",
        "cdl_inside_bar",
        "cdl_outside_bar",
        "cdl_dragonfly_doji",
        "cdl_gravestone_doji",
        "cdl_hanging_man",
        "cdl_harami",
        "cdl_harami_cross",
        "cdl_piercing",
        "cdl_dark_cloud_cover",
        "cdl_morning_star",
        "cdl_evening_star",
        "cdl_three_white_soldiers",
        "cdl_three_black_crows",
        "cdl_tweezer_top",
        "cdl_tweezer_bottom",
        # Round-7 gap closure: simple windowed sign-ratio / central moment and
        # group reducers are genuinely SQL-expressible (see emitter branches).
        "ts_positive_ratio",
        "ts_negative_ratio",
        "ts_zero_ratio",
        "ts_moment",
        "group_valid_count",
        "group_weighted_mean",
    }
)

# Drop emitter names that are neither active registry canonicals nor IR meta /
# historical alias targets still referenced by the compiler.  Keeps
# ``is_sql_implemented`` honest for coverage reports.
_SQL_IMPLEMENTED_DEAD: frozenset[str] = frozenset(
    {
        "div_or_default",
        "log_fill_invalid",
        "nan_to_num",
        "protected_log",
        "protected_sqrt",
        "ewm_mean",
        "standardize",
        "volatility",
        "vwap",
        "WMA",
        "c_percentile",
        "rank_pct",
        "count",
        # keep rolling_beta — alias target still asserted by production SQL tests
    }
)
SQL_IMPLEMENTED_CANONICALS = frozenset(
    c for c in SQL_IMPLEMENTED_CANONICALS if c not in _SQL_IMPLEMENTED_DEAD
) | frozenset({"column", "literal"})
# 2026-08 geometry/math expansion: SQL pushdown subset with emitter branches in
# backend/sql_pushdown/emitter.py (exact-parity DuckDB window aggregates).
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "intraday_volatility_concentration",
    "intraday_volatility_entropy",
    "intraday_realized_semivariance_balance",
    "ts_crossing_speed",
    "ts_crossing_acceleration",
})

# 2026-08-08 Gemini-recommended primitives: SQL pushdown subset with emitter
# branches in backend/sql_pushdown/emitter.py (exact-parity DuckDB window
# aggregates / correlated top-k selection).
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "group_topk_mean",
    "cs_weighted_percentile_rank",
    "ts_cov_if",
    "ts_value_at_argextreme",
    "ts_weighted_standardized_moment",
    # 2026-08-08 Gemini V2 round: PIT-safe Abdi-Ranaldo spread (lead()ed
    # eta_{s+1} with an exclusive-end rolling frame, see emitter branch).
    "ts_abdi_ranaldo_spread",
})

# 2026-08-09 三后端一致性轮：新增可下推的简单/窗口/分组算子（emitter 分支见
# backend/sql_pushdown/emitter.py，含 candle 滚动 zscore/percentile、组内矩、
# leave-one-out 均值、截面 max-abs 归一化、展开 rank、abs-return 滚动 corr）。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "identity",
    "unitize",
    "candle_body_zscore",
    "candle_range_zscore",
    "candle_upper_shadow_zscore",
    "candle_lower_shadow_zscore",
    "candle_body_percentile",
    "candle_range_percentile",
    "group_skewness",
    "group_kurtosis",
    "group_quantile_spread",
    "group_ex_self_mean",
    "abs_return_volume_corr",
    "expanding_rank",
})

# EWMA/Wilder 平滑族（RSI/ATR/DMI/DX/ADX/MACD/DEMA/TEMA/PPO/PVO/TSI/Keltner/
# ADL/CMF/ChaikinOscillator/ForceIndex）：pandas ewm(adjust=False) 的 NaN 缺口
# 语义（绝对位置衰减 + 有效观测重新归一化）无法在 SQL 中精确复刻，emitter 返回
# None 走 polars 回退（与 pandas 完全一致）。从白名单移除以保持 sync 一致。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS - frozenset({
    "RSI_WILDER", "ATR_WILDER", "DMI_plus", "DMI_minus", "DX", "ADX",
    "MACD_line", "MACD_signal", "MACD_hist",
    "DEMA", "TEMA", "PPO", "PPO_signal", "PPO_hist",
    "PVO", "PVO_signal", "PVO_hist", "TSI", "TSI_signal",
    "KeltnerMid", "KeltnerUpper", "KeltnerLower", "KeltnerPosition",
    "ADL", "ChaikinOscillator", "CMF", "ForceIndex",
    "cdl_hammer", "cdl_hanging_man",
    "ts_time_slope", "ts_upside_deviation", "ts_weighted_standardized_moment",
    "ts_abdi_ranaldo_spread", "ts_value_at_argextreme",
    "industry_size_neutralize",
})

# DuckDB 分层
DUCKDB_SQL_PARITY_VERIFIED: frozenset[str] = frozenset()
DUCKDB_SQL_PRODUCTION_SAFE: frozenset[str] = frozenset()

# ClickHouse 独立分层（无真实 integration test 前默认空）
CLICKHOUSE_SQL_PARITY_VERIFIED: frozenset[str] = frozenset()
CLICKHOUSE_SQL_PRODUCTION_SAFE: frozenset[str] = frozenset()

from backend.primitive_evidence import (
    DUCKDB_REAL_SQL_VERIFIED,
    POLARS_REFERENCE_PARITY_VERIFIED,
    PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
)

SQL_PARITY_VERIFIED_CANONICALS: frozenset[str] = frozenset(
    {"column", "literal"}
) | DUCKDB_REAL_SQL_VERIFIED

from cleaned_operators.operator_surface import DAILY_CANONICALS as _DAILY_CANONICALS

_STATIC_SQL_CANDIDATES: frozenset[str] = (
    frozenset({"column", "literal", "protected_div"}) | frozenset(_DAILY_CANONICALS)
)

SQL_PRODUCTION_SAFE_CANONICALS: frozenset[str] = frozenset(
    c for c in _STATIC_SQL_CANDIDATES if c in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
) | frozenset({"column", "literal"})

DUCKDB_SQL_PARITY_VERIFIED = SQL_PARITY_VERIFIED_CANONICALS
DUCKDB_SQL_PRODUCTION_SAFE = SQL_PRODUCTION_SAFE_CANONICALS

# ClickHouse 暂不与 DuckDB 共享认证
CLICKHOUSE_SQL_PARITY_VERIFIED = frozenset()
CLICKHOUSE_SQL_PRODUCTION_SAFE = frozenset()

# 有 emitter 实现但暂不默认 production SQL 下推
SQL_PRODUCTION_DEFERRED_CANONICALS: frozenset[str] = frozenset(
    FASTPATH_DEFERRED_CANONICALS
)

# Cheap elementwise emitters that exist in SQL_IMPLEMENTED but stay on the
# *extended* surface (not daily).  They push down in research/validation mode
# via ``is_sql_capable``; promoting them to production SQL would require daily
# surface + six-way dual evidence.  Not forced — pandas (and Polars for
# ``scale``) is fine when the formula stays off the daily allowlist.
SQL_RESEARCH_SPEED_CANDIDATES: frozenset[str] = frozenset(
    {
        "cbrt",
        "truncate",
        "is_nan",
        "scale",
    }
)

# 向后兼容
SQL_CAPABLE_CANONICALS = SQL_IMPLEMENTED_CANONICALS


def is_sql_implemented(canon: str) -> bool:
    """判断 canonical 是否有 SQL emitter 实现。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否在 ``SQL_IMPLEMENTED_CANONICALS`` 内。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_IMPLEMENTED_CANONICALS


def is_sql_parity_verified(canon: str) -> bool:
    """判断 canonical 是否已通过 SQL parity 验证。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否在 ``SQL_PARITY_VERIFIED_CANONICALS`` 内。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_PARITY_VERIFIED_CANONICALS


def is_sql_production_safe(canon: str) -> bool:
    """判断 canonical 是否在静态 SQL production-safe 白名单内。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否在 ``SQL_PRODUCTION_SAFE_CANONICALS`` 内（不含运行时降级）。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_PRODUCTION_SAFE_CANONICALS


_DUCKDB_DOWNGRADE_CACHE: frozenset[str] | None = None


def duckdb_downgraded_canonicals(*, refresh: bool = False) -> frozenset[str]:
    """按当前 DuckDB 部署能力应从 production SQL 降级的 canonical。"""
    global _DUCKDB_DOWNGRADE_CACHE
    if _DUCKDB_DOWNGRADE_CACHE is None or refresh:
        try:
            from backend.sql_pushdown.duckdb_capabilities import (
                downgrade_sql_canonicals,
                get_duckdb_capability_report,
            )

            report = get_duckdb_capability_report(refresh=refresh)
            _DUCKDB_DOWNGRADE_CACHE = downgrade_sql_canonicals(report)
        except Exception:
            _DUCKDB_DOWNGRADE_CACHE = frozenset()
    return _DUCKDB_DOWNGRADE_CACHE


def effective_sql_production_safe(canon: str, *, refresh_duckdb: bool = False) -> bool:
    """静态 SQL_PRODUCTION_SAFE 减去 DuckDB 运行时能力降级。"""
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    if name not in SQL_PRODUCTION_SAFE_CANONICALS:
        return False
    return name not in duckdb_downgraded_canonicals(refresh=refresh_duckdb)
