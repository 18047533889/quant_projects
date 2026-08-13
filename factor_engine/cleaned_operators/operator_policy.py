# -*- coding: utf-8 -*-
"""算子执行策略（Operator Policy）：企业级语义的标准化描述。

与 ``OperatorMetadata``（文档/catalog）互补：
- metadata：人类可读、LLM 提示词
- policy：机器可读、lookback 推断、PIT 审计、lineage hash
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

Scope = Literal[
    "ts", "cs", "group", "fundamental_period", "session_intraday",
    "elementwise", "aggregate", "hypothesis", "unknown",
]
NanPolicy = Literal["propagate", "ignore", "zero", "ffill_only"]

# research 常用：须有显式 policy；与 operator_surface.RESEARCH_ONLY_CANONICALS 对齐。
# 已迁出的 micro_* / ttm / recipe 名不再列入（避免假阳性 research 白名单）。
RESEARCH_CORE_CANONICALS: frozenset[str] = frozenset(
    {
        "group_decay_linear",
        "trade_when",
        "ts_poly2_coeff",
        "ts_poly2_resid",
        "ts_moment",
        "ts_max_buildup",
        "ts_sum_decay",
        "tail_beta",
        "idio_skew",
        "residual_momentum_capm",
        "coskewness_to_market",
        "rolling_beta_to_market",
        "digital_count",
        "intraday_vwap_deviation",
    }
)

# 须有显式 policy 的基础设施算子（非 production_core / research_core）
_POLICY_EXTENSION_CANONICALS: frozenset[str] = frozenset(
    {
        "WMA",
        "ts_ema",
        "ts_ewm_corr",
        "neutralize",
        "quantile",
        "standardize",
        "group_winsorize",
        "cum_prod",
        "cum_delta",
        "cum_first",
        "expanding_rank",
        "col",
        "add",
        "subtract",
        "multiply",
        "divide",
        # Filter Layer (2026-08-12)
        "ts_hampel_filter_causal",
        "ts_median3_causal",
        "ts_rolling_median_causal",
    }
)


# 向后兼容：Tier-1 = policy_required（延迟求值，避免与 operator_spec 循环 import）
def policy_required_canonicals() -> frozenset[str]:
    """获取必须有显式 OperatorPolicy 的 canonical 并集。

返回:
    production core、research core 与 policy 扩展算子的 ``frozenset``。
"""
    from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS
    from cleaned_operators.operator_surface import (
        DAILY_CANONICALS,
        EXTENDED_ONLY_CANONICALS,
        INTERNAL_ONLY_CANONICALS,
        LEGACY_ONLY_CANONICALS,
        RESEARCH_ONLY_CANONICALS,
    )

    active = (
        DAILY_CANONICALS | EXTENDED_ONLY_CANONICALS | INTERNAL_ONLY_CANONICALS
        | LEGACY_ONLY_CANONICALS | RESEARCH_ONLY_CANONICALS
    )

    return (
        PRODUCTION_CORE_CANONICALS
        | RESEARCH_CORE_CANONICALS
        | (_POLICY_EXTENSION_CANONICALS & active)
    ) & active


def tier1_canonicals() -> frozenset[str]:
    """向后兼容别名：等同 ``policy_required_canonicals()``。

返回:
    Tier-1 / policy_required canonical 集合。
"""
    return policy_required_canonicals()


def __getattr__(name: str):
    """模块级延迟属性访问（向后兼容 ``TIER1_CANONICALS``）。

参数:
    name: 属性名。

返回:
    请求的属性值。

异常:
    AttributeError: 未知属性名。
"""
    if name == "TIER1_CANONICALS":
        return policy_required_canonicals()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Tier-1 DSL 别名 → canonical（CI / policy 校验前先 resolve）
TIER1_ALIASES: dict[str, str] = {
    "SMA": "ts_mean",
    "EMA": "ts_ema",
    "ema": "ts_ema",
    "ewm_mean": "ts_ema",
    "decay_linear": "ts_decay_linear",
    "cap": "clip",
    "delay": "ts_delay",
    "ts_regression": "ts_regression_slope",
    "safe_div": "safe_div_null",
}

# production ``auto`` backend 下允许走 Polars 的 canonical 白名单
#
# 三层覆盖（与 registry / SQL emitter 分离）：
#   1. runtime 实现：OperatorRegistry 有 pandas_numpy / polars / sql
#   2. POLARS_PARITY_VERIFIED：parity CI 通过（Tier-1…9，见 tests/operators/test_polars_parity_tier*.py）
#   3. POLARS_PRODUCTION_SAFE：production auto 默认走 Polars（= CORE + PARITY_VERIFIED）
#
# 晋级路径：有 Polars 实现 → parity test → 并入 POLARS_PARITY_VERIFIED → 自动进入 PRODUCTION_SAFE
POLARS_PRODUCTION_SAFE_CORE: frozenset[str] = frozenset({
    "ts_mean",
    "ts_sum",
    "ts_min",
    "ts_max",
    "ts_delta",
    "ts_delay",
    "add",
    "subtract",
    "multiply",
    "divide",
    "abs",
    "log",
    "clip",
    "neg",
    "exp",
    "sqrt",
    "sign",
})

# parity golden 已通过、可安全走 auto→polars 的扩展层（Tier-1）
POLARS_PARITY_VERIFIED_TIER1: frozenset[str] = frozenset({
    "rank",
    "zscore",
    "ts_std",
    "ffill",
    "ts_pct",
    "winsorize",
})

# Tier-2：双序列 / 截面变换 / 价量（见 tests/operators/test_polars_parity_tier2.py）
POLARS_PARITY_VERIFIED_TIER2: frozenset[str] = frozenset({
    "ts_corr",
    "ts_rank",
    "coalesce",
    "protected_div",
    "safe_div_null",
    "protected_log",
    "where",
    "cs_demean",
    "scale",
    "normalize",
    "group_rank",
    "rolling_beta",
    "vwap",
})

# Tier-3：EMA / decay / fillna / group 统计（见 tests/operators/test_polars_parity_tier3.py）
POLARS_PARITY_VERIFIED_TIER3: frozenset[str] = frozenset({
    "ts_ema",
    "ts_decay_linear",
    "fillna_const",
    "fillna",
    "group_zscore",
    "group_mean",
    "group_neutralize",
    "ts_var",
    "ts_median",
})

# Tier-4：Sharpe / 自相关（见 tests/operators/test_polars_parity_tier4_candidates.py）
POLARS_PARITY_VERIFIED_TIER4: frozenset[str] = frozenset({
    "ts_sharpe",
    "ts_autocorr",
})

# Tier-5：Core 缺口 — ts_zscore / protected_sqrt（见 tests/operators/test_polars_parity_tier5.py）
POLARS_PARITY_VERIFIED_TIER5: frozenset[str] = frozenset({
    "ts_zscore",
    "protected_sqrt",
})

# Tier-6：双序列回归 / Beta（见 tests/operators/test_polars_parity_tier6.py）
POLARS_PARITY_VERIFIED_TIER6: frozenset[str] = frozenset({
    "ts_beta",
    "cs_resid",
    "cs_regression",
})

# Tier-7：PRODUCTION_CORE TA Wilder（见 tests/operators/test_polars_parity_tier7.py）
POLARS_PARITY_VERIFIED_TIER7: frozenset[str] = frozenset({
    "RSI_WILDER",
    "ATR_WILDER",
})

# Tier-8：截面聚合 / EWM·WMA / 清洗·group 扩展（见 tests/operators/test_polars_parity_tier8.py）
# Names are post-governance canonicals (c_mean → cs_mean, etc.).
POLARS_PARITY_VERIFIED_TIER8: frozenset[str] = frozenset({
    "cs_mean",
    "cs_std",
    "cs_sum",
    "cs_count",
    "ts_ema",
    "WMA",
    "nan_to_num",
    "is_finite",
    "group_std",
    "group_normalize",
    "ts_cov",
})

# Tier-9：扩展 ts / group / cum / EWM / 比较逻辑 / 数学（见 tests/operators/test_polars_parity_tier9.py）
POLARS_PARITY_VERIFIED_TIER9: frozenset[str] = frozenset({
    # 时序扩展
    "ts_mad",
    "ts_skew",
    "ts_kurt",
    "ts_quantile",
    "ts_product",
    "ts_argmax",
    "ts_argmin",
    "ts_regression_slope",
    "ts_ratio",
    "ts_max_buildup",
    "ts_moment",
    "ts_ratio",
    "Slope",
    # 截面 / 分组
    "cs_quantile",
    "cs_mad",
    "cs_mad_zscore",
    "quantile",
    "group_percentile",
    "group_decay_linear",
    "group_winsorize",
    # EWM 矩
    "ewm_std",
    "ewm_var",
    "ewm_corr",
    "ewm_cov",
    # 累计 / 扩展
    "cum_sum",
    "cum_max",
    "cum_min",
    "cum_prod",
    "cum_delta",
    "expanding_std",
    "expanding_rank",
    "expanding_mean",
    "expanding_sum",
    "count",
    # 元素 / 比较 / 逻辑
    "power",
    "floor",
    "ceil",
    "inverse",
    "minimum",
    "maximum",
    "signed_sqrt",
    "signed_log",
    "log_abs",
    "is_nan",
    "is_null",
    "is_not_null",
    "gt",
    "lt",
    "ge",
    "le",
    "eq",
    "ne",
    "and_",
    "or_",
    "not_",
})

# Tier-10：价量 / 截面百分位（见 tests/operators/test_polars_parity_tier10.py）
POLARS_PARITY_VERIFIED_TIER10: frozenset[str] = frozenset({
    "log_returns",
    "volatility",
    "rank_pct",
    "cs_pct_rank",
    "cs_quantile",
})

from backend.primitive_evidence import (
    POLARS_EDGE_VERIFIED as _POLARS_EDGE_VERIFIED,
    POLARS_NO_FALLBACK_VERIFIED as _POLARS_NO_FALLBACK_VERIFIED,
    POLARS_REFERENCE_PARITY_VERIFIED as _POLARS_REFERENCE_PARITY_VERIFIED,
)

# Deprecated Tier exports remain import-compatible for old test modules, but
# their values are now derived from the current surface and certified evidence.
# Historical/deleted names in the source-era partitions cannot leak into policy,
# routing, capability reports, or production admission.
from cleaned_operators.operator_surface import (  # noqa: E402
    DAILY_CANONICALS as _DAILY_SURFACE,
    EXTENDED_ONLY_CANONICALS as _EXTENDED_SURFACE,
    INTERNAL_ONLY_CANONICALS as _INTERNAL_SURFACE,
    LEGACY_ONLY_CANONICALS as _LEGACY_SURFACE,
    RESEARCH_ONLY_CANONICALS as _RESEARCH_SURFACE,
)

_CURRENT_ACTIVE_SURFACE = (
    _DAILY_SURFACE | _EXTENDED_SURFACE | _INTERNAL_SURFACE
    | _LEGACY_SURFACE | _RESEARCH_SURFACE
)
for _tier_number in range(1, 11):
    _tier_name = f"POLARS_PARITY_VERIFIED_TIER{_tier_number}"
    globals()[_tier_name] = frozenset(
        globals()[_tier_name]
        & _CURRENT_ACTIVE_SURFACE
        & _POLARS_REFERENCE_PARITY_VERIFIED
    )
POLARS_PRODUCTION_SAFE_CORE = frozenset(
    POLARS_PRODUCTION_SAFE_CORE
    & _DAILY_SURFACE
    & _POLARS_REFERENCE_PARITY_VERIFIED
    & _POLARS_EDGE_VERIFIED
    & _POLARS_NO_FALLBACK_VERIFIED
)

# Mutable in place during registry finalisation so modules which imported these
# objects before load_all() observe the same final evidence-backed truth.
POLARS_PARITY_VERIFIED: set[str] = set(_POLARS_REFERENCE_PARITY_VERIFIED)
POLARS_PRODUCTION_SAFE: set[str] = set(
    _POLARS_REFERENCE_PARITY_VERIFIED
    & _POLARS_EDGE_VERIFIED
    & _POLARS_NO_FALLBACK_VERIFIED
)


def polars_implemented_canonicals() -> frozenset[str]:
    """动态查询 Registry 中已有 polars backend 的 canonical。

返回:
    已实现 polars 后端的 canonical ``frozenset``（非 production 白名单）。
"""
    from cleaned_operators.registry import OperatorRegistry

    return frozenset(
        c
        for c in OperatorRegistry.list_canonical()
        if "polars" in OperatorRegistry.backends_for(c)
    )


def check_polars_production_gate() -> list[str]:
    """校验 Polars production 准入集合的内部一致性。

返回:
    违规描述字符串列表。
"""
    errors: list[str] = []
    implemented = polars_implemented_canonicals()
    if not POLARS_PRODUCTION_SAFE <= POLARS_PARITY_VERIFIED:
        errors.append("POLARS_PRODUCTION_SAFE 必须是 reference parity 子集")
    if not POLARS_PRODUCTION_SAFE <= implemented:
        errors.append("POLARS_PRODUCTION_SAFE 含未注册原生 Polars backend 的算子")
    if not POLARS_PARITY_VERIFIED <= implemented:
        errors.append("POLARS_PARITY_VERIFIED 含未注册 Polars backend 的算子")
    return errors


def resolve_tier1_canonical(name: str) -> str:
    """Tier-1 校验前将 DSL 别名解析为 canonical。

参数:
    name: DSL 名或 canonical 名。

返回:
    解析后的 canonical 名。
"""
    from cleaned_operators.registry import OperatorRegistry

    if name in TIER1_ALIASES:
        return TIER1_ALIASES[name]
    return OperatorRegistry._aliases.get(name, name)


def tier1_policy_keys() -> frozenset[str]:
    """获取 policy_required 解析后的 canonical policy 键集合。

返回:
    含 canonical 与 Tier-1 别名的 ``frozenset``。
"""
    return frozenset(resolve_tier1_canonical(name) for name in policy_required_canonicals())

# 不做 Polars 移植且通常非 PIT 安全（FFT/矩阵/CDF-PDF 等）。
# R30 §2: shuffle / rand_* 已物理删除（tombstones 单点），不再属于任何算子集合。
INTENTIONALLY_PANDAS_ONLY: frozenset[str] = frozenset(
    {
        "fft",
        "ifft",
        "convolve",
        "filter_bandpass",
        "filter_highpass",
        "filter_lowpass",
        "filter_notch",
        "wavelet",
        "wavelet_denoise",
        "mat_add",
        "mat_subtract",
        "mat_multiply",
        "mat_transpose",
        "mat_inverse",
        "mat_determinant",
        "mat_rank",
        "eig",
        "svd",
        "pca",
        "qr_decompose",
        "lu_decompose",
        "cdf_chi2",
        "cdf_f",
        "cdf_normal",
        "cdf_t",
        "pdf_chi2",
        "pdf_f",
        "pdf_normal",
        "pdf_t",
        # experimental / 待 Polars 移植（daily 已 native 的不要放这里）
        "open_gap",
        "close_gap",
        "cs_quantile",
        "cs_rank_01",
        "rank_pct",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "operating_margin",
        "intraday_vwap_deviation",
        "div_or_default",
        "log_fill_invalid",
        "div_or_null",
    }
)

# 暂无 Polars、但 metadata 仍 PIT 安全（不应强制 pit_safe=False）
PANDAS_ONLY_PIT_SAFE: frozenset[str] = frozenset(
    {
        "open_gap",
        "close_gap",
        "cs_quantile",
        "cs_rank_01",
        "rank_pct",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "operating_margin",
        "intraday_vwap_deviation",
    }
)

# Backend availability is not a temporal-safety property.  In particular,
# deterministic elementwise transforms must not become PIT-unsafe merely
# because one execution backend is missing.
# Temporal safety is independent of backend availability.  Keep this set
# limited to operators whose semantics actually consume future observations or
# deliberately destroy time ordering.
# R30 §2: Lead/next/bfill/causal_bfill/fillna_interpolate/shuffle were removed
# from the executable system (see ``tombstones``) — they are not PIT-unsafe
# operators, they are deleted operators.  The set is now empty; any future
# future-referencing candidate must instead go through the tombstone audit.
PIT_UNSAFE_CANONICALS: frozenset[str] = frozenset()

# 破坏 panel shape 的算子（即使 PIT-safe 也不进 production）
NON_SHAPE_PRESERVING_CANONICALS: frozenset[str] = frozenset({"dropna"})

# WS4 P0-07: minute -> daily grain-changing canonicals (``intra_*`` /
# ``intraday_*`` minute→daily aggregation operators in the intraday package).
# These consume a minute-frequency panel and emit one scalar per (TradeDate,
# Symbol), so they are NOT shape/index-preserving — the daily-aggregation axis
# is strictly shorter than the input axis.  ``shape_preserving`` /
# ``index_preserving`` / ``columns_preserving`` must therefore be ``False`` for
# them, even though they are PIT-safe.  Ops carrying the
# ``grain_minute_to_daily`` metadata tag are treated the same way regardless of
# whether their canonical is enumerated here.
GRAIN_CHANGING_CANONICALS: frozenset[str] = frozenset(
    {
        "intra_bar_range_deviation", "intra_bar_range_persistence",
        "intra_beta_asymmetry", "intra_close_participation",
        "intra_continuous_variance", "intra_down_down_semibeta",
        "intra_down_up_semibeta", "intra_drawdown_depth",
        "intra_drawdown_duration", "intra_drawdown_recovery_half_life",
        "intra_high_low_affinity", "intra_idiosyncratic_kurtosis",
        "intra_idiosyncratic_kurtosis_ex_self", "intra_idiosyncratic_skewness",
        "intra_idiosyncratic_skewness_ex_self", "intra_idiosyncratic_variance",
        "intra_idiosyncratic_variance_ex_self", "intra_industry_lead_lag_ex_self",
        "intra_interval_amount_share", "intra_interval_illiquidity",
        "intra_interval_realized_variance", "intra_interval_return",
        "intra_interval_volume_share", "intra_interval_vwap_deviation",
        "intra_jump_clustering", "intra_jump_concentration", "intra_jump_count",
        "intra_jump_first_time", "intra_jump_last_time", "intra_jump_variation",
        "intra_longest_above_vwap_streak", "intra_longest_below_vwap_streak",
        "intra_market_lead_lag_ex_self", "intra_market_model_r2",
        "intra_market_model_r2_ex_self", "intra_max_drawdown", "intra_max_drawup",
        "intra_negative_jump_variation", "intra_positive_jump_variation",
        "intra_price_vwap_max_negative_excursion",
        "intra_price_vwap_max_positive_excursion", "intra_realized_beta",
        "intra_realized_beta_ex_self", "intra_realized_correlation",
        "intra_realized_correlation_ex_self", "intra_realized_kurtosis",
        "intra_realized_quarticity", "intra_realized_skewness",
        "intra_same_slot_momentum", "intra_same_slot_reversal",
        "intra_session_return_asymmetry", "intra_signed_jump_ratio",
        "intra_signed_tail_variation_ratio", "intra_slot_amount_surprise",
        "intra_slot_volatility_surprise", "intra_slot_volume_surprise",
        "intra_tail_volume_share", "intra_time_above_vwap",
        "intra_tripower_quarticity", "intra_up_down_semibeta",
        "intra_up_up_semibeta", "intra_ute_high", "intra_ute_low",
        "intra_volume_price_alignment", "intra_vwap_path_curvature",
        "intra_vwap_path_curvature_pct", "intra_vwap_path_slope",
        "intra_vwap_path_slope_pct", "intra_vwap_reversion_speed",
        "intraday_rv_signature_slope",
        # R11 unusable-operators sweep: concurrent minute-source additions that
        # carry the ``minute`` tag (minute -> daily aggregation) but were not yet
        # classified grain-changing, so their policy wrongly said shape-preserving.
        "intraday_bvc_imbalance",
        "intraday_impact_asymmetry",
        "intraday_impact_beta",
        "intraday_return_wasserstein_shift",
        "intraday_volume_clock_path_efficiency",
        "intraday_volume_clock_roughness",
        "session_event_recovery_score",
    }
)

# 显式声明的核心算子策略（Tier-1 + 特殊语义；其余由 infer_operator_policy 推断）
_EXPLICIT_POLICIES: dict[str, dict[str, Any]] = {
    "tanh": {"scope": "elementwise", "pit_safe": True},
    "sigmoid": {"scope": "elementwise", "pit_safe": True},
    "signed_log": {"scope": "elementwise", "pit_safe": True},
    "log_abs": {"scope": "elementwise", "pit_safe": True},
    "signed_sqrt": {"scope": "elementwise", "pit_safe": True},
    "square": {"scope": "elementwise", "pit_safe": True},
    "log10": {"scope": "elementwise", "pit_safe": True},
    "log2": {"scope": "elementwise", "pit_safe": True},
    "ashare_limit_distance": {"scope": "elementwise", "pit_safe": True},
    "ashare_limit_up_touch": {"scope": "elementwise", "pit_safe": True},
    "ashare_limit_down_touch": {"scope": "elementwise", "pit_safe": True},
    "ashare_open_at_upper_limit": {"scope": "elementwise", "pit_safe": True},
    "ashare_limit_open_failed": {"scope": "elementwise", "pit_safe": True},
    "ashare_limit_one_price": {"scope": "elementwise", "pit_safe": True},
    "ashare_limit_failed": {"scope": "elementwise", "pit_safe": True},
    "cs_neutralize": {"scope": "cs", "pit_safe": True},
    "size_neutralize": {"scope": "cs", "pit_safe": True},
    "industry_size_neutralize": {"scope": "cs", "pit_safe": True},
    "ffill_limit": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_argmax": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_argmin": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_time_slope": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_regression_intercept": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_regression_resid": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_regression_r2": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_topk_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_topk_std": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_bottomk_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_bottomk_sum": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_bottomk_std": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_tail_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "period_average": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
    "period_change": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
    "period_cagr": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 2},
    "period_stability": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 2},
    "revision_delta": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
    "fundamental_staleness": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
    "ts_delay": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_delta": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_pct": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_log_return": {"scope": "ts", "lag": 1, "pit_safe": True, "min_periods": 2},
    "ts_sharpe": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_autocorr": {"scope": "ts", "pit_safe": True, "min_periods": 3, "lag": 1},
    # Explicit causal contracts for operators promoted from extended/research review.
    "price_spread_deviation": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "saturate": {"scope": "elementwise", "pit_safe": True, "min_periods": 1},
    "signed_power": {"scope": "elementwise", "pit_safe": True, "min_periods": 1},
    "ts_decay_exp_window": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_sum_decay": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_moment": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_ratio": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 2},
    "trade_when": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
    },
    "ts_max_buildup": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "digital_count": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 1},
    "intraday_vwap_deviation": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "operating_margin": {"scope": "elementwise", "pit_safe": True},
    "current_ratio": {"scope": "elementwise", "pit_safe": True},
    "quick_ratio": {"scope": "elementwise", "pit_safe": True},
    "debt_to_equity": {"scope": "elementwise", "pit_safe": True},
    "ts_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_std": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_sum": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_rank": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_corr": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_min": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_max": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_ema": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "SMA": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "WMA": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ewm_corr": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_beta": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_regression_slope": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_count_if": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_sum_if": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_mean_if": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_std_if": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_min_if": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_max_if": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_quantile_if": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_corr_if": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_beta_if": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_regression_resid_if": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_last_if": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_days_since": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_true_streak": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "period_lag": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
    "ts_regression_tstat": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_trend_tstat": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_max_drawdown": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_partial_corr": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_nth_value": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "rolling_beta": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "rolling_beta_to_market": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_poly2_coeff": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_poly2_resid": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    # R22-058 causal siblings: the model is fit STRICTLY on [t-d, t-1], the
    # current observation only evaluates (forecast error / prior coeff) — no
    # current row participates in its own fit.  These are causal by construction
    # and must be PIT-safe (R38 evidence-regen blocker: without this declaration
    # the factor certifier reports them "PIT policy is not causal").
    "ts_poly2_prior_coeff": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_poly2_forecast_error": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_poly2_forecast_error_z": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "idio_vol": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "idio_skew": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "downside_beta": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "tail_beta": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "residual_momentum_capm": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "coskewness_to_market": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "open_gap": {"scope": "ts", "pit_safe": True, "lag": 1},
    "close_gap": {"scope": "elementwise", "pit_safe": True},
    "cs_mad": {"scope": "cs", "pit_safe": True},
    "cs_mad_zscore": {"scope": "cs", "pit_safe": True},
    "cs_quantile": {"scope": "cs", "pit_safe": True},
    "ADXR": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "AROON": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "AROON_up": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "AROON_down": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "RSI_WILDER": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ATR_WILDER": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_decay_linear": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "group_decay_linear": {"scope": "cs", "pit_safe": True},
    "hump_decay": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_topk_sum": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "quantile": {"scope": "cs", "pit_safe": True},
    "scale": {"scope": "cs", "pit_safe": True},
    "normalize": {"scope": "cs", "pit_safe": True},
    "standardize": {"scope": "cs", "pit_safe": True},
    "col": {"scope": "elementwise", "lookback_window": 0, "pit_safe": True},
    "ffill": {"scope": "elementwise", "pit_safe": True, "nan_policy": "ffill_only"},
    "fillna_const": {"scope": "elementwise", "pit_safe": True},
    "causal_linear_extrapolate": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 2,
        "nan_policy": "ffill_only",
    },
    "rank": {"scope": "cs", "pit_safe": True},
    "rank_pct": {"scope": "cs", "pit_safe": True},
    "cs_pct_rank": {"scope": "cs", "pit_safe": True},
    "cs_rank_01": {"scope": "cs", "pit_safe": True},
    "cs_quantile": {"scope": "cs", "pit_safe": True},
    "cs_mean": {"scope": "cs", "pit_safe": True},
    "cs_std": {"scope": "cs", "pit_safe": True},
    "cs_sum": {"scope": "cs", "pit_safe": True},
    "cs_count": {"scope": "cs", "pit_safe": True},
    "is_infinite": {"scope": "elementwise", "pit_safe": True},
    "cbrt": {"scope": "elementwise", "pit_safe": True},
    "round": {"scope": "elementwise", "pit_safe": True},
    "truncate": {"scope": "elementwise", "pit_safe": True},
    "ts_quantile": {"scope": "ts", "pit_safe": True},
    "ts_skew": {"scope": "ts", "pit_safe": True},
    "ts_kurt": {"scope": "ts", "pit_safe": True},
    "group_count": {"scope": "group", "pit_safe": True},
    "group_max": {"scope": "group", "pit_safe": True},
    "group_min": {"scope": "group", "pit_safe": True},
    "group_sum": {"scope": "group", "pit_safe": True},
    "constant": {"scope": "elementwise", "pit_safe": True},
    "identity": {"scope": "elementwise", "pit_safe": True},
    "protected_div": {"scope": "elementwise", "pit_safe": True},
    "safe_div_null": {"scope": "elementwise", "pit_safe": True},
    "protected_log": {"scope": "elementwise", "pit_safe": True},
    "protected_sqrt": {"scope": "elementwise", "pit_safe": True},
    "vwap": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "volatility": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "zscore": {"scope": "cs", "pit_safe": True},
    "winsorize": {"scope": "cs", "pit_safe": True},
    "neutralize": {"scope": "cs", "pit_safe": True},
    "group_rank": {"scope": "cs", "pit_safe": True},
    "group_neutralize": {"scope": "cs", "pit_safe": True},
    "group_winsorize": {"scope": "cs", "pit_safe": True},
    "group_zscore": {"scope": "cs", "pit_safe": True},
    "group_mean": {"scope": "cs", "pit_safe": True},
    "cs_regression": {"scope": "cs", "pit_safe": True},
    "cs_resid": {"scope": "cs", "pit_safe": True},
    "cs_bucket": {"scope": "cs", "pit_safe": True},
    "cs_multi_resid": {"scope": "cs", "pit_safe": True},
    "cs_wls_resid": {"scope": "cs", "pit_safe": True},
    "add": {"scope": "elementwise", "pit_safe": True},
    "subtract": {"scope": "elementwise", "pit_safe": True},
    "multiply": {"scope": "elementwise", "pit_safe": True},
    "divide": {"scope": "elementwise", "pit_safe": True},
    "cum_prod": {"scope": "ts", "pit_safe": True, "includes_current_bar": True},
    "cum_delta": {"scope": "ts", "pit_safe": True},
    "cum_first": {"scope": "ts", "pit_safe": True},
    "expanding_rank": {"scope": "ts", "pit_safe": True},
    "rank_corr": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "vp_weighted_price": {"scope": "ts", "pit_safe": True},
    # NOTE: real_turnover_rate (volume / float_shares) is a per-day elementwise
    # rate and is defined once below under the elementwise section.  A previous
    # duplicate ts-scoped entry was silently overriding it; removed.
    "micro_realized_vol": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 2,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_spread": {"scope": "elementwise", "pit_safe": True},
    "micro_amihud_hf": {
        "scope": "ts",
        "pit_safe": True,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_mid_return": {
        "scope": "ts",
        "pit_safe": True,
        "lag": 1,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_bipower_var": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 2,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_jump_indicator": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 2,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_trade_imbalance": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_vpin": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_kyle_lambda": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 2,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "avg": {"scope": "aggregate", "pit_safe": True},
    "corr_test": {"scope": "hypothesis", "pit_safe": True},
    "abs": {"scope": "elementwise", "pit_safe": True},
    "log": {"scope": "elementwise", "pit_safe": True},
    "clip": {"scope": "elementwise", "pit_safe": True},
    "neg": {"scope": "elementwise", "pit_safe": True},
    "power": {"scope": "elementwise", "pit_safe": True},
    "floor": {"scope": "elementwise", "pit_safe": True},
    "ceil": {"scope": "elementwise", "pit_safe": True},
    "inverse": {"scope": "elementwise", "pit_safe": True},
    "maximum": {"scope": "elementwise", "pit_safe": True},
    "minimum": {"scope": "elementwise", "pit_safe": True},
    "flex_max": {"scope": "elementwise", "pit_safe": True},
    "flex_min": {"scope": "elementwise", "pit_safe": True},
    "gt": {"scope": "elementwise", "pit_safe": True},
    "lt": {"scope": "elementwise", "pit_safe": True},
    "ge": {"scope": "elementwise", "pit_safe": True},
    "le": {"scope": "elementwise", "pit_safe": True},
    "eq": {"scope": "elementwise", "pit_safe": True},
    "ne": {"scope": "elementwise", "pit_safe": True},
    "and_": {"scope": "elementwise", "pit_safe": True},
    "or_": {"scope": "elementwise", "pit_safe": True},
    "not_": {"scope": "elementwise", "pit_safe": True},
    "is_nan": {"scope": "elementwise", "pit_safe": True},
    "is_null": {"scope": "elementwise", "pit_safe": True},
    "is_not_null": {"scope": "elementwise", "pit_safe": True},
    "is_finite": {"scope": "elementwise", "pit_safe": True},
    "nan_to_num": {"scope": "elementwise", "pit_safe": True},
    "fillna": {"scope": "elementwise", "pit_safe": True},
    "log_returns": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 2},
    "group_std": {"scope": "cs", "pit_safe": True},
    "group_normalize": {"scope": "cs", "pit_safe": True},
    "group_percentile": {"scope": "cs", "pit_safe": True},
    "ts_cov": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "exp": {"scope": "elementwise", "pit_safe": True},
    "sqrt": {"scope": "elementwise", "pit_safe": True},
    "sign": {"scope": "elementwise", "pit_safe": True},
    "where": {"scope": "elementwise", "pit_safe": True},
    "if_else": {"scope": "elementwise", "pit_safe": True},
    "coalesce": {"scope": "elementwise", "pit_safe": True},
    "cs_demean": {"scope": "cs", "pit_safe": True},
    "ts_var": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_median": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "quarter": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "quarter_from_cumulative": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
    "ttm": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ttm_from_quarterly": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
    "ttm_from_cumulative": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
    "yoy": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "yoy_by_period": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 5},
    "ts_ewm_std": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_ewm_var": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_ewm_cov": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_ewm_corr": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "avg2": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_zscore": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Phase-1 composite lowerings（evidence 齐全且非 micro_* 可 production）
    "MOM": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ROC": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "BollingerBands": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "BollingerUpper": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "BollingerLower": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "DPO": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "WilliamsR": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "StochasticK": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "StochasticD": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "OBV": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "operating_margin": {"scope": "elementwise", "pit_safe": True},
    "current_ratio": {"scope": "elementwise", "pit_safe": True},
    "quick_ratio": {"scope": "elementwise", "pit_safe": True},
    "debt_to_equity": {"scope": "elementwise", "pit_safe": True},
    "real_turnover_rate": {"scope": "elementwise", "pit_safe": True},
    "micro_spread": {"scope": "elementwise", "pit_safe": True},
    # 2026-08 advanced: rolling first-digit distribution is a time-series
    # transform (the ``report_`` prefix does not infer ``ts`` scope).
    "report_benford_js_divergence": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Phase 1: A-class atomic operators - robust statistics family (2026-08)
    "ts_quantile_range": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_trimmed_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_robust_zscore_inclusive": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_robust_zscore_prior": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Phase 1: A-class atomic operators - downside risk family (2026-08)
    "ts_downside_deviation": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_upside_deviation": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_current_drawdown_duration": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_time_under_water": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_best_lag_corr": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_price_delay": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    # Filter Layer (2026-08-12)
    "ts_hampel_filter_causal": {"scope": "ts", "pit_safe": True, "min_periods": "window"},
    "ts_median3_causal": {"scope": "ts", "pit_safe": True},
    "ts_rolling_median_causal": {"scope": "ts", "pit_safe": True, "min_periods": "min_periods"},
    "ts_super_smoother": {"scope": "ts", "pit_safe": True, "min_periods": "period", "tags": ["stateful", "filter_role:low_pass"]},
    "ts_kama": {"scope": "ts", "pit_safe": True, "min_periods": "er_window", "tags": ["stateful", "filter_role:adaptive_low_pass"]},
    "state_adaptive_slew_limit": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:rate_limit"]},
    "state_adaptive_deadband": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:hysteresis"]},
    "state_rank_deadband": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:hysteresis"]},
    "state_quantile_hysteresis": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:hysteresis"]},
    "state_l1_turnover_prox": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:rate_limit"]},
    "state_l2_partial_adjustment": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:rate_limit"]},
    # Relation Jaccard (2026-08-13) - fiscal period scope
    "relation_jaccard": {"scope": "fundamental_period", "pit_safe": True},
}


def _reviewed_active_policy(canonical: str, *, pit_safe: bool) -> dict[str, Any]:
    """Materialize a fail-closed policy row for an active canonical."""
    if canonical.startswith("group_"):
        scope: Scope = "group"
    elif canonical.startswith("cs_") or canonical in {
        "rank", "normalize", "winsorize", "zscore",
    }:
        scope = "cs"
    elif canonical.startswith("ts_"):
        scope = "ts"
    elif canonical.startswith("period_") or canonical in {
        "quarter_from_cumulative", "ttm_from_cumulative",
        "ttm_from_quarterly", "yoy_by_period",
    }:
        scope = "fundamental_period"
    else:
        scope = "elementwise"
    policy: dict[str, Any] = {"scope": scope, "pit_safe": pit_safe}
    if scope in {"ts", "fundamental_period"}:
        policy["min_periods"] = 1
    return policy


# Primitive policy is constrained to the reviewed runtime surfaces.  Recipes
# and historical/deleted names are governed by their own registries and cannot
# influence primitive PIT or backend admission.
from cleaned_operators.operator_surface import (  # noqa: E402
    DAILY_CANONICALS,
    EXTENDED_ONLY_CANONICALS,
    INTERNAL_ONLY_CANONICALS,
    LEGACY_ONLY_CANONICALS,
    RESEARCH_ONLY_CANONICALS,
)

_ACTIVE_POLICY_CANONICALS = (
    DAILY_CANONICALS
    | EXTENDED_ONLY_CANONICALS
    | INTERNAL_ONLY_CANONICALS
    | LEGACY_ONLY_CANONICALS
    | RESEARCH_ONLY_CANONICALS
)
_EXPLICIT_POLICIES = {
    name: policy
    for name, policy in _EXPLICIT_POLICIES.items()
    if name in _ACTIVE_POLICY_CANONICALS
}
for _daily_name in DAILY_CANONICALS:
    _EXPLICIT_POLICIES.setdefault(
        _daily_name,
        _reviewed_active_policy(_daily_name, pit_safe=True),
    )
for _active_name in _ACTIVE_POLICY_CANONICALS - DAILY_CANONICALS:
    _EXPLICIT_POLICIES.setdefault(
        _active_name,
        _reviewed_active_policy(_active_name, pit_safe=False),
    )

# Explicit scope overrides for the 2026-08 operator expansion.  Prefix-based
# auto-inference would label non-``ts_``/``cs_``/``group_`` names elementwise;
# these operators carry real time-series / cross-sectional semantics.
_EXPLICIT_POLICIES.update({
    "relation_entry_count": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "relation_exit_count": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "relation_weighted_change": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "index_weight_change": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "index_entry_exit_event": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "index_membership_age": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "event_cumulative_return_past": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "event_abnormal_return_past": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "relation_hhi": {"scope": "cs", "pit_safe": True},
    "relation_entropy": {"scope": "cs", "pit_safe": True},
    "relation_topk_sum": {"scope": "cs", "pit_safe": True},
    "relation_rank_weighted_sum": {"scope": "cs", "pit_safe": True},
    "relation_category_share": {"scope": "group", "pit_safe": True},
    "relation_peer_weighted_mean_ex_self": {"scope": "group", "pit_safe": True},
    "fin_announcement_lag": {"scope": "elementwise", "pit_safe": True},
    "fin_applicability_mask": {"scope": "elementwise", "pit_safe": True},
    "index_member": {"scope": "elementwise", "pit_safe": True},
    # ts_model / state-space / GARCH / HAR / wavelet / complexity families
    "ts_kalman_level": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_kalman_innovation_z": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # round-3: MAD split (one common center per window) — reviewed statistics
    "ts_mean_abs_deviation": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_median_abs_deviation": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_kalman_beta_uncertainty": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_kalman_trend": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_kalman_beta": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_kalman_beta_change": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_garch_vol_forecast": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_garch_persistence": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_garch_standardized_shock": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_gjr_garch_vol_forecast": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_gjr_leverage": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_har_rv_forecast": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_har_rv_innovation_z": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_change_point_probability": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_cusum_vol_break_score": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_dfa_hurst": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_lz_complexity": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_multiscale_entropy_slope": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_permutation_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_regime_duration": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_sample_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_turning_point_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_two_state_regime_probability": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_path_leadlag_area": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_path_signature_area": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_path_signature_depth2_norm": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_matrix_profile_discord_score": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_matrix_profile_motif_distance": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_motif_recurrence_count": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_spectral_low_frequency_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_wavelet_energy_slope": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_wavelet_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_wavelet_high_frequency_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_wavelet_low_frequency_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_ar_forecast": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_ar_innovation": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_ar_innovation_z": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_mean_reversion_half_life": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_variance_ratio_slope": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_huber_regression_coeff": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_huber_regression_resid_z": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_multi_regression_coeff": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_multi_regression_r2": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_multi_regression_resid": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_multi_regression_resid_z": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_quantile_beta_spread": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_quantile_regression_coeff": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_quantile_regression_resid": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_ridge_regression_coeff": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_ridge_regression_resid_z": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_rolling_pca_loading": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_rolling_pca_score": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_rolling_pca_explained_var": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_rolling_pca_idio_share": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_rolling_pca_rank_residual": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_dynamic_factor_r2": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_dynamic_factor_innovation": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_dynamic_factor_load": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_svar_impact": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_svar_forecast": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_granger_pvalue": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_cointegration_spread": {"scope": "ts", "pit_safe": True, "min_periods": 1},
})

# 只保留 active surface 的显式 policy；未分类的 experimental/WIP canonical 由
# ``infer_operator_policy`` 的名称前缀回退推断 scope，避免 inactive 项残留。
_EXPLICIT_POLICIES = {
    name: policy
    for name, policy in _EXPLICIT_POLICIES.items()
    if name in _ACTIVE_POLICY_CANONICALS
}

# 2026-08 final pack (61 atomics).  These explicit policies must SURVIVE the
# active-surface filter above (the surface unions are only populated when the
# operator modules import, which may happen after ``operator_policy`` itself),
# so they are applied here, unconditionally, after the filter.  ``infer_operator_policy``
# reads ``_EXPLICIT_POLICIES`` at call time, so layer_governance / hardening see them.
_FINAL_PACK_POLICIES = {
    # Group 1 — robust tail (7)
    "ts_lower_partial_moment": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_upper_partial_moment": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_expected_shortfall": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_quantile_skew": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_quantile_kurtosis": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_tail_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_extreme_cluster_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Group 2 — nonlinear dependence (6)
    "ts_distance_corr": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_distance_cov": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_mutual_information": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_lagged_mutual_information": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_upper_tail_dependence": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_lower_tail_dependence": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Group 3 — complexity / long memory (8)
    "ts_permutation_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_weighted_permutation_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_permutation_transition_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_sample_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_hurst_dfa": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_higuchi_fractal_dimension": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_variogram_slope": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_autocorr_decay_half_life": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Group 4 — A-share limit/suspension state machine (14)
    "ashare_limit_up_streak": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_limit_down_streak": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_days_since_limit_up": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_days_since_limit_down": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_limit_touch_count": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_failed_limit_count": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_one_price_limit_streak": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_limit_event_density": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_limit_asymmetry": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_suspension_episode_length": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_limit_open_up_streak": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_limit_open_down_streak": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_limit_up_volume_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ashare_limit_down_volume_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Group 5 — relation / group distribution (12)
    "relation_topk_concentration": {"scope": "cs", "pit_safe": True},
    "relation_distribution_skew": {"scope": "cs", "pit_safe": True},
    "relation_distribution_kurtosis": {"scope": "cs", "pit_safe": True},
    "relation_distribution_pearson_kurtosis": {"scope": "cs", "pit_safe": True},
    "relation_distribution_excess_kurtosis": {"scope": "cs", "pit_safe": True},
    "relation_hhi_change": {"scope": "cs", "pit_safe": True},
    "relation_entropy_change": {"scope": "cs", "pit_safe": True},
    "relation_concentration_acceleration": {"scope": "cs", "pit_safe": True},
    "relation_rank_mobility": {"scope": "cs", "pit_safe": True},
    "relation_share_mobility": {"scope": "cs", "pit_safe": True},
    "group_skewness": {"scope": "group", "pit_safe": True},
    "group_kurtosis": {"scope": "group", "pit_safe": True},
    "group_quantile_spread": {"scope": "group", "pit_safe": True},
    "group_tail_ratio": {"scope": "group", "pit_safe": True},
    # Group 6 — intraday time-structure v2 (14, minute in -> daily out)
    "intra_bar_range_persistence": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_bar_range_deviation": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_tail_volume_share": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_volume_price_alignment": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_ute_high": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_ute_low": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_slot_volume_surprise": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_slot_amount_surprise": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_slot_volatility_surprise": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_market_lead_lag_ex_self": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_industry_lead_lag_ex_self": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_session_return_asymmetry": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_close_participation": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_high_low_affinity": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    # Filter Layer (2026-08-12): robust EMA with innovation clipping.
    "ts_robust_ema": {"scope": "ts", "pit_safe": True, "min_periods": "span", "tags": ["stateful", "filter_role:low_pass"]},
    "ts_super_smoother": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "checkpointable", "time_shard_safe:false", "filter_role:low_pass"]},
    "ts_kama": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "checkpointable", "time_shard_safe:false", "filter_role:adaptive_low_pass"]},
    "ts_butterworth_lowpass_causal": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "checkpointable", "time_shard_safe:false", "filter_role:low_pass"]},
    "ts_causal_local_linear_smoother": {"scope": "ts", "pit_safe": True, "min_periods": 10, "tags": ["filter_role:low_pass"]},
    # Filter Layer: hysteresis and turnover control (2026-08-12 P0).
    "state_adaptive_deadband": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:hysteresis"]},
    "state_rank_deadband": {"scope": "group", "pit_safe": True, "tags": ["stateful", "filter_role:hysteresis"]},
    "state_quantile_hysteresis": {"scope": "group", "pit_safe": True, "tags": ["stateful", "filter_role:hysteresis"]},
    "state_adaptive_slew_limit": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:rate_limit"]},
    "state_confidence_weighted_ema": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:adaptive_low_pass"]},
    "state_uncertainty_deadband": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:hysteresis"]},
    "state_l1_turnover_prox": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:rate_limit"]},
    "state_l2_partial_adjustment": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:rate_limit"]},
    "state_cost_aware_deadband": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:rate_limit"]},
    "state_cost_aware_slew": {"scope": "ts", "pit_safe": True, "tags": ["stateful", "filter_role:rate_limit"]},
    # Phase 1: A-class atomic operators - return decomposition family
    "overnight_return": {"scope": "elementwise", "pit_safe": True, "lag": 1},
    "open_close_return": {"scope": "elementwise", "pit_safe": True},
    "open_to_vwap_return": {"scope": "elementwise", "pit_safe": True},
    "vwap_to_close_return": {"scope": "elementwise", "pit_safe": True},
}
_EXPLICIT_POLICIES.update(_FINAL_PACK_POLICIES)

# R11 unusable-operators sweep: structural policies that must SURVIVE the
# active-surface filter.  These trailing-window / elementwise operators are
# surfaced into the daily surface DURING module import (after ``operator_policy``
# has already run its filter), so they have no entry in ``_ACTIVE_POLICY_CANONICALS``
# at filter time and would otherwise fail closed to ``pit_safe=False``, blocking
# the certifier bootstrap (review §2.4).  Applied unconditionally, like
# ``_FINAL_PACK_POLICIES`` above.
_R11_UNUSABLE_SWEEP_POLICIES = {
    "ts_abs_entropy_nats": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_abs_entropy_normalized": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_effective_turning_rate": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_argmax_age": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_argmin_age": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_argmax_index_from_oldest": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_argmin_index_from_oldest": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_coverage_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_ffill_limited": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_staleness": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_valid_count": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_regression_forecast_error": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_regression_forecast_error_z": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_regression_in_sample_resid": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_regression_resid_mean": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_mean_abs_deviation": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_median_abs_deviation": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ADX": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "MACD_line": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "MACD_signal": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "MACD_hist": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "true_range": {"scope": "ts", "pit_safe": True, "min_periods": 1},
}
_EXPLICIT_POLICIES.update(_R11_UNUSABLE_SWEEP_POLICIES)

# Phase 2: relation/index/event operators (2026-08-12)
_PHASE2_RELATION_POLICIES = {
    "trading_day_diff": {"scope": "elementwise", "pit_safe": True},
    "relation_distinct_count": {"scope": "cs", "pit_safe": True},
    "relation_overlap_ratio": {"scope": "cs", "pit_safe": True},
}
_EXPLICIT_POLICIES.update(_PHASE2_RELATION_POLICIES)

# Research quality operators (2026-08-13): experimental fundamental quality measures
_RESEARCH_QUALITY_POLICIES = {
    "accounting_comparability_score": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 12},
}
_EXPLICIT_POLICIES.update(_RESEARCH_QUALITY_POLICIES)

# Alpha-language expansion (2026-08): run/hysteresis state, path geometry,
# distribution shift, volatility structure, cs locality, events, report seq.
# All are causal trailing-window / sequential transforms -> pit_safe.
_ALPHA_LANGUAGE_POLICIES = {
    # Module 1 — temporal state + signed pattern
    "ts_run_strength": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_run_efficiency": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_run_concentration": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_hysteresis_state": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_hysteresis_age": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_state_integral": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_state_entry_strength": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_transition_intensity": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_sign_persistence": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_sign_cluster_index": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    # State event族 (Phase 1 A类原子算子) — 布尔条件序列的状态转换和事件模式
    "ts_transition_count": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_time_since_change": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_event_spacing_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_event_spacing_cv": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Direction concentration族 (Phase 1 A类原子算子) — 方向浓度和分布统计
    "ts_positive_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_negative_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_zero_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_abs_concentration": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_abs_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Module 2 — path / shape geometry
    "ts_monotonicity": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_turning_rate": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_turning_intensity": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_path_efficiency": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_roughness": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_trend_break_score": {"scope": "ts", "pit_safe": True, "min_periods": 4},
    "ts_weighted_time_centroid": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_endpoint_deviation": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_mass_concentration": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    # Module 3 — distribution shape / shift
    "ts_tail_imbalance": {"scope": "ts", "pit_safe": True, "min_periods": 4},
    "ts_expected_shortfall_asymmetry": {"scope": "ts", "pit_safe": True, "min_periods": 6},
    "ts_wasserstein_shift": {"scope": "ts", "pit_safe": True, "min_periods": 6},
    "ts_ks_shift": {"scope": "ts", "pit_safe": True, "min_periods": 6},
    "ts_location_shift": {"scope": "ts", "pit_safe": True, "min_periods": 6},
    "ts_scale_shift": {"scope": "ts", "pit_safe": True, "min_periods": 6},
    # Module 4 — volatility structure
    "ts_vol_of_vol": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_vol_acceleration": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_vol_term_structure": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_semivariance_balance": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_realized_quarticity": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_vol_clustering": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_leverage_effect": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_jump_bipower_proxy": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    # Module 5 — cross-sectional locality + group ex-self
    "cs_neighbor_gap": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    "cs_local_density": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    "cs_isolation": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    "cs_local_curvature": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    "group_ex_self_std": {"scope": "group", "pit_safe": True, "min_periods": 1},
    "group_ex_self_mad": {"scope": "group", "pit_safe": True, "min_periods": 1},
    "group_ex_self_quantile": {"scope": "group", "pit_safe": True, "min_periods": 1},
    # Phase 1: A-class atomic operators - group extension family
    "group_ex_self_mean": {"scope": "group", "pit_safe": True},
    "group_ex_self_weighted_mean": {"scope": "group", "pit_safe": True},
    "hierarchical_group_neutralize": {"scope": "group", "pit_safe": True},
    "group_multi_resid": {"scope": "group", "pit_safe": True},
    "cs_trimmed_ols_resid": {"scope": "cs", "pit_safe": True},
    "cs_huber_resid": {"scope": "cs", "pit_safe": True},
    "cs_lad_resid": {"scope": "cs", "pit_safe": True},
    "relation_weighted_std_ex_self": {"scope": "group", "pit_safe": True, "min_periods": 1},
    # Module 6 — events + fundamental report sequence
    "event_frequency": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "event_cluster_count": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "event_cluster_mean_size": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "report_rolling_mean": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 2},
    "report_yoy_lag": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
}
_EXPLICIT_POLICIES.update(_ALPHA_LANGUAGE_POLICIES)

# Stateful rule / episode / rotation pack (2026-08 CTA): forward per-column
# recursions (latch / hold / slew / deadband / CUSUM / episode / survival) and
# cross-sectional rotation statistics.  All causal and PIT-safe; ts_*/state_*
# get a time-series scope, cs_* get cross-sectional.
_STATEFUL_PACK_POLICIES = {
    # Module 1 — rule language
    "state_latch": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "state_hold": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "state_slew_limit": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "state_deadband": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Module 2 — events
    "event_refractory": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "cross_event": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # Module 3 — sequential / conditional memory
    "ts_cusum_pressure": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_rank_if": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "state_ewm_if": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_lag_of_peak_corr": {"scope": "ts", "pit_safe": True, "min_periods": 6},
    # Module 4 — dynamic episode / directional change
    "state_since_reduce": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "directional_change_state": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "directional_change_extent": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "state_since_trend_tstat": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Module 5 — state survival / maturity
    "ts_state_age_percentile": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_state_exit_hazard": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_state_residual_life": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    # Module 6 — cross-sectional rotation
    "cs_rank_churn": {"scope": "cs", "pit_safe": True},
    "cs_tail_retention": {"scope": "cs", "pit_safe": True},
    # Module 7 — drawdown path / recovery
    "ts_recovery_fraction": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_current_drawdown_area": {"scope": "ts", "pit_safe": True, "min_periods": 2},
}
_EXPLICIT_POLICIES.update(_STATEFUL_PACK_POLICIES)

# Turnover-survival / weighted-tail / behavioural / order-flow families
# (2026-08).  The turnover-survival operators build their reference cost only
# from rows ``[t-W, t-1]``, so they carry ``lag=1`` (strictly past).  The
# minute→daily order-flow operators are session-aware daily aggregators.
_CHIP_FLOW_PACK_POLICIES = {
    "ts_turnover_reference_price": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_turnover_cost_dispersion": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_turnover_profit_share": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_turnover_holding_age": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_turnover_near_cost_mass": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_turnover_cost_quantile_distance": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_stratified_mean_spread": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_weighted_semivariance": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_weighted_expected_shortfall": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_weighted_drawdown_area": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_cpt_value": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "intraday_bvc_imbalance": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_impact_beta": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_impact_asymmetry": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_return_wasserstein_shift": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 30,
    },
    "micro_bvc_vpin": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
}
_EXPLICIT_POLICIES.update(_CHIP_FLOW_PACK_POLICIES)

# 2026-08 advanced information-theoretic / intraday-distribution / topology /
# structure pack (concurrent expansion).  These are applied AFTER the
# active-surface filter so they survive load order (the pack's own surface
# registration may import after ``operator_policy``).
_ADVANCED_PACK_POLICIES = {
    "ts_transfer_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_effective_transfer_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_score_rank_weighted_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "report_benford_js_divergence": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "intraday_wasserstein_pair_distance": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_barrier_approach_acceleration": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_quantile_curve_pca_score": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_quantile_curve_pca_residual": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "ts_betti_1_max_persistence": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_persistence_diagram_shift": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_fisher_information_shift": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_bures_corr_shift": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_kramers_moyal_drift": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_kramers_moyal_diffusion": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "cs_sliced_wasserstein_copula_shift": {"scope": "cs", "pit_safe": True, "min_periods": 2},
    "group_spd_feature_structure_shift": {"scope": "group", "pit_safe": True, "min_periods": 2},
    "holder_class_js_shift": {"scope": "ts", "pit_safe": True, "min_periods": 2},
}
_EXPLICIT_POLICIES.update(_ADVANCED_PACK_POLICIES)

# 2026-08-08 market-state description language pack (§18): quantile-hit /
# extreme dependence / expectile / directional-change / feature geometry /
# conditional dependence / spread estimators / local non-linear cross-section /
# systemic tail / marked event / update clock.  Explicit scopes keep the
# registry bridge and the production surface truthful.
_MARKET_LANGUAGE_PACK_POLICIES = {
    # Quantile-hit / extreme dependence.
    "ts_quantilogram": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_cross_quantilogram": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_quantile_crossing_spectral_concentration": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    "ts_extremogram": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_cross_extremogram": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_extremal_dependence_decay": {"scope": "ts", "pit_safe": True, "min_periods": 4},
    # Expectile (magnitude-sensitive tail).
    "ts_expectile": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_expectile_beta": {"scope": "ts", "pit_safe": True, "min_periods": 4},
    # Directional-change intrinsic time.
    "ts_dc_overshoot_ratio": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_dc_event_rate": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_dc_duration_asymmetry": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_dc_overshoot_asymmetry": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Multi-field covariance geometry.
    "ts_feature_mode_share": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_feature_effective_rank": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_feature_subspace_rotation": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    "ts_beta_break_score": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    # Conditional / band dependence.
    "ts_conditional_transfer_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_modwt_band_corr": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    # Sampling-scale / noise diagnostics and profile surprise (intraday).
    "intraday_subsampled_rv_dispersion": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_volatility_signature_slope": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_realized_power_variation": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_profile_surprise_energy": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    "intraday_profile_phase_shift": {
        "scope": "session_intraday", "pit_safe": True,
        "session_aware": True, "reset_at_session_boundary": True, "min_periods": 20,
    },
    # No-L2 spread estimators.
    "ohlc_corwin_schultz_spread": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_roll_effective_spread": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Local non-linear cross-section.
    "cs_knn_local_linear_residual": {"scope": "cs", "pit_safe": True, "min_periods": 2},
    "cs_knn_tangent_residual": {"scope": "cs", "pit_safe": True, "min_periods": 2},
    "cs_knn_local_gradient_norm": {"scope": "cs", "pit_safe": True, "min_periods": 2},
    "cs_rank_copula_mi": {"scope": "cs", "pit_safe": True, "min_periods": 2},
    "cs_rank_copula_entropy": {"scope": "cs", "pit_safe": True, "min_periods": 2},
    # Systemic tail / group relation diffusion.
    "group_tail_centrality": {"scope": "group", "pit_safe": True, "min_periods": 5},
    "group_tail_lead_score": {"scope": "group", "pit_safe": True, "min_periods": 5},
    "relation_diffusion_score": {"scope": "group", "pit_safe": True, "min_periods": 2},
    # Marked event.
    "event_mark_autocorr": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "event_interval_mark_coupling": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Update clock (sparse non-price fields).
    "update_path_efficiency": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "update_acceleration": {"scope": "ts", "pit_safe": True, "min_periods": 4},
    "update_surprise": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "update_direction_persistence": {"scope": "ts", "pit_safe": True, "min_periods": 3},
}
_EXPLICIT_POLICIES.update(_MARKET_LANGUAGE_PACK_POLICIES)

# Fiscal/event pack：fiscal_event_ops 把 date_diff_days / cash_flow_lifecycle_stage
# 并入 extended 表面（cash_flow_lifecycle_stage 已进 DAILY_FACTOR_MIGRATED），但它们的
# 显式 scope policy 不在过滤后的 _EXPLICIT_POLICIES，导致 load_all 的
# finalize_layer_governance._enrich_catalog 对全体会话抛「无显式 scope policy」。
# 这是幂等的加法补丁（与 _FINAL_PACK_POLICIES / _ADVANCED_PACK_POLICIES 同模式）。
_FISCAL_EVENT_PACK_POLICIES = {
    "date_diff_days": {"scope": "fundamental_period", "pit_safe": True},
    "cash_flow_lifecycle_stage": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_perpetual_inventory": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_standardized_surprise": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_sign_consistency": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_change_direction_agreement": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_sign_agreement": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_autocorr": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_ar_resid_std": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_asymmetric_elasticity": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_reversal_ratio": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_regression_resid_std": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_accrual_quality": {"scope": "fundamental_period", "pit_safe": True},
    "fin_seasonal_zscore": {"scope": "fundamental_period", "pit_safe": True},
    "fin_seasonal_percentile": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_true_streak": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_direction_consistency": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_pair_direction_agreement": {"scope": "fundamental_period", "pit_safe": True},
    "row_sum_skipna": {"scope": "fundamental_period", "pit_safe": True},
    "industry_fiscal_resid": {"scope": "fundamental_period", "pit_safe": True},
    "fiscal_asymmetric_timeliness": {"scope": "fundamental_period", "pit_safe": True},
}
_EXPLICIT_POLICIES.update(_FISCAL_EVENT_PACK_POLICIES)

# 2026-08 V2/V3 state-dynamics / event-response / spectral-crowding / minute
# volume-clock / dynamic-KNN / extreme-tail / report-timing families.  Applied
# after the active-surface filter so the policies survive load order (the pack's
# own surface registration may import after ``operator_policy``).
_DYNAMICS_PACK_POLICIES = {
    # State geometry + ordinal time asymmetry.
    "ts_ordinal_irreversibility": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_state_density": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_multiscale_permutation_entropy_slope": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Local Markov dynamics (strictly-past transition matrix; current value only
    # selects the state).
    "ts_markov_persistence": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_markov_state_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_markov_transition_surprisal": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_markov_entropy_production": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_active_information_storage": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_kramers_moyal_local_stability": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # First-passage (last anchor's path may read x_t).
    "ts_first_passage_bias": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Historical event-response learning.
    "event_historical_response_mean": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "event_historical_response_sign_balance": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "event_hawkes_branching_ratio_proxy": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    # Multivariate distribution break (shift windows end at t-1).
    "ts_joint_energy_shift": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 10},
    "ts_energy_break_score": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 10},
    "ts_copula_central_asymmetry": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Cross-sectional spectral crowding (per-date group spectrum).  P1-G
    # rename: the SVD is over the members' feature matrix (not a correlation
    # matrix), so the canonical names are group_feature_*.
    "group_feature_mode_share": {"scope": "group", "pit_safe": True},
    "group_feature_effective_rank": {"scope": "group", "pit_safe": True},
    "group_feature_mode_localization": {"scope": "group", "pit_safe": True},
    "group_feature_spectral_gap": {"scope": "group", "pit_safe": True},
    "group_feature_second_mode_localization": {"scope": "group", "pit_safe": True},
    # group_corr_* are aliases of the above (kept for compatibility).
    "group_corr_mode_share": {"scope": "group", "pit_safe": True},
    "group_corr_effective_rank": {"scope": "group", "pit_safe": True},
    "group_corr_mode_localization": {"scope": "group", "pit_safe": True},
    "group_corr_spectral_gap": {"scope": "group", "pit_safe": True},
    "group_corr_second_mode_localization": {"scope": "group", "pit_safe": True},
    # Session recovery + volume-clock path geometry (minute → daily, EOD).
    "session_event_recovery_score": {
        "scope": "session_intraday", "pit_safe": True, "session_aware": True,
        "reset_at_session_boundary": True, "min_periods": 2,
    },
    "intraday_volume_clock_path_efficiency": {
        "scope": "session_intraday", "pit_safe": True, "session_aware": True,
        "reset_at_session_boundary": True, "min_periods": 2,
    },
    "intraday_volume_clock_roughness": {
        "scope": "session_intraday", "pit_safe": True, "session_aware": True,
        "reset_at_session_boundary": True, "min_periods": 2,
    },
    # Dynamic cross-sectional KNN peers.
    "cs_knn_peer_mean_ex_self": {"scope": "cs", "pit_safe": True},
    "cs_knn_neighbor_retention": {"scope": "cs", "pit_safe": True},
    # Extreme tail + exact quantile regression + local divergence.
    "ts_hill_tail_index": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    "ts_quantile_regression_beta": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_local_lyapunov_exponent": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Report disclosure timing / revision magnitude.
    "report_filing_delay_surprise": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "report_revision_magnitude": {"scope": "ts", "pit_safe": True, "min_periods": 5},
}
_EXPLICIT_POLICIES.update(_DYNAMICS_PACK_POLICIES)

# 2026-08 vertical deepening pack — all strictly trailing / report-causal.
_DEEPENING_PACK_POLICIES = {
    # Markov state deep-dive (strictly-past transition matrix).
    "ts_markov_committor": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_markov_mean_first_passage_time": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_markov_spectral_gap": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_markov_stationary_surprisal": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Kramers-Moyal deep-dive.
    "ts_km_equilibrium_distance": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_km_diffusion_gradient": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_km_quasipotential_depth": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # First-passage decomposition (anchors s+H <= t).
    "ts_first_passage_hit_probability": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_first_passage_conditional_time": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Event-response curve shape (completed events only).
    "event_response_peak_lag": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "event_response_decay_rate": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "event_response_dispersion": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "event_response_reversal_strength": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Extreme-value tail shape.
    "ts_extremal_index": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_mean_excess_slope": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_gpd_shape_pwm": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Transfer-entropy peak (fused, lagged).
    "ts_transfer_entropy_peak_strength": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_transfer_entropy_peak_lag": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    # Distribution transport / MMD (two-window trailing).
    "ts_quantile_transport_slope": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_quantile_transport_curvature": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_mmd_rbf_shift": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # Chord geometry.
    "ts_chord_excursion_area": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_max_chord_excursion": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    # Chip-cost shape (turnover survival uses t-1 and before).
    "ts_turnover_cost_entropy": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_turnover_cost_mode_distance": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_turnover_cost_skew": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    "ts_turnover_age_dispersion": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 5},
    # Group spectral deep-dive.
    "group_corr_spectral_gap": {"scope": "group", "pit_safe": True},
    "group_corr_second_mode_localization": {"scope": "group", "pit_safe": True},
    # Style-graph smoothness.
    "cs_knn_graph_dirichlet_energy": {"scope": "cs", "pit_safe": True},
    # Intraday volatility signature (EOD).
    "intraday_rv_signature_slope": {
        "scope": "session_intraday", "pit_safe": True, "session_aware": True,
        "reset_at_session_boundary": True, "min_periods": 2,
    },
    # Report change breadth / coherence (period-aware).
    "report_change_breadth": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 5},
    "report_change_coherence": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 5},
    # Recurrence quantification analysis.
    "ts_recurrence_rate": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_recurrence_diagonal_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_recurrence_trapping_time": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_recurrence_divergence": {"scope": "ts", "pit_safe": True, "min_periods": 5},
}
_EXPLICIT_POLICIES.update(_DEEPENING_PACK_POLICIES)

# 2026-08-08 Gemini-recommended primitives: gathering/distribution, weighted
# moment / conditional covariance / robust multi-resid, activity clock, spectral
# shape, relation PageRank, intraday activity-duration curvature and
# research-surface transforms.  See gather_ext / weighted_moment_ext /
# activity_clock / spectral_ext / relation.ops_ext / intraday_activity_duration /
# research_transform.
_GEMINI_PACK_POLICIES: dict[str, dict[str, Any]] = {
    # group / cross-sectional routing (scope: group / cs).
    "group_topk_mean": {"scope": "group", "pit_safe": True, "min_periods": 1},
    "cs_weighted_percentile_rank": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    "group_distribution_js_divergence": {"scope": "group", "pit_safe": True, "min_periods": 1},
    "cs_multi_robust_resid": {"scope": "cs", "pit_safe": True, "min_periods": 3},
    # trailing window gather / weighted moment / conditional covariance.
    "ts_value_at_argextreme": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_weighted_standardized_moment": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_cov_if": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    # activity clock (strict past scale, lag-aware).
    "ts_activity_clock_lagged_value": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 2},
    "ts_activity_clock_age": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 2},
    "ts_max_drawdown_activity_cost": {"scope": "ts", "pit_safe": True, "lag": 1, "min_periods": 2},
    # spectral shape (trailing FFT).
    "ts_spectral_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 16},
    "ts_dominant_cycle_period": {"scope": "ts", "pit_safe": True, "min_periods": 16},
    # relation graph (group adjacency).
    "relation_pagerank_centrality": {"scope": "group", "pit_safe": True, "min_periods": 2},
    # P0-008: honest rename — group complete-graph signal share (research-only).
    "group_signal_attraction_share": {"scope": "group", "pit_safe": True, "min_periods": 2},
    # intraday activity-duration curvature (minute source → daily scalar).
    "intraday_activity_duration_curvature": {
        "scope": "ts", "pit_safe": True, "min_periods": 3, "session_aware": True,
    },
    # research-surface transforms.
    "ts_wavelet_lowpass_reconstruct": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    "ts_signature_mahalanobis_anomaly": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_persistence_birth_dispersion": {"scope": "ts", "pit_safe": True, "min_periods": 8},
}
_EXPLICIT_POLICIES.update(_GEMINI_PACK_POLICIES)

# 2026-08-08 Gemini V2 round policies (see operator_surface._DAILY_GEMINI_V2_PACK).
# HVG / RQA / GLR / spreads / robust-scale / cross-spectral / multifractal are
# trailing-window ts; composition is elementwise across panels; the dip is a
# cross-sectional global state; the barycenter distance is group-scoped; the
# intraday impact decay is a session-aware minute op.
_GEMINI_V2_PACK_POLICIES: dict[str, dict[str, Any]] = {
    # HVG network.
    "ts_hvg_degree_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    "ts_hvg_forward_backward_asymmetry": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    "ts_hvg_clustering_coefficient": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    "ts_hvg_assortativity": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    "ts_hvg_motif_entropy": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    # RQA line structure.
    "ts_recurrence_determinism": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    "ts_recurrence_laminarity": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    "ts_recurrence_mean_diagonal_length": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    "ts_recurrence_longest_vertical_length": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    # change-point scores.
    "ts_glr_mean_shift_score": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    "ts_glr_variance_shift_score": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    "ts_pettitt_change_score": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    # OHLC microstructure spreads.
    "ts_edge_effective_spread": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_abdi_ranaldo_spread": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_pastor_stambaugh_liquidity_gamma": {"scope": "ts", "pit_safe": True, "min_periods": 20},
    # robust scale / location.
    "ts_qn_scale": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    "ts_hodges_lehmann_location": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    # extreme-value.
    "ts_pickands_tail_index": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    "ts_evt_threshold_stability": {"scope": "ts", "pit_safe": True, "min_periods": 10},
    "event_allan_factor": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    # compositional data (elementwise across aligned panels).
    "composition_clr_component": {"scope": "elementwise", "pit_safe": True, "min_periods": 1},
    "composition_entropy": {"scope": "elementwise", "pit_safe": True, "min_periods": 1},
    "composition_aitchison_distance": {"scope": "elementwise", "pit_safe": True, "min_periods": 1},
    "composition_ilr_balance": {"scope": "elementwise", "pit_safe": True, "min_periods": 1},
    "composition_js_divergence": {"scope": "elementwise", "pit_safe": True, "min_periods": 1},
    # cross-sectional global state / transport.
    "cs_hartigan_dip": {"scope": "cs", "pit_safe": True, "min_periods": 100},
    "group_wasserstein_barycenter_distance": {"scope": "group", "pit_safe": True, "min_periods": 4},
    # cross-spectral.
    "ts_cross_spectral_coherence": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    "ts_cross_spectral_phase": {"scope": "ts", "pit_safe": True, "min_periods": 8},
    # intraday minute-source.
    "intraday_impact_decay_rate": {
        "scope": "ts", "pit_safe": True, "min_periods": 3, "session_aware": True,
    },
    # multifractal asymmetry.
    "ts_multifractal_asymmetry": {"scope": "ts", "pit_safe": True, "min_periods": 20},
    # research-surface (DMD / bicoherence / kernel / BDS / SR).
    "ts_dmd_dominant_growth_rate": {"scope": "ts", "pit_safe": True, "min_periods": 12},
    "ts_dmd_dominant_frequency": {"scope": "ts", "pit_safe": True, "min_periods": 12},
    "ts_dmd_mode_concentration": {"scope": "ts", "pit_safe": True, "min_periods": 12},
    "ts_bicoherence_top_decile_mean": {"scope": "ts", "pit_safe": True, "min_periods": 16},
    # R9-OP-028: deprecated alias kept for policy lookups on legacy recipes.
    "ts_bicoherence_max": {"scope": "ts", "pit_safe": True, "min_periods": 16},
    "ts_kernel_granger_score": {"scope": "ts", "pit_safe": True, "min_periods": 30},
    "ts_residualized_hsic": {"scope": "ts", "pit_safe": True, "min_periods": 24},
    "ts_bds_statistic": {"scope": "ts", "pit_safe": True, "min_periods": 30},
    "ts_rolling_sr_gaussian_mean_shift_score": {"scope": "ts", "pit_safe": True, "min_periods": 12},
}
_EXPLICIT_POLICIES.update(_GEMINI_V2_PACK_POLICIES)

# Compatibility export now reflects the reviewed research surface exactly.
RESEARCH_CORE_CANONICALS = RESEARCH_ONLY_CANONICALS


@dataclass
class OperatorPolicy:
    """算子执行语义（企业级 schema 子集）。

    描述 lookback、PIT 安全、NaN 策略、shape 契约等机器可读属性，
    供 lineage hash、数据加载缓冲与 production 审计使用。
    """

    scope: Scope = "unknown"
    lookback_window: int | None = None
    min_periods: int | None = None
    lag: int = 0
    pit_safe: bool = False
    domain_policy: str = "all_numeric"
    overflow_policy: str = "ieee_propagate"
    null_policy: str = "propagate"
    nan_policy: NanPolicy = "propagate"
    includes_current_bar: bool = True
    calendar: str = "bar"  # 本引擎 bar = 每标的连续行，非自然日
    session_aware: bool = False
    reset_at_session_boundary: bool = False
    shape_preserving: bool = True
    index_preserving: bool = True
    columns_preserving: bool = True
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """将 ``OperatorPolicy`` 序列化为普通字典。

        返回:
            策略字段的字典表示。
        """
        return asdict(self)


def infer_operator_policy(op: Any, *, canonical: str | None = None) -> OperatorPolicy:
    """从算子实例 metadata/category/tags 推断默认 policy。

参数:
    op: 算子实例。
    canonical: 可选 canonical 名覆盖。

返回:
    推断或查表得到的 ``OperatorPolicy`` 对象。
"""
    canon = canonical or (op if isinstance(op, str) else getattr(getattr(op, "metadata", None), "name", "")) or ""
    try:
        from cleaned_operators.registry import OperatorRegistry

        canon = OperatorRegistry._aliases.get(canon, canon)
    except Exception:
        pass
    if canon in _EXPLICIT_POLICIES:
        base = {"scope": "unknown", "pit_safe": False}
        # R11 P0-13: the authoritative grain-changing signal is the operator's
        # OWN contract (declared input_grain/output_grain).  The hand lists are
        # only fallbacks for legacy ops that never declared their grains — an
        # explicit-policy op that declares minute -> daily must be classified
        # non-shape-preserving even though it is not in the hand set.
        _emeta = getattr(op, "metadata", None)
        _eig = (getattr(_emeta, "input_grain", None) or "").lower()
        _eog = (getattr(_emeta, "output_grain", None) or "").lower()
        _declared = bool(_eig and _eog and _eig != _eog)
        if (
            canon in NON_SHAPE_PRESERVING_CANONICALS
            or canon in GRAIN_CHANGING_CANONICALS
            or _declared
        ):
            base.update(
                shape_preserving=False,
                index_preserving=False,
                columns_preserving=False,
            )
        policy = OperatorPolicy(**{**base, **_EXPLICIT_POLICIES[canon]})
        # An intentionally-experimental / isolated canonical (e.g. the
        # in-sample diagnostic family: ts_ar_forecast, ts_ar_innovation, the
        # *_resid variants) is reviewed *not to be admitted*: its explicit
        # structural policy may say pit_safe=True, but the fail-closed
        # classification wins so ``infer_operator_policy`` agrees with the
        # experimental lifecycle and the manifest/catalog surfaces stay
        # consistent (audit P0-A03 / convergence test).
        try:
            from cleaned_operators.semantic_certification import should_fail_closed

            if should_fail_closed(canon):
                policy.pit_safe = False
        except Exception:  # pragma: no cover - module not importable
            pass
        return policy

    meta = getattr(op, "metadata", None)
    category = (getattr(meta, "category", "") or "").lower()
    tags = [str(t).lower() for t in (getattr(meta, "tags", None) or [])]
    name = (getattr(meta, "name", "") or canon).lower()

    scope: Scope = "unknown"
    if category in ("cross_sectional",) or "cross_section" in tags:
        scope = "cs"
    elif category in ("group_neutralization",):
        scope = "cs"
    elif category in (
        "time_series",
        "shift_diff_cum",
        "technical_signal",
        "price_volume",
        "intraday_microstructure",
        "signal",
        "time_series_regression",
        "panel_model",
        "complexity",
        "sequence_anomaly",
        "path_signature",
        "wavelet_spectral",
        "state_space",
    ):
        scope = "ts"
    elif category in ("statistics",):
        if any(k in name for k in ("test", "corr_test", "granger", "ttest", "adf", "kpss")):
            scope = "hypothesis"
        else:
            scope = "aggregate"
    elif category in ("math", "elementwise_math", "data_handling"):
        scope = "elementwise"

    # 2026-08 expansion families (ts_model / panel / cross-section / fin /
    # holder / index / valuation) ship research surface entries without explicit
    # policies; infer scope from name prefix so governance does not fail closed.
    if scope == "unknown":
        if name.startswith(("ts_", "panel_", "event_", "index_", "fin_", "fundamental_", "holder_")):
            scope = "ts"
        elif name.startswith(("cs_", "cross_", "relation_", "peer_")):
            scope = "cs"
        elif name.startswith(("group_",)):
            scope = "group"
        elif "entropy" in name or "complexity" in name or "hurst" in name:
            scope = "ts"

    # Active daily/extended primitives are fail-closed: only an explicit policy
    # above may grant PIT safety.  Heuristics remain useful for migration reports
    # and third-party research operators but never for production admission.
    governed_without_policy = canon in (DAILY_CANONICALS | EXTENDED_ONLY_CANONICALS)
    pit_safe = False if governed_without_policy else (
        scope in {"elementwise", "cs", "group"}
        or "pit_safe" in tags
        or "causal" in tags
    )
    if canon in PIT_UNSAFE_CANONICALS:
        pit_safe = False

    if scope == "ts" and any(k in name for k in ("mean", "std", "sum", "corr", "rank", "decay")):
        lookback: int | None = None  # 运行时由 window 参数决定
        min_periods = 1
    elif scope == "cs":
        lookback = 0
        min_periods = 1
    elif scope == "aggregate":
        lookback = None  # expanding
        min_periods = 1
    else:
        lookback = None
        min_periods = None

    # WS4 P0-07 / R11 P0-13: grain-changing (minute -> daily) operators are not
    # shape / index preserving even though they are PIT-safe.  The AUTHORITATIVE
    # signal is the operator's own contract — a declared ``input_grain`` /
    # ``output_grain`` pair (e.g. minute -> daily).  The hand-maintained
    # ``GRAIN_CHANGING_CANONICALS`` set and the ``grain_minute_to_daily`` tag are
    # only fallbacks for legacy ops that never declared their grains.  A new
    # minute->daily operator that declares its grains in its own metadata is
    # classified correctly WITHOUT being added to any hand list.
    _ig = (getattr(meta, "input_grain", None) or "").lower()
    _og = (getattr(meta, "output_grain", None) or "").lower()
    _declared_grain_changing = bool(_ig and _og and _ig != _og)
    _grain_changing = (
        _declared_grain_changing
        or canon in GRAIN_CHANGING_CANONICALS
        or "grain_minute_to_daily" in tags
    )
    shape_preserving = canon not in NON_SHAPE_PRESERVING_CANONICALS and not _grain_changing
    index_preserving = shape_preserving
    columns_preserving = shape_preserving

    return OperatorPolicy(
        scope=scope,
        lookback_window=lookback,
        min_periods=min_periods,
        pit_safe=pit_safe,
        shape_preserving=shape_preserving,
        index_preserving=index_preserving,
        columns_preserving=columns_preserving,
        tags=list(getattr(meta, "tags", None) or []),
    )


def compute_operator_catalog_hash(*, backend: str = "pandas_numpy") -> str:
    """对所有已实现算子的 canonical + policy 做确定性 hash。

参数:
    backend: 目标 backend，默认 ``pandas_numpy``。

返回:
    SHA-256 十六进制摘要，用于 lineage 追踪。
"""
    from cleaned_operators.registry import OperatorRegistry

    entries: list[dict[str, Any]] = []
    for canon in sorted(OperatorRegistry._operators.keys()):
        op = OperatorRegistry.get(canon, backend=backend)
        if op is None:
            continue
        policy = infer_operator_policy(op, canonical=canon)
        entries.append(
            {
                "canonical": canon,
                "category": getattr(op.metadata, "category", ""),
                "policy": policy.to_dict(),
            }
        )
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_lookback(
    analysis_lookback: int,
    *,
    factor_freq: str | None = None,
    source_bar_freq: str | None = None,
    lag_buffer: int = 1,
    extra: int = 5,
) -> int:
    """将 IR 分析 lookback 转为数据加载历史缓冲 bar 数。

参数:
    analysis_lookback: 因子分析窗口 bar 数。
    factor_freq: 因子频率（如 ``1d``）。
    source_bar_freq: 数据源 bar 频率（如 ``5m``）。
    lag_buffer: 滞后缓冲 bar 数。
    extra: 额外安全缓冲 bar 数。

返回:
    数据加载所需的历史 bar 数。
"""
    base = max(0, int(analysis_lookback))
    bars = base + max(0, int(lag_buffer)) + max(0, int(extra))
    if factor_freq and source_bar_freq and factor_freq != source_bar_freq:
        f_bpd = bars_per_day(factor_freq)
        s_bpd = bars_per_day(source_bar_freq)
        if f_bpd > 0 and s_bpd > 0 and f_bpd != s_bpd:
            # lookback 按因子频率解释，换算为数据源 bar 数（如 20 日 × 78 个 5m bar）
            bars = np.where(f_bpd))) != 0, int(max(bars, round(bars * s_bpd / f_bpd))), np.nan)
    return bars


def bars_to_calendar_trading_days(bars: int, bar_freq: str | None) -> int:
    """将 bar 数换算为交易日数。

参数:
    bars: bar 数量。
    bar_freq: bar 频率字符串。

返回:
    对应的交易日数（至少 1）。
"""
    count = max(0, int(bars))
    if count <= 0:
        return 0
    s_bpd = bars_per_day(bar_freq)
    if s_bpd > 1:
        return max(1, (count + s_bpd - 1) // s_bpd)
    return count


def infer_source_bar_freq(data_source: Any, *, fallback: str | None = "1d") -> str:
    """从 DataSource 对象推断 bar 频率。

参数:
    data_source: 数据源对象（可嵌套 ``inner``/_``inner``）。
    fallback: 无频率信息时的回退值，默认 ``1d``。

返回:
    bar 频率字符串。
"""
    for attr in ("bar_freq", "freq"):
        value = getattr(data_source, attr, None)
        if value:
            return str(value)
    inner = getattr(data_source, "inner", None) or getattr(data_source, "_inner", None)
    if inner is not None:
        return infer_source_bar_freq(inner, fallback=fallback)
    return str(fallback or "1d")


# 常见 bar 频率 → 每交易日 bar 数（按市场 regular session）
# US: 6.5h RTH；CN: 4h（240min）；HK: 5.5h（330min）
_MARKET_BARS_PER_DAY: dict[str, dict[str, dict[str, int]]] = {
    "US": {
        "regular": {
            "1d": 1, "1D": 1, "d": 1,
            "1m": 390, "5m": 78, "15m": 26, "30m": 13, "1h": 7, "60m": 7,
        },
    },
    "CN": {
        "regular": {
            "1d": 1, "1D": 1, "d": 1,
            "1m": 240, "5m": 48, "15m": 16, "30m": 8, "1h": 4, "60m": 4,
        },
    },
    "HK": {
        "regular": {
            "1d": 1, "1D": 1, "d": 1,
            "1m": 330, "5m": 66, "15m": 22, "30m": 11, "1h": 6, "60m": 6,
        },
    },
}

# 向后兼容：默认美股 regular session
_BARS_PER_DAY: dict[str, int] = dict(_MARKET_BARS_PER_DAY["US"]["regular"])


def bars_per_day(
    freq: str | None,
    *,
    market: str = "US",
    session: str = "regular",
) -> int:
    """解析频率字符串为每交易日 bar 数。

参数:
    freq: 频率字符串（如 ``5m``、``1d``）。
    market: 市场代码，``US``/``CN``/``HK``。
    session: 交易时段，目前仅 ``regular``。

返回:
    每交易日 bar 数；未知频率默认 1（日频）。
"""
    if not freq:
        return 1
    text = str(freq).strip()
    mkt = str(market or "US").strip().upper()
    sess = str(session or "regular").strip().lower()
    table = _MARKET_BARS_PER_DAY.get(mkt, _MARKET_BARS_PER_DAY["US"]).get(
        sess, _MARKET_BARS_PER_DAY["US"]["regular"]
    )
    if text in table:
        return table[text]
    lowered = text.lower()
    if lowered in table:
        return table[lowered]
    # R32-P0-004: "1min"/"5min"/"15min" 长写法归一为 "1m"/"5m"/"15m" —— 否则
    # ``bars_per_day("1min")`` 抛错，日内 session clock 无法解析分钟频率。
    if lowered.endswith("min") and lowered[:-3].isdigit():
        lowered = lowered[:-3] + "m"
        if lowered in table:
            return table[lowered]
    if lowered.endswith("d"):
        try:
            days = int(lowered[:-1] or "1")
        except ValueError as exc:
            raise ValueError(f"unsupported bar frequency {freq!r}") from exc
        if days > 0:
            return 1
    raise ValueError(
        f"unsupported bar frequency {freq!r} for market={mkt!r} session={sess!r}"
    )


def normalize_bars_market(market: str | None) -> str:
    """将 universe/市场码规范化为 ``bars_per_day`` 市场键。

参数:
    market: 市场标识字符串。

返回:
    ``US``、``CN`` 或 ``HK``。
"""
    if not market:
        return "US"
    key = str(market).strip().lower()
    if key in {"ashare", "a_share", "cn", "china", "sse", "szse"}:
        return "CN"
    if key in {"hk", "hongkong", "hkg"}:
        return "HK"
    if key in {"us", "usa", "nyse", "nasdaq"}:
        return "US"
    upper = str(market).strip().upper()
    if upper in _MARKET_BARS_PER_DAY:
        return upper
    return "US"


def bar_freq_to_timedelta(freq: str | None) -> pd.Timedelta:
    """将 bar 频率字符串转为 ``pd.Timedelta``。

参数:
    freq: 频率字符串。

返回:
    对应的 ``pd.Timedelta``；未知时默认 1 日。
"""
    if not freq:
        return pd.Timedelta(days=1)
    text = str(freq).strip().lower()
    mapping = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "60m": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "d": pd.Timedelta(days=1),
    }
    if text in mapping:
        return mapping[text]
    if text.endswith("m") and text[:-1].isdigit():
        return pd.Timedelta(minutes=int(text[:-1]))
    # R32-P0-004: "1min"/"5min"/"15min" 等长写法缺失会让 SessionCalendar 的
    # bar_timedelta 退化成长达 1 天的错误量（分钟 lookback 全错）。统一归一。
    if text.endswith("min") and text[:-3].isdigit():
        return pd.Timedelta(minutes=int(text[:-3]))
    if text.endswith("h") and text[:-1].isdigit():
        return pd.Timedelta(hours=int(text[:-1]))
    return pd.Timedelta(days=1)
