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

Scope = Literal["ts", "cs", "elementwise", "aggregate", "hypothesis", "unknown"]
NanPolicy = Literal["propagate", "ignore", "zero", "ffill_only"]

# research 常用：须有显式 policy，但不进 production
RESEARCH_CORE_CANONICALS: frozenset[str] = frozenset(
    {
        # 微观结构
        "micro_realized_vol",
        "micro_spread",
        "micro_amihud_hf",
        "micro_mid_return",
        "micro_bipower_var",
        "micro_jump_indicator",
        "micro_trade_imbalance",
        "micro_vpin",
        "micro_kyle_lambda",
        # 价量 / 统计 research
        "real_turnover_rate",
        "rank_corr",
        "hump_decay",
        "corr_test",
        "vp_weighted_price",
        "ts_topk_sum",
        "ts_poly2_coeff",
        "ts_poly2_resid",
        # 基本面 period（research alias，待 period-aware 拆分）
        "ttm",
        "quarter",
        "yoy",
        "avg2",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
        # CAPM 扩展（production 仅 rolling_beta）
        "downside_beta",
        "tail_beta",
        "idio_vol",
        "idio_skew",
        "residual_momentum_capm",
        "coskewness_to_market",
        "rolling_beta_to_market",
        # 清洗 research
        "causal_linear_extrapolate",
    }
)

# 须有显式 policy 的基础设施算子（非 production_core / research_core）
_POLICY_EXTENSION_CANONICALS: frozenset[str] = frozenset(
    {
        "WMA",
        "ts_ema",
        "ewm_corr",
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
    }
)


# 向后兼容：Tier-1 = policy_required（延迟求值，避免与 operator_spec 循环 import）
def policy_required_canonicals() -> frozenset[str]:
    """获取必须有显式 OperatorPolicy 的 canonical 并集。

返回:
    production core、research core 与 policy 扩展算子的 ``frozenset``。
"""
    from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS

    return (
        PRODUCTION_CORE_CANONICALS
        | RESEARCH_CORE_CANONICALS
        | _POLICY_EXTENSION_CANONICALS
    )


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
POLARS_PARITY_VERIFIED_TIER8: frozenset[str] = frozenset({
    "c_mean",
    "c_std",
    "c_sum",
    "c_count",
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
    "c_percentile",
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

POLARS_PARITY_VERIFIED: frozenset[str] = (
    POLARS_PARITY_VERIFIED_TIER1
    | POLARS_PARITY_VERIFIED_TIER2
    | POLARS_PARITY_VERIFIED_TIER3
    | POLARS_PARITY_VERIFIED_TIER4
    | POLARS_PARITY_VERIFIED_TIER5
    | POLARS_PARITY_VERIFIED_TIER6
    | POLARS_PARITY_VERIFIED_TIER7
    | POLARS_PARITY_VERIFIED_TIER8
    | POLARS_PARITY_VERIFIED_TIER9
    | POLARS_PARITY_VERIFIED_TIER10
)

POLARS_PRODUCTION_SAFE: frozenset[str] = POLARS_PRODUCTION_SAFE_CORE | POLARS_PARITY_VERIFIED


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
    from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS

    errors: list[str] = []
    if POLARS_PRODUCTION_SAFE != POLARS_PRODUCTION_SAFE_CORE | POLARS_PARITY_VERIFIED:
        errors.append("POLARS_PRODUCTION_SAFE 须等于 CORE | PARITY_VERIFIED")
    if not POLARS_PARITY_VERIFIED <= POLARS_PRODUCTION_SAFE:
        errors.append("POLARS_PARITY_VERIFIED 必须是 POLARS_PRODUCTION_SAFE 子集")
    missing_core = sorted(PRODUCTION_CORE_CANONICALS - POLARS_PRODUCTION_SAFE)
    if missing_core:
        errors.append(
            f"PRODUCTION_CORE 未全部进入 POLARS_PRODUCTION_SAFE: {missing_core}"
        )
    unverified = sorted(
        POLARS_PRODUCTION_SAFE - POLARS_PRODUCTION_SAFE_CORE - POLARS_PARITY_VERIFIED
    )
    if unverified:
        errors.append(
            f"POLARS_PRODUCTION_SAFE 含未在 CORE|PARITY 中的算子: {unverified}"
        )
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

# 不做 Polars 移植且通常非 PIT 安全（FFT/矩阵/随机/CDF-PDF 等）
INTENTIONALLY_PANDAS_ONLY: frozenset[str] = frozenset(
    {
        "constant",
        "shuffle",
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
        "rand_exp",
        "rand_lognormal",
        "rand_normal",
        "rand_poisson",
        "rand_uniform",
        "cdf_chi2",
        "cdf_f",
        "cdf_normal",
        "cdf_t",
        "pdf_chi2",
        "pdf_f",
        "pdf_normal",
        "pdf_t",
        # experimental / 待 Polars 移植
        "ts_log_return",
        "open_gap",
        "close_gap",
        "cs_mad",
        "cs_mad_zscore",
        "cs_pct_rank",
        "cs_quantile",
        "cs_rank_01",
        "rank_pct",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "operating_margin",
        "intraday_vwap_deviation",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
        "div_or_default",
        "log_fill_invalid",
        "div_or_null",
    }
)

# 暂无 Polars、但 metadata 仍 PIT 安全（不应强制 pit_safe=False）
PANDAS_ONLY_PIT_SAFE: frozenset[str] = frozenset(
    {
        "ts_log_return",
        "open_gap",
        "close_gap",
        "cs_mad",
        "cs_mad_zscore",
        "cs_pct_rank",
        "cs_quantile",
        "cs_rank_01",
        "rank_pct",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "operating_margin",
        "intraday_vwap_deviation",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
    }
)

PIT_UNSAFE_CANONICALS: frozenset[str] = INTENTIONALLY_PANDAS_ONLY | frozenset(
    {"Lead", "next", "bfill", "causal_bfill", "fillna_interpolate", "shuffle"}
)

# 破坏 panel shape 的算子（即使 PIT-safe 也不进 production）
NON_SHAPE_PRESERVING_CANONICALS: frozenset[str] = frozenset({"dropna"})

# 显式声明的核心算子策略（Tier-1 + 特殊语义；其余由 infer_operator_policy 推断）
_EXPLICIT_POLICIES: dict[str, dict[str, Any]] = {
    "ts_delay": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_delta": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_pct": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_log_return": {"scope": "ts", "lag": 1, "pit_safe": True, "min_periods": 2},
    "ts_sharpe": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_autocorr": {"scope": "ts", "pit_safe": True, "min_periods": 3, "lag": 1},
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
    "ts_last_if": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_days_since": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_true_streak": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "period_lag": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_regression_tstat": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_trend_tstat": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_max_drawdown": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_partial_corr": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_nth_value": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "rolling_beta": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "rolling_beta_to_market": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_poly2_coeff": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_poly2_resid": {"scope": "ts", "pit_safe": True, "min_periods": 3},
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
    "Lead": {"scope": "ts", "lag": -1, "pit_safe": False},
    "next": {"scope": "ts", "lag": -1, "pit_safe": False},
    "bfill": {"scope": "elementwise", "pit_safe": False, "nan_policy": "ffill_only"},
    "causal_bfill": {"scope": "elementwise", "pit_safe": False, "nan_policy": "ffill_only"},
    "ffill": {"scope": "elementwise", "pit_safe": True, "nan_policy": "ffill_only"},
    "fillna_const": {"scope": "elementwise", "pit_safe": True},
    "fillna_interpolate": {"scope": "elementwise", "pit_safe": False},
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
    "c_percentile": {"scope": "cs", "pit_safe": True},
    "c_mean": {"scope": "cs", "pit_safe": True},
    "c_std": {"scope": "cs", "pit_safe": True},
    "c_sum": {"scope": "cs", "pit_safe": True},
    "c_count": {"scope": "cs", "pit_safe": True},
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
    "real_turnover_rate": {"scope": "ts", "pit_safe": True},
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
    "shuffle": {"scope": "unknown", "pit_safe": False},
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
    "quarter_from_cumulative": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ttm": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ttm_from_quarterly": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ttm_from_cumulative": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "yoy": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "yoy_by_period": {"scope": "ts", "pit_safe": True, "min_periods": 5},
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
}


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
    pit_safe: bool = True
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
    canon = canonical or getattr(getattr(op, "metadata", None), "name", "") or ""
    try:
        from cleaned_operators.registry import OperatorRegistry

        canon = OperatorRegistry._aliases.get(canon, canon)
    except Exception:
        pass
    if canon in _EXPLICIT_POLICIES:
        base = {"scope": "unknown", "pit_safe": True}
        if canon in NON_SHAPE_PRESERVING_CANONICALS:
            base.update(
                shape_preserving=False,
                index_preserving=False,
                columns_preserving=False,
            )
        return OperatorPolicy(**{**base, **_EXPLICIT_POLICIES[canon]})

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
    ):
        scope = "ts"
    elif category in ("statistics",):
        if any(k in name for k in ("test", "corr_test", "granger", "ttest", "adf", "kpss")):
            scope = "hypothesis"
        else:
            scope = "aggregate"
    elif category in ("math", "elementwise_math", "data_handling"):
        scope = "elementwise"

    pit_safe = "pit_safe" in tags or "causal" in tags
    if canon in PIT_UNSAFE_CANONICALS:
        pit_safe = False
    elif name in ("lead", "next"):
        pit_safe = False
    elif name == "shuffle":
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

    shape_preserving = canon not in NON_SHAPE_PRESERVING_CANONICALS
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
            bars = int(max(bars, round(bars * s_bpd / f_bpd)))
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
    if lowered.endswith("d"):
        return 1
    return 1


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
    if text.endswith("h") and text[:-1].isdigit():
        return pd.Timedelta(hours=int(text[:-1]))
    return pd.Timedelta(days=1)
