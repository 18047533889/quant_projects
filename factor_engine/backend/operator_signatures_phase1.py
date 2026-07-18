# -*- coding: utf-8
"""Phase-1 六证算子类型签名批量登记（驱动 operational_production_certified）。"""
from __future__ import annotations

from backend.operator_types import (
    ArgSpec,
    OperatorSignature,
    TypeKind,
)

_U = (ArgSpec("x", TypeKind.SERIES_FLOAT),)
_B = (
    ArgSpec("a", TypeKind.SERIES_FLOAT),
    ArgSpec("b", TypeKind.SERIES_FLOAT),
)
_W = (
    ArgSpec("x", TypeKind.SERIES_FLOAT),
    ArgSpec("window", TypeKind.WINDOW),
)
_G = (
    ArgSpec("x", TypeKind.SERIES_FLOAT),
    ArgSpec("group", TypeKind.GROUP_KEY),
)
_W3 = (
    ArgSpec("x", TypeKind.SERIES_FLOAT),
    ArgSpec("y", TypeKind.SERIES_FLOAT),
    ArgSpec("window", TypeKind.WINDOW),
)
_W4 = (
    ArgSpec("x", TypeKind.SERIES_FLOAT),
    ArgSpec("y", TypeKind.SERIES_FLOAT),
    ArgSpec("window", TypeKind.WINDOW),
    ArgSpec("lag", TypeKind.SCALAR_INT, required=False),
)

_UNARY = frozenset(
    """
    abs neg sign floor ceil exp log sqrt inverse rank rank_pct zscore normalize
    winsorize cum_sum cum_max cum_min cum_prod cum_delta expanding_sum expanding_mean
    count is_null is_not_null is_nan is_finite is_infinite nan_to_num
    cs_demean cs_mad cs_mad_zscore cs_pct_rank c_mean c_std c_sum c_count
    cs_mean cs_std cs_sum cs_count log_returns ts_log_return tanh
    protected_log protected_sqrt not_ log_abs signed_log signed_sqrt
    fillna_const ffill
    """.split()
)

_BINARY = frozenset(
    """
    add subtract multiply divide maximum minimum gt lt eq ge le ne and_ or_
    power protected_div safe_div_null coalesce
    """.split()
)

_WINDOW = frozenset(
    """
    ts_mean ts_std ts_var ts_sum ts_min ts_max ts_median ts_delay ts_delta ts_pct
    ts_zscore ts_rank ts_autocorr ts_sharpe volatility
    """.split()
)

_PAIR_WINDOW = frozenset({"ts_corr", "ts_cov", "ts_beta"})

_GROUP = frozenset(
    """
    group_mean group_std group_rank group_zscore group_neutralize group_normalize
    group_percentile group_winsorize group_sum group_min group_max group_count
    """.split()
)

_SPECIAL: dict[str, OperatorSignature] = {
    "where": OperatorSignature(
        "where",
        (
            ArgSpec("cond", TypeKind.SERIES_BOOL),
            ArgSpec("a", TypeKind.SERIES_FLOAT),
            ArgSpec("b", TypeKind.SERIES_FLOAT),
        ),
    ),
    "clip": OperatorSignature(
        "clip",
        (
            ArgSpec("x", TypeKind.SERIES_FLOAT),
            ArgSpec("lo", TypeKind.SCALAR_FLOAT, allow_scalar_broadcast=True),
            ArgSpec("hi", TypeKind.SCALAR_FLOAT, allow_scalar_broadcast=True),
        ),
    ),
    "scale": OperatorSignature(
        "scale",
        (
            ArgSpec("x", TypeKind.SERIES_FLOAT),
            ArgSpec("to", TypeKind.SCALAR_FLOAT, allow_scalar_broadcast=True),
        ),
    ),
    "vwap": OperatorSignature(
        "vwap",
        (
            ArgSpec("price", TypeKind.SERIES_FLOAT),
            ArgSpec("volume", TypeKind.SERIES_FLOAT),
            ArgSpec("window", TypeKind.WINDOW),
        ),
    ),
    "div_or_default": OperatorSignature("div_or_default", _B),
    "div_or_null": OperatorSignature("div_or_null", _B),
    "log_fill_invalid": OperatorSignature("log_fill_invalid", _U),
    "period_change": OperatorSignature(
        "period_change", (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("period_id", TypeKind.ANY)),
    ),
    "period_average": OperatorSignature(
        "period_average", (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("period_id", TypeKind.ANY)),
    ),
    "period_cagr": OperatorSignature(
        "period_cagr", (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("period_id", TypeKind.ANY)),
    ),
    "quarter_from_cumulative": OperatorSignature(
        "quarter_from_cumulative", (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("period_id", TypeKind.ANY)),
    ),
    "ttm_from_quarterly": OperatorSignature(
        "ttm_from_quarterly", (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("period_id", TypeKind.ANY)),
    ),
    "ttm_from_cumulative": OperatorSignature(
        "ttm_from_cumulative", (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("period_id", TypeKind.ANY)),
    ),
    "yoy_by_period": OperatorSignature(
        "yoy_by_period", (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("period_id", TypeKind.ANY)),
    ),
    "cs_regression": OperatorSignature(
        "cs_regression",
        (
            ArgSpec("y", TypeKind.SERIES_FLOAT),
            ArgSpec("x", TypeKind.SERIES_FLOAT),
            ArgSpec("mode", TypeKind.SCALAR_INT, allow_scalar_broadcast=True),
        ),
    ),
}


def phase1_operator_signatures() -> dict[str, OperatorSignature]:
    out = dict(_SPECIAL)
    for name in _UNARY:
        out[name] = OperatorSignature(name, _U)
    for name in _BINARY:
        out[name] = OperatorSignature(name, _B)
    for name in _WINDOW:
        out[name] = OperatorSignature(name, _W)
    for name in _PAIR_WINDOW:
        out[name] = OperatorSignature(name, _W3)
    for name in _GROUP:
        out[name] = OperatorSignature(name, _G)
    return out
