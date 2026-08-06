# -*- coding: utf-8 -*-
"""Polars backends for next-stage valuation / share-structure operators.

Real ``pl.Expr`` implementations on wide Polars panels.  Division by zero maps
to null; gaps are elementwise.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common._polars_bridge import align_cols, numeric_cols

_CANONICALS: list[str] = []


def _meta(name: str, description: str, params: list[str], *, unit: str = "ratio") -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="valuation",
        description=description,
        param_names=params,
        return_type="series",
        tags=["valuation", "polars", "typed_v2", f"signature:{','.join(params)}->series", f"unit:{unit}"],
    )


def _register(name: str, description: str, params: list[str], fn, *, unit: str = "ratio"):
    @register_operator(
        name=name,
        category="valuation",
        business_category="valuation",
        canonical=name,
        source="valuation.polars_ops_v2",
    )
    class _ValuationPolars(SeriesOperator):
        metadata = _meta(name, description, params, unit=unit)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _CANONICALS.append(name)
    return _ValuationPolars


def _binary(df_a, df_b, expr_fn):
    cols = align_cols(df_a, df_b)
    return df_a.with_columns([expr_fn(df_a[c], df_b[c]).alias(c) for c in cols])


_register(
    "free_float_ratio", "自由流通股 / 总股本。", ["free_cap", "capitalization"],
    lambda a, b: _binary(a, b, lambda x, y: x / y.fill_null(0.0).replace(0, None)),
)
_register(
    "free_to_circulating_ratio", "自由流通股 / 流通股。", ["free_cap", "circulating_cap"],
    lambda a, b: _binary(a, b, lambda x, y: x / y.fill_null(0.0).replace(0, None)),
)
_register(
    "a_share_cap_ratio", "A 股股本 / 总股本。", ["a_cap", "capitalization"],
    lambda a, b: _binary(a, b, lambda x, y: x / y.fill_null(0.0).replace(0, None)),
)
_register(
    "free_float_turnover", "成交量 / 自由流通股。", ["volume", "free_cap"],
    lambda a, b: _binary(a, b, lambda x, y: x / y.fill_null(0.0).replace(0, None)),
    unit="turnover",
)


def _log_abs(x):
    return x.abs().log().fill_null(0.0)  # abs(0)->log(0)=null->0 keeps subtraction defined


def _gap(a, b):
    cols = align_cols(a, b)
    return a.with_columns([(_log_abs(a[c]) - _log_abs(b[c])).alias(c) for c in cols])


_register(
    "valuation_pe_ttm_lyr_gap", "log|PeRatio| - log|PeRatioLyr|。", ["pe_ratio", "pe_ratio_lyr"],
    lambda a, b: _gap(a, b),
    unit="level",
)
_register(
    "valuation_pcf_definition_gap", "log|PcfRatio| - log|PcfRatio2|。", ["pcf_ratio", "pcf_ratio2"],
    lambda a, b: _gap(a, b),
    unit="level",
)
_register(
    "market_cap_free_cap_gap", "log(MarketCap) - log(FreeMarketCap)。", ["market_cap", "free_market_cap"],
    lambda a, b: _gap(a, b),
    unit="level",
)
_register(
    "valuation_growth_mismatch", "盈利收益率 - 标准化利润增长。", ["earnings_yield", "profit_growth", "scale"],
    lambda ey, g, scale=1.0: _binary(ey, g, lambda a, b: (a - b / float(scale))),
    unit="level",
)
_register(
    "capital_change_magnitude", "TotalCapital / 上期 - 1。", ["total_capital"],
    lambda tc: tc.with_columns([(tc[c] / tc[c].shift(1).fill_null(0.0).replace(0, None) - 1.0).alias(c) for c in numeric_cols(tc)]),
)


def _circulating_unlock_proxy(cc, tc):
    cols = align_cols(cc, tc)
    out = []
    for c in cols:
        ratio = cc[c] / tc[c].fill_null(0.0).replace(0, None)
        out.append((ratio - ratio.shift(1)).alias(c))
    return cc.with_columns(out)


_register(
    "circulating_cap_unlock_proxy", "流通股本/总股本 变化（解禁代理）。", ["circulating_capital", "total_capital"],
    _circulating_unlock_proxy,
)


def _valuation_disagreement(pe, pcf, pcf2, ocf_yield, earnings_yield):
    cols = align_cols(pe, pcf, pcf2, ocf_yield, earnings_yield)
    out = []
    for c in cols:
        v1 = -pe[c].abs().log().fill_null(0.0).to_numpy()
        v2 = -pcf[c].abs().log().fill_null(0.0).to_numpy()
        v3 = -pcf2[c].abs().log().fill_null(0.0).to_numpy()
        v4 = ocf_yield[c].to_numpy()
        v5 = earnings_yield[c].to_numpy()
        stacked = np.column_stack([v1, v2, v3, v4, v5])
        out.append(pl.Series(c, np.nanstd(stacked, axis=1)))
    return pe.with_columns(out)


_register(
    "valuation_cashflow_disagreement", "盈利/现金流收益率与 PCF 离散度。",
    ["pe_ratio", "pcf_ratio", "pcf_ratio2", "ocf_yield", "earnings_yield"],
    lambda pe, pcf, pcf2, oy, ey: _valuation_disagreement(pe, pcf, pcf2, oy, ey),
    unit="level",
)
