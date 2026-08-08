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
    # abs(0) -> log(0) -> null (matches the pandas reference which replaces 0
    # with NaN before logging); no fill_null, so a zero multiple yields null.
    return x.abs().log()


def _gap(a, b):
    cols = align_cols(a, b)
    return a.with_columns([(_log_abs(a[c]) - _log_abs(b[c])).alias(c) for c in cols])


def _signed_log_gap(a, b):
    cols = align_cols(a, b)
    return a.with_columns(
        [((a[c].sign() * a[c].abs().log1p()) - (b[c].sign() * b[c].abs().log1p())).alias(c) for c in cols]
    )


def _positive_log_gap(a, b):
    cols = align_cols(a, b)
    return a.with_columns(
        [
            pl.when((a[c] > 0) & (b[c] > 0)).then(a[c].log() - b[c].log()).otherwise(None).alias(c)
            for c in cols
        ]
    )


_register(
    "valuation_pe_ttm_lyr_gap", "log|PeRatio| - log|PeRatioLyr|（口径差异；log|x| 塌缩盈亏符号，建议用 *_gap_signed_log）。", ["pe_ratio", "pe_ratio_lyr"],
    lambda a, b: _gap(a, b),
    unit="level",
)
_register(
    "valuation_pcf_definition_gap", "log|PcfRatio| - log|PcfRatio2|（口径差异；建议用 *_gap_signed_log）。", ["pcf_ratio", "pcf_ratio2"],
    lambda a, b: _gap(a, b),
    unit="level",
)
_register(
    "valuation_pe_gap_signed_log", "sign(PeRatio)*log1p|PeRatio| - sign(PeRatioLyr)*log1p|PeRatioLyr|。", ["pe_ratio", "pe_ratio_lyr"],
    lambda a, b: _signed_log_gap(a, b),
    unit="level",
)
_register(
    "valuation_pe_gap_positive", "log(PeRatio) - log(PeRatioLyr)，仅两值均 >0（亏损侧 NaN）。", ["pe_ratio", "pe_ratio_lyr"],
    lambda a, b: _positive_log_gap(a, b),
    unit="level",
)
_register(
    "valuation_pcf_gap_signed_log", "sign(PcfRatio)*log1p|PcfRatio| - sign(PcfRatio2)*log1p|PcfRatio2|。", ["pcf_ratio", "pcf_ratio2"],
    lambda a, b: _signed_log_gap(a, b),
    unit="level",
)
_register(
    "valuation_pcf_gap_positive", "log(PcfRatio) - log(PcfRatio2)，仅两值均 >0（负侧 NaN）。", ["pcf_ratio", "pcf_ratio2"],
    lambda a, b: _positive_log_gap(a, b),
    unit="level",
)
_register(
    "market_cap_free_cap_gap", "log(MarketCap) - log(FreeMarketCap)。", ["market_cap", "free_market_cap"],
    lambda a, b: _gap(a, b),
    unit="level",
)
def _growth_mismatch(ey, g, scale=1.0):
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("valuation_growth_mismatch scale must be a positive finite number")
    # P1-140: ``scale`` removed from the searchable surface (only 1.0 is
    # production-meaningful); kept as a backward-compat positional that
    # validates the unit-honest default.
    return _binary(ey, g, lambda a, b: (a - b / scale))


_register(
    "valuation_growth_mismatch", "盈利收益率 - 利润增长（输入须为小数比率，字段层 /100；scale 已从搜索面移除，仅向后兼容 1.0）。",
    ["earnings_yield", "profit_growth"],
    lambda ey, g, scale=1.0: _growth_mismatch(ey, g, scale),
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
    "circulating_cap_ratio_change", "流通股本/总股本 日间变化（状态变化，非解禁事件）。", ["circulating_capital", "total_capital"],
    _circulating_unlock_proxy,
)
_register(
    "circulating_cap_unlock_proxy", "流通股本/总股本 变化（旧名；名称过度解释为解禁，改用 circulating_cap_ratio_change）。", ["circulating_capital", "total_capital"],
    _circulating_unlock_proxy,
)


def _valuation_disagreement(pe, pcf, pcf2, ocf_yield, earnings_yield):
    cols = align_cols(pe, pcf, pcf2, ocf_yield, earnings_yield)
    out = []
    for c in cols:
        v1 = -pe[c].abs().log().to_numpy()
        v2 = -pcf[c].abs().log().to_numpy()
        v3 = -pcf2[c].abs().log().to_numpy()
        v4 = ocf_yield[c].to_numpy()
        v5 = earnings_yield[c].to_numpy()
        stacked = np.column_stack([v1, v2, v3, v4, v5])
        # Match the pandas reference: winsorize (1/99) then z-score each row so the
        # log-transformed terms (~1-5) and the yields (~0.01-0.10) are comparable
        # before the dispersion is measured.  Rows with <5 finite components keep
        # their raw values (nanstd then ignores NaN), exactly like the pandas path.
        standardized = np.full_like(stacked, np.nan, dtype=float)
        for row in range(stacked.shape[0]):
            finite = stacked[row][np.isfinite(stacked[row])]
            if finite.size < 5:
                standardized[row] = stacked[row]
                continue
            lo, hi = np.quantile(finite, [0.01, 0.99])
            clipped = np.clip(stacked[row], lo, hi)
            sd = np.nanstd(clipped)
            if sd > 0:
                standardized[row] = (clipped - np.nanmean(clipped)) / sd
        out.append(pl.Series(c, np.nanstd(standardized, axis=1)))
    return pe.with_columns(out)


_register(
    "valuation_cashflow_disagreement", "盈利/现金流收益率与 PCF 离散度。",
    ["pe_ratio", "pcf_ratio", "pcf_ratio2", "ocf_yield", "earnings_yield"],
    lambda pe, pcf, pcf2, oy, ey: _valuation_disagreement(pe, pcf, pcf2, oy, ey),
    unit="level",
)


# Deprecation notes mirroring the pandas backend (audit §5.1 / §5.5).
from cleaned_operators.registry import OperatorRegistry as _registry  # noqa: E402

for _old, _new in (
    ("valuation_pe_ttm_lyr_gap", "valuation_pe_gap_signed_log"),
    ("valuation_pcf_definition_gap", "valuation_pcf_gap_signed_log"),
    ("circulating_cap_unlock_proxy", "circulating_cap_ratio_change"),
):
    _entry = _registry._catalog.get(_old)
    if _entry is not None:
        _entry["compatibility_only"] = True
        _entry["preferred_replacements"] = [_new]
        _entry.setdefault("semantic_note", "旧公式 log|x| 塌缩盈亏符号 / 名称过度解释；使用 *_gap_signed_log 或 circulating_cap_ratio_change")
