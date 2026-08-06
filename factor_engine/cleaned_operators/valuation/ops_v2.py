# -*- coding: utf-8 -*-
"""Valuation-ratio gaps, free-float structure and capital-change operators (P0).

Daily as-of panels in, daily panels out.  Ratios are elementwise; change terms
are plain period-over-period differences over the as-of daily panel.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12
_CANONICALS: list[str] = []


def _meta(name: str, description: str, params: list[str], *, unit: str = "ratio") -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="valuation",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "valuation", "ashare", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:valuation",
            f"unit:{unit}", "cost:1",
        ],
    )


def _safe_div(num, den):
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _mk(name: str, description: str, params: list[str], fn, *, unit: str = "ratio"):
    metadata = _meta(name, description, params, unit=unit)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"Valuation_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="valuation",
        business_category="valuation",
        canonical=name,
        source="valuation.ops_v2",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {name}
    )
    return cls


def _log_abs(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.abs().replace(0, np.nan))


_mk(
    "valuation_pe_ttm_lyr_gap",
    "log|PeRatio| - log|PeRatioLyr|（口径差异）。",
    ["pe_ratio", "pe_ratio_lyr"],
    lambda a, b: _log_abs(a) - _log_abs(b),
    unit="level",
)
_mk(
    "valuation_pcf_definition_gap",
    "log|PcfRatio| - log|PcfRatio2|（口径差异）。",
    ["pcf_ratio", "pcf_ratio2"],
    lambda a, b: _log_abs(a) - _log_abs(b),
    unit="level",
)
_mk(
    "free_float_ratio",
    "自由流通股 / 总股本。",
    ["free_cap", "capitalization"],
    lambda a, b: _safe_div(a, b),
)
_mk(
    "free_to_circulating_ratio",
    "自由流通股 / 流通股。",
    ["free_cap", "circulating_cap"],
    lambda a, b: _safe_div(a, b),
)
_mk(
    "a_share_cap_ratio",
    "A 股股本 / 总股本。",
    ["a_cap", "capitalization"],
    lambda a, b: _safe_div(a, b),
)
_mk(
    "free_float_turnover",
    "成交量 / 自由流通股。",
    ["volume", "free_cap"],
    lambda a, b: _safe_div(a, b),
    unit="turnover",
)
_mk(
    "market_cap_free_cap_gap",
    "log(MarketCap) - log(FreeMarketCap)。",
    ["market_cap", "free_market_cap"],
    lambda a, b: _log_abs(a) - _log_abs(b),
    unit="level",
)


def _valuation_cashflow_disagreement(pe_ratio, pcf_ratio, pcf_ratio2, ocf_yield, earnings_yield):
    # The five components are scale-heterogeneous: the log transforms sit around
    # 1-5 while the yields are ~0.01-0.10, so a raw nanstd would be dominated by
    # the log-transformed terms.  Standardize every component cross-sectionally
    # (row-wise winsorize at 1/99, then z-score) before measuring dispersion.
    frames = {
        "pe": np.log(pe_ratio.abs().replace(0, np.nan)) * -1.0,
        "pcf": np.log(pcf_ratio.abs().replace(0, np.nan)) * -1.0,
        "pcf2": np.log(pcf_ratio2.abs().replace(0, np.nan)) * -1.0,
        "ocf": ocf_yield,
        "ey": earnings_yield,
    }
    standardized = []
    for frame in frames.values():
        arr = frame.to_numpy(dtype=float).copy()
        for row in range(arr.shape[0]):
            finite = arr[row][np.isfinite(arr[row])]
            if finite.size < 5:
                continue
            lo, hi = np.quantile(finite, [0.01, 0.99])
            clipped = np.clip(arr[row], lo, hi)
            sd = np.nanstd(clipped)
            if sd > 0:
                arr[row] = (clipped - np.nanmean(clipped)) / sd
        standardized.append(arr)
    with np.errstate(invalid="ignore"):
        return pd.DataFrame(
            np.nanstd(np.stack(standardized, axis=0), axis=0),
            index=pe_ratio.index,
            columns=pe_ratio.columns,
            dtype=float,
        )


_mk(
    "valuation_cashflow_disagreement",
    "盈利收益率、现金流收益率与两套 PCF 之间的离散度。",
    ["pe_ratio", "pcf_ratio", "pcf_ratio2", "ocf_yield", "earnings_yield"],
    _valuation_cashflow_disagreement,
    unit="level",
)
_mk(
    "valuation_growth_mismatch",
    "盈利收益率 - 标准化利润增长（估值成长错配）。",
    ["earnings_yield", "profit_growth", "scale"],
    lambda ey, g, scale=1.0: ey - g / float(scale),
    unit="level",
)


def _valuation_quality_mismatch(valuation, quality, weight):
    """估值对质量横截面 WLS 回归残差（weight 真正参与加权）。"""
    v = valuation.to_numpy(dtype=float)
    q = quality.to_numpy(dtype=float)
    w = weight.to_numpy(dtype=float)
    rows, cols = v.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        valid = (
            np.isfinite(v[row])
            & np.isfinite(q[row])
            & np.isfinite(w[row])
            & (w[row] > 0)
        )
        if valid.sum() < 5 or np.var(q[row][valid]) <= _EPS:
            continue
        y, x, wts = v[row][valid], q[row][valid], w[row][valid]
        root = np.sqrt(wts)
        design = np.column_stack([np.ones(valid.sum()), x])
        # WLS: scale design and target by sqrt(weight) so the weight actually
        # enters the fit instead of serving only as a positive-mask filter.
        beta, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        out[row] = np.where(np.isfinite(q[row]), v[row] - (beta[0] + beta[1] * q[row]), np.nan)
    return pd.DataFrame(out, index=valuation.index, columns=valuation.columns, dtype=float)


_mk(
    "valuation_quality_mismatch",
    "估值相对财务质量的横截面回归残差。",
    ["valuation", "quality", "weight"],
    _valuation_quality_mismatch,
    unit="level",
)


def _capital_change_age(change_date, index_dates):
    """距最近股本变动日的交易日数。"""
    pos = {ts: i for i, ts in enumerate(index_dates)}
    out = pd.DataFrame(np.nan, index=change_date.index, columns=change_date.columns, dtype=float)
    for col in change_date.columns:
        last_change: pd.Timestamp | None = None
        last_pos: int | None = None
        for i, (dt, cd) in enumerate(zip(change_date.index, change_date[col])):
            if pd.notna(cd):
                try:
                    cd_ts = pd.Timestamp(cd).normalize()
                    if cd_ts in pos:
                        last_pos = pos[cd_ts]
                    else:
                        last_pos = None
                except Exception:
                    last_pos = None
            if last_pos is not None:
                out.iloc[i, change_date.columns.get_loc(col)] = float(i - last_pos)
    return out


_mk(
    "capital_change_age",
    "距 ChangeDate 的交易日数。",
    ["change_date"],
    lambda cd: _capital_change_age(cd, cd.index),
    unit="count",
)
_mk(
    "capital_change_magnitude",
    "TotalCapital / 上期 - 1。",
    ["total_capital"],
    lambda tc: tc / tc.shift(1).replace(0, np.nan) - 1.0,
)
_mk(
    "circulating_cap_unlock_proxy",
    "流通股本 / 总股本 的变化（解禁代理）。",
    ["circulating_capital", "total_capital"],
    lambda cc, tc: (_safe_div(cc, tc) - _safe_div(cc, tc).shift(1)),
)
