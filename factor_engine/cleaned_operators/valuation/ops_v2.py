# -*- coding: utf-8 -*-
"""Valuation-ratio gaps, free-float structure and capital-change operators (P0).

Daily as-of panels in, daily panels out.  Ratios are elementwise; change terms
are plain period-over-period differences over the as-of daily panel.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator

_EPS = 1e-12
_CANONICALS: list[str] = []


def _meta(name: str, description: str, params: list[str], *, unit: str = "ratio", param_specs: dict | None = None) -> OperatorMetadata:
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
        param_specs=dict(param_specs or {}),
    )


def _safe_div(num, den):
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _mk(name: str, description: str, params: list[str], fn, *, unit: str = "ratio", param_specs: dict | None = None):
    metadata = _meta(name, description, params, unit=unit, param_specs=param_specs)

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

    _surface.extend_extended_only({name})
    return cls


def _log_abs(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.abs().replace(0, np.nan))


def _signed_log(x: pd.DataFrame) -> pd.DataFrame:
    """``sign(x) * log1p(|x|)``: preserves the profit/loss economic sign.

    ``log(|x|)`` collapses +PE and -PE onto the same value; the signed transform
    keeps gain vs loss distinguishable for the gap factor (audit §5.1).
    """
    return np.sign(x) * np.log1p(x.abs())


def _positive_log_gap(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Gap defined only when both multiples are strictly positive (loss → NaN)."""
    valid = a.gt(0) & b.gt(0)
    out = np.log(a.where(valid)) - np.log(b.where(valid))
    return out.where(valid)


_mk(
    "valuation_pe_ttm_lyr_gap",
    "log|PeRatio| - log|PeRatioLyr|（口径差异；log|x| 塌缩盈亏符号，建议用 *_gap_signed_log）。",
    ["pe_ratio", "pe_ratio_lyr"],
    lambda a, b: _log_abs(a) - _log_abs(b),
    unit="level",
)
_mk(
    "valuation_pcf_definition_gap",
    "log|PcfRatio| - log|PcfRatio2|（口径差异；建议用 *_gap_signed_log）。",
    ["pcf_ratio", "pcf_ratio2"],
    lambda a, b: _log_abs(a) - _log_abs(b),
    unit="level",
)
# Sign-aware and positive-only gap variants (audit §5.1): log|x| treats a
# profitable +10x and a loss-making -10x as identical, discarding the
# economic meaning of profit vs loss for long/short factor interpretation.
_mk(
    "valuation_pe_gap_signed_log",
    "sign(PeRatio)*log1p|PeRatio| - sign(PeRatioLyr)*log1p|PeRatioLyr|。",
    ["pe_ratio", "pe_ratio_lyr"],
    lambda a, b: _signed_log(a) - _signed_log(b),
    unit="level",
)
_mk(
    "valuation_pe_gap_positive",
    "log(PeRatio) - log(PeRatioLyr)，仅两值均 >0（亏损侧 NaN）。",
    ["pe_ratio", "pe_ratio_lyr"],
    _positive_log_gap,
    unit="level",
)
_mk(
    "valuation_pcf_gap_signed_log",
    "sign(PcfRatio)*log1p|PcfRatio| - sign(PcfRatio2)*log1p|PcfRatio2|。",
    ["pcf_ratio", "pcf_ratio2"],
    lambda a, b: _signed_log(a) - _signed_log(b),
    unit="level",
)
_mk(
    "valuation_pcf_gap_positive",
    "log(PcfRatio) - log(PcfRatio2)，仅两值均 >0（负侧 NaN）。",
    ["pcf_ratio", "pcf_ratio2"],
    _positive_log_gap,
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
                # R5 P0-18: fewer than 5 valid stocks means no cross-section to
                # standardize.  Fail closed (whole row -> NaN) instead of leaving
                # the raw unscaled values, which would fabricate a fake
                # "no disagreement" cell.
                arr[row, :] = np.nan
                continue
            lo, hi = np.quantile(finite, [0.01, 0.99])
            clipped = np.clip(arr[row], lo, hi)
            sd = np.nanstd(clipped)
            if sd > 0:
                arr[row] = (clipped - np.nanmean(clipped)) / sd
        standardized.append(arr)
    stacked = np.stack(standardized, axis=0)  # (components, rows, cols)
    out = np.full((stacked.shape[1], stacked.shape[2]), np.nan, dtype=float)
    for r in range(stacked.shape[1]):
        # R5 P0-18: a day whose component rows are all-NaN (fewer than 5 valid
        # stocks) must stay NaN; require at least two standardised components.
        for c in range(stacked.shape[2]):
            finite = stacked[:, r, c]
            finite = finite[np.isfinite(finite)]
            if finite.size >= 2:
                out[r, c] = float(np.std(finite))
    return pd.DataFrame(
        out,
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
def _growth_mismatch(ey, g, scale=1.0):
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("valuation_growth_mismatch scale must be a positive finite number")
    # P1-140: ``scale`` is a dead searchable parameter — the only production-
    # meaningful value is 1.0.  Both inputs are decimal ratios by contract (the
    # field layer divides percent-based StockIndicator growth by 100); a free
    # ``scale`` would silently mask a percent/ratio unit mismatch (audit §5.2).
    # Removed from the searchable metadata surface; kept only as a
    # backward-compat positional that validates the unit-honest default.
    return ey - g / scale


_mk(
    "valuation_growth_mismatch",
    "盈利收益率 - 利润增长（输入须为小数比率，字段层 /100；scale 已从搜索面移除，仅向后兼容 1.0）。",
    ["earnings_yield", "profit_growth", "scale"],
    _growth_mismatch,
    unit="level",
    param_specs={"scale": ParamSpec(dtype=float, default=1.0, searchable=False, min=1e-6)},
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
    """距最近股本变动生效日的交易日数。

    A ChangeDate on a weekend/holiday is mapped to the first trading day on or
    after it (audit §5.3) instead of dropping the change and producing NaN until
    the next in-index date.  A future-dated ChangeDate (not yet knowable) yields
    NaN rather than a negative age.
    """
    index_array = np.asarray(index_dates, dtype="datetime64[ns]")
    out = pd.DataFrame(np.nan, index=change_date.index, columns=change_date.columns, dtype=float)
    for col in change_date.columns:
        last_pos: int | None = None
        for i, cd in enumerate(change_date[col]):
            if pd.notna(cd):
                try:
                    cd_ts = pd.Timestamp(cd).normalize().to_datetime64()
                    pos_arr = np.searchsorted(index_array, cd_ts, side="left")
                    if 0 <= pos_arr < len(index_array) and pos_arr <= i:
                        last_pos = int(pos_arr)
                    else:
                        last_pos = None
                except Exception:
                    last_pos = None
            if last_pos is not None:
                out.iloc[i, change_date.columns.get_loc(col)] = float(i - last_pos)
    return out


_mk(
    "capital_change_age",
    "距最近股本变动生效日的交易日数（周末/节假日 ChangeDate 顺延至下一交易日）。",
    ["change_date"],
    lambda cd: _capital_change_age(cd, cd.index),
    unit="count",
)
_mk(
    "capital_change_magnitude",
    "较上一交易日的股本变化（日状态差分，非快照事件）。",
    ["total_capital"],
    lambda tc: tc / tc.shift(1).replace(0, np.nan) - 1.0,
)
_mk(
    "circulating_cap_ratio_change",
    "流通股本 / 总股本 的日间变化（状态变化，非解禁事件；解禁需真实事件源）。",
    ["circulating_capital", "total_capital"],
    lambda cc, tc: (_safe_div(cc, tc) - _safe_div(cc, tc).shift(1)),
)
_mk(
    "circulating_cap_unlock_proxy",
    "流通股本 / 总股本 的变化（旧名；名称过度解释为解禁，改用 circulating_cap_ratio_change）。",
    ["circulating_capital", "total_capital"],
    lambda cc, tc: (_safe_div(cc, tc) - _safe_div(cc, tc).shift(1)),
)

# Deprecation notes for the sign-collapsing / over-named legacy variants so the
# registry advertises the honest replacements (audit §5.1 / §5.5).
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
        _entry.setdefault(
            "semantic_note",
            "旧公式 log|x| 塌缩盈亏符号 / 名称过度解释；新因子使用 *_gap_signed_log 或 "
            "circulating_cap_ratio_change",
        )
