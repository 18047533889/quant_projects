# -*- coding: utf-8 -*-
"""Cross-sectional / panel-model operators (2026-08 R47).

Daily wide panels (date x instrument) in, daily wide panels out.  All kernels
are causal, trailing-only and deterministic.

  1. ``panel_async_beta_ex_self``     -- beta on the value-weighted ex-self
     market return, refitted on a fixed refresh grid (async).
  2. ``panel_factor_pocket_strength`` -- persistence of factor-payout regimes
     (in-pocket AND above cross-sectional-median return).
  3. ``cs_predictability_mosaic_score``  -- per-stock state-bucketed historical
     rank-IC mosaic score.
  4. ``panel_predictability_mosaic_score`` -- cross-sectional mean of the mosaic
     score each date (reuses the #3 kernel).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
    register_operator,
)

_EPS = 1e-12
_CANONICALS: list[str] = []

_INT_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2),
    "min_periods": ParamSpec(dtype=int, min=2),
    "min_history": ParamSpec(dtype=int, min=2),
    "refresh_freq": ParamSpec(dtype=int, min=1),
    "clusters": ParamSpec(dtype=int, min=2),
    "lag": ParamSpec(dtype=int, min=1),
}


def _meta(name: str, description: str, params: list[str], *, unit: str = "level") -> OperatorMetadata:
    param_specs = {n: _INT_SPECS[n] for n in params if n in _INT_SPECS}
    return OperatorMetadata(
        name=name,
        category="panel_model",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "panel_model", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:panel_gap",
            f"unit:{unit}", "cost:10",
        ],
        param_specs=param_specs,
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _cs_rank_pct(row: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank (0..1, higher = higher value, avg ties)."""
    return pd.Series(row).rank(method="average", pct=True).to_numpy()


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 2:
        return np.nan
    ra = pd.Series(a[valid]).rank(method="average").to_numpy()
    rb = pd.Series(b[valid]).rank(method="average").to_numpy()
    if np.std(ra) <= _EPS or np.std(rb) <= _EPS:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# 1. panel_async_beta_ex_self
# ---------------------------------------------------------------------------
def _ex_self_market_returns(ret: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Value-weighted ex-self market return per stock/date (total-minus-self)."""
    rows, cols = ret.shape
    total = np.zeros(rows)
    total_w = np.zeros(rows)
    valid = np.isfinite(ret) & np.isfinite(weight) & (weight > 0.0)
    wv = np.where(valid, weight, 0.0)
    rv = np.where(valid, ret, 0.0)
    total = (wv * rv).sum(axis=1)
    total_w = wv.sum(axis=1)
    mkt = np.full((rows, cols), np.nan)
    for c in range(cols):
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = total_w - wv[:, c]
            num = total - wv[:, c] * rv[:, c]
            out = np.where(denom > _EPS, num / np.where(denom > _EPS, denom, np.nan), np.nan)
        mkt[:, c] = np.where(valid[:, c] & (denom > _EPS), out, np.nan)
    return mkt


def _async_beta_column(
    r: np.ndarray,
    mkt: np.ndarray,
    window: int,
    min_periods: int,
    refresh_freq: int,
) -> np.ndarray:
    n = len(r)
    out = np.full(n, np.nan)
    for t in range(n):
        anchor = (t // refresh_freq) * refresh_freq
        start = max(0, anchor - window + 1)
        seg_r = r[start : anchor + 1]
        seg_m = mkt[start : anchor + 1]
        valid = np.isfinite(seg_r) & np.isfinite(seg_m)
        if int(valid.sum()) < min_periods:
            continue
        x = seg_m[valid]
        y = seg_r[valid]
        xbar = float(np.mean(x))
        varx = float(np.mean((x - xbar) ** 2))
        if varx <= _EPS:
            continue
        ybar = float(np.mean(y))
        out[t] = float(np.mean((x - xbar) * (y - ybar)) / varx)
    return out


def _async_beta_panel(ret: pd.DataFrame, weight: pd.DataFrame, window: int, min_periods: int, refresh_freq: int) -> pd.DataFrame:
    rv = ret.to_numpy(dtype=float)
    wv = weight.to_numpy(dtype=float)
    mkt = _ex_self_market_returns(rv, wv)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan)
    for c in range(cols):
        out[:, c] = _async_beta_column(rv[:, c], mkt[:, c], window, min_periods, refresh_freq)
    return _frame_like(ret, out)


@register_operator(
    name="panel_async_beta_ex_self",
    category="panel_model",
    business_category="panel_model",
    canonical="panel_async_beta_ex_self",
    source="cross_section.panel_gap",
    backend="pandas_numpy",
    status="implemented",
)
class PanelAsyncBetaExSelf(SeriesOperator):
    """Rolling beta on the value-weighted ex-self market return, refit asynchronously.

    Per date the ex-self market return is ``(sum_{j!=i} w_j r_j)/(sum_{j!=i}
    w_j)`` (total-minus-self).  Beta is estimated with an intercept on the most
    recent ``window`` rows, refitted every ``refresh_freq`` days on a fixed
    grid ``{0, refresh_freq, 2*refresh_freq, ...}`` (the latest grid point <=
    date); between refreshes the beta is carried forward.  NaN when fewer than
    ``min_periods`` valid regression rows exist or the predictor variance is ~0.
    """

    metadata = _meta(
        "panel_async_beta_ex_self",
        "剔除自身的市值加权市场收益上的异步重拟合滚动 Beta。",
        ["ret", "weight", "window", "min_periods", "refresh_freq"],
        unit="beta",
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        weight: pd.DataFrame,
        window: int = 252,
        min_periods: int = 126,
        refresh_freq: int = 21,
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        min_periods = max(2, int(min_periods))
        refresh_freq = max(1, int(refresh_freq))
        return _async_beta_panel(ret, weight, window, min_periods, refresh_freq)


# ---------------------------------------------------------------------------
# 2. panel_factor_pocket_strength
# ---------------------------------------------------------------------------
def _pocket_strength_panel(
    ret: pd.DataFrame,
    factor: pd.DataFrame,
    window: int,
    min_periods: int,
    threshold: float,
) -> pd.DataFrame:
    rv = ret.to_numpy(dtype=float)
    fv = factor.to_numpy(dtype=float)
    rows, cols = rv.shape
    out_days = np.full((rows, cols), np.nan)  # per-day 0/1 indicator
    cs_median = np.full(rows, np.nan)
    for s in range(rows):
        fr = _cs_rank_pct(fv[s])
        r = rv[s]
        valid_r = np.isfinite(r)
        if valid_r.sum() >= 1:
            cs_median[s] = float(np.nanmedian(r[valid_r]))
        valid = np.isfinite(fv[s]) & valid_r
        in_pocket = fr >= threshold
        for c in range(cols):
            if not np.isfinite(fv[s, c]) or not valid_r[c]:
                out_days[s, c] = np.nan
            elif in_pocket[c] and r[c] > cs_median[s]:
                out_days[s, c] = 1.0
            else:
                out_days[s, c] = 0.0
    out = np.full((rows, cols), np.nan)
    for t in range(rows):
        start = max(0, t - window + 1)
        seg = out_days[start : t + 1, :]
        for c in range(cols):
            col_seg = seg[:, c]
            valid = np.isfinite(col_seg)
            if int(valid.sum()) < min_periods:
                continue
            out[t, c] = float(np.mean(col_seg[valid]))
    return _frame_like(ret, out)


@register_operator(
    name="panel_factor_pocket_strength",
    category="panel_model",
    business_category="panel_model",
    canonical="panel_factor_pocket_strength",
    source="cross_section.panel_gap",
    backend="pandas_numpy",
    status="implemented",
)
class PanelFactorPocketStrength(SeriesOperator):
    """Persistence of factor-payout outperformance regimes (documented formula).

    Each date every stock is ranked cross-sectionally on ``factor`` (percentile
    rank 0..1, higher = higher factor).  A stock is *in-pocket* when its factor
    rank is >= ``threshold`` (the top ``1-threshold`` quantile).  The output is
    the trailing ``window`` share of days on which the stock was in-pocket AND
    its contemporaneous return exceeded the cross-sectional median return that
    day.  NaN until ``min_periods`` valid days accumulate in the window.
    """

    metadata = _meta(
        "panel_factor_pocket_strength",
        "因子支付(跑赢中位数)状态的滚动持续性，0~1。",
        ["ret", "factor", "window", "min_periods", "threshold"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        factor: pd.DataFrame,
        window: int = 60,
        min_periods: int = 40,
        threshold: float = 0.5,
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        min_periods = max(2, int(min_periods))
        threshold = float(threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"panel_factor_pocket_strength: threshold must be in (0,1), got {threshold!r}")
        return _pocket_strength_panel(ret, factor, window, min_periods, threshold)


# ---------------------------------------------------------------------------
# 3. cs_predictability_mosaic_score  (+ panel variant)
# ---------------------------------------------------------------------------
def _mosaic_kernel(
    base_signal: np.ndarray,
    realized_return: np.ndarray,
    state_feature: np.ndarray,
    window: int,
    min_history: int,
    clusters: int,
    lag: int,
) -> np.ndarray:
    rows, cols = base_signal.shape
    # per-date cross-sectional quantile bucket labels (0..clusters-1), -1 = invalid
    labels = np.full((rows, cols), -1, dtype=int)
    for s in range(rows):
        fr = _cs_rank_pct(state_feature[s])
        valid = np.isfinite(state_feature[s])
        for c in range(cols):
            if not valid[c]:
                labels[s, c] = -1
            else:
                labels[s, c] = min(clusters - 1, int(fr[c] * clusters))
    # per-(date, bucket) historical rank-IC of signal[s-lag] vs return[s]
    ic = np.full((rows, clusters), np.nan)
    for s in range(rows):
        if s < lag:
            continue
        sig_s = base_signal[s - lag]
        ret_s = realized_return[s]
        for b in range(clusters):
            mask = labels[s] == b
            if int(mask.sum()) < 2:
                continue
            ic[s, b] = _rank_ic(sig_s[mask], ret_s[mask])
    out = np.full((rows, cols), np.nan)
    for t in range(rows):
        start = max(0, t - window + 1)
        for c in range(cols):
            b = labels[t, c]
            if b < 0:
                continue
            ic_series = ic[start : t + 1, b]
            valid_ic = np.isfinite(ic_series)
            if int(valid_ic.sum()) < min_history:
                continue
            out[t, c] = float(np.mean(ic_series[valid_ic]))
    return out


def _mosaic_score_panel(
    base_signal: pd.DataFrame,
    realized_return: pd.DataFrame,
    state_feature: pd.DataFrame,
    window: int,
    min_history: int,
    clusters: int,
    lag: int,
) -> pd.DataFrame:
    bv = base_signal.to_numpy(dtype=float)
    rv = realized_return.to_numpy(dtype=float)
    sv = state_feature.to_numpy(dtype=float)
    return _frame_like(base_signal, _mosaic_kernel(bv, rv, sv, window, min_history, clusters, lag))


@register_operator(
    name="cs_predictability_mosaic_score",
    category="panel_model",
    business_category="panel_model",
    canonical="cs_predictability_mosaic_score",
    source="cross_section.panel_gap",
    backend="pandas_numpy",
    status="implemented",
)
class CsPredictabilityMosaicScore(SeriesOperator):
    """Per-stock state-bucketed historical rank-IC mosaic score (documented).

    Each date the ``state_feature`` is binned cross-sectionally into ``clusters``
    quantile buckets (rank-based percentile, deterministic on ties).  For bucket
    ``b`` on date ``s`` the historical rank-IC of ``base_signal[s-lag]`` vs
    ``realized_return[s]`` is computed within the bucket's members (>= 2 valid
    names required).  The mosaic score of a stock at date ``t`` is the mean over
    the trailing ``window`` of the rank-ICs of the bucket the stock occupies at
    ``t``.  NaN when the bucket accumulated fewer than ``min_history`` valid
    per-date rank-ICs.  Purged: only returns realized by ``t`` and signals dated
    ``<= t - lag`` are used.
    """

    metadata = _meta(
        "cs_predictability_mosaic_score",
        "状态分桶的历史 rank-IC 马赛克可预测性得分。",
        ["base_signal", "realized_return", "state_feature", "window", "min_history", "clusters", "lag"],
        unit="ic",
    )

    def _calculate_series(
        self,
        base_signal: pd.DataFrame,
        realized_return: pd.DataFrame,
        state_feature: pd.DataFrame,
        window: int = 504,
        min_history: int = 160,
        clusters: int = 6,
        lag: int = 1,
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        min_history = max(2, int(min_history))
        clusters = max(2, int(clusters))
        lag = max(1, int(lag))
        return _mosaic_score_panel(base_signal, realized_return, state_feature, window, min_history, clusters, lag)


@register_operator(
    name="panel_predictability_mosaic_score",
    category="panel_model",
    business_category="panel_model",
    canonical="panel_predictability_mosaic_score",
    source="cross_section.panel_gap",
    backend="pandas_numpy",
    status="implemented",
)
class PanelPredictabilityMosaicScore(SeriesOperator):
    """Cross-sectional mean of the per-stock mosaic score each date.

    Reuses the ``cs_predictability_mosaic_score`` kernel and returns, for each
    date, the cross-sectional mean of the per-stock scores broadcast over the
    instrument axis (a date-level regime/panel signal).
    """

    metadata = _meta(
        "panel_predictability_mosaic_score",
        "马赛克可预测性得分的截面均值（按日广播到全体标的）。",
        ["base_signal", "realized_return", "state_feature", "window", "min_history", "clusters", "lag"],
        unit="ic",
    )

    def _calculate_series(
        self,
        base_signal: pd.DataFrame,
        realized_return: pd.DataFrame,
        state_feature: pd.DataFrame,
        window: int = 504,
        min_history: int = 160,
        clusters: int = 6,
        lag: int = 1,
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        min_history = max(2, int(min_history))
        clusters = max(2, int(clusters))
        lag = max(1, int(lag))
        mosaic = _mosaic_score_panel(base_signal, realized_return, state_feature, window, min_history, clusters, lag)
        mv = mosaic.to_numpy(dtype=float)
        rows = mv.shape[0]
        out = np.full(mv.shape, np.nan)
        for t in range(rows):
            row = mv[t]
            valid = np.isfinite(row)
            if int(valid.sum()) < 1:
                continue
            out[t, :] = float(np.mean(row[valid]))
        return _frame_like(base_signal, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_CANONICALS))


_CANONICALS.extend(
    [
        "panel_async_beta_ex_self",
        "panel_factor_pocket_strength",
        "cs_predictability_mosaic_score",
        "panel_predictability_mosaic_score",
    ]
)
_register_surface()
