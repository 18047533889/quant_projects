# -*- coding: utf-8 -*-
"""Minute-frequency realized market beta and idiosyncratic moments (P0).

The market minute return is built *inside* the operator from the cross-section
of stock minute returns, value-weighted by a daily free-float capitalisation
panel supplied by the caller (as-of previous trading day).  No index minute
data is required.

Contract: ``close`` is a minute panel (row=minute, col=instrument),
``free_market_cap`` is a daily panel (row=date, col=instrument).  Output is one
scalar per (TradeDate, Symbol).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import register_operator
from cleaned_operators.intraday._core import (
    _EPS,
    DataDegeneracy,
    SessionAggregationOperator,
    as_panel,
    broadcast_daily_panel,
    log_returns,
    metadata,
    np_errstate,
    register_surface,
    require_same_session_grid,
)

_CANONICALS: list[str] = []


def _aligned_market(
    close: pd.DataFrame, weights: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return (per-stock minute returns, value-weighted market minute return,
    validated per-stock weight panel, returns-masked weight panel).

    P1-104: the cap weights must be FINITE and POSITIVE — a negative / zero /
    NaN capitalisation would poison the value-weighted market return.  The
    weights panel is a *daily* capitalisation that the data layer must provide
    already lagged as-of (previous-day cap), so the day's weights are known at
    the day's open.  The validated ``w_bc`` / ``w_ret`` panels are returned
    once and reused by every realized-beta series (including the ex-self path)
    so an invalid cap can never sneak back into a leave-one-out market return.

    P0: minute log-returns are computed per trading session.  The first minute
    of every calendar-day session is NaN (its return is undefined within the
    session), so an overnight gap never leaks into the intraday market return
    or any realized beta / correlation / semibeta / market-model estimate.
    """
    close = as_panel(close)
    weights = as_panel(weights)
    raw = close.to_numpy(dtype=float)
    with np_errstate():
        logr = np.full_like(raw, np.nan, dtype=float)
        logr[1:, :] = np.log(raw[1:, :] / raw[:-1, :])
        # P0: hard-break at calendar-day boundaries only; the am/pm lunch-break
        # policy stays governed by the session contract (no invented break).
        days = close.index.normalize().to_numpy()
        day_change = np.zeros(len(days), dtype=bool)
        if len(days) > 1:
            day_change[1:] = days[1:] != days[:-1]
        logr[day_change, :] = np.nan
    rets = pd.DataFrame(logr, index=close.index, columns=close.columns)
    w_bc = broadcast_daily_panel(close, weights)
    w_bc = w_bc.where(np.isfinite(w_bc) & (w_bc > 0.0))  # P1-104: invalid cap -> excluded
    w_ret = w_bc.where(rets.notna())
    with np_errstate():
        num = (rets * w_bc).sum(axis=1)
        den = w_ret.sum(axis=1)
        mkt = pd.Series(
            np.where(np.isfinite(den) & (np.abs(den) > _EPS), num / den, np.nan),
            index=close.index,
            dtype=float,
        )
    return rets, mkt, w_bc, w_ret


def _market_return_ex_self(rets: pd.DataFrame, w_bc: pd.DataFrame, w_ret: pd.DataFrame, inst: str) -> pd.Series:
    """Value-weighted market minute return excluding ``inst`` itself.

    ``w_ret`` is ``w_bc`` masked to minutes where the stock return is finite,
    mirroring the ``den`` convention of ``_aligned_market``.  Removing the
    instrument's own contribution prevents large-capitalisation names from
    mechanically inflating their own market beta / commonality.
    """
    num = (rets * w_bc).sum(axis=1) - rets[inst] * w_bc[inst]
    den = w_ret.sum(axis=1) - w_ret[inst]
    with np_errstate():
        m = pd.Series(
            np.where(np.isfinite(den) & (np.abs(den) > _EPS), num / den, np.nan),
            index=rets.index,
            dtype=float,
        )
    return m


def _beta_daily(close: pd.DataFrame, weights: pd.DataFrame, fn: Callable[[np.ndarray, np.ndarray], float], *, ex_self: bool = False) -> pd.DataFrame:
    rets, mkt, w_bc, w_ret = _aligned_market(close, weights)
    # P0: reuse the ONE validated market-weight panel built by ``_aligned_market``
    # (finite-and-positive caps only).  Re-broadcasting weights from scratch here
    # would let NaN/0/negative caps that were excluded from the full-market return
    # re-enter the leave-one-out (ex-self) market return.
    # P0-08: the concat+dropna below would silently compress a mismatched
    # session grid; require the same grid first (once — every per-stock market
    # return shares the minute index of ``rets``), then keep the dropna (it
    # only removes rows where the PRIMARY column is NaN).
    require_same_session_grid(rets, mkt)
    out: dict[str, pd.Series] = {}
    for inst in close.columns:
        m = _market_return_ex_self(rets, w_bc, w_ret, inst) if ex_self else mkt
        joined = pd.concat([rets[inst], m], axis=1, keys=["r", "m"]).dropna(subset=["m"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            rr = np.asarray(group["r"], dtype=float)
            mm = np.asarray(group["m"], dtype=float)
            if int(np.sum(np.isfinite(mm)) < 2:
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(rr, mm))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def _realized_beta(r: np.ndarray, m: np.ndarray) -> float:
    valid = np.isfinite(r) & np.isfinite(m)
    r, m = r[valid], m[valid]
    if len(r) < 3:
        return np.nan
    var_m = float(np.var(m))
    if var_m <= _EPS:
        return np.nan
    cov = float(np.mean((r - np.mean(r)) * (m - np.mean(m))))
    return cov / var_m


def _realized_corr(r: np.ndarray, m: np.ndarray) -> float:
    valid = np.isfinite(r) & np.isfinite(m)
    r, m = r[valid], m[valid]
    if len(r) < 3 or np.std(r) <= _EPS or np.std(m) <= _EPS:
        return np.nan
    return float(np.corrcoef(r, m)[0, 1])


def _quadrant_beta(r: np.ndarray, m: np.ndarray, r_cond: np.ndarray, m_cond: np.ndarray) -> float:
    valid = np.isfinite(r) & np.isfinite(m)
    num_mask = valid & r_cond & m_cond
    den_mask = valid & m_cond
    num = float(np.sum(r[num_mask] * m[num_mask]))
    den = float(np.sum(m[den_mask] * m[den_mask]))
    if den <= _EPS:
        return np.nan
    return num / den


def _down_down(r: np.ndarray, m: np.ndarray) -> float:
    return _quadrant_beta(r, m, r < 0, m < 0)


def _up_up(r: np.ndarray, m: np.ndarray) -> float:
    return _quadrant_beta(r, m, r > 0, m > 0)


def _down_up(r: np.ndarray, m: np.ndarray) -> float:
    return _quadrant_beta(r, m, r < 0, m > 0)


def _up_down(r: np.ndarray, m: np.ndarray) -> float:
    return _quadrant_beta(r, m, r > 0, m < 0)


def _market_model(r: np.ndarray, m: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    """Fit r = a + b*m; return (a, b, residuals) or None if degenerate.

    P0: ``np.cov`` defaults to ddof=1 (divide by n-1) while ``np.var`` defaults
    to ddof=0 (divide by n); mixing them scaled beta by (n-1)/n and polluted
    every derived market-model quantity (idiosyncratic variance / skewness /
    kurtosis and market R²).  Unify on population moments — the OLS slope — so
    every downstream consumer inherits the corrected value.
    """
    valid = np.isfinite(r) & np.isfinite(m)
    r, m = r[valid], m[valid]
    if len(r) < 3:
        return None
    rbar = float(np.mean(r))
    mbar = float(np.mean(m))
    var_m = float(np.mean((m - mbar) ** 2))
    if var_m <= _EPS:
        return None
    cov = float(np.mean((r - rbar) * (m - mbar)))
    b = cov / var_m
    a = rbar - b * mbar
    with np_errstate():
        e = r - (a + b * m)
    return a, b, e


def _idio_variance(r: np.ndarray, m: np.ndarray) -> float:
    fit = _market_model(r, m)
    if fit is None:
        return np.nan
    return float(np.mean(fit[2] * fit[2]))


def _idio_skewness(r: np.ndarray, m: np.ndarray) -> float:
    fit = _market_model(r, m)
    if fit is None:
        return np.nan
    e = fit[2]
    sd = float(np.std(e))
    if sd <= _EPS:
        return np.nan
    return float(np.mean(((e - np.mean(e)) / sd) ** 3))


def _idio_kurtosis(r: np.ndarray, m: np.ndarray) -> float:
    fit = _market_model(r, m)
    if fit is None:
        return np.nan
    e = fit[2]
    sd = float(np.std(e))
    if sd <= _EPS:
        return np.nan
    return float(np.mean(((e - np.mean(e)) / sd) ** 4))


def _market_r2(r: np.ndarray, m: np.ndarray) -> float:
    fit = _market_model(r, m)
    if fit is None:
        return np.nan
    e = fit[2]
    ss_res = float(np.sum(e * e))
    ss_tot = float(np.sum((r - np.mean(r)) ** 2))
    if ss_tot <= _EPS:
        return np.nan
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def _op(name: str, description: str, unit: str):
    def decorator(cls):
        # ``free_market_cap`` is a daily panel alongside the minute panel; the
        # allow_panel_broadcast tag exempts it from same-frequency alignment.
        cls.metadata = metadata(
            name, description, ["close", "free_market_cap"], unit=unit,
            domain="intraday_beta", extra_tags=["allow_panel_broadcast"],
        )
        return register_operator(
            name=name,
            category="intraday_microstructure",
            business_category="intraday_microstructure",
            canonical=name,
            source="intraday.realized_beta",
            backend="pandas_numpy",
            status="experimental",
        )(cls)

    return decorator


@_op("intra_realized_beta", "日内已实现 Beta：分钟收益对市场分钟收益回归。", "level")
class IntraRealizedBeta(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _realized_beta)


@_op("intra_realized_correlation", "日内已实现相关系数。", "corr")
class IntraRealizedCorrelation(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _realized_corr)


@_op("intra_down_down_semibeta", "市场下跌且个股下跌象限协同 Beta。", "level")
class IntraDownDownSemibeta(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _down_down)


@_op("intra_up_up_semibeta", "市场上涨且个股上涨象限协同 Beta。", "level")
class IntraUpUpSemibeta(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _up_up)


@_op("intra_down_up_semibeta", "市场上涨、个股下跌象限协同暴露。", "level")
class IntraDownUpSemibeta(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _down_up)


@_op("intra_up_down_semibeta", "市场下跌、个股上涨象限协同暴露。", "level")
class IntraUpDownSemibeta(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _up_down)


@_op("intra_beta_asymmetry", "下-下半 Beta 减 上-上半 Beta。", "level")
class IntraBetaAsymmetry(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        out = _beta_daily(close, free_market_cap, _down_down)
        up = _beta_daily(close, free_market_cap, _up_up)
        return out - up


@_op("intra_idiosyncratic_variance", "分钟市场模型残差方差。", "variance")
class IntraIdiosyncraticVariance(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _idio_variance)


@_op("intra_idiosyncratic_skewness", "分钟市场模型残差偏度。", "level")
class IntraIdiosyncraticSkewness(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _idio_skewness)


@_op("intra_idiosyncratic_kurtosis", "分钟市场模型残差峰度。", "level")
class IntraIdiosyncraticKurtosis(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _idio_kurtosis)


@_op("intra_market_model_r2", "分钟市场模型 R²。", "r2")
class IntraMarketModelR2(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _market_r2)


# ---------------------------------------------------------------------------
# Leave-one-out market return variants.  Each stock's market minute return is
# recomputed excluding the stock itself, so large caps no longer mechanically
# inflate their own beta / correlation / market-model fit.
# ---------------------------------------------------------------------------
@_op("intra_realized_beta_ex_self", "日内已实现 Beta(市场收益剔除自身)。", "level")
class IntraRealizedBetaExSelf(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _realized_beta, ex_self=True)


@_op("intra_realized_correlation_ex_self", "日内已实现相关系数(市场收益剔除自身)。", "corr")
class IntraRealizedCorrelationExSelf(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _realized_corr, ex_self=True)


@_op("intra_idiosyncratic_variance_ex_self", "分钟市场模型残差方差(市场收益剔除自身)。", "variance")
class IntraIdiosyncraticVarianceExSelf(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _idio_variance, ex_self=True)


@_op("intra_idiosyncratic_skewness_ex_self", "分钟市场模型残差偏度(市场收益剔除自身)。", "level")
class IntraIdiosyncraticSkewnessExSelf(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _idio_skewness, ex_self=True)


@_op("intra_idiosyncratic_kurtosis_ex_self", "分钟市场模型残差峰度(市场收益剔除自身)。", "level")
class IntraIdiosyncraticKurtosisExSelf(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _idio_kurtosis, ex_self=True)


@_op("intra_market_model_r2_ex_self", "分钟市场模型 R²(市场收益剔除自身)。", "r2")
class IntraMarketModelR2ExSelf(SessionAggregationOperator):
    def _calculate_series(self, close, free_market_cap, **_):
        return _beta_daily(close, free_market_cap, _market_r2, ex_self=True)


_CANONICALS.extend(
    [
        "intra_realized_beta",
        "intra_realized_correlation",
        "intra_down_down_semibeta",
        "intra_up_up_semibeta",
        "intra_down_up_semibeta",
        "intra_up_down_semibeta",
        "intra_beta_asymmetry",
        "intra_idiosyncratic_variance",
        "intra_idiosyncratic_skewness",
        "intra_idiosyncratic_kurtosis",
        "intra_market_model_r2",
        "intra_realized_beta_ex_self",
        "intra_realized_correlation_ex_self",
        "intra_idiosyncratic_variance_ex_self",
        "intra_idiosyncratic_skewness_ex_self",
        "intra_idiosyncratic_kurtosis_ex_self",
        "intra_market_model_r2_ex_self",
    ]
)

register_surface(_CANONICALS)
