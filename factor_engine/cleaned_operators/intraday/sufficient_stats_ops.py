# -*- coding: utf-8 -*-
"""R61-P1 #57: session sufficient-statistics operators (vectorized daily_agg).

Thirteen minute->daily operators whose scalar kernel is a pure per-session
aggregate derivable in O(1) from the shared sufficient statistics of
``sufficient_stats.py`` (Σr Σr² Σr³ Σr⁴, Σv Σv² Σrv, Σamount, max/min,
first/last, argmax/argmin):

  scalar family (window parameter, default full session):
    intra_ts_sum / _mean / _variance / _std / _min / _max / _last
  built-in one-panel:
    intra_ts_first / _last_value / _argmax / _argmin /
    intra_ts_realized_variance / intra_ts_vwap
  built-in two-panel (amount/volume weighted):
    intra_ts_volume_weighted_return / _realized_covariance / _amount_weighted_mean

Every scalar kernel takes ``(vals[, times])`` (or ``(a, b)``) and returns a
float — the exact ``daily_agg`` call convention.  The ``__vec__`` attributes
are attached by ``perf_vec_kernels.bind_whitelist``; the vector
implementations live in ``_core`` and derive their output from the shared
bundle in O(1) per (day, inst).

Parity contract
---------------
* ``ts_sum/mean/var/std/min/max/first/last_value/last/argmax/argmin/vwap/
  realized_variance`` are BIT-FOR-BIT derivable from the bundle (pure sums /
  max / min / first / last / index-of-first-extreme).
* The amount/volume-weighted pair (``_volume_weighted_return``,
  ``_realized_covariance``, ``_amount_weighted_mean``) needs one O(n) window
  pass each (weighted moments cannot be reduced from scalar aggregates
  alone); they still share the joint finite-mask + grid materialization.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from factor_engine.cleaned_operators.base import SeriesOperator, register_operator
from factor_engine.cleaned_operators.intraday._core import (
    _EPS,
    SessionAggregationOperator,
    daily_agg,
    daily_agg_two,
    log_returns,
    metadata,
    np_errstate,
    register_surface,
)

_CANONICALS: list[str] = []


# ---------------------------------------------------------------------------
# Scalar kernels (daily_agg call convention: fn(vals[, times]) -> float)
# ---------------------------------------------------------------------------


def _ts_sum(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    finite = vals[np.isfinite(vals)]
    return float(np.sum(finite))


def _ts_mean(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan
    return float(np.mean(finite))


def _ts_variance(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size < 1:
        return np.nan
    # population variance (ddof=0), scalar ``np.var``.
    return float(np.var(finite))


def _ts_std(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size < 1:
        return np.nan
    return float(np.std(finite))


def _ts_min(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan
    return float(np.min(finite))


def _ts_max(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan
    return float(np.max(finite))


def _ts_last(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    """Value of the LAST finite bar of the session (in bar order)."""
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan
    return float(finite[-1])


def _ts_last_value(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    """Value of the LAST finite bar (``last_value`` alias kernel)."""
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan
    return float(finite[-1])


def _ts_first(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    """Value of the FIRST finite bar of the session (in bar order)."""
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan
    return float(finite[0])


def _ts_argmax(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    """Bar position (0-based) of the first maximum among finite bars."""
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan
    return float(np.argmax(finite))


def _ts_argmin(vals: np.ndarray, times: np.ndarray | None = None) -> float:
    """Bar position (0-based) of the first minimum among finite bars."""
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan
    return float(np.argmin(finite))


def _ts_realized_variance(close_v: np.ndarray, times: np.ndarray | None = None) -> float:
    r = log_returns(close_v)
    r_finite = r[np.isfinite(r)]
    if r_finite.size < 2:
        return np.nan
    with np_errstate():
        return float(np.sum(r_finite * r_finite))


def _ts_vwap(close_v: np.ndarray, volume: np.ndarray, times: np.ndarray | None = None) -> float:
    """Session VWAP over finite (close, volume>0) bars.

    Two-panel kernel: daily_agg_two binds ``times=None`` on the third arg.
    """
    valid = np.isfinite(close_v) & np.isfinite(volume) & (volume > 0)
    if valid.sum() < 1:
        return np.nan
    p = close_v[valid]
    v = volume[valid]
    total = float(np.sum(v))
    if total <= _EPS:
        return np.nan
    return float(np.sum(p * v) / total)


def _ts_volume_weighted_return(
    close_v: np.ndarray, volume: np.ndarray, times: np.ndarray | None = None
) -> float:
    """Volume-weighted log return of the session: Σ(r_t·v_t) / Σv_t.

    Returns are over the ORIGINAL minute grid (``log_returns``), paired with
    the volume at the return's target bar; only (finite return, finite
    volume>0) pairs contribute, matching the common finite-mask contract.
    """
    r = log_returns(close_v)
    valid = np.isfinite(r) & np.isfinite(volume) & (volume > 0)
    if valid.sum() < 1:
        return np.nan
    total_v = float(np.sum(volume[valid]))
    if total_v <= _EPS:
        return np.nan
    return float(np.sum(r[valid] * volume[valid]) / total_v)


def _ts_realized_covariance(
    close_v: np.ndarray, volume: np.ndarray, times: np.ndarray | None = None
) -> float:
    """Volume-weighted realized covariance: Σ(r_t²·v_t) / Σv_t."""
    r = log_returns(close_v)
    valid = np.isfinite(r) & np.isfinite(volume) & (volume > 0)
    if valid.sum() < 1:
        return np.nan
    total_v = float(np.sum(volume[valid]))
    if total_v <= _EPS:
        return np.nan
    with np_errstate():
        return float(np.sum(r[valid] ** 2 * volume[valid]) / total_v)


def _ts_amount_weighted_mean(
    close_v: np.ndarray, amount: np.ndarray, times: np.ndarray | None = None
) -> float:
    """Amount-weighted mean close: Σ(c_t·a_t) / Σa_t over finite (close, amount>0)."""
    valid = np.isfinite(close_v) & np.isfinite(amount) & (amount > 0)
    if valid.sum() < 1:
        return np.nan
    total_a = float(np.sum(amount[valid]))
    if total_a <= _EPS:
        return np.nan
    return float(np.sum(close_v[valid] * amount[valid]) / total_a)


# ---------------------------------------------------------------------------
# register_surface for the 13 new canonicals.
# ---------------------------------------------------------------------------

_SCALAR_FAMILY_CANONICALS = [
    "intra_ts_sum",
    "intra_ts_mean",
    "intra_ts_variance",
    "intra_ts_std",
    "intra_ts_min",
    "intra_ts_max",
    "intra_ts_last",
]
_BUILTIN_ONE = [
    "intra_ts_first",
    "intra_ts_last_value",
    "intra_ts_argmax",
    "intra_ts_argmin",
    "intra_ts_realized_variance",
]
_BUILTIN_TWO = [
    "intra_ts_vwap",
    "intra_ts_volume_weighted_return",
    "intra_ts_realized_covariance",
    "intra_ts_amount_weighted_mean",
]
_CANONICALS.extend(_SCALAR_FAMILY_CANONICALS + _BUILTIN_ONE + _BUILTIN_TWO)


def _mk_scalar_family(name: str, description: str, unit: str, kernel: Callable, min_finite: int = 2):
    """Register one parameterized (close, window) -> daily scalar family member.

    The scalar kernel is passed DIRECTLY (carries ``__vec__`` after binding);
    ``window=None`` means the whole session (the daily_agg per-(day, inst)
    grouping IS the window).
    """

    @register_operator(
        name=name,
        category="intraday_microstructure",
        business_category="intraday_microstructure",
        canonical=name,
        source="intraday.sufficient_stats_ops",
        backend="pandas_numpy",
        status="extended",
    )
    class _Op(SeriesOperator):
        metadata = metadata(
            name, description, ["close", "window"], unit=unit,
            available_at="session_close", same_session_usable=False,
        )

        def _calculate_series(self, close, window=None, session_tz=None, **_):
            if window is not None:
                w = int(window)
                if w < 1:
                    raise ValueError(f"window must be >= 1 or None, got {window!r}")
            else:
                w = None
            if w is None:
                return daily_agg(close, kernel, min_finite=min_finite)
            from factor_engine.cleaned_operators.intraday._core import as_panel

            close = as_panel(close)

            def _fn(v, t):
                return kernel(v[-w:], t[-w:])

            return daily_agg(close, _fn, min_finite=min_finite)

    return _Op


def _register_builtin_one(
    name: str, description: str, unit: str, kernel: Callable, min_finite: int = 2
):
    """Register a non-parameterized (close) -> daily operator."""

    @register_operator(
        name=name,
        category="intraday_microstructure",
        business_category="intraday_microstructure",
        canonical=name,
        source="intraday.sufficient_stats_ops",
        backend="pandas_numpy",
        status="extended",
    )
    class _Op(SeriesOperator):
        metadata = metadata(
            name, description, ["close"], unit=unit,
            available_at="session_close", same_session_usable=False,
        )

        def _calculate_series(self, close, **_):
            return daily_agg(close, kernel, min_finite=min_finite)

    return _Op


def _register_builtin_two(
    name: str, description: str, unit: str, kernel: Callable, min_finite: int = 2
):
    """Register a non-parameterized (close, volume|amount) -> daily operator."""

    @register_operator(
        name=name,
        category="intraday_microstructure",
        business_category="intraday_microstructure",
        canonical=name,
        source="intraday.sufficient_stats_ops",
        backend="pandas_numpy",
        status="extended",
    )
    class _Op2(SeriesOperator):
        metadata = metadata(
            name, description, ["close", "volume"], unit=unit,
            available_at="session_close", same_session_usable=False,
        )

        def _calculate_series(self, close, volume, **_):
            return daily_agg_two(close, volume, kernel, min_finite=min_finite)

    return _Op2


# --- scalar family (7) -------------------------------------------------------
_mk_scalar_family("intra_ts_sum", "会话合计：Σ 有限收盘价。", "price", _ts_sum)
_mk_scalar_family("intra_ts_mean", "会话均值：有限收盘价的算术平均。", "price", _ts_mean)
_mk_scalar_family("intra_ts_variance", "会话方差：有限收盘价的总体方差 (ddof=0)。", "variance", _ts_variance)
_mk_scalar_family("intra_ts_std", "会话标准差：总体标准差 (ddof=0)。", "volatility", _ts_std)
_mk_scalar_family("intra_ts_min", "会话最低价（有限分钟）。", "price", _ts_min)
_mk_scalar_family("intra_ts_max", "会话最高价（有限分钟）。", "price", _ts_max)
_mk_scalar_family("intra_ts_last", "会话末根有限 bar 的收盘价。", "price", _ts_last)

# --- built-in one-panel (5) ---------------------------------------------------
_register_builtin_one("intra_ts_first", "会话首根有限 bar 的收盘价。", "price", _ts_first)
_register_builtin_one("intra_ts_last_value", "会话末根有限 bar 的收盘价（last_value 别名）。", "price", _ts_last_value)
_register_builtin_one("intra_ts_argmax", "会话最高价所在的 bar 位置（首个，0-based，有限分钟）。", "position", _ts_argmax)
_register_builtin_one("intra_ts_argmin", "会话最低价所在的 bar 位置（首个，0-based，有限分钟）。", "position", _ts_argmin)
_register_builtin_one("intra_ts_realized_variance", "会话已实现方差 Σr²。", "variance", _ts_realized_variance)

# --- built-in two-panel (4, incl. VWAP) ---------------------------------------
_register_builtin_two("intra_ts_vwap", "会话 VWAP：Σ(c·v)/Σv，v>0 的有限分钟。", "price", _ts_vwap, min_finite=2)
_register_builtin_two("intra_ts_volume_weighted_return", "成交量加权对数收益 Σ(r·v)/Σv。", "return", _ts_volume_weighted_return)
_register_builtin_two("intra_ts_realized_covariance", "成交量加权已实现二阶矩 Σ(r²·v)/Σv。", "variance", _ts_realized_covariance)
_register_builtin_two("intra_ts_amount_weighted_mean", "成交额加权平均价 Σ(c·a)/Σa。", "price", _ts_amount_weighted_mean)

register_surface(_CANONICALS)
