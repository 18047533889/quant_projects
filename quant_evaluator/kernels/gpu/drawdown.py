"""Batched GPU portfolio risk metrics (spec §19, Wave 3).

Vectorized over the factor axis F (no ``for f in range(F)``).  Each metric
matches the CPU reference in ``metrics/portfolio_stats.py`` exactly:

  - annualized return / volatility  (probe_portfolio.sharpe)
  - Sharpe / Sortino / Calmar        (portfolio_stats)
  - max drawdown with wealth<=0 wipeout guard (portfolio_stats)
  - win rate                          (portfolio_stats)
  - positive month ratio              (probe_portfolio.sharpe)

CuPy lacks ``maximum.accumulate``; a Hillis-Steele running-max is used for
the drawdown running maximum (bitwise-identical to ``np.maximum.accumulate``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

EPS = 1e-12


def _import_cp():
    import cupy as cp
    return cp


def _running_max(x, axis: int = 0):
    """Hillis-Steele inclusive running maximum along ``axis`` (float64)."""
    cp = _import_cp()
    y = cp.asarray(x, dtype=cp.float64)
    step = 1
    while step < y.shape[axis]:
        prefix = cp.full_like(y, -cp.inf)
        sl_src = [slice(None)] * y.ndim
        sl_src[axis] = slice(None, -step)
        sl_dst = [slice(None)] * y.ndim
        sl_dst[axis] = slice(step, None)
        prefix[tuple(sl_dst)] = y[tuple(sl_src)]
        y = cp.maximum(y, prefix)
        step *= 2
    return y


def _gather_valid_first(returns):
    """Return (sorted_ret, n_valid) where valid entries are moved to the front.

    ``sorted_ret`` is (T, F): for each column the finite returns appear first
    (in original order), followed by the non-finite entries (as 0).  ``n_valid``
    is (F,).
    """
    cp = _import_cp()
    ret = cp.asarray(returns, dtype=cp.float64)
    if ret.ndim == 1:
        ret = ret[:, None]
    valid = cp.isfinite(ret)
    n_valid = cp.sum(valid, axis=0).astype(cp.int64)  # (F,)
    order = cp.argsort(~valid, axis=0, kind="stable")  # valid first
    sorted_ret = cp.take_along_axis(cp.where(valid, ret, 0.0), order, axis=0)
    return sorted_ret, n_valid


def compute_annualized_return_batch(returns, periods_per_year: int = 252):
    """(T, F) -> (F,) annualized compounded return (CPU parity)."""
    cp = _import_cp()
    sorted_ret, n_valid = _gather_valid_first(returns)
    T, F = sorted_ret.shape
    rows = cp.arange(T)[:, None]
    keep = rows < n_valid[None, :]  # (T, F)
    total = cp.prod(cp.where(keep, 1.0 + sorted_ret, 1.0), axis=0)  # (F,)
    ann = cp.where(total > 0.0, total ** (periods_per_year / n_valid) - 1.0, cp.nan)
    # Bankruptcy is path-absorbing.  Two returns below -100% can make the raw
    # product positive again, but they cannot restore exhausted capital.
    wipeout = cp.any(keep & (sorted_ret <= -1.0), axis=0)
    ann = cp.where(wipeout, -1.0, ann)
    ann = cp.where(n_valid >= 2, ann, cp.nan)
    return ann


def compute_annualized_volatility_batch(returns, periods_per_year: int = 252):
    """(T, F) -> (F,) annualized volatility (CPU parity)."""
    cp = _import_cp()
    ret = cp.asarray(returns, dtype=cp.float64)
    if ret.ndim == 1:
        ret = ret[:, None]
    n_valid = cp.sum(cp.isfinite(ret), axis=0)
    # CPU authority drops every non-finite value, not only NaN.  Convert both
    # infinities to NaN before the reduction so n_valid and the sample agree.
    finite_ret = cp.where(cp.isfinite(ret), ret, cp.nan)
    std = cp.nanstd(finite_ret, axis=0, ddof=1)  # (F,)
    vol = cp.where(std <= EPS, 0.0, std * cp.sqrt(periods_per_year))
    vol = cp.where(n_valid >= 2, vol, cp.nan)
    return vol


def compute_sharpe_batch(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 20,
):
    """(T, F) -> (F,) annualized Sharpe (CPU parity)."""
    cp = _import_cp()
    sorted_ret, n_valid = _gather_valid_first(returns)
    T, F = sorted_ret.shape
    rows = cp.arange(T)[:, None]
    keep = rows < n_valid[None, :]
    ret_valid = cp.where(keep, sorted_ret, 0.0)  # (T, F)
    rf_per = risk_free_rate / periods_per_year
    excess = ret_valid - rf_per
    mean_ex = cp.sum(excess, axis=0) / cp.maximum(n_valid, 1)
    # std ddof=1 over the valid subset
    dev = excess - mean_ex[None, :]
    var = cp.sum(cp.where(keep, dev * dev, 0.0), axis=0) / cp.maximum(n_valid - 1, 1)
    std = cp.sqrt(var)
    sharpe = mean_ex / std * cp.sqrt(periods_per_year)
    bad = (n_valid < min_periods) | (~cp.isfinite(std)) | (std <= 1e-10)
    sharpe = cp.where(bad, cp.nan, sharpe)
    return sharpe


def compute_sortino_batch(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 20,
):
    """(T, F) -> (F,) annualized Sortino (CPU parity)."""
    cp = _import_cp()
    sorted_ret, n_valid = _gather_valid_first(returns)
    T, F = sorted_ret.shape
    rows = cp.arange(T)[:, None]
    keep = rows < n_valid[None, :]
    ret_valid = cp.where(keep, sorted_ret, 0.0)
    rf_per = risk_free_rate / periods_per_year
    excess = ret_valid - rf_per
    mean_ex = cp.sum(excess, axis=0) / cp.maximum(n_valid, 1)
    downside = cp.where((excess < 0) & keep, excess, 0.0)
    n_down = cp.sum((excess < 0) & keep, axis=0)
    downside_std = cp.sqrt(cp.sum(downside * downside, axis=0) / cp.maximum(n_down, 1))
    sortino = mean_ex / downside_std * cp.sqrt(periods_per_year)
    bad = (
        (n_valid < min_periods)
        | (n_down == 0)
        | (~cp.isfinite(downside_std))
        | (downside_std <= 1e-12)
    )
    sortino = cp.where(bad, cp.nan, sortino)
    return sortino


def compute_max_drawdown_batch(returns):
    """(T, F) -> (F,) max drawdown magnitude (positive), CPU parity.

    Includes the wealth<=0 wipeout guard: from the first nonpositive wealth
    onward the drawdown is -1 (total loss).  A negative wealth times (1+r)
    must not flip positive and fabricate a recovery.
    """
    cp = _import_cp()
    ret = cp.asarray(returns, dtype=cp.float64)
    if ret.ndim == 1:
        ret = ret[:, None]
    filled = cp.where(cp.isfinite(ret), ret, 0.0)
    cum = cp.cumprod(1.0 + filled, axis=0)  # (T, F)
    # Match the CPU high-water mark, including capital before the first return.
    running_max = cp.maximum(1.0, _running_max(cum, axis=0))
    invalid = _running_max((cum <= 0).astype(cp.float64), axis=0) > 0.0
    dd = cp.where(
        ~invalid,
        (cum - running_max) / running_max,
        -1.0,
    )
    max_dd = -cp.nanmin(dd, axis=0)
    max_dd = cp.where(cp.isfinite(max_dd), max_dd, cp.nan)
    return max_dd


def compute_calmar_batch(
    returns,
    periods_per_year: int = 252,
    min_periods: int = 20,
):
    """(T, F) -> (F,) Calmar ratio (CPU parity)."""
    cp = _import_cp()
    _, n_valid = _gather_valid_first(returns)
    # Calmar's numerator is CAGR, not arithmetic mean annualization.  Reuse
    # the batch compound authority so missing-value compaction and nonpositive
    # terminal wealth semantics stay identical to the CPU contract.
    ann_ret = compute_annualized_return_batch(returns, periods_per_year)
    max_dd = compute_max_drawdown_batch(returns)
    calmar = ann_ret / max_dd
    bad = (n_valid < min_periods) | (~cp.isfinite(max_dd)) | (max_dd <= 1e-12)
    calmar = cp.where(bad, cp.nan, calmar)
    return calmar


def compute_win_rate_batch(returns):
    """(T, F) -> (F,) win rate in [0, 1] (CPU parity)."""
    cp = _import_cp()
    ret = cp.asarray(returns, dtype=cp.float64)
    if ret.ndim == 1:
        ret = ret[:, None]
    valid = cp.isfinite(ret)
    n_valid = cp.sum(valid, axis=0)
    wins = cp.sum((ret > 0) & valid, axis=0)
    win_rate = wins / cp.maximum(n_valid, 1)
    win_rate = cp.where(n_valid > 0, win_rate, cp.nan)
    return win_rate


def compute_positive_month_ratio_batch(returns, periods_per_month: int = 21):
    """(T, F) -> (F,) positive month ratio (CPU parity)."""
    cp = _import_cp()
    sorted_ret, n_valid = _gather_valid_first(returns)
    T, F = sorted_ret.shape
    M = n_valid // periods_per_month  # (F,) full months
    M_max = int(cp.max(M)) if F else 0
    if M_max < 1:
        return cp.full(F, cp.nan)
    idx = cp.arange(M_max * periods_per_month)[:, None]  # (M_max*ppm, 1)
    col_mask = idx < (M * periods_per_month)[None, :]  # (M_max*ppm, F)
    vals = cp.where(col_mask, sorted_ret[: M_max * periods_per_month], 1.0)
    vals = vals.reshape(M_max, periods_per_month, F)
    month_prod = cp.prod(1.0 + vals, axis=1) - 1.0  # (M_max, F)
    month_win = month_prod > 0.0
    valid_month = cp.arange(M_max)[:, None] < M[None, :]  # (M_max, F)
    pmr = cp.sum(month_win & valid_month, axis=0) / cp.maximum(M, 1)
    pmr = cp.where(M > 0, pmr, cp.nan)
    return pmr


def compute_portfolio_metrics_batch(
    returns,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    min_periods: int = 20,
    periods_per_month: int = 21,
) -> dict:
    """(T, F) -> dict of (F,) risk metrics (CPU parity)."""
    return {
        "annualized_return": compute_annualized_return_batch(returns, periods_per_year),
        "annualized_volatility": compute_annualized_volatility_batch(returns, periods_per_year),
        "sharpe": compute_sharpe_batch(returns, risk_free_rate, periods_per_year, min_periods),
        "sortino": compute_sortino_batch(returns, risk_free_rate, periods_per_year, min_periods),
        "calmar": compute_calmar_batch(returns, periods_per_year, min_periods),
        "max_drawdown": compute_max_drawdown_batch(returns),
        "win_rate": compute_win_rate_batch(returns),
        "positive_month_ratio": compute_positive_month_ratio_batch(returns, periods_per_month),
    }
