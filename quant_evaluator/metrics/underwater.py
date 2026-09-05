"""
Underwater / drawdown-extent evidence kernels (R61-FI-023, plan §13.9).

These kernels consume a *probe* daily PnL / return series (T,) — the output
of :func:`quant_evaluator.metrics.probe_portfolio.compute_cohort_pnl`
(``pnl_net`` / ``active_ret``) or any cleaned dot-frequency series.  They
extend the existing drawdown family (already registered in ``registry/metrics.py``:

- ``portfolio_stats.compute_maximum_drawdown`` — max drawdown magnitude +
  drawdown series + peak index (the drawdown authority);
- ``risk/drawdown_analysis.compute_drawdown_statistics`` — avg drawdown,
  ``time_underwater_pct``, dd_vol, dd_99;
- ``risk/drawdown_analysis.compute_drawdown_duration`` — max/avg/current
  drawdown duration;
- ``probe_portfolio/sharpe.compute_portfolio_metrics`` — the daily-PnL family
  used in production report cards.

The new kernels add the *extent* semantics the production plan requires:
max/mean underwater duration, time-to-recovery, calendar worst-period
returns, rolling-1y Sharpe tail stats, return skew, downside deviation and
CVaR expected shortfall.  Every kernel follows the same fail-closed
convention: when there are not enough finite periods the value is NaN —
never a fabricated 0.0.

Units / conventions:
    - ``returns`` is a (T,) dot-frequency PnL series (fractions, not bp).
    - Durations are counted in *periods* (trading days), consistent with the
      existing ``drawdown_duration`` registry metric.  Calendar aggregation
      for ``worst_month`` / ``worst_quarter`` / ``worst_12m`` uses the fixed
      trading-period calendar that the codebase already uses everywhere else
      (252 per year / 63 per quarter / 21 per month).
    - Downside measures follow `risk/var_cvar.py` (positive loss magnitude,
      0.0 when no period breaches the threshold); ``cvar_expected_shortfall``
      is the historical (non-parametric) expected shortfall at 95%.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "compute_max_underwater_duration",
    "compute_mean_underwater_duration",
    "compute_time_to_recovery",
    "compute_worst_period_return",
    "compute_rolling_sharpe_tail",
    "compute_return_skew",
    "compute_downside_deviation",
    "compute_cvar_expected_shortfall",
    "compute_underwater_evidence",
]

EPS = 1e-12

#: Fixed trading-period calendar used by the codebase for annualization /
#: calendar aggregation.  Kept as module constants (single authority) so the
#: calendar-worst kernels never invent their own lookback lengths.
_PERIODS_PER_MONTH = 21
_PERIODS_PER_QUARTER = 63
_PERIODS_PER_YEAR = 252


def _as_1d(returns: object, name: str = "returns") -> np.ndarray:
    """Coerce to a finite-filtered 1-D float64 array (fail closed on ndim)."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional series, got ndim={arr.ndim}")
    return arr[np.isfinite(arr)]


def _drawdown_curve(ret: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (drawdown_series, running_max) over the *compounded* wealth curve.

    Reuses the same wealth-curve semantics as
    ``portfolio_stats.compute_maximum_drawdown`` (zero-fill missing handled by
    the caller's finite pre-filter; wealth<=0 wipeout guard: from the first
    nonpositive wealth onward the drawdown is NaN so a negative wealth cannot
    fabricate a fake recovery).
    """
    cum = np.cumprod(1.0 + ret)
    running_max = np.maximum.accumulate(cum)
    invalid = np.maximum.accumulate(cum <= 0).astype(bool)
    dd = np.full(cum.shape, np.nan)
    np.divide(cum - running_max, running_max, out=dd, where=~invalid)
    return dd, running_max


def _drawdown_episodes(returns: np.ndarray) -> np.ndarray:
    """Return a boolean array marking periods that are *in* a drawdown.

    A period is underwater when wealth is below its running maximum beyond a
    tiny floating-point tolerance.  ``NaN`` drawdown (post-wipeout) counts as
    in-drawdown (staying underwater).
    """
    dd, _ = _drawdown_curve(returns)
    in_dd = np.isnan(dd) | (dd < -EPS)
    return in_dd


def compute_max_underwater_duration(returns: np.ndarray, min_periods: int = 10) -> float:
    """Longest continuous stretch of being underwater (periods).

    An underwater episode is a maximal run of "wealth below running max" (or
    post-wipeout NaN).  Returns NaN when fewer than ``min_periods`` finite
    periods exist, 0.0 when the series is never underwater.

    Like the existing ``drawdown_duration`` registry metric, duration is in
    periods (trading days) of the *compounded* wealth curve.
    """
    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    in_dd = _drawdown_episodes(ret)
    if not np.any(in_dd):
        return 0.0
    if np.all(in_dd):
        return float(ret.size)
    # Longest run of True in the boolean mask.
    runs = np.diff(np.flatnonzero(np.concatenate(([False], in_dd, [False]))))
    # The runs array alternates; odd-length entries are the True runs.  Simpler:
    # count consecutive Trues.
    best = 0
    cur = 0
    for flag in in_dd:
        cur = cur + 1 if flag else 0
        if cur > best:
            best = cur
    return float(best)


def compute_mean_underwater_duration(returns: np.ndarray, min_periods: int = 10) -> float:
    """Mean length of underwater episodes, in periods.

    Returns NaN when fewer than ``min_periods`` finite periods exist, 0.0 when
    the series is never underwater.
    """
    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    in_dd = _drawdown_episodes(ret)
    if not np.any(in_dd):
        return 0.0
    lengths: list[int] = []
    cur = 0
    for flag in in_dd:
        if flag:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    return float(np.mean(lengths))


def compute_time_to_recovery(
    returns: np.ndarray,
    min_periods: int = 10,
    max_recovery_lookback: Optional[int] = None,
) -> float:
    """Mean time (periods) from an underwater episode's trough back to a new high.

    Recovery time is measured from the trough index to the first subsequent
    index where wealth returns to (or exceeds) the running maximum that
    preceded the episode.  For ongoing drawdowns with no recovery yet, only
    *completed* recoveries are counted (episodes still underwater contribute
    nothing — they are reported separately via ``max_underwater_duration``);
    when *every* episode is ongoing, the value is the observed lookback (the
    length of the ongoing episode), reflecting "not yet recovered".

    Returns NaN when fewer than ``min_periods`` finite periods exist or when
    the series is never underwater.  ``max_recovery_lookback`` caps the scan
    (defaults to the series length).
    """
    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    dd, running_max = _drawdown_curve(ret)
    cum = np.cumprod(1.0 + ret)
    n = ret.size
    lookback = n if max_recovery_lookback is None else int(max_recovery_lookback)
    in_dd = np.isnan(dd) | (dd < -EPS)

    recovered: list[float] = []
    ongoing_len: list[float] = []
    i = 0
    while i < n:
        if not in_dd[i]:
            i += 1
            continue
        # Find episode start (peak = last pre-episode index with wealth == running max).
        start = i
        while start > 0 and in_dd[start - 1]:
            start -= 1
        # Trough within [start, ...]
        finite = np.isfinite(dd)
        search = np.arange(start, min(n, start + lookback))
        dd_vals = dd[search]
        ok = finite[search]
        if not np.any(ok):
            ongoing_len.append(float(len(search)))
            break
        trough_idx = int(search[ok][int(np.argmin(dd_vals[ok]))])
        peak_value = running_max[max(start - 1, 0)]
        # Recovery = first index >= trough+1 with cum >= peak_value (finite).
        rec = None
        for j in range(trough_idx + 1, min(n, trough_idx + 1 + lookback)):
            if np.isfinite(cum[j]) and cum[j] >= peak_value:
                rec = j
                break
        if rec is not None:
            recovered.append(float(max(rec - trough_idx, 1)))
            i = rec + 1  # restart search after the recovery point
        else:
            # Episode never recovered within lookback — jump past it.
            k = trough_idx + 1
            while k < n and in_dd[k]:
                k += 1
            ongoing_len.append(float(k - trough_idx))
            i = k
    if recovered:
        return float(np.mean(recovered))
    if ongoing_len:
        return float(np.mean(ongoing_len))
    return 0.0


def compute_worst_period_return(
    returns: np.ndarray,
    period: str = "month",
    min_periods: int = 10,
) -> float:
    """Calendar worst block return: the minimum compounded return over any
    fixed-length trading-period block (``month``=21 / ``quarter``=63 /
    ``year``=252 periods).

    Returns NaN when fewer than ``min_periods`` finite periods exist, or when
    the series has fewer than one full block.

    Conventions mirror the existing codebase calendar: binary backtest /
    report cards fixed period lengths (21 / 63 / 252), no calendar-month
    index dependence — a pure mathematical "worst 21-period block" over the
    dot-frequency PnL series (kept calendar-window-agnostic because the probe
    PnL series carries only a dot index, not calendar dates).
    """
    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    if period == "month":
        block = _PERIODS_PER_MONTH
    elif period == "quarter":
        block = _PERIODS_PER_QUARTER
    elif period == "year":
        block = _PERIODS_PER_YEAR
    else:
        raise ValueError(
            f"period must be one of 'month'|'quarter'|'year', got {period!r}"
        )
    if ret.size < block:
        return np.nan
    # Compounded block returns: prod(1+r) over each length-block window.
    blocks = np.full(ret.size - block + 1, np.nan)
    for t in range(ret.size - block + 1):
        blocks[t] = np.prod(1.0 + ret[t : t + block]) - 1.0
    return float(np.min(blocks))


def compute_rolling_sharpe_tail(
    returns: np.ndarray,
    window: int = _PERIODS_PER_YEAR,
    quantile: float = 0.10,
    min_periods: int = 30,
    periods_per_year: int = _PERIODS_PER_YEAR,
) -> float:
    """Rolling-window annualized Sharpe tail statistic.

    Computes the rolling-window Sharpe (``quant_evaluator.metrics.
    portfolio_stats.compute_sharpe_ratio``) over every aligned window with at
    least ``min_periods`` finite returns, then returns the requested quantile:

    - ``quantile=0.0``   -> ``rolling_1y_sharpe_min`` (minimum attained);
    - ``quantile=0.10``  -> ``rolling_1y_sharpe_q10``.

    NaN when there are no valid rolling windows.
    """
    from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio

    ret = _as_1d(returns)
    if ret.size < window:
        return np.nan
    roll: list[float] = []
    for t in range(window - 1, ret.size):
        seg = ret[t - window + 1 : t + 1]
        if np.sum(np.isfinite(seg)) >= min_periods:
            val = float(
                compute_sharpe_ratio(
                    seg,
                    risk_free_rate=0.0,
                    periods_per_year=periods_per_year,
                    min_periods=min_periods,
                )
            )
            if np.isfinite(val):
                roll.append(val)
    if not roll:
        return np.nan
    if quantile == 0.0:
        return float(min(roll))
    return float(np.quantile(roll, quantile))


def compute_return_skew(returns: np.ndarray, min_periods: int = 20) -> float:
    """Sample skewness of the return series (population ``scipy.stats.skew``
    convention, bias=False; for the tail-risk registry family the existing
    ``metrics/distribution.compute_skewness`` is the same statistic).

    NaN when fewer than ``min_periods`` finite returns or when std is 0.
    """
    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    mu = np.mean(ret)
    std = np.std(ret, ddof=0)
    if std <= EPS:
        return np.nan
    return float(np.mean(((ret - mu) / std) ** 3))


def compute_downside_deviation(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _PERIODS_PER_YEAR,
    min_periods: int = 20,
) -> float:
    """Annualized downside deviation (semicovariance) of the return series.

    Definition matches the existing ``portfolio_stats.compute_sortino_ratio``
    family: RMS of *negative excess* returns only (no ddof), annualized by
    sqrt(periods_per_year).  NaN when there are fewer than ``min_periods``
    finite returns or no negative excess returns.
    """
    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    rf_per = risk_free_rate / periods_per_year
    excess = ret - rf_per
    downside = excess[excess < 0.0]
    if downside.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year))


def compute_cvar_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    min_periods: int = 20,
) -> float:
    """Historical CVaR / expected shortfall at ``confidence_level`` (95%).

    Reuses the existing ``risk/var_cvar.compute_cvar`` historical method; a
    wrapper so the evidence output has its own registered id while the
    numeric semantics stay the single-source policy of ``var_cvar.py``.
    Returns positive loss magnitude; 0.0 when no returns fall at/below the
    VaR threshold; NaN when insufficient observations.
    """
    from quant_evaluator.metrics.risk.var_cvar import compute_cvar

    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    val = compute_cvar(
        ret,
        confidence_level=confidence_level,
        method="historical",
        min_periods=min_periods,
    )
    return float(val)


def compute_underwater_evidence(
    returns: np.ndarray,
    min_periods: int = 20,
    worst_quarter_period: str = "quarter",
    rolling_window: int = _PERIODS_PER_YEAR,
    rolling_min_periods: int = 60,
) -> dict:
    """Bundle all underwater/drawdown-extent evidence for one dot-frequency
    PnL series into a plain dict of scalars (all NaN-safe).

    Keys (mirror the registry metric ids in ``registry/metrics.py``):
        max_drawdown / max_underwater_duration / mean_underwater_duration /
        time_to_recovery / worst_month / worst_quarter / worst_12m /
        rolling_1y_sharpe_min / rolling_1y_sharpe_q10 / return_skew /
        downside_deviation / cvar_expected_shortfall
    """
    from quant_evaluator.metrics.portfolio_stats import compute_maximum_drawdown

    ret = _as_1d(returns)
    if ret.size < min_periods:
        return {
            "max_drawdown": np.nan,
            "max_underwater_duration": np.nan,
            "mean_underwater_duration": np.nan,
            "time_to_recovery": np.nan,
            "worst_month": np.nan,
            "worst_quarter": np.nan,
            "worst_12m": np.nan,
            "rolling_1y_sharpe_min": np.nan,
            "rolling_1y_sharpe_q10": np.nan,
            "return_skew": np.nan,
            "downside_deviation": np.nan,
            "cvar_expected_shortfall": np.nan,
        }
    return {
        "max_drawdown": float(
            compute_maximum_drawdown(ret, missing_return_policy="zero_fill")[0]
        ),
        "max_underwater_duration": compute_max_underwater_duration(ret, min_periods),
        "mean_underwater_duration": compute_mean_underwater_duration(ret, min_periods),
        "time_to_recovery": compute_time_to_recovery(ret),
        "worst_month": compute_worst_period_return(ret, "month", min_periods),
        "worst_quarter": compute_worst_period_return(ret, "quarter", min_periods),
        "worst_12m": compute_worst_period_return(ret, "year", min_periods),
        "rolling_1y_sharpe_min": compute_rolling_sharpe_tail(
            ret, window=rolling_window, quantile=0.0,
            min_periods=rolling_min_periods,
        ),
        "rolling_1y_sharpe_q10": compute_rolling_sharpe_tail(
            ret, window=rolling_window, quantile=0.10,
            min_periods=rolling_min_periods,
        ),
        "return_skew": compute_return_skew(ret, min_periods),
        "downside_deviation": compute_downside_deviation(ret),
        "cvar_expected_shortfall": compute_cvar_expected_shortfall(ret),
    }