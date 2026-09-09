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
    """Coerce without deleting positions from the original observation grid."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional series, got ndim={arr.ndim}")
    return arr


def _path_events(returns, min_periods):
    from .risk.drawdown_analysis import drawdown_events
    ret = _as_1d(returns)
    if np.isfinite(ret).sum() < min_periods:
        return None
    events = drawdown_events(ret)
    # Scalar APIs cannot express an interval estimate or unknown NAV path.
    # Fail closed, while drawdown_events retains aligned censoring evidence.
    if any(e["status"] == "INVALID_VALUATION" for e in events):
        return None
    return events


def compute_max_underwater_duration(returns: np.ndarray, min_periods: int = 10) -> float:
    """Longest underwater grid span; unknown valuation paths return NaN."""
    events = _path_events(returns, min_periods)
    if events is None:
        return np.nan
    return float(max((e["duration"] for e in events), default=0))


def compute_mean_underwater_duration(returns: np.ndarray, min_periods: int = 10) -> float:
    """Mean observed underwater span, including explicitly censored events."""
    events = _path_events(returns, min_periods)
    if events is None:
        return np.nan
    return float(np.mean([e["duration"] for e in events])) if events else 0.0


def compute_time_to_recovery(
    returns: np.ndarray,
    min_periods: int = 10,
    max_recovery_lookback: Optional[int] = None,
) -> float:
    """Mean trough-to-recovery intervals of completed events only.

    Censored age is available separately in drawdown_events; it never becomes
    a measured recovery. The optional limit is applied per completed event.
    """
    if max_recovery_lookback is not None and (
        isinstance(max_recovery_lookback, bool)
        or not isinstance(max_recovery_lookback, (int, np.integer))
        or max_recovery_lookback < 1
    ):
        raise ValueError("max_recovery_lookback must be a positive integer")
    events = _path_events(returns, min_periods)
    if events is None:
        return np.nan
    recovered = [
        e["recovery_idx"] - e["trough_idx"] for e in events
        if not e["censored"] and (
            max_recovery_lookback is None
            or e["recovery_idx"] - e["trough_idx"] <= max_recovery_lookback
        )
    ]
    return float(np.mean(recovered)) if recovered else np.nan


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


def compute_return_skew(returns: np.ndarray, min_periods: int = 20, *, bias: bool = False) -> float:
    """Sample skewness of the return series (population ``scipy.stats.skew``
    convention, bias=False; for the tail-risk registry family the existing
    ``metrics/distribution.compute_skewness`` is the same statistic).

    NaN when fewer than ``min_periods`` finite returns or when std is 0.
    """
    ret = _as_1d(returns)
    ret = ret[np.isfinite(ret)]
    if ret.size < max(min_periods, 3):
        return np.nan
    mu = np.mean(ret)
    std = np.std(ret, ddof=0)
    if std <= EPS:
        return np.nan
    skew = float(np.mean(((ret - mu) / std) ** 3))
    return skew if bias else float(np.sqrt(ret.size * (ret.size - 1)) / (ret.size - 2) * skew)


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
    ret = ret[np.isfinite(ret)]
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
