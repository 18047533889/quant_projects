"""Long-only benchmark-relative metrics over an execution trajectory leg."""
from __future__ import annotations

import numpy as np
from quant_evaluator.metrics.portfolio_stats import compute_maximum_drawdown


def _matrix(returns):
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim == 1:
        return values[:, None], True
    if values.ndim != 2:
        raise ValueError("returns must be (T,) or (T,F)")
    return values, False


def compute_tracking_error(returns, periods_per_year: int = 252, min_periods: int = 2):
    """Annualized sample standard deviation of net active returns."""
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 2:
        raise ValueError("min_periods must be an integer >= 2")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    x, squeeze = _matrix(returns)
    out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        v = x[np.isfinite(x[:, f]), f]
        if len(v) >= min_periods:
            out[f] = np.std(v, ddof=1) * np.sqrt(periods_per_year)
    return out[0] if squeeze else out


def compute_information_ratio(returns, periods_per_year: int = 252, min_periods: int = 2):
    """Annualized mean(active)/sample-std(active); zero risk is undefined."""
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 2:
        raise ValueError("min_periods must be an integer >= 2")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    x, squeeze = _matrix(returns)
    out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        v = x[np.isfinite(x[:, f]), f]
        if len(v) >= min_periods:
            scale = np.std(v, ddof=1)
            if np.isfinite(scale) and scale > 0:
                out[f] = np.mean(v) / scale * np.sqrt(periods_per_year)
    return out[0] if squeeze else out


def compute_relative_max_drawdown(returns, min_periods: int = 1):
    """Positive drawdown magnitude from relative-wealth return increments."""
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 1:
        raise ValueError("min_periods must be an integer >= 1")
    x, squeeze = _matrix(returns); out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        v = x[:, f]
        if np.count_nonzero(np.isfinite(v)) >= min_periods:
            out[f] = np.asarray(compute_maximum_drawdown(v, missing_return_policy="unknown")[0]).item()
    return out[0] if squeeze else out


def compute_mean_investment_fraction(returns, min_periods: int = 1):
    """Mean actual invested-capital fraction; all-cash is explicitly 0, not missing."""
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 1:
        raise ValueError("min_periods must be an integer >= 1")
    x, squeeze = _matrix(returns)
    out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        v = x[np.isfinite(x[:, f]), f]
        if len(v) >= min_periods:
            if np.any(v < 0):
                raise ValueError("investment fraction must be nonnegative")
            out[f] = np.mean(v)
    return out[0] if squeeze else out


__all__ = ["compute_tracking_error", "compute_information_ratio",
           "compute_relative_max_drawdown", "compute_mean_investment_fraction"]
