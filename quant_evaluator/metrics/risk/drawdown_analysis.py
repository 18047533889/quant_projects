"""
Detailed drawdown analysis for portfolio returns.

Provides drawdown series, statistics, period identification, duration metrics,
and pain indices (Ulcer Index, Pain Index).
"""

from typing import Tuple, List, Dict, Optional
import numpy as np


def drawdown_events(returns: np.ndarray) -> List[Dict]:
    """Single-pass, observation-aligned events (definition version 0.3).

    A missing return makes subsequent NAV unknown, not a flat day. Such an
    event is censored at the first valuation gap; no recovery is inferred
    across it. Durations are grid intervals, never finite-observation counts.
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("drawdown_events requires 1D returns")
    if np.any(values[np.isfinite(values)] < -1):
        raise ValueError("returns below -100% require an explicit negative-capital contract")
    events = []
    wealth = high = 1.0
    peak = -1
    event = None
    for t, value in enumerate(values):
        if not np.isfinite(value):
            if event is None:
                event = dict(peak_idx=peak, start_idx=t, trough_idx=-1,
                             drawdown=np.nan)
            event.update(recovery_idx=-1, end_idx=len(values) - 1,
                         censored=True, status="INVALID_VALUATION",
                         first_missing_idx=t, duration=len(values) - event["start_idx"],
                         valid_observations=int(np.isfinite(values[event["start_idx"]:]).sum()))
            events.append(event)
            return events
        wealth *= 1 + value
        dd = max(0., 1 - wealth / high)
        # Same exact highwater contract as maximum-drawdown peak indices.
        # A measured underwater loss cannot be relabelled as a new peak.
        if wealth >= high:
            if event is not None:
                event.update(recovery_idx=t, end_idx=t, censored=False,
                             status="RECOVERED", duration=t - event["start_idx"],
                             valid_observations=t - event["start_idx"] + 1)
                events.append(event)
                event = None
            high, peak = max(high, wealth), t
        else:
            if event is None:
                event = dict(peak_idx=peak, start_idx=t, trough_idx=t, drawdown=dd)
            if dd > event["drawdown"]:
                event.update(trough_idx=t, drawdown=dd)
    if event is not None:
        event.update(recovery_idx=-1, end_idx=len(values) - 1, censored=True,
                     status="DEFAULTED" if wealth == 0 else "ACTIVE",
                     duration=len(values) - event["start_idx"],
                     valid_observations=len(values) - event["start_idx"])
        events.append(event)
    return events


def compute_drawdown_series(
    returns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute drawdown series from returns.

    Args:
        returns: Return series (T,) or (T, F)

    Returns:
        (drawdown_series, cumulative_returns, running_max)
        drawdown_series: Drawdown at each time (negative values), shape (T,) or (T, F)
        cumulative_returns: Cumulative wealth curve, shape (T,) or (T, F)
        running_max: Running maximum wealth, shape (T,) or (T, F)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    if np.any(returns[np.isfinite(returns)] < -1.0):
        raise ValueError("returns below -100% require an explicit negative-capital contract")
    # Replace NaN with 0 for cumulative computation
    returns_filled = np.where(np.isfinite(returns), returns, 0.0)

    # Cumulative wealth curve
    cum_returns = np.cumprod(1.0 + returns_filled, axis=0)

    # Initial capital is a high-water mark too: a loss on the first
    # observation must count even before an observed wealth peak exists.
    running_max = np.maximum(1.0, np.maximum.accumulate(cum_returns, axis=0))

    # Zero NAV is an absorbing default with an observed 100% loss, not missing
    # evidence. Negative capital is rejected above before compounding.
    drawdown_series = (cum_returns - running_max) / running_max

    if squeeze:
        return drawdown_series[:, 0], cum_returns[:, 0], running_max[:, 0]
    else:
        return drawdown_series, cum_returns, running_max


def compute_drawdown_statistics(
    returns: np.ndarray,
    min_periods: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive drawdown statistics.

    Args:
        returns: Return series (T,) or (T, F)
        min_periods: Minimum periods required

    Returns:
        Dictionary with keys:
            - max_drawdown: Maximum drawdown (positive magnitude)
            - max_drawdown_idx: Index where maximum drawdown occurred
            - avg_drawdown: Average drawdown when in drawdown
            - drawdown_volatility: Standard deviation of drawdowns
            - drawdown_99: 99th percentile drawdown
            - time_underwater_pct: Percentage of time in drawdown
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape

    max_dd = np.full(F, np.nan)
    max_dd_idx = np.full(F, -1, dtype=np.int64)
    avg_dd = np.full(F, np.nan)
    dd_vol = np.full(F, np.nan)
    dd_99 = np.full(F, np.nan)
    time_underwater = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        # Compute drawdown series
        dd_series, _, _ = compute_drawdown_series(ret_f)

        # Maximum drawdown (most negative, convert to positive), NaN-safe:
        # after a wipeout the dd_series is NaN from that index onward, so
        # plain np.argmin would point at the NaN (QE-P1-28). Use the
        # first-occurrence finite-min loop from portfolio_stats.
        finite_idx = np.nonzero(np.isfinite(dd_series))[0]
        if finite_idx.size == 0:
            # All-NaN drawdown (wipeout from the very start): no finite
            # trough, max_dd stays NaN.
            max_dd[f] = np.nan
            max_dd_idx[f] = -1
        else:
            vals = dd_series[finite_idx]
            min_val = np.min(vals)
            max_dd[f] = -min_val
            max_dd_idx[f] = int(
                finite_idx[np.nonzero(vals == min_val)[0][0]]
            )

        # Average drawdown when in drawdown (negative values)
        in_drawdown = dd_series < 0  # Same exact highwater event definition
        if np.sum(in_drawdown) > 0:
            avg_dd[f] = -np.mean(dd_series[in_drawdown])
            dd_vol[f] = np.std(dd_series[in_drawdown], ddof=1) if np.sum(in_drawdown) > 1 else 0.0

        # 99th percentile drawdown (convert to positive)
        dd_99[f] = -np.percentile(dd_series, 1.0)

        # Time underwater (percentage)
        time_underwater[f] = np.sum(in_drawdown) / len(dd_series) * 100.0

    result = {
        "max_drawdown": max_dd[0] if squeeze else max_dd,
        "max_drawdown_idx": max_dd_idx[0] if squeeze else max_dd_idx,
        "avg_drawdown": avg_dd[0] if squeeze else avg_dd,
        "drawdown_volatility": dd_vol[0] if squeeze else dd_vol,
        "drawdown_99": dd_99[0] if squeeze else dd_99,
        "time_underwater_pct": time_underwater[0] if squeeze else time_underwater,
    }

    return result


def identify_drawdown_periods(
    returns: np.ndarray,
    threshold: float = 0.05,
    min_duration: int = 1,
) -> List[Dict[str, int]]:
    """
    Identify distinct drawdown periods exceeding threshold.

    Args:
        returns: Return series (T,) - single series only
        threshold: Minimum drawdown magnitude to consider (e.g., 0.05 for 5%)
        min_duration: Minimum duration in periods

    Returns:
        List of drawdown periods, each with:
            - peak_idx: Index of peak before drawdown (-1 for initial capital)
            - trough_idx: Index of trough (maximum drawdown)
            - recovery_idx: Index of recovery (NaN if not recovered)
            - duration: Duration from peak to recovery
            - drawdown: Maximum drawdown magnitude
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    return [
        event for event in drawdown_events(returns)
        if event["duration"] >= min_duration
        and (event["status"] == "INVALID_VALUATION" or event["drawdown"] > threshold)
    ]


def compute_drawdown_duration(
    returns: np.ndarray,
    min_periods: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute drawdown duration statistics.

    Args:
        returns: Return series (T,) or (T, F)
        min_periods: Minimum periods required

    Returns:
        Dictionary with:
            - max_drawdown_duration: Longest drawdown period
            - avg_drawdown_duration: Average drawdown duration
            - current_drawdown_duration: Current drawdown length (0 if recovered)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape

    max_duration = np.full(F, np.nan)
    avg_duration = np.full(F, np.nan)
    current_duration = np.full(F, 0.0)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        # Identify periods
        periods = identify_drawdown_periods(ret_f, threshold=0.0, min_duration=1)
        if any(p["status"] == "INVALID_VALUATION" for p in periods):
            current_duration[f] = np.nan
            continue

        if len(periods) == 0:
            max_duration[f] = 0.0
            avg_duration[f] = 0.0
            current_duration[f] = 0.0
            continue

        # Extract durations
        durations = [p["duration"] for p in periods]
        max_duration[f] = np.max(durations)
        avg_duration[f] = np.mean(durations)

        # Check if currently in drawdown
        if periods[-1]["recovery_idx"] == -1:
            current_duration[f] = periods[-1]["duration"]
        else:
            current_duration[f] = 0.0

    result = {
        "max_drawdown_duration": max_duration[0] if squeeze else max_duration,
        "avg_drawdown_duration": avg_duration[0] if squeeze else avg_duration,
        "current_drawdown_duration": current_duration[0] if squeeze else current_duration,
    }

    return result


def compute_ulcer_index(
    returns: np.ndarray,
    min_periods: int = 10,
) -> np.ndarray:
    """
    Compute Ulcer Index (square root of mean squared drawdown).

    Measures the depth and duration of drawdowns.

    Args:
        returns: Return series (T,) or (T, F)
        min_periods: Minimum periods required

    Returns:
        Ulcer Index, scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    ulcer = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        dd_series, _, _ = compute_drawdown_series(ret_f)

        # Convert to percentage drawdowns (negative values)
        dd_pct = dd_series * 100.0

        # Ulcer Index = sqrt(mean(drawdown^2))
        ulcer[f] = np.sqrt(np.mean(dd_pct ** 2))

    return ulcer[0] if squeeze else ulcer


def compute_pain_index(
    returns: np.ndarray,
    min_periods: int = 10,
) -> np.ndarray:
    """
    Compute Pain Index (mean absolute drawdown).

    Similar to Ulcer Index but uses mean instead of RMS.

    Args:
        returns: Return series (T,) or (T, F)
        min_periods: Minimum periods required

    Returns:
        Pain Index, scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    pain = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        dd_series, _, _ = compute_drawdown_series(ret_f)

        # Pain Index = mean(abs(drawdown)) in percentage
        pain[f] = np.mean(np.abs(dd_series)) * 100.0

    return pain[0] if squeeze else pain
