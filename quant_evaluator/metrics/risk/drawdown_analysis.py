"""
Detailed drawdown analysis for portfolio returns.

Provides drawdown series, statistics, period identification, duration metrics,
and pain indices (Ulcer Index, Pain Index).
"""

from typing import Tuple, List, Dict, Optional
import numpy as np


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

    # Replace NaN with 0 for cumulative computation
    returns_filled = np.where(np.isfinite(returns), returns, 0.0)

    # Cumulative wealth curve
    cum_returns = np.cumprod(1.0 + returns_filled, axis=0)

    # Running maximum
    running_max = np.maximum.accumulate(cum_returns, axis=0)

    # Drawdown (negative values)
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

        # Maximum drawdown (most negative, convert to positive)
        max_dd[f] = -np.min(dd_series)
        max_dd_idx[f] = np.argmin(dd_series)

        # Average drawdown when in drawdown (negative values)
        in_drawdown = dd_series < -1e-10  # Small threshold to avoid floating point issues
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
            - peak_idx: Index of peak before drawdown
            - trough_idx: Index of trough (maximum drawdown)
            - recovery_idx: Index of recovery (NaN if not recovered)
            - duration: Duration from peak to recovery
            - drawdown: Maximum drawdown magnitude
    """
    if returns.ndim != 1:
        raise ValueError("identify_drawdown_periods requires 1D returns")

    dd_series, cum_returns, running_max = compute_drawdown_series(returns)

    periods = []
    in_drawdown = False
    peak_idx = 0

    for t in range(len(dd_series)):
        if not in_drawdown:
            # Check if entering drawdown
            if dd_series[t] < -threshold:
                in_drawdown = True
                peak_idx = t - 1 if t > 0 else 0
                # Find actual peak by searching backward
                for i in range(t - 1, -1, -1):
                    if cum_returns[i] >= running_max[t]:
                        peak_idx = i
                        break
        else:
            # Check if recovered (back to running max)
            if abs(dd_series[t]) < 1e-10:
                # Find trough in this period
                trough_idx = peak_idx
                max_dd_in_period = 0.0
                for i in range(peak_idx, t + 1):
                    if dd_series[i] < -max_dd_in_period:
                        max_dd_in_period = -dd_series[i]
                        trough_idx = i

                duration = t - peak_idx + 1

                if duration >= min_duration:
                    periods.append({
                        "peak_idx": peak_idx,
                        "trough_idx": trough_idx,
                        "recovery_idx": t,
                        "duration": duration,
                        "drawdown": max_dd_in_period,
                    })

                in_drawdown = False

    # Handle ongoing drawdown at end
    if in_drawdown:
        trough_idx = peak_idx
        max_dd_in_period = 0.0
        for i in range(peak_idx, len(dd_series)):
            if dd_series[i] < -max_dd_in_period:
                max_dd_in_period = -dd_series[i]
                trough_idx = i

        duration = len(dd_series) - peak_idx

        if duration >= min_duration:
            periods.append({
                "peak_idx": peak_idx,
                "trough_idx": trough_idx,
                "recovery_idx": -1,  # Not recovered
                "duration": duration,
                "drawdown": max_dd_in_period,
            })

    return periods


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
