"""
Short-halflife event-decay persistence (causal, one-sided).

Output at time ``t`` uses the strictly-past input ``x[t-1]``. The first finite
lagged value seeds the EMA; each later finite lagged value updates the
recurrence ``y = (1 - alpha)*y + alpha*x`` with
``alpha = ln(2)/halflife`` clipped to ``(0, 1]``. Warmup hides outputs but
does not pause the state updates.

Causality: each output uses only observations ``<= t-1`` (current excluded via
``shift(1)``), so the transform is prefix-invariant and production-causal —
the natural treatment for EVENT factors.
"""
import numpy as np
import pandas as pd
from typing import Optional
from numbers import Integral


def _check_sort(values, asset_col, time_col):
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")


def event_decay(
    values: pd.DataFrame,
    halflife: float,
    min_periods: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal short-halflife event decay.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col] sorted by
        [asset_col, time_col].
    halflife : float
        Decay halflife in observations (alpha = ln(2)/halflife, clipped to
        (0, 1]).
    min_periods : int
        Minimum finite lagged observations before an output is produced
        (warmup). Default 1.
    asset_col : str
        Asset identifier column.
    time_col : str
        Time column (for sorting verification).
    value_col : str
        Value column to decay.

    Returns
    -------
    pd.Series
        Event-decayed persistence aligned with the input index. Warmup and
        NaN lags produce NaN.

    Notes
    -----
    A non-finite lagged input resets the decay memory to NaN (no event
    signal). A finite lagged observation after a non-finite value re-seeds the output from
    that value.
    """
    if isinstance(halflife, (bool, np.bool_)):
        raise ValueError("halflife must be finite and > 0")
    try:
        halflife = float(halflife)
    except (TypeError, ValueError) as exc:
        raise ValueError("halflife must be finite and > 0") from exc
    if not np.isfinite(halflife) or halflife <= 0:
        raise ValueError(f"halflife must be finite and > 0, got {halflife}")
    if (isinstance(min_periods, (bool, np.bool_))
            or not isinstance(min_periods, Integral) or min_periods < 1):
        raise ValueError(f"min_periods must be a positive integer, got {min_periods}")
    min_periods = int(min_periods)

    _check_sort(values, asset_col, time_col)
    alpha = np.log(2.0) / halflife
    alpha = max(min(alpha, 1.0), 0.0)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        lagged = values.iloc[positions][value_col].shift(1).to_numpy()
        n = len(lagged)
        out = np.full(n, np.nan)
        y = np.nan
        finite_count = 0
        for i in range(n):
            x = lagged[i]
            if not np.isfinite(x):
                out[i] = np.nan
                y = np.nan
                finite_count = 0
                continue
            finite_count += 1
            if finite_count == 1:
                y = x
            else:
                y = (1 - alpha) * y + alpha * x
            if finite_count >= min_periods:
                out[i] = y
        result.iloc[positions] = out
    return result