"""
Causal one-sided signal smoothers.

All smoothers in this module:
- Operate on strictly ``<= t - 1`` information (current observation excluded
  via ``shift(1)``), so no future/current observation leaks into the output.
- Operate per-asset (isolated groups).
- Verify monotonic sort order per asset.
- Return a ``pd.Series`` aligned with the input index.
- Propagate NaN warmup (output is NaN until enough lagged history exists).

These are prefix-invariant: recomputing a smoother over a prefix of the
sample yields identical output for that prefix, because each output depends
only on strictly-past observations.
"""
import numpy as np
import pandas as pd
from typing import Optional


def _check_sort(values, asset_col, time_col):
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")


def _verified_series(values):
    return pd.Series(np.nan, index=values.index, dtype=float)


def trailing_sma(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Lagged trailing simple moving average (SMA).

    Output at time ``t`` is the mean of observations in the closed window
    ``[t-window, t-1]``. The current observation is never included.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of lagged periods to average (excluding current).
    min_periods : int, optional
        Minimum lagged observations required. Defaults to ``window``.
    asset_col : str
        Asset identifier column.
    time_col : str
        Time column (for sorting verification).
    value_col : str
        Value column to smooth.

    Returns
    -------
    pd.Series
        Trailing SMA aligned with input index. First ``min_periods``
        observations per asset are NaN.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_periods is None:
        min_periods = window

    _check_sort(values, asset_col, time_col)
    result = _verified_series(values)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        result.iloc[positions] = (
            values.iloc[positions][value_col]
            .shift(1)
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .to_numpy()
        )
    return result


def trailing_median(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Trailing rolling median (robust to spikes/outliers).

    Causal: output at time ``t`` is the median of the closed window
    ``[t-window, t-1]``. Robust against single-point spikes.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col].
    window : int
        Number of lagged periods (excluding current).
    min_periods : int, optional
        Minimum lagged observations required. Defaults to ``window``.
    asset_col : str
        Asset identifier column.
    time_col : str
        Time column (for sorting verification).
    value_col : str
        Value column to smooth.

    Returns
    -------
    pd.Series
        Trailing median aligned with input index. First ``min_periods``
        observations per asset are NaN.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_periods is None:
        min_periods = window

    _check_sort(values, asset_col, time_col)
    result = _verified_series(values)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        result.iloc[positions] = (
            values.iloc[positions][value_col]
            .shift(1)
            .rolling(window=window, min_periods=min_periods)
            .median()
            .to_numpy()
        )
    return result


def robust_ewma(
    values: pd.DataFrame,
    halflife: float,
    winsor_std: float = 4.0,
    min_periods: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Lagged EWMA computed on winsorized values (robust to outliers).

    The value series is winsorized per-asset against a lagged rolling
    mean/std estimate (a causal clip), then an EWMA is applied. Causality is
    preserved because both the clip statistics and the EWMA use only
    ``<= t - 1`` information.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col].
    halflife : float
        EWMA halflife in units of observations.
    winsor_std : float, optional
        Winsorization clip width in lagged rolling std units.
    min_periods : int
        Minimum observations required for the EWMA.
    asset_col : str
        Asset identifier column.
    time_col : str
        Time column (for sorting verification).
    value_col : str
        Value column to smooth.

    Returns
    -------
    pd.Series
        Robust EWMA aligned with input index.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be > 0, got {halflife}")
    if winsor_std <= 0:
        winsor_std = 1e-12

    _check_sort(values, asset_col, time_col)
    result = _verified_series(values)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        asset_df = values.iloc[positions]
        lagged = asset_df[value_col].shift(1)

        # Winsorize the lagged series using lagged rolling stats (still causal).
        rmean = lagged.rolling(window=10, min_periods=2).mean()
        rstd = lagged.rolling(window=10, min_periods=2).std()
        rstd = rstd.clip(lower=1e-12)
        robust = lagged.clip(lower=rmean - winsor_std * rstd, upper=rmean + winsor_std * rstd)

        result.iloc[positions] = (
            robust
            .ewm(halflife=halflife, min_periods=min_periods, adjust=False)
            .mean()
            .to_numpy()
        )
    return result


def kama(
    values: pd.DataFrame,
    period_fast: int = 2,
    period_slow: int = 30,
    period_er: int = 10,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Kaufman Adaptive Moving Average (KAMA), built recursively forward only.

    Causality: the KAMA recursion ``kama[t] = kama[t-1] + sc[t]*(x[t-1] - kama[t-1])``
    uses only observations up to and including ``t-1`` (the current observation
    ``x[t]`` is excluded by the ``shift(1)``), so each output depends only on
    the past. It is never centered.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col].
    period_fast : int
        Fast smoothing constant period (lower = more responsive).
    period_slow : int
        Slow smoothing constant period (lower = more responsive).
    period_er : int
        Efficiency-ratio lookback period.
    asset_col : str
        Asset identifier column.
    time_col : str
        Time column (for sorting verification).
    value_col : str
        Value column to smooth.

    Returns
    -------
    pd.Series
        KAMA aligned with input index. NaN warmup for the recursion seed.
    """
    for p in (period_fast, period_slow, period_er):
        if p < 1:
            raise ValueError(f"periods must be >= 1, got {p}")
    if period_fast == period_slow:
        period_slow = period_fast + 1

    fast_sc = 2.0 / (period_fast + 1.0)
    slow_sc = 2.0 / (period_slow + 1.0)

    _check_sort(values, asset_col, time_col)
    result = _verified_series(values)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        lagged = values.iloc[positions][value_col].shift(1).to_numpy()
        n = len(lagged)
        out = np.full(n, np.nan)

        # Efficiency ratio over the closed window [t-period_er, t-1].
        change = np.abs(lagged - np.roll(lagged, 1))
        change[0] = np.nan
        volatility = np.full(n, np.nan)
        for i in range(n):
            if i - period_er + 1 < 1 or np.isnan(change[i]):
                continue
            window = change[i - period_er + 1 : i + 1]
            if np.any(np.isnan(window)):
                continue
            volatility[i] = window.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            efficiency = np.where(
                (volatility > 0) & ~np.isnan(volatility),
                np.abs(change) / volatility,
                np.nan,
            )

        # Seed KAMA from the first valid value (after warmup).
        sc = (efficiency * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_val = np.nan
        for i in range(n):
            if i < period_er:
                out[i] = np.nan
                continue
            if kama_val is None or not np.isfinite(kama_val):
                if not np.isnan(lagged[i]):
                    kama_val = lagged[i]
                out[i] = np.nan
                continue
            if np.isnan(sc[i]):
                out[i] = kama_val
                continue
            kama_val = kama_val + sc[i] * (lagged[i] - kama_val)
            out[i] = kama_val
        result.iloc[positions] = out
    return result


def one_sided_iir_lowpass(
    values: pd.DataFrame,
    alpha: float,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    One-pole IIR low-pass filter applied forward only.

    Recurrence (on lagged observations, so still causal):
        ``y[t] = (1 - alpha) * y[t-1] + alpha * x[t-1]``

    ``alpha`` in ``(0, 1]``; smaller alpha smooths more. Because the
    recursion only ever consumes past observations, the output is
    prefix-invariant and strictly one-sided.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col].
    alpha : float
        Smoothing coefficient in ``(0, 1]``.
    asset_col : str
        Asset identifier column.
    time_col : str
        Time column (for sorting verification).
    value_col : str
        Value column to smooth.

    Returns
    -------
    pd.Series
        One-sided IIR low-pass aligned with input index.
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    _check_sort(values, asset_col, time_col)
    result = _verified_series(values)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        lagged = values.iloc[positions][value_col].shift(1).to_numpy()
        n = len(lagged)
        out = np.full(n, np.nan)
        y = np.nan
        for i in range(n):
            x = lagged[i]
            if np.isnan(x):
                y = np.nan
                out[i] = np.nan
                continue
            if not np.isfinite(y):
                y = x
            else:
                y = (1 - alpha) * y + alpha * x
            out[i] = y
        result.iloc[positions] = out
    return result


def kalman_local_level(
    values: pd.DataFrame,
    process_noise: float,
    measurement_noise: float,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    One-sided Kalman local-level smoother.

    Model: ``x[t] = level[t] + v``, ``level[t] = level[t-1] + w`` with
    ``var(v)=measurement_noise`` and ``var(w)=process_noise``. A standard
    scalar Kalman filter runs forward only over the lagged observations
    (current observation excluded via ``shift(1)``). The filtered-level
    estimate at each step is prefix-invariant and strictly causal.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col].
    process_noise : float
        Variance of the level process noise ``w`` (>= 0).
    measurement_noise : float
        Variance of the measurement noise ``v`` (> 0).
    asset_col : str
        Asset identifier column.
    time_col : str
        Time column (for sorting verification).
    value_col : str
        Value column to smooth.

    Returns
    -------
    pd.Series
        Kalman filtered local level, aligned with input index.
    """
    if process_noise < 0:
        raise ValueError(f"process_noise must be >= 0, got {process_noise}")
    if measurement_noise <= 0:
        raise ValueError(f"measurement_noise must be > 0, got {measurement_noise}")

    _check_sort(values, asset_col, time_col)
    result = _verified_series(values)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        lagged = values.iloc[positions][value_col].shift(1).to_numpy()
        n = len(lagged)
        out = np.full(n, np.nan)
        q = process_noise
        r = measurement_noise
        x_hat = np.nan
        p = np.nan
        for i in range(n):
            z = lagged[i]
            if np.isnan(z):
                out[i] = np.nan
                x_hat = np.nan
                p = np.nan
                continue
            # Predict
            if not np.isfinite(x_hat):
                x_hat = z
                p = q
            else:
                p = p + q
            # Update
            K = p / (p + r)
            x_hat = x_hat + K * (z - x_hat)
            p = (1 - K) * p
            out[i] = x_hat
        result.iloc[positions] = out
    return result
