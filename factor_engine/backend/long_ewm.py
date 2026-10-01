"""Shared FactorEngine recipes for lagged EWM transforms on long panels."""
from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd


def halflife_to_alpha(halflife: float) -> float:
    """Convert an observation half-life to binary64 EWMA alpha stably."""
    if (isinstance(halflife, (bool, np.bool_))
            or not isinstance(halflife, numbers.Real)):
        raise ValueError("halflife must be a finite positive real number")
    value = float(halflife)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("halflife must be a finite positive real number")
    try:
        exponent = -math.log(2.0) / value
    except OverflowError:
        exponent = -math.inf
    alpha = -math.expm1(exponent)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("halflife is outside the supported binary64 EWMA domain")
    return alpha


def _validate_min_periods(min_periods: int) -> int:
    if (isinstance(min_periods, (bool, np.bool_))
            or not isinstance(min_periods, numbers.Integral)
            or min_periods < 0):
        raise ValueError("min_periods must be a non-negative integer")
    return int(min_periods)


def lagged_ewma(frame: pd.DataFrame, *, halflife: float, min_periods: int = 1,
                asset_col: str = "asset_id", time_col: str = "date",
                value_col: str = "value") -> pd.Series:
    """Compute a per-asset pandas-compatible EWMA over rows strictly before t."""
    alpha = halflife_to_alpha(halflife)
    min_periods = _validate_min_periods(min_periods)
    missing = {asset_col, time_col, value_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"long panel is missing columns: {sorted(missing)}")

    output = np.full(len(frame), np.nan, dtype=np.float64)
    if frame.empty:
        return pd.Series(output, index=frame.index, name=value_col)

    assets = frame[asset_col].reset_index(drop=True)
    observed = assets.notna().to_numpy()
    if not observed.any():
        return pd.Series(output, index=frame.index, name=value_col)
    times = frame[time_col].reset_index(drop=True)
    if times.iloc[np.flatnonzero(observed)].isna().any():
        raise ValueError("per-asset dates must be non-null and monotone increasing")
    dates = frame.loc[observed, [asset_col, time_col]].reset_index(drop=True)
    if not dates.groupby(
        asset_col, sort=False, observed=True
    )[time_col].is_monotonic_increasing.all():
        raise ValueError("per-asset dates must be monotone increasing")

    group_codes, _ = pd.factorize(assets[observed], sort=False)
    positions = np.flatnonzero(observed).astype(np.int64, copy=False)
    values = frame[value_col].reset_index(drop=True).to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True
    )
    values[~np.isfinite(values)] = np.nan

    from factor_engine.backend.native_long_ewm import collect_lagged_ewma
    result = collect_lagged_ewma(
        positions, group_codes.astype(np.int64, copy=False), values[observed],
        alpha=alpha, min_periods=min_periods,
    )
    actual_positions = result.get_column("ts").to_numpy()
    actual_groups = result.get_column("inst").to_numpy()
    if (not np.array_equal(actual_positions, positions)
            or not np.array_equal(actual_groups, group_codes)):
        raise RuntimeError("FactorEngine lagged EWMA changed row identity or order")
    output[positions] = result.get_column("_ema").to_numpy()
    return pd.Series(output, index=frame.index, name=value_col)
