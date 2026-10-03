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


def lagged_iir_lowpass(frame: pd.DataFrame, *, alpha: float,
                       asset_col: str = "asset_id", time_col: str = "date",
                       value_col: str = "value") -> pd.Series:
    """Run a prior-row one-pole IIR, resetting state at non-finite gaps.

    ``alpha`` is accepted directly to avoid lossy conversion through a
    half-life. Run labels let the native EWMA restart after each gap while
    preserving the output that consumes the last finite pre-gap value.
    """
    # Keep the preprocessing transform's historical True == 1.0 behavior.
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    alpha = float(alpha)
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
    )[observed]
    finite = np.isfinite(values)
    values[~finite] = np.nan

    # One stable sort preserves within-asset input order. Start a new EWMA
    # group on the row after a gap, whose shifted sample is the gap itself.
    order = np.argsort(group_codes, kind="stable")
    sorted_groups = group_codes[order]
    breaks = np.empty(len(values), dtype=bool)
    breaks[0] = True
    breaks[1:] = (
        (sorted_groups[1:] != sorted_groups[:-1])
        | ~finite[order[:-1]]
    )
    run_codes = np.empty(len(values), dtype=np.int64)
    run_codes[order] = np.cumsum(breaks, dtype=np.int64) - 1

    from factor_engine.backend.native_long_ewm import collect_lagged_ewma
    result = collect_lagged_ewma(
        positions, run_codes, values, alpha=alpha, min_periods=1,
    )
    actual_positions = result.get_column("ts").to_numpy()
    actual_groups = result.get_column("inst").to_numpy()
    if (not np.array_equal(actual_positions, positions)
            or not np.array_equal(actual_groups, run_codes)):
        raise RuntimeError("FactorEngine lagged IIR changed row identity or order")
    output[positions] = result.get_column("_ema").to_numpy()
    return pd.Series(output, index=frame.index, name=value_col)


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

def event_decay_native(frame: pd.DataFrame, *, halflife: float,
                       min_periods: int = 1, asset_col: str = "asset_id",
                       time_col: str = "date", value_col: str = "value") -> pd.Series:
    """Execute FP event_decay semantics through FactorEngine's Polars collector.

    This wrapper uses pandas only to validate/order the long-panel contract,
    encode asset identity, and restore the caller's index. All lag segmentation,
    EMA state, warmup masking, and numeric updates run as Polars expressions.
    """
    if isinstance(halflife, (bool, np.bool_)):
        raise ValueError("halflife must be a finite positive real number")
    try:
        half = float(halflife)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("halflife must be a finite positive real number") from exc
    if not math.isfinite(half) or half <= 0.0:
        raise ValueError("halflife must be a finite positive real number")
    if (isinstance(min_periods, (bool, np.bool_))
            or not isinstance(min_periods, numbers.Integral)
            or min_periods < 1):
        raise ValueError("min_periods must be a positive integer")
    min_periods = int(min_periods)
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
    if not dates.groupby(asset_col, sort=False, observed=True)[time_col].is_monotonic_increasing.all():
        raise ValueError("per-asset dates must be monotone increasing")

    group_codes, _ = pd.factorize(assets[observed], sort=False)
    positions = np.flatnonzero(observed).astype(np.int64, copy=False)
    values = frame[value_col].reset_index(drop=True).to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True
    )[observed]
    alpha = min(math.log(2.0) / half, 1.0)
    from factor_engine.backend.native_event_decay import collect_event_decay
    result = collect_event_decay(
        positions, group_codes.astype(np.int64, copy=False), values,
        alpha=alpha, min_periods=min_periods,
    )
    actual_positions = result.get_column("pos").to_numpy()
    if not np.array_equal(actual_positions, positions):
        raise RuntimeError("FactorEngine event decay changed row identity or order")
    output[positions] = result.get_column("event_decay").to_numpy()
    return pd.Series(output, index=frame.index, name=value_col)
