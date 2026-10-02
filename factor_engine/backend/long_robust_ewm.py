"""Pandas boundary wrapper for the native long-panel robust EWMA recipe."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.backend.long_ewm import (
    _validate_min_periods,
    halflife_to_alpha,
)


def lagged_robust_ewma(
    frame: pd.DataFrame,
    *,
    halflife: float,
    winsor_std: float = 4.0,
    min_periods: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """Compute lagged robust EWMA with native Polars math and pandas alignment.

    The returned series preserves the caller's original index, including
    duplicate labels. Rows with a missing asset key follow pandas groupby
    ``dropna=True`` behavior and remain missing in the result.
    """
    # Reuse the FE EWMA parameter contracts used by the sibling long recipes.
    halflife_to_alpha(halflife)
    min_periods = _validate_min_periods(min_periods)
    from factor_engine.backend.native_long_robust_ewma import (
        _winsor_width,
        lagged_robust_ewma as native,
    )
    _winsor_width(winsor_std)
    for name, value in (("asset_col", asset_col), ("time_col", time_col),
                        ("value_col", value_col)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty column name")
    if len({asset_col, time_col, value_col}) != 3:
        raise ValueError("asset_col, time_col, and value_col must be distinct")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
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
        raise ValueError("per-asset dates must be non-null and monotone increasing")

    positions = np.flatnonzero(observed).astype(np.int64, copy=False)
    group_codes, _ = pd.factorize(assets[observed], sort=False)
    values = frame[value_col].reset_index(drop=True).to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True
    )[observed]

    # Factor codes preserve pandas categorical/null grouping semantics. The
    # compact frame uses source row positions as its synthetic chronological
    # axis after the original per-asset dates have been validated above.
    panel = pl.DataFrame({
        "__asset_code": pl.Series(group_codes.astype(np.int64, copy=False)),
        "__time": pl.Series(positions),
        "__value": pl.Series(values, dtype=pl.Float64),
    })
    native_result = native(
        panel,
        halflife=halflife,
        winsor_std=winsor_std,
        min_periods=min_periods,
        asset_col="__asset_code",
        time_col="__time",
        value_col="__value",
        output_col="__robust_ewma",
    )
    output[positions] = native_result.get_column("__robust_ewma").to_numpy()
    return pd.Series(output, index=frame.index, name=value_col)


__all__ = ["lagged_robust_ewma"]
