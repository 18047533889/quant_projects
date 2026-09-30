"""Shared FactorEngine recipes for lagged rolling transforms on long panels."""
from __future__ import annotations

import numbers

import numpy as np
import pandas as pd


def _validate_rolling_params(window: int, min_periods: int | None) -> tuple[int, int]:
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, numbers.Integral) or window < 1:
        raise ValueError("window must be a positive integer")
    if min_periods is None:
        min_periods = int(window)
    if isinstance(min_periods, (bool, np.bool_)) or not isinstance(min_periods, numbers.Integral):
        raise ValueError("min_periods must be an integer between 0 and window")
    if min_periods < 0 or min_periods > window:
        raise ValueError("min_periods must be an integer between 0 and window")
    return int(window), int(min_periods)


def _mean_expression(value_col: str, window: int, min_periods: int, CleanedCall, ColumnRef):
    delayed = CleanedCall("ts_delay", (ColumnRef(value_col),), (("n", 1),))
    return CleanedCall("ts_mean", (delayed,), (
        ("window", window), ("min_periods", max(1, min_periods))
    ))


def _median_expression(value_col: str, window: int, min_periods: int,
                       CleanedCall, ColumnRef, Literal):
    delayed = CleanedCall("ts_delay", (ColumnRef(value_col),), (("n", 1),))
    median = CleanedCall("ts_median", (delayed,), (("window", window),))
    delayed_valid = CleanedCall(
        "ts_delay", (ColumnRef("_valid"),), (("n", 1),)
    )
    count = CleanedCall("ts_sum", (delayed_valid,), (
        ("window", window), ("min_periods", 1)
    ))
    enough = CleanedCall("ge", (count, Literal(max(1, min_periods))))
    return CleanedCall("where", (enough, median, Literal(np.nan)))


def _lagged_rolling(frame: pd.DataFrame, *, window: int, min_periods: int | None,
                    asset_col: str, time_col: str, value_col: str,
                    reducer: str) -> pd.Series:
    """Validate, compile, and restore a lagged rolling statistic by row position."""
    window, min_periods = _validate_rolling_params(window, min_periods)
    missing = {asset_col, time_col, value_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"long panel is missing columns: {sorted(missing)}")
    output = np.full(len(frame), np.nan, dtype=float)
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

    # Integer group codes and row ordinals preserve nullable/categorical IDs,
    # interleaved assets, duplicate dates, and duplicate caller index labels.
    group_codes, _ = pd.factorize(assets[observed], sort=False)
    positions = np.flatnonzero(observed).astype(np.int64, copy=False)
    numeric_values = frame[value_col].reset_index(drop=True).to_numpy(
        dtype=float, na_value=np.nan, copy=True
    )
    valid_values = np.isfinite(numeric_values)
    numeric_values[~valid_values] = np.nan

    import polars as pl
    from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef
    from factor_engine.expr.literal import Literal
    from factor_engine.ir.analyzer import Analyzer

    panel = {
        "ts": positions,
        "inst": group_codes.astype(np.int64, copy=False),
        "_v": numeric_values[observed],
    }
    if reducer == "median":
        panel["_valid"] = valid_values.astype(np.float64, copy=False)[observed]
    base = pl.DataFrame(panel).lazy()
    builders = {"mean": _mean_expression, "median": _median_expression}
    try:
        builder = builders[reducer]
    except KeyError as exc:
        raise ValueError(f"unknown lagged rolling reducer {reducer!r}") from exc
    if reducer == "mean":
        expression = builder("_v", window, min_periods, CleanedCall, ColumnRef)
    else:
        expression = builder("_v", window, min_periods, CleanedCall, ColumnRef, Literal)
    plan = Analyzer(production=False).lower(expression).ir
    compiled = compile_plan_to_polars(plan, base)
    if compiled is None:
        raise RuntimeError(f"FactorEngine could not compile lagged {reducer} on a long panel")
    result = compiled.frame.collect()
    actual_positions = result.get_column("ts").to_numpy()
    actual_groups = result.get_column("inst").to_numpy()
    if not np.array_equal(actual_positions, positions) or not np.array_equal(actual_groups, group_codes):
        raise RuntimeError(f"FactorEngine long rolling {reducer} changed row identity or order")
    output[positions] = result.get_column("_v").to_numpy()
    return pd.Series(output, index=frame.index, name=value_col)


def lagged_mean(frame: pd.DataFrame, *, window: int, min_periods: int | None = None,
                asset_col: str = "asset_id", time_col: str = "date",
                value_col: str = "value") -> pd.Series:
    """Compute per-asset mean over prior observed rows on a long panel."""
    return _lagged_rolling(
        frame, window=window, min_periods=min_periods, asset_col=asset_col,
        time_col=time_col, value_col=value_col, reducer="mean",
    )


def lagged_median(frame: pd.DataFrame, *, window: int, min_periods: int | None = None,
                  asset_col: str = "asset_id", time_col: str = "date",
                  value_col: str = "value") -> pd.Series:
    """Compute a finite-only per-asset median over prior observed rows."""
    return _lagged_rolling(
        frame, window=window, min_periods=min_periods, asset_col=asset_col,
        time_col=time_col, value_col=value_col, reducer="median",
    )
