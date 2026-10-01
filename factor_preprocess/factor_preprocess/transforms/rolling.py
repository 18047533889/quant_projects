"""
Rolling (time-series) transforms with explicit causality.

All rolling transforms:
- Exclude current observation (strict lag)
- Operate per-asset (isolated groups)
- Have explicit warmup periods
- Preserve NaN propagation
"""
import numpy as np
import pandas as pd
from typing import Optional


def _groupwise_lagged_stat_reference(values, window, min_periods, asset_col, value_col, operation):
    """Apply a lagged window operation independently for each asset.

    Reference (scalar, per-asset Python loop) implementation retained as the
    numerical oracle for the vectorized paths in this module. Output is
    bit-for-bit equivalent to the vectorized kernels on sorted input.
    """
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        lagged = values.iloc[positions][value_col].shift(1)
        result.iloc[positions] = operation(lagged, window, min_periods)
    return result


def _lagged_rolling_stat(values, window, min_periods, asset_col, value_col, stat, ddof=1):
    """Vectorized per-asset lagged rolling statistic.

    Numerically equivalent to ``_groupwise_lagged_stat_reference`` but groups
    with pandas ``groupby`` + ``rolling`` instead of a Python loop over assets.
    A positional index is used while grouping so duplicate input index labels
    do not disturb alignment; results are scattered back to the original row
    order afterwards (mirrors the vectorized ``trailing_sma`` pattern).
    """
    keys = values[asset_col].reset_index(drop=True)
    lagged = values[value_col].reset_index(drop=True).groupby(
        keys, sort=False, observed=True
    ).shift(1)
    if stat == "mean":
        rolled = lagged.groupby(keys, sort=False, observed=True).rolling(
            window=window, min_periods=min_periods
        ).mean()
    elif stat == "std":
        rolled = lagged.groupby(keys, sort=False, observed=True).rolling(
            window=window, min_periods=min_periods
        ).std(ddof=ddof)
    else:
        raise ValueError(f"unknown stat {stat!r}")
    result = np.full(len(values), np.nan, dtype=float)
    if len(rolled):
        positions = rolled.index.get_level_values(-1).to_numpy()
        result[positions] = rolled.to_numpy()
    return result


def _lagged_ewm(values, halflife, min_periods, asset_col, value_col):
    """Vectorized per-asset lagged EWMA (see :func:`_lagged_rolling_stat`)."""
    keys = values[asset_col].reset_index(drop=True)
    lagged = values[value_col].reset_index(drop=True).groupby(
        keys, sort=False, observed=True
    ).shift(1)
    ewmr = lagged.groupby(keys, sort=False, observed=True).ewm(
        halflife=halflife, min_periods=min_periods, adjust=False
    ).mean()
    result = np.full(len(values), np.nan, dtype=float)
    if len(ewmr):
        positions = ewmr.index.get_level_values(-1).to_numpy()
        result[positions] = ewmr.to_numpy()
    return result


# --- Reference (scalar) oracles retained for A/B equivalence tests -----------

def _rolling_mean_reference(values, window, min_periods=None, asset_col="asset_id",
                            time_col="date", value_col="value"):
    if min_periods is None:
        min_periods = window
    return _groupwise_lagged_stat_reference(
        values, window, min_periods, asset_col, value_col,
        lambda series, w, mp: series.rolling(window=w, min_periods=mp).mean().to_numpy(),
    )


def _rolling_std_reference(values, window, min_periods=None, ddof=1, asset_col="asset_id",
                           time_col="date", value_col="value"):
    if min_periods is None:
        min_periods = window
    return _groupwise_lagged_stat_reference(
        values, window, min_periods, asset_col, value_col,
        lambda series, w, mp: series.rolling(window=w, min_periods=mp).std(ddof=ddof).to_numpy(),
    )


def _rolling_zscore_reference(values, window, min_periods=None, ddof=1, asset_col="asset_id",
                              time_col="date", value_col="value"):
    if min_periods is None:
        min_periods = window
    mean = _groupwise_lagged_stat_reference(
        values, window, min_periods, asset_col, value_col,
        lambda series, w, mp: series.rolling(window=w, min_periods=mp).mean().to_numpy(),
    )
    std = _groupwise_lagged_stat_reference(
        values, window, min_periods, asset_col, value_col,
        lambda series, w, mp: series.rolling(window=w, min_periods=mp).std(ddof=ddof).to_numpy(),
    )
    current = values[value_col]
    return (current - mean) / std


def _ewma_reference(values, halflife, min_periods=1, asset_col="asset_id",
                    time_col="date", value_col="value"):
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        result.iloc[positions] = (
            values.iloc[positions][value_col]
            .shift(1)
            .ewm(halflife=halflife, min_periods=min_periods, adjust=False)
            .mean()
            .to_numpy()
        )
    return result


def rolling_mean(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal rolling mean per asset.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to look back (excluding current)
    min_periods : int, optional
        Minimum observations required. Defaults to window.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to transform

    Returns
    -------
    pd.Series
        Rolling mean, aligned with input index.
        First `window` observations per asset are NaN.

    Notes
    -----
    Uses shift(1) to exclude current observation, ensuring no future leakage.
    """
    if min_periods is None:
        min_periods = window

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Shift to exclude current observation, then rolling (vectorized per-asset).
    return pd.Series(
        _lagged_rolling_stat(values, window, min_periods, asset_col, value_col, stat="mean"),
        index=values.index,
    )


def rolling_std(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal rolling standard deviation per asset.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to look back (excluding current)
    min_periods : int, optional
        Minimum observations required. Defaults to window.
    ddof : int
        Delta degrees of freedom
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to transform

    Returns
    -------
    pd.Series
        Rolling std, aligned with input index.
        First `window` observations per asset are NaN.
    """
    if min_periods is None:
        min_periods = window

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Shift to exclude current observation, then rolling (vectorized per-asset).
    return pd.Series(
        _lagged_rolling_stat(
            values, window, min_periods, asset_col, value_col, stat="std", ddof=ddof
        ),
        index=values.index,
    )


def _rolling_zscore_fp_research(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal rolling z-score normalization per asset.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to look back (excluding current)
    min_periods : int, optional
        Minimum observations required. Defaults to window.
    ddof : int
        Delta degrees of freedom
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to transform

    Returns
    -------
    pd.Series
        Rolling z-score: (current - rolling_mean) / rolling_std.
        First `window` observations per asset are NaN.
        Division follows IEEE semantics: 0/0 produces NaN, while a nonzero
        numerator over zero std produces a signed infinity.

    Notes
    -----
    Current observation is normalized against lagged statistics.
    """
    if min_periods is None:
        min_periods = window

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Compute lagged statistics (excluding current), vectorized per-asset.
    rolling_mean_val = _lagged_rolling_stat(
        values, window, min_periods, asset_col, value_col, stat="mean"
    )
    rolling_std_val = _lagged_rolling_stat(
        values, window, min_periods, asset_col, value_col, stat="std", ddof=ddof
    )

    # Z-score: (current - mean) / std
    current = values[value_col]
    result = (current - rolling_mean_val) / rolling_std_val

    return result


def rolling_zscore(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """Compute rolling z-scores through the FactorEngine authority route.

    The numerically fragile FP pandas implementation remains available only
    through the registry's explicit research fallback. Public calls fail
    closed when the FE composite is unavailable; they never silently fall
    back to the legacy rolling kernel.
    """
    from factor_preprocess.adapters.fe_composite import get_fe_composite_executor
    from factor_preprocess.adapters.fe_smoothing import ROLLING_ZSCORE_RECIPE
    from factor_preprocess.errors import GovernanceError

    executor = get_fe_composite_executor("rolling_zscore", ROLLING_ZSCORE_RECIPE)
    if executor is None:
        raise GovernanceError(
            "FE composite authority unavailable for 'rolling_zscore'; "
            "no implicit FP-native fallback"
        )
    return executor(
        values, window=window, min_periods=min_periods, ddof=ddof,
        asset_col=asset_col, time_col=time_col, value_col=value_col,
    )


def _ewma_fp_research(
    values: pd.DataFrame,
    halflife: float,
    min_periods: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal exponentially-weighted moving average per asset.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    halflife : float
        Halflife in units of observations
    min_periods : int
        Minimum observations required
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to transform

    Returns
    -------
    pd.Series
        EWMA, aligned with input index.

    Notes
    -----
    Uses shift(1) to exclude current observation.

    Research-only fallback. Public EWMA calls use the FactorEngine
    composite authority and fail closed when that route is unavailable.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be > 0, got {halflife}")

    valid_min_periods = (
        isinstance(min_periods, (int, np.integer))
        and not isinstance(min_periods, (bool, np.bool_))
        and min_periods >= 0
    )
    if not valid_min_periods:
        raise ValueError("min_periods must be a non-negative integer")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Shift to exclude current observation, then EWMA (vectorized per-asset).
    return pd.Series(
        _lagged_ewm(values, halflife, min_periods, asset_col, value_col),
        index=values.index,
    )


def ewma(
    values: pd.DataFrame,
    halflife: float,
    min_periods: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """Causal EWMA per asset, delegated to FactorEngine authority.

    Parameters
    ----------
    values : pd.DataFrame
        Long panel containing asset_col, time_col and value_col.
        Dates must be non-null and monotone within each observed asset.
    halflife : float
        Finite positive half-life in units of observations.
    min_periods : int
        Non-negative minimum number of valid prior observations; default 1.
    asset_col : str
        Asset identifier column; rows with null assets produce missing output.
    time_col : str
        Time column used for per-asset ordering validation.
    value_col : str
        Numeric value column; nonfinite values count as missing observations.

    Returns
    -------
    pd.Series
        Lagged EWMA aligned with the original row order and index.

    Notes
    -----
    The FE kernel uses adjust=False and ignore_nulls=False. Missing positions
    affect weights. Public calls fail closed if FE authority is unavailable;
    the registry permits an explicit research fallback via allow_research=True.
    Halflife is measured in observations and maps to
    alpha = 1 - exp(-log(2) / halflife). The current row is excluded:
    each output uses only earlier rows from the same asset. Min_periods
    counts valid prior observations before an output is emitted. Production
    registry admission further restricts halflife to its declared domain.
    """
    from factor_preprocess.adapters.fe_composite import get_fe_composite_executor
    from factor_preprocess.adapters.fe_smoothing import EWMA_RECIPE
    from factor_preprocess.errors import GovernanceError

    executor = get_fe_composite_executor("ewma", EWMA_RECIPE)
    if executor is None:
        raise GovernanceError(
            "FE composite authority unavailable for 'ewma'; "
            "no implicit FP-native fallback"
        )
    return executor(
        values, halflife=halflife, min_periods=min_periods,
        asset_col=asset_col, time_col=time_col, value_col=value_col,
    )
