"""
Polars backend for high-performance cross-sectional transforms.

Targets 3-5x speedup over pandas/numpy reference for large panels.
Uses polars expressions for vectorized cross-sectional operations.
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal, Union
from datetime import datetime

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


def _require_polars():
    """Check polars availability."""
    if not POLARS_AVAILABLE:
        from factor_preprocess.errors import OptionalDependencyMissing
        raise OptionalDependencyMissing(
            "polars backend requires polars>=0.19.0",
            package="polars",
            feature="polars_backend",
        )


def cs_rank_polars(
    df: Union[pd.DataFrame, pl.DataFrame],
    value_col: str,
    group_col: str = "date",
    method: Literal["average", "min", "max", "dense", "ordinal"] = "average",
    pct: bool = False,
) -> Union[pd.Series, pl.Series]:
    """
    Cross-sectional rank using polars for high performance.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input data with columns [group_col, value_col, ...]
    value_col : str
        Column to rank
    group_col : str
        Grouping column (typically date)
    method : str
        Tie-breaking method
    pct : bool
        If True, return percentile ranks [0, 1]

    Returns
    -------
    pd.Series or pl.Series
        Ranked values. NaN inputs produce NaN outputs.

    Notes
    -----
    Operates per group (typically per date) for cross-sectional ranking.
    Uses polars for 3-5x speedup on large panels.
    """
    _require_polars()

    # Convert to polars if needed
    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df)
        return_pandas = True
    else:
        pl_df = df
        return_pandas = False

    # Map method to polars rank method
    method_map = {
        "average": "average",
        "min": "min",
        "max": "max",
        "dense": "dense",
        "ordinal": "ordinal",
    }
    pl_method = method_map.get(method, "average")

    # Rank within each group
    result = (
        pl_df
        .with_columns([
            pl.col(value_col)
            .rank(method=pl_method)
            .over(group_col)
            .alias("_rank")
        ])
    )

    if pct:
        # Convert to percentile: (rank - 1) / (n - 1)
        result = result.with_columns([
            pl.when(pl.col("_rank").is_not_null())
            .then(
                (pl.col("_rank") - 1.0) /
                (pl.col(value_col).count().over(group_col) - 1.0)
            )
            .otherwise(None)
            .alias("_rank")
        ])

    if return_pandas:
        return result.select("_rank").to_pandas()["_rank"]
    else:
        return result.select("_rank").to_series()


def cs_zscore_polars(
    df: Union[pd.DataFrame, pl.DataFrame],
    value_col: str,
    group_col: str = "date",
    ddof: int = 1,
    constant_value: float = 0.0,
) -> Union[pd.Series, pl.Series]:
    """
    Cross-sectional z-score using polars for high performance.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input data
    value_col : str
        Column to normalize
    group_col : str
        Grouping column (typically date)
    ddof : int
        Delta degrees of freedom for std
    constant_value : float
        Value for constant groups (zero std)

    Returns
    -------
    pd.Series or pl.Series
        Z-scored values. NaN inputs produce NaN outputs.
    """
    _require_polars()

    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df)
        return_pandas = True
    else:
        pl_df = df
        return_pandas = False

    # Compute mean and std per group
    result = (
        pl_df
        .with_columns([
            pl.col(value_col).mean().over(group_col).alias("_mean"),
            pl.col(value_col).std(ddof=ddof).over(group_col).alias("_std"),
        ])
        .with_columns([
            pl.when(pl.col("_std") > 0)
            .then((pl.col(value_col) - pl.col("_mean")) / pl.col("_std"))
            .otherwise(constant_value)
            .alias("_zscore")
        ])
    )

    if return_pandas:
        return result.select("_zscore").to_pandas()["_zscore"]
    else:
        return result.select("_zscore").to_series()


def cs_demean_polars(
    df: Union[pd.DataFrame, pl.DataFrame],
    value_col: str,
    group_col: str = "date",
) -> Union[pd.Series, pl.Series]:
    """
    Cross-sectional demean using polars for high performance.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input data
    value_col : str
        Column to demean
    group_col : str
        Grouping column (typically date)

    Returns
    -------
    pd.Series or pl.Series
        Demeaned values. NaN inputs produce NaN outputs.
    """
    _require_polars()

    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df)
        return_pandas = True
    else:
        pl_df = df
        return_pandas = False

    result = (
        pl_df
        .with_columns([
            (pl.col(value_col) - pl.col(value_col).mean().over(group_col))
            .alias("_demean")
        ])
    )

    if return_pandas:
        return result.select("_demean").to_pandas()["_demean"]
    else:
        return result.select("_demean").to_series()


def cs_winsor_polars(
    df: Union[pd.DataFrame, pl.DataFrame],
    value_col: str,
    group_col: str = "date",
    lower: float = 0.01,
    upper: float = 0.99,
) -> Union[pd.Series, pl.Series]:
    """
    Cross-sectional winsorization using polars for high performance.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input data
    value_col : str
        Column to winsorize
    group_col : str
        Grouping column (typically date)
    lower : float
        Lower quantile [0, 1]
    upper : float
        Upper quantile [0, 1]

    Returns
    -------
    pd.Series or pl.Series
        Winsorized values. NaN inputs produce NaN outputs.
    """
    _require_polars()

    if not (0 <= lower < upper <= 1):
        raise ValueError(f"Invalid quantiles: lower={lower}, upper={upper}")

    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df)
        return_pandas = True
    else:
        pl_df = df
        return_pandas = False

    result = (
        pl_df
        .with_columns([
            pl.col(value_col).quantile(lower, interpolation="linear").over(group_col).alias("_lower"),
            pl.col(value_col).quantile(upper, interpolation="linear").over(group_col).alias("_upper"),
        ])
        .with_columns([
            pl.col(value_col).clip(pl.col("_lower"), pl.col("_upper")).alias("_winsor")
        ])
    )

    if return_pandas:
        return result.select("_winsor").to_pandas()["_winsor"]
    else:
        return result.select("_winsor").to_series()


def cs_scale_polars(
    df: Union[pd.DataFrame, pl.DataFrame],
    value_col: str,
    group_col: str = "date",
    target_std: float = 1.0,
    ddof: int = 1,
) -> Union[pd.Series, pl.Series]:
    """
    Cross-sectional scaling to target std using polars for high performance.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input data
    value_col : str
        Column to scale
    group_col : str
        Grouping column (typically date)
    target_std : float
        Target standard deviation
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    pd.Series or pl.Series
        Scaled values. NaN inputs produce NaN outputs.
    """
    _require_polars()

    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df)
        return_pandas = True
    else:
        pl_df = df
        return_pandas = False

    result = (
        pl_df
        .with_columns([
            pl.col(value_col).std(ddof=ddof).over(group_col).alias("_std"),
        ])
        .with_columns([
            pl.when(pl.col("_std") > 0)
            .then(pl.col(value_col) * (target_std / pl.col("_std")))
            .otherwise(pl.col(value_col))
            .alias("_scaled")
        ])
    )

    if return_pandas:
        return result.select("_scaled").to_pandas()["_scaled"]
    else:
        return result.select("_scaled").to_series()


def ols_neutralize_polars(
    values: Union[pd.DataFrame, pl.DataFrame],
    exposures: Union[pd.DataFrame, pl.DataFrame],
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    min_observations: int = 10,
    add_intercept: bool = True,
) -> Union[pd.Series, pl.Series]:
    """
    Cross-sectional OLS neutralization using polars for high performance.

    Parameters
    ----------
    values : pd.DataFrame or pl.DataFrame
        Factor values to neutralize. Columns: [date_col, asset_col, value_col]
    exposures : pd.DataFrame or pl.DataFrame
        Exposure matrix. Columns: [date_col, asset_col, exposure1, exposure2, ...]
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Factor value column to neutralize
    min_observations : int
        Minimum valid observations per date to fit
    add_intercept : bool
        Whether to add intercept column

    Returns
    -------
    pd.Series or pl.Series
        Residuals aligned with values index.
        Dates with insufficient data produce NaN.

    Notes
    -----
    Uses polars for groupby operations, then numpy lstsq for OLS.
    Operates per-date (no time-series leakage).
    NaN handling: pairwise deletion per date.
    """
    _require_polars()

    # Convert to polars; carry a positional row key through the join so the
    # residuals can be realigned to the ORIGINAL values row order.  The
    # polars join resets row identity, and reindexing by the merged frame's
    # index misaligns rows whenever the values index is non-unique or a
    # duplicate (date, asset) key in exposures multiplies merged rows (same
    # class of defect fixed in the pandas neutralizers via
    # ``neutralization._alignment``).
    if isinstance(values, pd.DataFrame):
        pl_values = pl.from_pandas(values)
        return_pandas = True
    else:
        pl_values = values
        return_pandas = False
    pl_values = pl_values.with_row_index("__values_pos__")

    if isinstance(exposures, pd.DataFrame):
        pl_exposures = pl.from_pandas(exposures)
    else:
        pl_exposures = exposures

    # Merge values and exposures
    merged = pl_values.join(
        pl_exposures,
        on=[date_col, asset_col],
        how="left",
    )

    # Get exposure columns
    exposure_cols = [c for c in pl_exposures.columns if c not in [date_col, asset_col]]

    if not exposure_cols:
        raise ValueError("No exposure columns found")

    # Convert to pandas for groupby + numpy lstsq
    # (polars doesn't have native OLS)
    merged_pd = merged.to_pandas()

    results = []

    for date, group in merged_pd.groupby(date_col):
        # Extract y and X
        y = group[value_col].values
        X = group[exposure_cols].values

        # Drop rows with any NaN in y or X
        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        n_valid = np.sum(valid_mask)

        if n_valid < min_observations or n_valid <= X.shape[1]:
            # Insufficient data or rank deficient
            residuals = np.full_like(y, np.nan)
        else:
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            if add_intercept:
                X_valid = np.column_stack([np.ones(n_valid), X_valid])

            # Fit OLS using lstsq
            try:
                coef, _, rank, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)

                # Predict and compute residuals
                if add_intercept:
                    X_all = np.column_stack([np.ones(len(y)), X])
                else:
                    X_all = X

                y_pred = X_all @ coef
                residuals = y - y_pred

                # NaN inputs produce NaN residuals
                residuals[~valid_mask] = np.nan

            except np.linalg.LinAlgError:
                # Singular matrix
                residuals = np.full_like(y, np.nan)

        # Build result frame keyed by the ORIGINAL values row position
        result_df = pd.DataFrame({
            "__values_pos__": group["__values_pos__"].to_numpy(),
            "residual": residuals,
        })
        result_df = result_df.set_index("__values_pos__")

        results.append(result_df)

    n = len(pl_values)
    # Concatenate and realign to the original values row order (positionally).
    # Rows absent from the merge become NaN; duplicated positions (duplicate
    # (date, asset) exposure keys multiply merged rows) keep the first.
    if not results:
        if return_pandas:
            return pd.Series(np.nan, index=values.index if return_pandas else None)
        else:
            return pl.Series("residual", [np.nan] * n)

    all_results = pd.concat(results)
    all_results = all_results[~all_results.index.duplicated(keep="first")]
    residuals_series = all_results["residual"].reindex(range(n))
    residuals_series.name = "residual"

    if return_pandas:
        residuals_series.index = values.index
        return residuals_series
    else:
        return pl.from_pandas(residuals_series.to_frame()).to_series()


__all__ = [
    "cs_rank_polars",
    "cs_zscore_polars",
    "cs_demean_polars",
    "cs_winsor_polars",
    "cs_scale_polars",
    "ols_neutralize_polars",
    "POLARS_AVAILABLE",
]
