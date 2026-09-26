"""
Polars-based backend for large-scale IC and quantile computation.

Leverages polars lazy evaluation, streaming execution, and efficient groupby/window
operations to achieve 5-10x memory efficiency and 2-5x speed vs pandas.
"""

from typing import Tuple, Optional
import numpy as np

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


class PolarsBackendUnavailable(ImportError):
    """Raised when polars backend is requested but not installed."""
    def __init__(self):
        super().__init__(
            "Polars backend requires polars>=0.19.0. Install with: pip install polars"
        )


def require_polars():
    """Check polars availability and raise if missing."""
    if not POLARS_AVAILABLE:
        raise PolarsBackendUnavailable()


def factorbatch_to_lazyframe(
    factor_values: np.ndarray,
    factor_ids: Tuple[str, ...],
    time_values: Optional[np.ndarray] = None,
    asset_values: Optional[np.ndarray] = None,
    factor_validity: Optional[np.ndarray] = None,
) -> pl.LazyFrame:
    """
    Convert FactorBatch to polars LazyFrame for efficient processing.

    Converts wide format (T, N, F) to long format with columns:
    [time_idx, asset_idx, factor_id, value]

    Args:
        factor_values: Factor batch (T, N, F)
        factor_ids: Factor identifiers
        time_values: Optional time axis values
        asset_values: Optional asset axis values
        factor_validity: Optional validity mask (T, N, F)

    Returns:
        LazyFrame in long format ready for groupby operations
    """
    require_polars()

    T, N, F = factor_values.shape

    # Apply validity mask
    if factor_validity is not None:
        values = np.where(factor_validity, factor_values, np.nan)
    else:
        values = factor_values

    # Build long format data
    # Create indices
    time_idx = np.repeat(np.arange(T), N * F)
    asset_idx = np.tile(np.repeat(np.arange(N), F), T)
    factor_idx = np.tile(np.arange(F), T * N)
    factor_id_arr = np.array([factor_ids[i] for i in factor_idx])

    # Flatten values
    value_flat = values.ravel()

    # Build dataframe
    data = {
        "time_idx": time_idx,
        "asset_idx": asset_idx,
        "factor_id": factor_id_arr,
        "value": value_flat,
    }

    # Add actual time/asset values if provided
    if time_values is not None:
        data["time"] = time_values[time_idx]
    if asset_values is not None:
        data["asset"] = asset_values[asset_idx]

    # Create eager dataframe then convert to lazy
    df = pl.DataFrame(data)
    return df.lazy()


def polars_ic_batch(
    factor_values: np.ndarray,
    label_values: np.ndarray,
    factor_ids: Tuple[str, ...],
    method: str = "pearson",
    min_obs: int = 10,
    factor_validity: Optional[np.ndarray] = None,
    label_validity: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Polars-based IC computation using lazy evaluation and streaming.

    Converts to long format, computes correlations using polars groupby,
    then reshapes back to (T, F) format.

    Args:
        factor_values: Factor batch (T, N, F)
        label_values: Labels (T, N) or (T,)
        factor_ids: Factor identifiers
        method: "pearson" or "spearman"
        min_obs: Minimum valid observations per period
        factor_validity: Optional validity mask (T, N, F)
        label_validity: Optional validity mask (T, N)

    Returns:
        (ic_matrix, valid_counts)
        ic_matrix: shape (T, F) with IC per day per factor
        valid_counts: shape (T, F) with count of valid obs per day per factor
    """
    require_polars()

    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}")

    T, N, F = factor_values.shape

    # Broadcast labels if needed
    if label_values.ndim == 1:
        labels = np.broadcast_to(label_values[:, np.newaxis], (T, N))
    elif label_values.shape == (T, N):
        labels = label_values
    else:
        raise ValueError(f"Invalid label shape: {label_values.shape}")

    # Apply validity masks
    # The canonical IC reference promotes compressed cross-sections to
    # float64 before correlation.  Keeping float32 columns here makes the
    # grouped moment calculation differ beyond the accepted parity bound
    # for tied/near-zero IC panels.
    if np.asarray(factor_values).dtype.kind not in "iuf" or np.asarray(labels).dtype.kind not in "iuf":
        raise TypeError("factor and label values must be real numeric arrays")
    factors = np.asarray(factor_values, dtype=np.float64)
    if factor_validity is not None:
        factors = np.where(factor_validity, factors, np.nan)

    labels_bcast = np.asarray(labels, dtype=np.float64)
    if label_validity is not None:
        if label_validity.ndim == 1:
            label_validity = np.broadcast_to(label_validity[:, np.newaxis], (T, N))
        elif label_validity.shape != (T, N):
            raise ValueError(
                f"Invalid label validity shape: {label_validity.shape}, "
                f"expected (T,) or (T, N)"
            )
        labels_bcast = np.where(label_validity, labels_bcast, np.nan)

    # Convert to long format.
    # Performance note (2026-09-25 AB benchmark): grouping used to include a
    # per-row string ``factor_id`` column built with a Python list
    # comprehension over T*N*F rows.  ``factor_idx`` is already a group key
    # that identifies the same partition, so the string column was pure
    # overhead and has been removed.  ``factor_ids`` remains a required
    # parameter for signature compatibility and output labeling.
    time_idx = np.repeat(np.arange(T), N * F)
    asset_idx = np.tile(np.repeat(np.arange(N), F), T)
    factor_idx = np.tile(np.arange(F), T * N)

    value_flat = factors.ravel()

    # Expand labels to match factor dimensions
    label_expanded = np.repeat(labels_bcast.ravel(), F)

    # Build dataframe with both factor values and labels
    df = pl.DataFrame({
        "time_idx": time_idx,
        "factor_idx": factor_idx,
        "value": value_flat,
        "label": label_expanded,
    })

    # Filter out NaN values for pairwise-complete correlations
    df_clean = df.filter(
        pl.col("value").is_finite() & pl.col("label").is_finite()
    )

    # Correlation via grouped SUM aggregations + vectorized post-arithmetic
    # (2026-09-25 rewrite).  Per-group ``pl.corr`` was 2.5x slower than one
    # groupby emitting the raw moment sums; and the moments are computed on
    # WINDOW-CENTERED deviations (native ``mean().over`` pass) so offset data
    # cannot trigger catastrophic cancellation in the raw-moment formula.
    # This is pure polars end to end: numpy panels enter once as columns,
    # every reduction after that is a native expression.
    lf = df.lazy().filter(pl.col("value").is_finite() & pl.col("label").is_finite())

    # F == 1 collapses to a single group key: hashing (time_idx, factor_idx)
    # with a constant factor column measurably regressed the dominant
    # single-factor case (39ms -> matches the 11ms prototype with one key).
    group_keys = ["time_idx"] if F == 1 else ["time_idx", "factor_idx"]

    if method == "spearman":
        lf = lf.with_columns([
            pl.col("value").rank(method="average").over(group_keys).alias("value"),
            pl.col("label").rank(method="average").over(group_keys).alias("label"),
        ])

    lf = lf.with_columns([
        (pl.col("value") - pl.col("value").mean().over(group_keys)).alias("value"),
        (pl.col("label") - pl.col("label").mean().over(group_keys)).alias("label"),
    ])

    ic_result = (
        lf.group_by(group_keys)
        .agg([
            pl.len().alias("n_obs"),
            (pl.col("value") * pl.col("value")).sum().alias("svv"),
            (pl.col("label") * pl.col("label")).sum().alias("syy"),
            (pl.col("value") * pl.col("label")).sum().alias("svy"),
        ])
        .filter(pl.col("n_obs") >= min_obs)
        .collect()
    )

    # Convert back to (T, F) format
    ic_matrix = np.full((T, F), np.nan, dtype=np.float64)
    valid_counts = np.zeros((T, F), dtype=np.int32)

    if len(ic_result) > 0:
        times = ic_result["time_idx"].to_numpy()
        if F == 1:
            factor_indices = np.zeros(len(times), dtype=np.int64)
        else:
            factor_indices = ic_result["factor_idx"].to_numpy()
        n_obs = ic_result["n_obs"].to_numpy().astype(np.float64)
        svv = ic_result["svv"].to_numpy()
        syy = ic_result["syy"].to_numpy()
        svy = ic_result["svy"].to_numpy()

        denom = np.sqrt(svv * syy)
        with np.errstate(invalid="ignore", divide="ignore"):
            ic_vals = np.where(denom > 0, svy / np.where(denom > 0, denom, 1.0), np.nan)
        ic_vals = np.where(n_obs >= min_obs, ic_vals, np.nan)

        ic_matrix[times, factor_indices] = ic_vals
        valid_counts[times, factor_indices] = n_obs.astype(np.int32)

    return ic_matrix, valid_counts


def polars_quantile_binning(
    factor_values: np.ndarray,
    factor_ids: Tuple[str, ...],
    n_quantiles: int = 5,
    min_valid: Optional[int] = None,
) -> np.ndarray:
    """
    Polars-based quantile binning using lazy evaluation.

    Args:
        factor_values: Factor batch (T, N, F)
        factor_ids: Factor identifiers
        n_quantiles: Number of quantiles
        min_valid: Minimum valid assets per period

    Returns:
        Quantile assignments (T, N, F) with dtype int32, -1 for invalid
    """
    require_polars()

    if min_valid is None:
        min_valid = n_quantiles

    T, N, F = factor_values.shape

    # Convert to long format
    time_idx = np.repeat(np.arange(T), N * F)
    asset_idx = np.tile(np.repeat(np.arange(N), F), T)
    factor_idx = np.tile(np.arange(F), T * N)

    value_flat = factor_values.ravel()

    df = pl.DataFrame({
        "time_idx": time_idx,
        "asset_idx": asset_idx,
        "factor_idx": factor_idx,
        "value": value_flat,
    })

    # QE-Q-P0-002: percentile boundaries + tie-policy comparison (reference
    # semantics, QuantileTiePolicy.MAX default). The previous rank-floor
    # formula disagreed with the reference whenever n_valid was not divisible
    # by n_quantiles.  Boundaries are computed in NumPy via the shared
    # _percentile_boundaries helper (single source of truth, including the
    # 1e-9 integer-position snap: polars' own quantile interpolation can land
    # one ulp off the data value there, silently flipping the tie policy for
    # the value sitting exactly on the boundary) and joined in as per-group
    # literal columns.
    from quant_evaluator.metrics.quantile import _percentile_boundaries

    n_boundaries = n_quantiles - 1

    # Per-(t, f) boundaries computed in NumPy (parity with the reference).
    boundary_rows = []
    for t in range(T):
        for f in range(F):
            v = factor_values[t, :, f]
            v_finite = v[np.isfinite(v)]
            if v_finite.shape[0] < min_valid:
                continue
            bounds = _percentile_boundaries(v_finite, n_quantiles)
            row = {"time_idx": t, "factor_idx": f}
            for b in range(n_boundaries):
                row[f"boundary_{b}"] = float(bounds[b])
            boundary_rows.append(row)

    if not boundary_rows:
        return np.full((T, N, F), -1, dtype=np.int32)

    boundaries_df = pl.DataFrame(boundary_rows)

    quantile_result = (
        df
        .lazy()
        .filter(pl.col("value").is_finite())
        .join(boundaries_df.lazy(), on=["time_idx", "factor_idx"], how="inner")
        .with_columns([
            # searchsorted side='right': boundary values go to the HIGHER bin
            pl.sum_horizontal([
                (pl.col("value") >= pl.col(f"boundary_{b}")).cast(pl.Int32)
                for b in range(n_boundaries)
            ]).clip(0, n_quantiles - 1).alias("quantile")
            if n_boundaries > 0
            else pl.lit(0, dtype=pl.Int32).alias("quantile"),
        ])
        .select(["time_idx", "asset_idx", "factor_idx", "quantile"])
        .collect()
    )

    # Convert back to (T, N, F) format
    quantiles = np.full((T, N, F), -1, dtype=np.int32)

    if len(quantile_result) > 0:
        times = quantile_result["time_idx"].to_numpy()
        assets = quantile_result["asset_idx"].to_numpy()
        factors = quantile_result["factor_idx"].to_numpy()
        q_vals = quantile_result["quantile"].to_numpy()

        quantiles[times, assets, factors] = q_vals

    return quantiles


def polars_quantile_returns(
    factor_values: np.ndarray,
    label_values: np.ndarray,
    factor_ids: Tuple[str, ...],
    n_quantiles: int = 5,
    min_assets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Polars-based quantile return computation.

    Args:
        factor_values: Factor batch (T, N, F)
        label_values: Forward returns (T, N)
        factor_ids: Factor identifiers
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: shape (T, n_quantiles, F)
        quantile_counts: shape (T, n_quantiles, F)
    """
    require_polars()

    T, N, F = factor_values.shape

    # Broadcast labels if needed
    if label_values.ndim == 1:
        labels = np.broadcast_to(label_values[:, np.newaxis], (T, N))
    else:
        labels = label_values

    # Get quantile assignments
    q_assignments = polars_quantile_binning(factor_values, factor_ids, n_quantiles=n_quantiles)

    # Convert to long format
    time_idx = np.repeat(np.arange(T), N * F)
    factor_idx = np.tile(np.arange(F), T * N)

    quantile_flat = q_assignments.ravel()
    label_expanded = np.repeat(labels.ravel(), F)

    df = pl.DataFrame({
        "time_idx": time_idx,
        "factor_idx": factor_idx,
        "quantile": quantile_flat,
        "label": label_expanded,
    })

    # Compute mean returns per quantile
    # NOTE: the count threshold must be applied AFTER aggregating (and the
    # raw count reported regardless), matching compute_quantile_returns_fast.
    quantile_stats = (
        df
        .lazy()
        .filter((pl.col("quantile") >= 0) & pl.col("label").is_finite())
        .group_by(["time_idx", "factor_idx", "quantile"])
        .agg([
            pl.col("label").mean().alias("mean_return"),
            pl.len().alias("count"),
        ])
        .collect()
    )

    # Convert back to (T, n_quantiles, F) format
    quantile_returns = np.full((T, n_quantiles, F), np.nan, dtype=np.float64)
    quantile_counts = np.zeros((T, n_quantiles, F), dtype=np.int32)

    if len(quantile_stats) > 0:
        times = quantile_stats["time_idx"].to_numpy()
        factors = quantile_stats["factor_idx"].to_numpy()
        quantiles = quantile_stats["quantile"].to_numpy()
        means = quantile_stats["mean_return"].to_numpy()
        counts = quantile_stats["count"].to_numpy()

        quantile_counts[times, quantiles, factors] = counts
        sufficient = counts >= min_assets
        quantile_returns[times[sufficient], quantiles[sufficient], factors[sufficient]] = (
            means[sufficient]
        )

    return quantile_returns, quantile_counts
