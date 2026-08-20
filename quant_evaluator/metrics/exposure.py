"""
Factor exposure analysis: loadings, sector/style exposure decomposition.

Reference implementation for attribution and risk factor exposure measurement.
"""

from typing import Optional, Tuple
import numpy as np


def compute_factor_loadings(
    factor_values: np.ndarray,
    risk_factors: np.ndarray,
    intercept: bool = True,
    min_obs: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute factor loadings via cross-sectional OLS regression.

    For each time period, regress factor_values on risk_factors:
    factor[t, i] = alpha[t] + sum_k(beta[t, k] * risk_factor[t, i, k]) + epsilon[t, i]

    Args:
        factor_values: Factor values (T, N)
        risk_factors: Risk factor matrix (T, N, K) where K is number of risk factors
        intercept: Include intercept in regression
        min_obs: Minimum valid observations per period

    Returns:
        (loadings, r_squared, residuals)
        loadings: shape (T, K) or (T, K+1) if intercept=True
        r_squared: shape (T,)
        residuals: shape (T, N)
    """
    T, N = factor_values.shape
    K = risk_factors.shape[2]

    num_coefs = K + 1 if intercept else K
    loadings = np.full((T, num_coefs), np.nan, dtype=np.float64)
    r_squared = np.full(T, np.nan, dtype=np.float64)
    residuals = np.full((T, N), np.nan, dtype=np.float64)

    for t in range(T):
        y = factor_values[t, :]  # (N,)
        X = risk_factors[t, :, :]  # (N, K)

        # Filter finite observations
        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        y_valid = y[valid_mask]
        X_valid = X[valid_mask, :]

        if len(y_valid) < min_obs:
            continue

        # Add intercept column
        if intercept:
            X_valid = np.column_stack([np.ones(len(y_valid)), X_valid])

        # OLS: beta = (X'X)^-1 X'y
        try:
            XtX = X_valid.T @ X_valid
            Xty = X_valid.T @ y_valid
            beta = np.linalg.solve(XtX, Xty)

            loadings[t, :] = beta

            # Compute R^2
            y_pred = X_valid @ beta
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)

            if ss_tot > 0:
                r_squared[t] = 1.0 - ss_res / ss_tot
            else:
                r_squared[t] = np.nan

            # Compute residuals for all assets (NaN for invalid)
            resid_t = np.full(N, np.nan, dtype=np.float64)
            X_full = risk_factors[t, :, :]
            if intercept:
                X_full = np.column_stack([np.ones(N), X_full])
            resid_t[valid_mask] = y_valid - (X_valid @ beta)
            residuals[t, :] = resid_t

        except np.linalg.LinAlgError:
            # Singular matrix, skip this period
            continue

    return loadings, r_squared, residuals


def compute_sector_exposure(
    factor_values: np.ndarray,
    sector_labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    label_time: Optional[np.ndarray] = None,
    factor_time: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted factor exposure by sector.

    QE-METRIC P0-13 — point-in-time (PIT) sector labels:

    This function needs sector/industry labels *as of each factor
    observation date*. Static labels (a single (N,) vector applied to all
    T periods) embed an assumption that sector membership never changed
    over the sample — if a stock moved from Energy to Technology in 2023
    and you use 2024 labels, the 2021 sector exposures are attributed
    using look-ahead information.

    Full PIT support (a (T, N) label panel) is out of scope for this
    reference implementation; instead this function:

    - accepts ``label_time`` (the as-of timestamp of the provided static
      labels) and ``factor_time`` (the per-period timestamps of
      ``factor_values``, length T);
    - if both are provided, REJECTS the call (ValueError) whenever any
      factor period predates the label as-of time — i.e. it fails closed
      on detectable look-ahead rather than silently attributing exposures
      with future labels;
    - if they are not provided, the caller implicitly asserts labels are
      valid PIT for the whole sample; that assumption is documented here
      and is the caller's responsibility.

    Timestamps may be numpy datetime64, pandas Timestamps, or any
    totally-ordered 1D values; comparison is elementwise ``<``.

    Args:
        factor_values: Factor values (T, N)
        sector_labels: Sector membership (N,) - integer sector codes,
            assumed point-in-time for the sample unless ``label_time`` is
            given
        weights: Asset weights (T, N), defaults to equal-weight within sector
        label_time: As-of timestamp of ``sector_labels`` (scalar or
            length-1 array). Optional.
        factor_time: Per-period timestamps of ``factor_values``, length T.
            Optional; required for look-ahead validation.

    Returns:
        (sector_exposure, sector_counts)
        sector_exposure: shape (T, num_sectors) - mean factor value per sector
        sector_counts: shape (T, num_sectors) - valid asset count per sector

    Raises:
        ValueError: If ``factor_time`` is provided without ``label_time``
            (cannot validate look-ahead), if shapes are inconsistent, or
            if any factor period predates the label as-of time (detected
            look-ahead)
    """
    if factor_time is not None and label_time is None:
        raise ValueError(
            "compute_sector_exposure: factor_time provided without "
            "label_time — cannot validate sector labels against "
            "look-ahead. Pass label_time (the as-of date of "
            "sector_labels) or omit both to accept the documented "
            "static-label assumption."
        )

    if label_time is not None and factor_time is not None:
        factor_time_arr = np.asarray(factor_time)
        if factor_time_arr.shape != (factor_values.shape[0],):
            raise ValueError(
                f"factor_time must have shape ({factor_values.shape[0]},), "
                f"got {factor_time_arr.shape}"
            )
        label_time_arr = np.asarray(label_time)
        if label_time_arr.size != 1:
            raise ValueError(
                f"label_time must be a scalar as-of timestamp, got "
                f"shape {label_time_arr.shape}"
            )
        look_ahead = factor_time_arr < label_time_arr.reshape(1)[0]
        n_ahead = int(np.sum(look_ahead))
        if n_ahead > 0:
            first_idx = int(np.nonzero(look_ahead)[0][0])
            raise ValueError(
                f"compute_sector_exposure: {n_ahead} of "
                f"{factor_values.shape[0]} factor periods (first at index "
                f"{first_idx}) predate the sector-label as-of time — using "
                "these labels for those periods is look-ahead. Provide "
                "point-in-time labels or restrict the factor sample."
            )

    T, N = factor_values.shape

    unique_sectors = np.unique(sector_labels[~np.isnan(sector_labels)])
    num_sectors = len(unique_sectors)

    sector_exposure = np.full((T, num_sectors), np.nan, dtype=np.float64)
    sector_counts = np.zeros((T, num_sectors), dtype=np.int32)

    if weights is None:
        weights = np.ones((T, N), dtype=np.float64)

    for s_idx, sector in enumerate(unique_sectors):
        sector_mask = sector_labels == sector  # (N,)

        for t in range(T):
            factor_t = factor_values[t, :]
            weight_t = weights[t, :]

            # Valid assets in this sector
            valid_mask = sector_mask & np.isfinite(factor_t) & np.isfinite(weight_t) & (weight_t > 0)

            if not np.any(valid_mask):
                continue

            factor_sector = factor_t[valid_mask]
            weight_sector = weight_t[valid_mask]

            # Weighted mean
            weight_sum = np.sum(weight_sector)
            if weight_sum > 0:
                sector_exposure[t, s_idx] = np.sum(factor_sector * weight_sector) / weight_sum
                sector_counts[t, s_idx] = int(np.sum(valid_mask))

    return sector_exposure, sector_counts


def compute_style_exposure(
    factor_values: np.ndarray,
    style_factors: np.ndarray,
    style_names: Tuple[str, ...],
) -> np.ndarray:
    """
    Compute time-series average exposure to style factors.

    Regresses factor on style factors at each time period and averages loadings.

    Args:
        factor_values: Factor values (T, N)
        style_factors: Style factor matrix (T, N, K)
        style_names: Names of K style factors

    Returns:
        mean_exposures: shape (K,) - average loading on each style factor
    """
    T, N = factor_values.shape
    K = style_factors.shape[2]

    if K != len(style_names):
        raise ValueError(f"style_factors has {K} factors but {len(style_names)} names provided")

    loadings, _, _ = compute_factor_loadings(
        factor_values, style_factors, intercept=True, min_obs=10
    )

    # loadings shape: (T, K+1) with intercept as first column
    # Extract factor loadings (skip intercept)
    factor_loadings = loadings[:, 1:]  # (T, K)

    # Average across time
    with np.errstate(invalid='ignore'):
        mean_exposures = np.nanmean(factor_loadings, axis=0)

    return mean_exposures


def compute_concentration_hhi(
    factor_values: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Herfindahl-Hirschman Index (HHI) concentration metric.

    QE-METRIC P0-12: HHI is computed on GROSS (absolute) exposure. The
    previous implementation normalized the signed weighted exposure
    (weight_i * factor_i / sum_j(weight_j * factor_j)); when long and
    short exposures nearly cancel, that signed denominator approaches 0
    and the "shares" explode, producing arbitrary huge HHI values that
    say nothing about concentration. Using gross exposure,

        share_i = |weight_i * factor_i| / sum_j |weight_j * factor_j|
        HHI_t = sum_i share_i^2

    gives a well-defined concentration index in (0, 1]: 1/N when N assets
    hold equal absolute exposure, 1.0 when a single asset holds all of it.
    It is invariant to sign flips of any factor value (only the
    distribution of absolute exposure matters).

    Args:
        factor_values: Factor values (T, N)
        weights: Asset weights (T, N), defaults to equal-weight

    Returns:
        hhi: shape (T,) - concentration index per period, in (0, 1];
            NaN where no valid (finite, positive-weight) assets exist
    """
    T, N = factor_values.shape

    if weights is None:
        weights = np.ones((T, N), dtype=np.float64)

    hhi = np.full(T, np.nan, dtype=np.float64)

    for t in range(T):
        factor_t = factor_values[t, :]
        weight_t = weights[t, :]

        valid_mask = np.isfinite(factor_t) & np.isfinite(weight_t) & (weight_t > 0)

        if not np.any(valid_mask):
            continue

        factor_valid = factor_t[valid_mask]
        weight_valid = weight_t[valid_mask]

        # Normalize weights
        weight_valid = weight_valid / np.sum(weight_valid)

        # Gross (absolute) exposure: the denominator is the total absolute
        # exposure, strictly positive whenever at least one asset has
        # non-zero |factor|, so no near-zero-division explosion.
        gross_exposure = np.abs(weight_valid * factor_valid)
        total_gross = np.sum(gross_exposure)

        if total_gross > 0:
            exposure_shares = gross_exposure / total_gross
            hhi[t] = np.sum(exposure_shares ** 2)

    return hhi
