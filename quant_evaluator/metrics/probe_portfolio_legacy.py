"""
Portfolio construction and backtesting probe metrics.

Provides equal-weighted and cap-weighted long-short portfolio construction
with detailed position tracking and performance analytics.
"""

from typing import Tuple, Optional, Dict
import numpy as np


def _bucket_guard(
    factor_valid: np.ndarray,
    long_mask: np.ndarray,
    short_mask: np.ndarray,
    min_bucket_size: int,
) -> Optional[str]:
    """QE-METRIC P0-11: fail-closed guards for degenerate quantile cuts.

    Returns None when the long/short construction is sound, otherwise a
    diagnostic string describing the violation:
      - quantile cutoffs collapse to the same value (e.g. a constant
        factor: long and short cutoffs are identical, buckets overlap and
        the long-short spread is a fake 0);
      - long and short buckets are not disjoint;
      - either bucket has fewer than ``min_bucket_size`` members.
    """
    if np.sum(long_mask & short_mask) > 0:
        return (
            f"long/short buckets overlap on {int(np.sum(long_mask & short_mask))} "
            "assets (quantile cutoffs collapsed)"
        )
    if np.sum(long_mask) < min_bucket_size:
        return (
            f"long bucket has {int(np.sum(long_mask))} members, "
            f"below floor {min_bucket_size}"
        )
    if np.sum(short_mask) < min_bucket_size:
        return (
            f"short bucket has {int(np.sum(short_mask))} members, "
            f"below floor {min_bucket_size}"
        )
    return None


def construct_long_short_portfolio(
    factor_values: np.ndarray,
    long_threshold: float = 0.8,
    short_threshold: float = 0.2,
    validity_mask: Optional[np.ndarray] = None,
    min_bucket_size: int = 1,
    strict: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct long/short portfolio positions based on factor quantiles.

    QE-METRIC P0-11 degenerate-bucket guards: a constant (or
    low-cardinality) factor makes the long and short cutoff quantiles
    identical, so the "long" and "short" buckets select the same assets
    and any long-short spread computed from them is a fake 0. Guards
    applied per cross-section:

    1. Disjointness: if the long and short buckets share any asset (which
       also catches cutoff collapse on constant factors), or either bucket
       is smaller than ``min_bucket_size``, the period is treated as
       degenerate.
    2. Low cardinality: if the valid factor values have fewer than 2
       unique values, the quantile cut is meaningless and the period is
       degenerate.

    Degenerate periods leave the position masks empty (no positions). With
    ``strict=True`` a ValueError is raised instead, naming the first
    offending period — use this to fail closed in pipelines that would
    otherwise silently consume a zero spread.

    Args:
        factor_values: Factor values (T, N) or (T, N, F)
        long_threshold: Quantile threshold for long positions (default 0.8 = top 20%)
        short_threshold: Quantile threshold for short positions (default 0.2 = bottom 20%)
        validity_mask: Optional boolean mask (T, N) or (T, N, F)
        min_bucket_size: Minimum members required in each bucket
            (default 1 = only disjointness/overlap enforcement; set higher,
            e.g. 2-5, to require statistically meaningful buckets)
        strict: If True, raise ValueError on the first degenerate
            cross-section instead of returning empty positions for it

    Returns:
        (long_positions, short_positions)
        long_positions: Boolean mask for long positions, shape (T, N) or (T, N, F)
        short_positions: Boolean mask for short positions, shape (T, N) or (T, N, F)

    Raises:
        ValueError: In ``strict`` mode, when any cross-section fails the
            bucket guards
    """
    factor_values = np.asarray(factor_values, dtype=float)
    if factor_values.ndim not in (2, 3):
        raise ValueError("factor_values must have layout (T,N) or (T,N,F)")
    if validity_mask is not None:
        validity_mask = np.asarray(validity_mask)
        if validity_mask.dtype != np.bool_:
            raise TypeError("validity_mask must be boolean")
        allowed = {(factor_values.shape[0], factor_values.shape[1])}
        if factor_values.ndim == 3:
            allowed.add(factor_values.shape)
        if validity_mask.shape not in allowed:
            raise ValueError(f"validity_mask shape {validity_mask.shape} incompatible with {factor_values.shape}")
    # Handle 3D factor values
    if factor_values.ndim == 3:
        T, N, F = factor_values.shape
        long_pos = np.zeros((T, N, F), dtype=bool)
        short_pos = np.zeros((T, N, F), dtype=bool)

        for f in range(F):
            fv = factor_values[:, :, f]
            vm = (validity_mask if validity_mask.ndim == 2 else validity_mask[:, :, f]) if validity_mask is not None else None
            long_pos[:, :, f], short_pos[:, :, f] = construct_long_short_portfolio(
                fv, long_threshold, short_threshold, vm,
                min_bucket_size=min_bucket_size, strict=strict,
            )
        return long_pos, short_pos

    # 2D case
    T, N = factor_values.shape
    long_positions = np.zeros((T, N), dtype=bool)
    short_positions = np.zeros((T, N), dtype=bool)

    for t in range(T):
        factor_t = factor_values[t, :]

        # Apply validity mask
        if validity_mask is not None:
            valid = validity_mask[t, :]
            factor_t = np.where(valid, factor_t, np.nan)

        # Filter finite values
        finite_mask = np.isfinite(factor_t)
        if np.sum(finite_mask) < 2:
            continue

        factor_valid = factor_t[finite_mask]

        # P0-11 guard: low-cardinality cross-section — fewer than 2 unique
        # factor values means the quantile cut is meaningless.
        if np.unique(factor_valid).size < 2:
            if strict:
                raise ValueError(
                    f"construct_long_short_portfolio: period {t} has "
                    f"{np.unique(factor_valid).size} unique factor value(s); "
                    "quantile cut is degenerate (constant factor?)"
                )
            continue

        # Compute quantiles
        long_cutoff = np.quantile(factor_valid, long_threshold)
        short_cutoff = np.quantile(factor_valid, short_threshold)

        # Select long/short positions
        long_mask = finite_mask & (factor_t >= long_cutoff)
        short_mask = finite_mask & (factor_t <= short_cutoff)

        # P0-11 guards: disjointness + bucket-size floor.
        diag = _bucket_guard(
            factor_valid, long_mask[finite_mask], short_mask[finite_mask],
            min_bucket_size,
        )
        if diag is not None:
            if strict:
                raise ValueError(
                    f"construct_long_short_portfolio: period {t} failed "
                    f"bucket guard: {diag}"
                )
            continue

        long_positions[t, :] = long_mask
        short_positions[t, :] = short_mask

    return long_positions, short_positions


def compute_equal_weighted_returns(
    positions: np.ndarray,
    forward_returns: np.ndarray,
) -> np.ndarray:
    """
    Compute equal-weighted portfolio returns.

    Args:
        positions: Boolean position mask (T, N) or (T, N, F)
        forward_returns: Forward returns (T, N)

    Returns:
        Portfolio returns (T,) or (T, F)
    """
    positions = np.asarray(positions)
    forward_returns = np.asarray(forward_returns, dtype=float)
    if positions.dtype != np.bool_:
        raise TypeError("positions must be boolean")
    if positions.ndim not in (2, 3) or forward_returns.shape != positions.shape[:2]:
        raise ValueError("positions/returns axes do not align")
    if positions.ndim == 3:
        T, N, F = positions.shape
        portfolio_returns = np.full((T, F), np.nan)

        for f in range(F):
            pos_f = positions[:, :, f]
            portfolio_returns[:, f] = compute_equal_weighted_returns(pos_f, forward_returns)

        return portfolio_returns

    # 2D case
    T, N = positions.shape
    portfolio_returns = np.full(T, np.nan)

    for t in range(T):
        pos_t = positions[t, :]
        ret_t = forward_returns[t, :]

        # Valid positions with finite returns
        if np.any(pos_t & ~np.isfinite(ret_t)):
            continue  # Unknown held return cannot be removed then reweighted.
        valid_mask = pos_t & np.isfinite(ret_t)
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            continue

        # Equal-weighted average
        portfolio_returns[t] = np.mean(ret_t[valid_mask])

    return portfolio_returns


def compute_cap_weighted_returns(
    positions: np.ndarray,
    forward_returns: np.ndarray,
    market_caps: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute cap-weighted portfolio returns.

    Args:
        positions: Boolean position mask (T, N) or (T, N, F)
        forward_returns: Forward returns (T, N)
        market_caps: Market capitalization values (T, N)
        normalize: If True, normalize weights to sum to 1 within each period

    Returns:
        Portfolio returns (T,) or (T, F)
    """
    positions=np.asarray(positions); forward_returns=np.asarray(forward_returns,dtype=float); market_caps=np.asarray(market_caps,dtype=float)
    if positions.dtype!=np.bool_: raise TypeError("positions must be boolean")
    if positions.ndim not in (2,3) or forward_returns.shape!=positions.shape[:2] or market_caps.shape!=positions.shape[:2]: raise ValueError("positions/returns/caps axes do not align")
    if positions.ndim == 3:
        T, N, F = positions.shape
        portfolio_returns = np.full((T, F), np.nan)

        for f in range(F):
            pos_f = positions[:, :, f]
            portfolio_returns[:, f] = compute_cap_weighted_returns(
                pos_f, forward_returns, market_caps, normalize
            )

        return portfolio_returns

    # 2D case
    T, N = positions.shape
    portfolio_returns = np.full(T, np.nan)

    for t in range(T):
        pos_t = positions[t, :]
        ret_t = forward_returns[t, :]
        cap_t = market_caps[t, :]

        # Valid positions with finite returns and caps
        if np.any(pos_t & (~np.isfinite(ret_t) | ~np.isfinite(cap_t) | (cap_t <= 0))):
            continue
        valid_mask = pos_t & np.isfinite(ret_t) & np.isfinite(cap_t) & (cap_t > 0)
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            continue

        weights = cap_t[valid_mask]

        if normalize:
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)

        # Cap-weighted return
        portfolio_returns[t] = np.sum(ret_t[valid_mask] * weights)

    return portfolio_returns


def compute_long_short_equal_weighted(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    long_threshold: float = 0.8,
    short_threshold: float = 0.2,
    validity_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute long/short equal-weighted portfolio returns.

    Convenience function combining portfolio construction and equal-weighted returns.

    Args:
        factor_values: Factor values (T, N) or (T, N, F)
        forward_returns: Forward returns (T, N)
        long_threshold: Quantile threshold for long positions
        short_threshold: Quantile threshold for short positions
        validity_mask: Optional boolean mask (T, N) or (T, N, F)

    Returns:
        (long_returns, short_returns, long_short_returns)
        Each shape (T,) or (T, F) for time series of portfolio returns
    """
    long_pos, short_pos = construct_long_short_portfolio(
        factor_values, long_threshold, short_threshold, validity_mask
    )

    long_returns = compute_equal_weighted_returns(long_pos, forward_returns)
    short_returns = compute_equal_weighted_returns(short_pos, forward_returns)

    # Compute long-short spread
    if long_returns.ndim == 1:
        long_short_returns = np.where(
            np.isfinite(long_returns) & np.isfinite(short_returns),
            long_returns - short_returns,
            np.nan
        )
    else:
        long_short_returns = np.where(
            np.isfinite(long_returns) & np.isfinite(short_returns),
            long_returns - short_returns,
            np.nan
        )

    from quant_evaluator.metrics.portfolio_stats import equal_gross_long_short_returns
    if long_pos.ndim == 2:
        long_short_returns = equal_gross_long_short_returns(long_pos, short_pos, forward_returns)
    else:
        long_short_returns = np.column_stack([
            equal_gross_long_short_returns(long_pos[:, :, f], short_pos[:, :, f], forward_returns)
            for f in range(long_pos.shape[2])])
    return long_returns, short_returns, long_short_returns


def compute_long_short_cap_weighted(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    market_caps: np.ndarray,
    long_threshold: float = 0.8,
    short_threshold: float = 0.2,
    validity_mask: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute long/short cap-weighted portfolio returns.

    Convenience function combining portfolio construction and cap-weighted returns.

    Args:
        factor_values: Factor values (T, N) or (T, N, F)
        forward_returns: Forward returns (T, N)
        market_caps: Market capitalization values (T, N)
        long_threshold: Quantile threshold for long positions
        short_threshold: Quantile threshold for short positions
        validity_mask: Optional boolean mask (T, N) or (T, N, F)
        normalize: If True, normalize weights to sum to 1 within each period

    Returns:
        (long_returns, short_returns, long_short_returns)
        Each shape (T,) or (T, F) for time series of portfolio returns
    """
    long_pos, short_pos = construct_long_short_portfolio(
        factor_values, long_threshold, short_threshold, validity_mask
    )

    long_returns = compute_cap_weighted_returns(long_pos, forward_returns, market_caps, normalize)
    short_returns = compute_cap_weighted_returns(short_pos, forward_returns, market_caps, normalize)

    # Compute long-short spread
    if long_returns.ndim == 1:
        long_short_returns = np.where(
            np.isfinite(long_returns) & np.isfinite(short_returns),
            long_returns - short_returns,
            np.nan
        )
    else:
        long_short_returns = np.where(
            np.isfinite(long_returns) & np.isfinite(short_returns),
            long_returns - short_returns,
            np.nan
        )

    return long_returns, short_returns, long_short_returns


def compute_portfolio_weights(
    positions: np.ndarray,
    market_caps: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute portfolio weights from positions.

    Args:
        positions: Boolean position mask (T, N) or (T, N, F)
        market_caps: Optional market cap values (T, N). If None, equal-weighted.
        normalize: If True, normalize weights to sum to 1 per period

    Returns:
        Portfolio weights (T, N) or (T, N, F), same shape as positions
    """
    weights = positions.astype(np.float64)

    if market_caps is not None:
        # Cap-weighted
        if positions.ndim == 2:
            # Broadcast market caps to positions
            weights = weights * market_caps
        elif positions.ndim == 3:
            # Broadcast market caps to (T, N, F)
            weights = weights * market_caps[:, :, np.newaxis]

    if normalize:
        # Normalize weights to sum to 1 per time period (and factor if 3D)
        if weights.ndim == 2:
            weight_sums = np.sum(weights, axis=1, keepdims=True)
            weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
            weights = weights / weight_sums
        elif weights.ndim == 3:
            weight_sums = np.sum(weights, axis=1, keepdims=True)
            weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
            weights = weights / weight_sums

    # Set zero positions to exactly zero
    weights = np.where(positions, weights, 0.0)

    return weights


def compute_portfolio_concentration(
    positions: np.ndarray,
    market_caps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Herfindahl-Hirschman Index (HHI) for portfolio concentration.

    HHI = sum of squared weights. Range [0, 1], where 1 = single asset.

    Args:
        positions: Boolean position mask (T, N) or (T, N, F)
        market_caps: Optional market cap values (T, N). If None, equal-weighted.

    Returns:
        HHI concentration index (T,) or (T, F)
    """
    weights = compute_portfolio_weights(positions, market_caps, normalize=True)

    if weights.ndim == 2:
        # Sum of squared weights per time period
        hhi = np.sum(weights ** 2, axis=1)
    elif weights.ndim == 3:
        # Sum of squared weights per time period and factor
        hhi = np.sum(weights ** 2, axis=1)

    # Set to NaN where no positions
    if weights.ndim == 2:
        n_positions = np.sum(positions, axis=1)
    else:
        n_positions = np.sum(positions, axis=1)

    hhi = np.where(n_positions > 0, hhi, np.nan)

    return hhi


def compute_turnover_from_positions(
    positions: np.ndarray,
    market_caps: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute portfolio turnover from position changes.

    Turnover = sum of absolute weight changes between periods.

    Args:
        positions: Boolean position mask (T, N) or (T, N, F)
        market_caps: Optional market cap values (T, N). If None, equal-weighted.
        normalize: If True, use normalized weights

    Returns:
        Turnover series (T-1,) or (T-1, F)
    """
    weights = compute_portfolio_weights(positions, market_caps, normalize)

    # Compute weight changes
    if weights.ndim == 2:
        weight_changes = np.diff(weights, axis=0)
        turnover = np.sum(np.abs(weight_changes), axis=1)
    elif weights.ndim == 3:
        weight_changes = np.diff(weights, axis=0)
        turnover = np.sum(np.abs(weight_changes), axis=1)

    return turnover
