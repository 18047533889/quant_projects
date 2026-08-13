"""Weight calculation utilities for portfolio construction.

Provides various weighting schemes:
- Equal weight
- Market capitalization weighted
- Risk parity (inverse variance)
- Inverse volatility
- Rank-based weights (signal-driven)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


def normalize_weights(
    weights: Union[pd.Series, np.ndarray],
    target_sum: float = 1.0,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
) -> Union[pd.Series, np.ndarray]:
    """Normalize weights to sum to target, with optional bounds.

    Args:
        weights: Raw weights (Series or array)
        target_sum: Target sum (default 1.0 for long-only, 0.0 for market-neutral)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset (None = no limit)

    Returns:
        Normalized weights with same type as input
    """
    is_series = isinstance(weights, pd.Series)
    w = weights.values if is_series else weights

    # Handle all-zero or all-nan
    if np.all(np.isnan(w)) or np.sum(np.abs(w[~np.isnan(w)])) == 0:
        result = np.full_like(w, np.nan)
        return pd.Series(result, index=weights.index) if is_series else result

    # Replace nan with zero
    w = np.nan_to_num(w, nan=0.0)

    # Apply bounds iteratively
    max_iter = 100
    for _ in range(max_iter):
        # Normalize
        current_sum = np.sum(w)
        if abs(current_sum) < 1e-12:
            break
        w = w * (target_sum / current_sum)

        # Clip bounds
        w = np.maximum(w, min_weight)
        if max_weight is not None:
            w = np.minimum(w, max_weight)

        # Check convergence
        if abs(np.sum(w) - target_sum) < 1e-8:
            break

    return pd.Series(w, index=weights.index) if is_series else w


def equal_weights(
    assets: Union[pd.Index, list, int],
    target_sum: float = 1.0,
) -> pd.Series:
    """Equal weight portfolio.

    Args:
        assets: Asset identifiers (Index/list) or number of assets (int)
        target_sum: Target weight sum (default 1.0)

    Returns:
        Series of equal weights
    """
    if isinstance(assets, int):
        n = assets
        assets = pd.Index(range(n))
    else:
        n = len(assets)

    if n == 0:
        return pd.Series(dtype=float)

    weight = target_sum / n
    return pd.Series(weight, index=assets)


def market_cap_weights(
    market_caps: pd.Series,
    target_sum: float = 1.0,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
) -> pd.Series:
    """Market capitalization weighted portfolio.

    Args:
        market_caps: Market capitalizations per asset
        target_sum: Target weight sum
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Cap-weighted portfolio
    """
    # Filter positive caps
    caps = market_caps.copy()
    caps[caps <= 0] = np.nan

    return normalize_weights(caps, target_sum, min_weight, max_weight)


def inverse_volatility_weights(
    volatilities: pd.Series,
    target_sum: float = 1.0,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
) -> pd.Series:
    """Inverse volatility weighted portfolio.

    Args:
        volatilities: Volatility (stddev) per asset
        target_sum: Target weight sum
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Inverse-vol weighted portfolio
    """
    vols = volatilities.copy()
    vols[vols <= 0] = np.nan

    # Inverse volatility
    inv_vol = 1.0 / vols

    return normalize_weights(inv_vol, target_sum, min_weight, max_weight)


def risk_parity_weights(
    covariance: pd.DataFrame,
    target_sum: float = 1.0,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> pd.Series:
    """Risk parity portfolio (equal risk contribution).

    Uses iterative method to find weights where each asset contributes
    equally to portfolio variance.

    Args:
        covariance: Covariance matrix (DataFrame)
        target_sum: Target weight sum
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Risk parity weights
    """
    n = len(covariance)
    assets = covariance.index

    # Start with inverse-vol as initial guess
    vols = np.sqrt(np.diag(covariance.values))
    w = 1.0 / vols
    w = w / np.sum(w)

    cov = covariance.values

    for iteration in range(max_iter):
        # Portfolio variance and marginal contributions
        port_var = w @ cov @ w
        if port_var < 1e-12:
            break

        # Marginal risk contribution: d(portfolio_std)/d(w_i)
        mrc = (cov @ w) / np.sqrt(port_var)

        # Risk contribution: w_i * MRC_i
        rc = w * mrc

        # Target: equal risk contribution
        target_rc = port_var / n

        # Update weights (gradient-like step)
        ratio = target_rc / (rc + 1e-12)
        w_new = w * ratio

        # Normalize
        w_new = w_new / np.sum(w_new)

        # Check convergence
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break

        w = w_new

    # Apply bounds
    weights = pd.Series(w, index=assets)
    return normalize_weights(weights, target_sum, min_weight, max_weight)


def rank_weights(
    signals: pd.Series,
    method: str = "linear",
    target_sum: float = 1.0,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    long_short: bool = False,
) -> pd.Series:
    """Rank-based weights from signals.

    Args:
        signals: Signal values per asset (higher = better)
        method: Ranking method
            - 'linear': weight proportional to rank
            - 'quadratic': weight proportional to rank^2
            - 'exponential': exponentially decaying weights
        target_sum: Target weight sum (use 0.0 for market-neutral)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        long_short: If True, long top half, short bottom half

    Returns:
        Rank-based weights
    """
    # Remove NaN
    sig = signals.dropna()
    if len(sig) == 0:
        return pd.Series(dtype=float)

    # Rank (1 = worst, n = best)
    ranks = sig.rank(method='average')
    n = len(ranks)

    if long_short:
        # Long top half, short bottom half
        median_rank = (n + 1) / 2
        raw_weights = ranks - median_rank

        if method == 'linear':
            weights = raw_weights
        elif method == 'quadratic':
            weights = np.sign(raw_weights) * (raw_weights ** 2)
        elif method == 'exponential':
            # Exponential on absolute rank distance
            weights = np.sign(raw_weights) * (np.exp(np.abs(raw_weights) / n) - 1)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # Long only
        if method == 'linear':
            weights = ranks
        elif method == 'quadratic':
            weights = ranks ** 2
        elif method == 'exponential':
            # More weight on top ranks
            weights = np.exp(ranks / n) - 1
        else:
            raise ValueError(f"Unknown method: {method}")

    return normalize_weights(pd.Series(weights, index=sig.index),
                           target_sum, min_weight, max_weight)
