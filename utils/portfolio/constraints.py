"""Portfolio constraint specification and enforcement.

Provides constraint checking and projection for:
- Position limits (min/max weights per asset)
- Leverage limits (sum of absolute weights)
- Turnover limits (sum of absolute weight changes)
- Sector/group exposure limits
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class PortfolioConstraints:
    """Portfolio constraint specification.

    Attributes:
        min_weight: Minimum weight per asset (default 0.0 for long-only)
        max_weight: Maximum weight per asset (None = no limit)
        max_leverage: Maximum sum of absolute weights (None = no limit)
        max_turnover: Maximum sum of absolute weight changes (None = no limit)
        target_sum: Target sum of weights (1.0 = fully invested, 0.0 = market-neutral)
        group_limits: Dict mapping group name to (min_exposure, max_exposure)
        group_membership: Dict mapping asset to group names (list)
    """
    min_weight: float = 0.0
    max_weight: Optional[float] = None
    max_leverage: Optional[float] = None
    max_turnover: Optional[float] = None
    target_sum: float = 1.0
    group_limits: Dict[str, tuple[float, float]] = field(default_factory=dict)
    group_membership: Dict[str, list] = field(default_factory=dict)


def check_constraints(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    current_weights: Optional[pd.Series] = None,
    tol: float = 1e-6,
) -> Dict[str, bool]:
    """Check if portfolio satisfies constraints.

    Args:
        weights: Target portfolio weights
        constraints: Constraint specification
        current_weights: Current weights (for turnover check)
        tol: Numerical tolerance

    Returns:
        Dict mapping constraint name to whether it's satisfied
    """
    results = {}

    # Weight sum
    weight_sum = weights.sum()
    results['target_sum'] = abs(weight_sum - constraints.target_sum) < tol

    # Min weight
    if constraints.min_weight is not None:
        results['min_weight'] = (weights >= constraints.min_weight - tol).all()

    # Max weight
    if constraints.max_weight is not None:
        results['max_weight'] = (weights <= constraints.max_weight + tol).all()

    # Leverage
    if constraints.max_leverage is not None:
        leverage = weights.abs().sum()
        results['max_leverage'] = leverage <= constraints.max_leverage + tol

    # Turnover
    if constraints.max_turnover is not None and current_weights is not None:
        # Align indices
        aligned_current = current_weights.reindex(weights.index, fill_value=0.0)
        turnover = (weights - aligned_current).abs().sum()
        results['max_turnover'] = turnover <= constraints.max_turnover + tol

    # Group limits
    for group_name, (min_exp, max_exp) in constraints.group_limits.items():
        if group_name not in constraints.group_membership:
            results[f'group_{group_name}'] = False
            continue

        group_assets = constraints.group_membership[group_name]
        group_weights = weights.reindex(group_assets, fill_value=0.0)
        group_exposure = group_weights.sum()

        satisfied = (min_exp - tol <= group_exposure <= max_exp + tol)
        results[f'group_{group_name}'] = satisfied

    return results


def project_to_constraints(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    current_weights: Optional[pd.Series] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> pd.Series:
    """Project weights to satisfy constraints.

    Uses iterative projection method. May not converge for infeasible constraints.

    Args:
        weights: Initial weights
        constraints: Constraint specification
        current_weights: Current weights (for turnover constraint)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Projected weights (best effort)
    """
    w = weights.copy()

    for iteration in range(max_iter):
        w_old = w.copy()

        # Project to position limits
        if constraints.min_weight is not None:
            w = w.clip(lower=constraints.min_weight)

        if constraints.max_weight is not None:
            w = w.clip(upper=constraints.max_weight)

        # Project to target sum
        current_sum = w.sum()
        if abs(current_sum) > 1e-12:
            w = w * (constraints.target_sum / current_sum)

        # Project to leverage limit
        if constraints.max_leverage is not None:
            leverage = w.abs().sum()
            if leverage > constraints.max_leverage:
                # Scale down proportionally
                w = w * (constraints.max_leverage / leverage)

        # Project to turnover limit
        if constraints.max_turnover is not None and current_weights is not None:
            aligned_current = current_weights.reindex(w.index, fill_value=0.0)
            trades = w - aligned_current
            turnover = trades.abs().sum()

            if turnover > constraints.max_turnover:
                # Scale trades proportionally
                scale = constraints.max_turnover / turnover
                w = aligned_current + trades * scale

        # Project to group limits
        for group_name, (min_exp, max_exp) in constraints.group_limits.items():
            if group_name not in constraints.group_membership:
                continue

            group_assets = constraints.group_membership[group_name]
            group_mask = w.index.isin(group_assets)
            group_exposure = w[group_mask].sum()

            if group_exposure < min_exp:
                # Increase group weights proportionally
                if group_mask.any():
                    shortfall = min_exp - group_exposure
                    w[group_mask] += shortfall / group_mask.sum()

            elif group_exposure > max_exp:
                # Decrease group weights proportionally
                if group_mask.any():
                    excess = group_exposure - max_exp
                    w[group_mask] -= excess / group_mask.sum()

        # Check convergence
        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


def apply_position_limits(
    weights: pd.Series,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    target_sum: float = 1.0,
) -> pd.Series:
    """Simple position limit enforcement with renormalization.

    Args:
        weights: Input weights
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        target_sum: Target weight sum after clipping

    Returns:
        Clipped and renormalized weights
    """
    w = weights.clip(lower=min_weight)
    if max_weight is not None:
        w = w.clip(upper=max_weight)

    # Renormalize
    current_sum = w.sum()
    if abs(current_sum) > 1e-12:
        w = w * (target_sum / current_sum)

    return w


def calculate_leverage(weights: pd.Series) -> float:
    """Calculate portfolio leverage (sum of absolute weights).

    Args:
        weights: Portfolio weights

    Returns:
        Leverage ratio
    """
    return weights.abs().sum()


def calculate_turnover(
    target_weights: pd.Series,
    current_weights: pd.Series,
) -> float:
    """Calculate portfolio turnover (sum of absolute weight changes).

    Args:
        target_weights: Target portfolio weights
        current_weights: Current portfolio weights

    Returns:
        Turnover (one-way)
    """
    # Align indices
    all_assets = target_weights.index.union(current_weights.index)
    target = target_weights.reindex(all_assets, fill_value=0.0)
    current = current_weights.reindex(all_assets, fill_value=0.0)

    return (target - current).abs().sum()


def calculate_net_exposure(weights: pd.Series) -> float:
    """Calculate net exposure (sum of weights).

    Args:
        weights: Portfolio weights

    Returns:
        Net exposure
    """
    return weights.sum()


def calculate_long_short_exposure(weights: pd.Series) -> tuple[float, float]:
    """Calculate long and short exposure separately.

    Args:
        weights: Portfolio weights

    Returns:
        Tuple of (long_exposure, short_exposure)
    """
    long_exp = weights[weights > 0].sum()
    short_exp = weights[weights < 0].sum()
    return long_exp, short_exp
