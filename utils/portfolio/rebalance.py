"""Portfolio rebalancing with transaction costs.

Provides rebalancing logic that accounts for:
- Transaction costs (linear, quadratic, or custom)
- Minimum trade sizes
- Constraints on trades
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Union
from dataclasses import dataclass


@dataclass
class TransactionCostModel:
    """Transaction cost model specification.

    Attributes:
        linear_bps: Linear cost in basis points (e.g., 10 = 10bps = 0.1%)
        quadratic_bps: Quadratic cost coefficient (for market impact)
        fixed_cost: Fixed cost per trade (in currency units)
        min_trade_size: Minimum trade size (ignore trades below this)
    """
    linear_bps: float = 0.0
    quadratic_bps: float = 0.0
    fixed_cost: float = 0.0
    min_trade_size: float = 0.0


def calculate_trades(
    target_weights: pd.Series,
    current_weights: pd.Series,
    min_trade_size: float = 0.0,
) -> pd.Series:
    """Calculate required trades to reach target weights.

    Args:
        target_weights: Target portfolio weights
        current_weights: Current portfolio weights
        min_trade_size: Ignore trades smaller than this (absolute value)

    Returns:
        Series of trades (positive = buy, negative = sell)
    """
    # Align indices
    all_assets = target_weights.index.union(current_weights.index)
    target = target_weights.reindex(all_assets, fill_value=0.0)
    current = current_weights.reindex(all_assets, fill_value=0.0)

    trades = target - current

    # Filter small trades
    if min_trade_size > 0:
        trades[trades.abs() < min_trade_size] = 0.0

    # Remove zero trades
    trades = trades[trades != 0.0]

    return trades


def apply_transaction_costs(
    trades: pd.Series,
    cost_model: Union[TransactionCostModel, float],
    portfolio_value: float = 1.0,
) -> pd.Series:
    """Calculate transaction costs for trades.

    Args:
        trades: Trade sizes (as fraction of portfolio)
        cost_model: Cost model or linear bps (if float)
        portfolio_value: Portfolio value for absolute cost calculation

    Returns:
        Series of transaction costs per asset (always positive)
    """
    if isinstance(cost_model, (int, float)):
        cost_model = TransactionCostModel(linear_bps=float(cost_model))

    costs = pd.Series(0.0, index=trades.index)

    for asset, trade in trades.items():
        abs_trade = abs(trade)

        if abs_trade == 0:
            continue

        # Linear cost
        if cost_model.linear_bps > 0:
            costs[asset] += abs_trade * (cost_model.linear_bps / 10000.0)

        # Quadratic cost (market impact)
        if cost_model.quadratic_bps > 0:
            costs[asset] += (abs_trade ** 2) * (cost_model.quadratic_bps / 10000.0)

        # Fixed cost
        if cost_model.fixed_cost > 0:
            costs[asset] += cost_model.fixed_cost / portfolio_value

    return costs


def rebalance_portfolio(
    target_weights: pd.Series,
    current_weights: pd.Series,
    cost_model: Optional[Union[TransactionCostModel, float]] = None,
    max_turnover: Optional[float] = None,
    portfolio_value: float = 1.0,
    optimize_costs: bool = False,
) -> dict:
    """Rebalance portfolio from current to target weights.

    Args:
        target_weights: Target portfolio weights
        current_weights: Current portfolio weights
        cost_model: Transaction cost model (None = no costs)
        max_turnover: Maximum allowed turnover (None = unlimited)
        portfolio_value: Portfolio value for cost calculation
        optimize_costs: If True, optimize trade-off between tracking and costs

    Returns:
        Dict with keys:
            - 'trades': Series of trades
            - 'final_weights': Actual weights after rebalancing
            - 'transaction_costs': Series of costs per asset
            - 'total_cost': Total transaction cost
            - 'turnover': Realized turnover
    """
    # Calculate ideal trades
    trades = calculate_trades(
        target_weights,
        current_weights,
        min_trade_size=cost_model.min_trade_size if cost_model else 0.0,
    )

    # Check turnover constraint
    turnover = trades.abs().sum()
    if max_turnover is not None and turnover > max_turnover:
        # Scale down trades proportionally
        scale = max_turnover / turnover
        trades = trades * scale
        turnover = max_turnover

    # Calculate costs
    if cost_model is not None:
        costs = apply_transaction_costs(trades, cost_model, portfolio_value)
        total_cost = costs.sum()
    else:
        costs = pd.Series(0.0, index=trades.index)
        total_cost = 0.0

    # Final weights after rebalancing
    all_assets = target_weights.index.union(current_weights.index)
    current_aligned = current_weights.reindex(all_assets, fill_value=0.0)
    trades_aligned = trades.reindex(all_assets, fill_value=0.0)
    final_weights = current_aligned + trades_aligned

    # Subtract costs from cash position (assuming cash is residual)
    # In practice, this would be handled by the execution system

    return {
        'trades': trades,
        'final_weights': final_weights,
        'transaction_costs': costs,
        'total_cost': total_cost,
        'turnover': turnover,
    }


def optimize_rebalance_with_costs(
    target_weights: pd.Series,
    current_weights: pd.Series,
    cost_model: Union[TransactionCostModel, float],
    lambda_cost: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.Series:
    """Optimize rebalance considering transaction costs.

    Minimizes: tracking_error + lambda_cost * transaction_costs

    Uses simple gradient descent on trade sizes.

    Args:
        target_weights: Target portfolio weights
        current_weights: Current portfolio weights
        cost_model: Transaction cost model
        lambda_cost: Cost penalty weight
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Optimized trades
    """
    if isinstance(cost_model, (int, float)):
        cost_model = TransactionCostModel(linear_bps=float(cost_model))

    # Initial trades (full rebalance)
    all_assets = target_weights.index.union(current_weights.index)
    target = target_weights.reindex(all_assets, fill_value=0.0)
    current = current_weights.reindex(all_assets, fill_value=0.0)

    trades = target - current

    # Gradient descent
    learning_rate = 0.1

    for iteration in range(max_iter):
        trades_old = trades.copy()

        # Current weights after trades
        w = current + trades

        # Tracking error gradient: 2 * (w - target)
        grad_tracking = 2.0 * (w - target)

        # Cost gradient
        grad_cost = pd.Series(0.0, index=trades.index)

        # Linear cost gradient: sign(trade) * linear_bps
        if cost_model.linear_bps > 0:
            grad_cost += np.sign(trades) * (cost_model.linear_bps / 10000.0)

        # Quadratic cost gradient: 2 * trade * quadratic_bps
        if cost_model.quadratic_bps > 0:
            grad_cost += 2.0 * trades * (cost_model.quadratic_bps / 10000.0)

        # Total gradient
        grad = grad_tracking + lambda_cost * grad_cost

        # Update trades
        trades = trades - learning_rate * grad

        # Check convergence
        if np.max(np.abs(trades - trades_old)) < tol:
            break

    return trades


def schedule_rebalance(
    target_weights: pd.Series,
    current_weights: pd.Series,
    num_periods: int,
) -> list[pd.Series]:
    """Schedule rebalance over multiple periods.

    Splits trades into equal increments to reduce market impact.

    Args:
        target_weights: Target portfolio weights
        current_weights: Current portfolio weights
        num_periods: Number of periods to split rebalance

    Returns:
        List of trade schedules (one Series per period)
    """
    total_trades = calculate_trades(target_weights, current_weights)

    # Split into equal increments
    increment = total_trades / num_periods

    schedule = []
    for period in range(num_periods):
        schedule.append(increment.copy())

    return schedule
