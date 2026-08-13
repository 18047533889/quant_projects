"""Tests for portfolio rebalancing utilities."""

import numpy as np
import pandas as pd
import pytest
from utils.portfolio.rebalance import (
    TransactionCostModel,
    calculate_trades,
    apply_transaction_costs,
    rebalance_portfolio,
    optimize_rebalance_with_costs,
    schedule_rebalance,
)


def test_transaction_cost_model_defaults():
    """Test default cost model values."""
    model = TransactionCostModel()
    assert model.linear_bps == 0.0
    assert model.quadratic_bps == 0.0
    assert model.fixed_cost == 0.0


def test_calculate_trades_basic():
    """Test basic trade calculation."""
    current = pd.Series([0.3, 0.3, 0.4], index=['A', 'B', 'C'])
    target = pd.Series([0.4, 0.2, 0.4], index=['A', 'B', 'C'])

    trades = calculate_trades(target, current)

    assert np.isclose(trades['A'], 0.1)
    assert np.isclose(trades['B'], -0.1)
    assert 'C' not in trades  # Zero trade removed


def test_calculate_trades_new_assets():
    """Test trades with new assets."""
    current = pd.Series([0.5, 0.5], index=['A', 'B'])
    target = pd.Series([0.3, 0.3, 0.4], index=['A', 'B', 'C'])

    trades = calculate_trades(target, current)

    assert np.isclose(trades['A'], -0.2)
    assert np.isclose(trades['B'], -0.2)
    assert np.isclose(trades['C'], 0.4)


def test_calculate_trades_min_size():
    """Test minimum trade size filter."""
    current = pd.Series([0.33, 0.33, 0.34], index=['A', 'B', 'C'])
    target = pd.Series([0.35, 0.32, 0.33], index=['A', 'B', 'C'])

    trades = calculate_trades(target, current, min_trade_size=0.025)

    # All trades should be filtered out (< 0.025)
    assert len(trades) == 0


def test_apply_transaction_costs_linear():
    """Test linear transaction cost calculation."""
    trades = pd.Series([0.1, -0.2], index=['A', 'B'])
    model = TransactionCostModel(linear_bps=10)  # 10 bps = 0.1%

    costs = apply_transaction_costs(trades, model)

    assert np.isclose(costs['A'], 0.1 * 0.001)  # 0.1% of 0.1
    assert np.isclose(costs['B'], 0.2 * 0.001)  # 0.1% of 0.2 (absolute)


def test_apply_transaction_costs_quadratic():
    """Test quadratic transaction cost (market impact)."""
    trades = pd.Series([0.1, -0.2], index=['A', 'B'])
    model = TransactionCostModel(quadratic_bps=100)

    costs = apply_transaction_costs(trades, model)

    # Quadratic: trade^2 * coef
    assert np.isclose(costs['A'], (0.1 ** 2) * 0.01)
    assert np.isclose(costs['B'], (0.2 ** 2) * 0.01)


def test_apply_transaction_costs_fixed():
    """Test fixed transaction costs."""
    trades = pd.Series([0.1, -0.2], index=['A', 'B'])
    model = TransactionCostModel(fixed_cost=0.001)

    costs = apply_transaction_costs(trades, model, portfolio_value=1.0)

    # Fixed cost per trade
    assert np.isclose(costs['A'], 0.001)
    assert np.isclose(costs['B'], 0.001)


def test_apply_transaction_costs_combined():
    """Test combined cost model."""
    trades = pd.Series([0.1], index=['A'])
    model = TransactionCostModel(
        linear_bps=10,
        quadratic_bps=100,
        fixed_cost=0.001
    )

    costs = apply_transaction_costs(trades, model, portfolio_value=1.0)

    expected = 0.1 * 0.001 + (0.1 ** 2) * 0.01 + 0.001
    assert np.isclose(costs['A'], expected)


def test_apply_transaction_costs_float_model():
    """Test float cost model (shorthand for linear bps)."""
    trades = pd.Series([0.1], index=['A'])
    costs = apply_transaction_costs(trades, 20.0)  # 20 bps

    assert np.isclose(costs['A'], 0.1 * 0.002)


def test_rebalance_portfolio_basic():
    """Test basic rebalancing."""
    current = pd.Series([0.3, 0.3, 0.4], index=['A', 'B', 'C'])
    target = pd.Series([0.4, 0.2, 0.4], index=['A', 'B', 'C'])

    result = rebalance_portfolio(target, current)

    assert 'trades' in result
    assert 'final_weights' in result
    assert np.isclose(result['trades']['A'], 0.1)
    assert np.isclose(result['trades']['B'], -0.1)
    assert np.isclose(result['final_weights'].sum(), 1.0)


def test_rebalance_portfolio_with_costs():
    """Test rebalancing with transaction costs."""
    current = pd.Series([0.5, 0.5], index=['A', 'B'])
    target = pd.Series([0.6, 0.4], index=['A', 'B'])
    cost_model = TransactionCostModel(linear_bps=10)

    result = rebalance_portfolio(target, current, cost_model=cost_model)

    assert result['total_cost'] > 0
    assert 'transaction_costs' in result
    # Trade of 0.1 + 0.1 = 0.2 total, at 10bps = 0.0002
    assert np.isclose(result['total_cost'], 0.0002, atol=1e-6)


def test_rebalance_portfolio_turnover_constraint():
    """Test rebalancing with turnover limit."""
    current = pd.Series([0.25, 0.25, 0.25, 0.25], index=['A', 'B', 'C', 'D'])
    target = pd.Series([0.7, 0.1, 0.1, 0.1], index=['A', 'B', 'C', 'D'])

    result = rebalance_portfolio(target, current, max_turnover=0.4)

    assert result['turnover'] <= 0.4 + 1e-6
    # Trades should be scaled down
    assert result['trades'].abs().sum() <= 0.4 + 1e-6


def test_rebalance_portfolio_min_trade_size():
    """Test rebalancing with minimum trade size."""
    current = pd.Series([0.33, 0.33, 0.34], index=['A', 'B', 'C'])
    target = pd.Series([0.35, 0.32, 0.33], index=['A', 'B', 'C'])
    cost_model = TransactionCostModel(min_trade_size=0.025)

    result = rebalance_portfolio(target, current, cost_model=cost_model)

    # Small trades should be filtered
    assert len(result['trades']) == 0


def test_optimize_rebalance_with_costs():
    """Test cost-aware rebalance optimization."""
    current = pd.Series([0.5, 0.5], index=['A', 'B'])
    target = pd.Series([0.7, 0.3], index=['A', 'B'])
    cost_model = TransactionCostModel(linear_bps=50)  # High cost

    trades = optimize_rebalance_with_costs(
        target, current, cost_model, lambda_cost=10.0
    )

    # With high costs, should trade less than full rebalance
    ideal_trade = 0.2
    assert abs(trades['A']) < ideal_trade
    assert abs(trades['B']) < ideal_trade


def test_schedule_rebalance():
    """Test rebalancing over multiple periods."""
    current = pd.Series([0.5, 0.5], index=['A', 'B'])
    target = pd.Series([0.7, 0.3], index=['A', 'B'])

    schedule = schedule_rebalance(target, current, num_periods=4)

    assert len(schedule) == 4

    # Each period should have equal increments
    for period_trades in schedule:
        assert np.isclose(period_trades['A'], 0.05)  # 0.2 / 4
        assert np.isclose(period_trades['B'], -0.05)

    # Total should equal full rebalance
    total = sum(schedule)
    assert np.isclose(total['A'], 0.2)
    assert np.isclose(total['B'], -0.2)


def test_rebalance_portfolio_exit_position():
    """Test rebalancing when exiting a position."""
    current = pd.Series([0.3, 0.3, 0.4], index=['A', 'B', 'C'])
    target = pd.Series([0.5, 0.5], index=['A', 'B'])

    result = rebalance_portfolio(target, current)

    # Should sell all of C
    assert np.isclose(result['trades']['C'], -0.4)
    assert result['final_weights']['C'] == 0.0


def test_rebalance_portfolio_enter_position():
    """Test rebalancing when entering a new position."""
    current = pd.Series([0.5, 0.5], index=['A', 'B'])
    target = pd.Series([0.3, 0.3, 0.4], index=['A', 'B', 'C'])

    result = rebalance_portfolio(target, current)

    # Should buy C
    assert np.isclose(result['trades']['C'], 0.4)
    assert np.isclose(result['final_weights']['C'], 0.4)


def test_optimize_rebalance_convergence():
    """Test optimization convergence."""
    current = pd.Series([0.5, 0.5], index=['A', 'B'])
    target = pd.Series([0.6, 0.4], index=['A', 'B'])
    cost_model = TransactionCostModel(linear_bps=10)

    trades = optimize_rebalance_with_costs(
        target, current, cost_model, lambda_cost=0.0, max_iter=100
    )

    # With zero cost penalty, should converge to full rebalance
    assert np.isclose(trades['A'], 0.1, atol=0.01)
    assert np.isclose(trades['B'], -0.1, atol=0.01)


def test_transaction_cost_model_portfolio_value():
    """Test cost calculation with different portfolio values."""
    trades = pd.Series([0.1], index=['A'])
    model = TransactionCostModel(fixed_cost=100)  # $100 fixed cost

    # For $1000 portfolio, fixed cost is 10% of trade
    costs_1000 = apply_transaction_costs(trades, model, portfolio_value=1000)
    assert np.isclose(costs_1000['A'], 0.1)  # $100 / $1000

    # For $10000 portfolio, fixed cost is 1% of trade
    costs_10000 = apply_transaction_costs(trades, model, portfolio_value=10000)
    assert np.isclose(costs_10000['A'], 0.01)  # $100 / $10000
