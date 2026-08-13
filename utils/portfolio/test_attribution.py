"""Tests for performance attribution utilities."""

import numpy as np
import pandas as pd
import pytest
from utils.portfolio.attribution import (
    brinson_attribution,
    sector_brinson_attribution,
    factor_attribution,
    transaction_cost_attribution,
    rolling_attribution,
)
from utils.portfolio.rebalance import TransactionCostModel


def test_brinson_attribution_basic():
    """Test basic Brinson attribution."""
    portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=['A', 'B', 'C'])
    benchmark_weights = pd.Series([0.3, 0.4, 0.3], index=['A', 'B', 'C'])
    asset_returns = pd.Series([0.10, 0.05, -0.02], index=['A', 'B', 'C'])

    result = brinson_attribution(portfolio_weights, benchmark_weights, asset_returns)

    assert 'portfolio_return' in result
    assert 'benchmark_return' in result
    assert 'active_return' in result
    assert 'allocation_effect' in result
    assert 'selection_effect' in result

    # Portfolio return
    expected_port = 0.4 * 0.10 + 0.3 * 0.05 + 0.3 * (-0.02)
    assert np.isclose(result['portfolio_return'], expected_port)

    # Benchmark return
    expected_bench = 0.3 * 0.10 + 0.4 * 0.05 + 0.3 * (-0.02)
    assert np.isclose(result['benchmark_return'], expected_bench)

    # Active return
    assert np.isclose(result['active_return'], expected_port - expected_bench)


def test_brinson_attribution_equal_weights():
    """Test Brinson attribution with equal weights (no active return)."""
    weights = pd.Series([0.5, 0.5], index=['A', 'B'])
    returns = pd.Series([0.10, 0.05], index=['A', 'B'])

    result = brinson_attribution(weights, weights, returns)

    # No active return when weights are identical
    assert np.isclose(result['active_return'], 0.0, atol=1e-10)
    assert np.isclose(result['portfolio_return'], result['benchmark_return'])


def test_brinson_attribution_provided_benchmark_return():
    """Test Brinson attribution with provided benchmark return."""
    portfolio_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
    benchmark_weights = pd.Series([0.4, 0.6], index=['A', 'B'])
    asset_returns = pd.Series([0.10, 0.05], index=['A', 'B'])
    bench_ret = 0.07

    result = brinson_attribution(
        portfolio_weights, benchmark_weights, asset_returns, benchmark_return=bench_ret
    )

    assert np.isclose(result['benchmark_return'], bench_ret)


def test_sector_brinson_attribution():
    """Test sector-level Brinson attribution."""
    portfolio_weights = pd.Series([0.2, 0.2, 0.3, 0.3], index=['A', 'B', 'C', 'D'])
    benchmark_weights = pd.Series([0.1, 0.3, 0.3, 0.3], index=['A', 'B', 'C', 'D'])
    asset_returns = pd.Series([0.10, 0.08, 0.05, 0.04], index=['A', 'B', 'C', 'D'])
    sector_mapping = pd.Series(['Tech', 'Tech', 'Finance', 'Finance'], index=['A', 'B', 'C', 'D'])

    result = sector_brinson_attribution(
        portfolio_weights, benchmark_weights, asset_returns, sector_mapping
    )

    assert isinstance(result, pd.DataFrame)
    assert 'sector' in result.columns
    assert 'allocation_effect' in result.columns
    assert 'selection_effect' in result.columns
    assert len(result) == 2  # Two sectors


def test_sector_brinson_attribution_overweight():
    """Test sector attribution with overweight position."""
    portfolio_weights = pd.Series([0.6, 0.4], index=['A', 'B'])
    benchmark_weights = pd.Series([0.3, 0.7], index=['A', 'B'])
    asset_returns = pd.Series([0.10, 0.05], index=['A', 'B'])
    sector_mapping = pd.Series(['Tech', 'Finance'], index=['A', 'B'])

    result = sector_brinson_attribution(
        portfolio_weights, benchmark_weights, asset_returns, sector_mapping
    )

    # Portfolio overweights Tech (0.6 vs 0.3)
    tech_row = result[result['sector'] == 'Tech'].iloc[0]
    assert tech_row['portfolio_weight'] > tech_row['benchmark_weight']


def test_factor_attribution_basic():
    """Test basic factor attribution."""
    portfolio_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
    asset_returns = pd.Series([0.10, 0.06], index=['A', 'B'])

    # Factor exposures
    factor_exposures = pd.DataFrame({
        'momentum': [1.5, 0.5],
        'value': [0.3, 1.2]
    }, index=['A', 'B'])

    factor_returns = pd.Series([0.04, 0.02], index=['momentum', 'value'])

    result = factor_attribution(
        portfolio_weights, asset_returns, factor_exposures, factor_returns
    )

    assert 'factor_contributions' in result
    assert 'specific_return' in result
    assert 'total_return' in result
    assert 'factor_exposures' in result

    # Total return should match weighted returns
    expected_total = 0.5 * 0.10 + 0.5 * 0.06
    assert np.isclose(result['total_return'], expected_total)


def test_factor_attribution_exposures():
    """Test factor exposure calculation."""
    portfolio_weights = pd.Series([0.6, 0.4], index=['A', 'B'])

    factor_exposures = pd.DataFrame({
        'factor1': [1.0, 2.0],
        'factor2': [0.5, 1.5]
    }, index=['A', 'B'])

    asset_returns = pd.Series([0.05, 0.05], index=['A', 'B'])
    factor_returns = pd.Series([0.03, 0.02], index=['factor1', 'factor2'])

    result = factor_attribution(
        portfolio_weights, asset_returns, factor_exposures, factor_returns
    )

    # Portfolio factor1 exposure = 0.6 * 1.0 + 0.4 * 2.0 = 1.4
    assert np.isclose(result['factor_exposures']['factor1'], 1.4)

    # Portfolio factor2 exposure = 0.6 * 0.5 + 0.4 * 1.5 = 0.9
    assert np.isclose(result['factor_exposures']['factor2'], 0.9)


def test_factor_attribution_contributions():
    """Test factor contribution calculation."""
    portfolio_weights = pd.Series([1.0], index=['A'])

    factor_exposures = pd.DataFrame({
        'factor1': [2.0],
    }, index=['A'])

    asset_returns = pd.Series([0.10], index=['A'])
    factor_returns = pd.Series([0.04], index=['factor1'])

    result = factor_attribution(
        portfolio_weights, asset_returns, factor_exposures, factor_returns
    )

    # Factor contribution = exposure * factor_return = 2.0 * 0.04 = 0.08
    assert np.isclose(result['factor_contributions']['factor1'], 0.08)

    # Specific return = total - factor_explained = 0.10 - 0.08 = 0.02
    assert np.isclose(result['specific_return'], 0.02)


def test_transaction_cost_attribution():
    """Test transaction cost attribution."""
    trades = pd.Series([0.1, -0.1], index=['A', 'B'])
    cost_model = TransactionCostModel(
        linear_bps=10,
        quadratic_bps=50,
        fixed_cost=0.001
    )
    asset_returns = pd.Series([0.05, 0.03], index=['A', 'B'])

    result = transaction_cost_attribution(trades, cost_model, asset_returns)

    assert 'total_cost' in result
    assert 'linear_cost' in result
    assert 'quadratic_cost' in result
    assert 'fixed_cost' in result
    assert 'opportunity_cost' in result

    # All costs should be positive
    assert result['total_cost'] >= 0
    assert result['linear_cost'] >= 0
    assert result['quadratic_cost'] >= 0


def test_transaction_cost_attribution_breakdown():
    """Test transaction cost attribution breakdown."""
    trades = pd.Series([0.2], index=['A'])
    cost_model = TransactionCostModel(
        linear_bps=100,
        quadratic_bps=0,
        fixed_cost=0
    )
    asset_returns = pd.Series([0.05], index=['A'])

    result = transaction_cost_attribution(trades, cost_model, asset_returns)

    # Linear cost = 0.2 * 0.01 = 0.002
    assert np.isclose(result['linear_cost'], 0.002)
    assert result['quadratic_cost'] == 0.0
    assert result['fixed_cost'] == 0.0


def test_rolling_attribution():
    """Test rolling attribution over time."""
    dates = pd.date_range('2020-01-01', periods=30, freq='D')

    # Create time series data
    portfolio_weights = pd.DataFrame(
        np.random.dirichlet([1, 1, 1], size=30),
        index=dates,
        columns=['A', 'B', 'C']
    )

    benchmark_weights = pd.DataFrame(
        [[0.33, 0.33, 0.34]] * 30,
        index=dates,
        columns=['A', 'B', 'C']
    )

    asset_returns = pd.DataFrame(
        np.random.randn(30, 3) * 0.01,
        index=dates,
        columns=['A', 'B', 'C']
    )

    result = rolling_attribution(
        portfolio_weights, benchmark_weights, asset_returns, window=5
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 30 - 5  # window = 5
    assert 'portfolio_return' in result.columns
    assert 'active_return' in result.columns


def test_brinson_attribution_misaligned_assets():
    """Test Brinson attribution with misaligned asset sets."""
    portfolio_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
    benchmark_weights = pd.Series([0.4, 0.3, 0.3], index=['A', 'B', 'C'])
    asset_returns = pd.Series([0.10, 0.05, 0.02], index=['A', 'B', 'C'])

    result = brinson_attribution(portfolio_weights, benchmark_weights, asset_returns)

    # Should handle misalignment gracefully
    assert 'portfolio_return' in result
    assert 'benchmark_return' in result


def test_factor_attribution_zero_weights():
    """Test factor attribution with zero weights."""
    portfolio_weights = pd.Series([0.0, 1.0], index=['A', 'B'])
    asset_returns = pd.Series([0.10, 0.05], index=['A', 'B'])

    factor_exposures = pd.DataFrame({
        'factor1': [1.0, 2.0]
    }, index=['A', 'B'])

    factor_returns = pd.Series([0.03], index=['factor1'])

    result = factor_attribution(
        portfolio_weights, asset_returns, factor_exposures, factor_returns
    )

    # Portfolio exposure should only come from B
    assert np.isclose(result['factor_exposures']['factor1'], 2.0)


def test_transaction_cost_attribution_no_trades():
    """Test transaction cost attribution with no trades."""
    trades = pd.Series([], dtype=float)
    cost_model = TransactionCostModel(linear_bps=10)
    asset_returns = pd.Series([0.05, 0.03], index=['A', 'B'])

    result = transaction_cost_attribution(trades, cost_model, asset_returns)

    # All costs should be zero
    assert result['total_cost'] == 0.0
    assert result['linear_cost'] == 0.0
