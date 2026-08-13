"""Performance attribution utilities.

Provides attribution analysis for portfolio returns:
- Brinson attribution (allocation + selection effects)
- Factor-based attribution
- Transaction cost attribution
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


def brinson_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    asset_returns: pd.Series,
    benchmark_return: Optional[float] = None,
) -> Dict[str, float]:
    """Brinson-Fachler attribution analysis.

    Decomposes active return into allocation and selection effects.

    Args:
        portfolio_weights: Portfolio weights at period start
        benchmark_weights: Benchmark weights at period start
        asset_returns: Asset returns during period
        benchmark_return: Benchmark return (if None, calculated from weights)

    Returns:
        Dict with attribution components:
            - 'portfolio_return': Total portfolio return
            - 'benchmark_return': Benchmark return
            - 'active_return': Portfolio - benchmark
            - 'allocation_effect': Return from asset allocation decisions
            - 'selection_effect': Return from security selection
            - 'interaction_effect': Interaction between allocation and selection
    """
    # Align all series
    all_assets = portfolio_weights.index.union(benchmark_weights.index).union(asset_returns.index)
    pw = portfolio_weights.reindex(all_assets, fill_value=0.0)
    bw = benchmark_weights.reindex(all_assets, fill_value=0.0)
    ret = asset_returns.reindex(all_assets, fill_value=0.0)

    # Portfolio return
    port_ret = (pw * ret).sum()

    # Benchmark return
    if benchmark_return is None:
        bench_ret = (bw * ret).sum()
    else:
        bench_ret = benchmark_return

    # Active return
    active_ret = port_ret - bench_ret

    # Weight differences
    weight_diff = pw - bw

    # Return differences (vs benchmark average)
    ret_diff = ret - bench_ret

    # Allocation effect: (portfolio_weight - benchmark_weight) * (sector_return - benchmark_return)
    # Simplified: weight_diff * (asset_return - benchmark_return)
    allocation = (weight_diff * ret_diff).sum()

    # Selection effect: benchmark_weight * (asset_return - benchmark_return)
    selection = (bw * ret_diff).sum()

    # Interaction effect: (portfolio_weight - benchmark_weight) * (asset_return - benchmark_return)
    # This is double-counted in simple decomposition
    interaction = (weight_diff * ret_diff).sum()

    # Traditional Brinson: allocation uses benchmark returns per sector
    # Simplified version here uses asset-level returns directly

    return {
        'portfolio_return': port_ret,
        'benchmark_return': bench_ret,
        'active_return': active_ret,
        'allocation_effect': allocation,
        'selection_effect': selection,
        'interaction_effect': 0.0,  # Absorbed into allocation in this simplified version
    }


def sector_brinson_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    asset_returns: pd.Series,
    sector_mapping: pd.Series,
    benchmark_return: Optional[float] = None,
) -> pd.DataFrame:
    """Brinson attribution by sector.

    Args:
        portfolio_weights: Portfolio weights
        benchmark_weights: Benchmark weights
        asset_returns: Asset returns
        sector_mapping: Series mapping asset to sector
        benchmark_return: Benchmark return

    Returns:
        DataFrame with attribution by sector
    """
    # Calculate benchmark return if not provided
    if benchmark_return is None:
        all_assets = benchmark_weights.index.intersection(asset_returns.index)
        bw_aligned = benchmark_weights.reindex(all_assets, fill_value=0.0)
        ret_aligned = asset_returns.reindex(all_assets, fill_value=0.0)
        benchmark_return = (bw_aligned * ret_aligned).sum()

    sectors = sector_mapping.unique()
    results = []

    for sector in sectors:
        # Assets in this sector
        sector_assets = sector_mapping[sector_mapping == sector].index

        # Filter weights and returns
        pw_sector = portfolio_weights.reindex(sector_assets, fill_value=0.0)
        bw_sector = benchmark_weights.reindex(sector_assets, fill_value=0.0)
        ret_sector = asset_returns.reindex(sector_assets, fill_value=0.0)

        # Sector returns
        port_sector_weight = pw_sector.sum()
        bench_sector_weight = bw_sector.sum()

        if bench_sector_weight > 0:
            bench_sector_return = (bw_sector * ret_sector).sum() / bench_sector_weight
        else:
            bench_sector_return = 0.0

        if port_sector_weight > 0:
            port_sector_return = (pw_sector * ret_sector).sum() / port_sector_weight
        else:
            port_sector_return = 0.0

        # Allocation effect: (portfolio_weight - benchmark_weight) * (sector_return - benchmark_return)
        allocation = (port_sector_weight - bench_sector_weight) * (bench_sector_return - benchmark_return)

        # Selection effect: benchmark_weight * (portfolio_sector_return - benchmark_sector_return)
        selection = bench_sector_weight * (port_sector_return - bench_sector_return)

        # Interaction
        interaction = (port_sector_weight - bench_sector_weight) * (port_sector_return - bench_sector_return)

        results.append({
            'sector': sector,
            'portfolio_weight': port_sector_weight,
            'benchmark_weight': bench_sector_weight,
            'portfolio_return': port_sector_return,
            'benchmark_return': bench_sector_return,
            'allocation_effect': allocation,
            'selection_effect': selection,
            'interaction_effect': interaction,
            'total_effect': allocation + selection + interaction,
        })

    return pd.DataFrame(results)


def factor_attribution(
    portfolio_weights: pd.Series,
    asset_returns: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_returns: pd.Series,
) -> Dict[str, pd.Series]:
    """Factor-based return attribution.

    Attributes portfolio returns to factor exposures using:
    portfolio_return = sum(factor_exposure_i * factor_return_i) + specific_return

    Args:
        portfolio_weights: Portfolio weights
        asset_returns: Asset returns
        factor_exposures: DataFrame of factor exposures (assets x factors)
        factor_returns: Series of factor returns

    Returns:
        Dict with:
            - 'factor_contributions': Series of return contribution per factor
            - 'specific_return': Residual return not explained by factors
            - 'total_return': Total portfolio return
    """
    # Portfolio return
    all_assets = portfolio_weights.index.intersection(asset_returns.index)
    pw = portfolio_weights.reindex(all_assets, fill_value=0.0)
    ret = asset_returns.reindex(all_assets, fill_value=0.0)
    total_return = (pw * ret).sum()

    # Portfolio factor exposures (weighted average)
    exposures_aligned = factor_exposures.reindex(all_assets, fill_value=0.0)
    portfolio_exposures = (pw.values[:, np.newaxis] * exposures_aligned.values).sum(axis=0)
    portfolio_exposures = pd.Series(portfolio_exposures, index=exposures_aligned.columns)

    # Factor contributions to return
    factor_contribs = portfolio_exposures * factor_returns.reindex(portfolio_exposures.index, fill_value=0.0)

    # Specific return (residual)
    factor_return_explained = factor_contribs.sum()
    specific_return = total_return - factor_return_explained

    return {
        'factor_contributions': factor_contribs,
        'specific_return': specific_return,
        'total_return': total_return,
        'factor_exposures': portfolio_exposures,
    }


def transaction_cost_attribution(
    trades: pd.Series,
    cost_model: 'TransactionCostModel',
    asset_returns: pd.Series,
    portfolio_value: float = 1.0,
) -> Dict[str, float]:
    """Attribute transaction costs and their impact on returns.

    Args:
        trades: Trade sizes (as fraction of portfolio)
        cost_model: Transaction cost model
        asset_returns: Asset returns during period
        portfolio_value: Portfolio value

    Returns:
        Dict with cost attribution:
            - 'total_cost': Total transaction cost
            - 'linear_cost': Cost from linear component
            - 'quadratic_cost': Cost from quadratic (market impact) component
            - 'fixed_cost': Cost from fixed components
            - 'opportunity_cost': Forgone return from delayed execution
    """
    from .rebalance import apply_transaction_costs

    # Calculate costs by component
    costs_total = apply_transaction_costs(trades, cost_model, portfolio_value)

    # Break down by component
    linear_cost = 0.0
    quadratic_cost = 0.0
    fixed_cost = 0.0

    for asset, trade in trades.items():
        abs_trade = abs(trade)

        if abs_trade == 0:
            continue

        if cost_model.linear_bps > 0:
            linear_cost += abs_trade * (cost_model.linear_bps / 10000.0)

        if cost_model.quadratic_bps > 0:
            quadratic_cost += (abs_trade ** 2) * (cost_model.quadratic_bps / 10000.0)

        if cost_model.fixed_cost > 0:
            fixed_cost += cost_model.fixed_cost / portfolio_value

    # Opportunity cost: if we hadn't traded, what return would we have gotten?
    # This is the return on the cash used for trading
    trade_volume = trades.abs().sum()
    avg_return = asset_returns.mean() if len(asset_returns) > 0 else 0.0
    opportunity_cost = trade_volume * avg_return * 0.5  # Half the period on average

    return {
        'total_cost': costs_total.sum(),
        'linear_cost': linear_cost,
        'quadratic_cost': quadratic_cost,
        'fixed_cost': fixed_cost,
        'opportunity_cost': opportunity_cost,
    }


def rolling_attribution(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Rolling Brinson attribution over time.

    Args:
        portfolio_weights: DataFrame of portfolio weights (time x assets)
        benchmark_weights: DataFrame of benchmark weights (time x assets)
        asset_returns: DataFrame of asset returns (time x assets)
        window: Rolling window size

    Returns:
        DataFrame with rolling attribution metrics
    """
    results = []

    for i in range(window, len(portfolio_weights)):
        period_end = portfolio_weights.index[i]

        # Weights at period start
        pw = portfolio_weights.iloc[i - 1]
        bw = benchmark_weights.iloc[i - 1]

        # Returns during period
        ret = asset_returns.iloc[i]

        # Attribution
        attr = brinson_attribution(pw, bw, ret)
        attr['date'] = period_end

        results.append(attr)

    return pd.DataFrame(results).set_index('date')
