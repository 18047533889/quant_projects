"""
Example usage of risk metrics module.

This script demonstrates how to use the comprehensive risk metrics
for portfolio analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from quant_evaluator.metrics.risk import (
    compute_var_historical,
    compute_var_parametric,
    compute_var_cornish_fisher,
    compute_cvar,
    compute_drawdown_statistics,
    identify_drawdown_periods,
    compute_ulcer_index,
    compute_tail_ratio,
    compute_omega_ratio,
    compute_tail_dependence,
    apply_historical_scenario,
    compute_scenario_impact,
    compute_worst_case_scenarios,
)


def main():
    # Generate sample portfolio returns
    np.random.seed(42)
    T = 1000
    returns = np.random.randn(T) * 0.02 + 0.0005  # Daily returns with small positive drift

    print("=" * 60)
    print("RISK METRICS ANALYSIS")
    print("=" * 60)

    # 1. VaR and CVaR Analysis
    print("\n1. VALUE AT RISK (VaR) & CONDITIONAL VaR")
    print("-" * 60)

    var_95_hist = compute_var_historical(returns, confidence_level=0.95)
    var_95_param = compute_var_parametric(returns, confidence_level=0.95)
    var_95_cf = compute_var_cornish_fisher(returns, confidence_level=0.95)
    cvar_95 = compute_cvar(returns, confidence_level=0.95, method="historical")

    print(f"VaR (95%, Historical):      {var_95_hist:.4f}")
    print(f"VaR (95%, Parametric):      {var_95_param:.4f}")
    print(f"VaR (95%, Cornish-Fisher):  {var_95_cf:.4f}")
    print(f"CVaR (95%, Historical):     {cvar_95:.4f}")

    # 2. Drawdown Analysis
    print("\n2. DRAWDOWN ANALYSIS")
    print("-" * 60)

    dd_stats = compute_drawdown_statistics(returns)
    print(f"Max Drawdown:               {dd_stats['max_drawdown']:.4f}")
    print(f"Average Drawdown:           {dd_stats['avg_drawdown']:.4f}")
    print(f"Time Underwater:            {dd_stats['time_underwater_pct']:.2f}%")
    print(f"Ulcer Index:                {compute_ulcer_index(returns):.4f}")

    # Identify significant drawdown periods
    periods = identify_drawdown_periods(returns, threshold=0.05, min_duration=5)
    print(f"\nSignificant Drawdown Periods: {len(periods)}")
    if periods:
        worst = periods[0]
        print(f"  Worst Period: Drawdown={worst['drawdown']:.4f}, Duration={worst['duration']} days")

    # 3. Tail Risk Measures
    print("\n3. TAIL RISK MEASURES")
    print("-" * 60)

    tail_ratio = compute_tail_ratio(returns)
    omega = compute_omega_ratio(returns, threshold=0.0)

    print(f"Tail Ratio (95/5):          {tail_ratio:.4f}")
    print(f"Omega Ratio:                {omega:.4f}")

    # 4. Stress Testing
    print("\n4. STRESS TESTING")
    print("-" * 60)

    # Historical scenario: 2008-style crash
    crash_scenario = np.array([-0.10, -0.08, -0.12, -0.05, -0.07])
    stressed_returns = apply_historical_scenario(returns, crash_scenario, scaling_method="direct")

    print(f"Crash Scenario Applied: {len(stressed_returns)} days")
    print(f"Cumulative Loss:            {(np.prod(1 + stressed_returns) - 1):.4f}")

    # Compute scenario impact
    baseline = returns[:500]
    scenario = returns[500:600]
    impact = compute_scenario_impact(
        baseline, scenario,
        metrics=["mean", "sharpe", "max_drawdown"],
        periods_per_year=252
    )

    print(f"\nScenario Impact:")
    print(f"  Mean Return Change:       {np.asarray(impact['mean_change']).item():.6f}")
    print(f"  Sharpe Ratio Change:      {np.asarray(impact['sharpe_change']).item():.4f}")
    print(f"  Max Drawdown Change:      {np.asarray(impact['max_drawdown_change']).item():.4f}")

    # Identify worst-case historical periods
    worst_periods = compute_worst_case_scenarios(
        returns, n_scenarios=3, window_size=20, metric="cumulative"
    )

    print(f"\nWorst 3 Historical Periods (20-day windows):")
    for i, period in enumerate(worst_periods, 1):
        cum_ret = np.prod(1 + period['returns']) - 1
        print(f"  {i}. Days {period['start_idx']}-{period['end_idx']}: {cum_ret:.4f}")

    # 5. Multi-asset tail dependence (if comparing two assets)
    print("\n5. TAIL DEPENDENCE")
    print("-" * 60)

    # Generate correlated returns for demonstration
    asset1_returns = returns
    asset2_returns = 0.6 * returns + 0.4 * np.random.randn(T) * 0.02

    lower_dep, upper_dep = compute_tail_dependence(
        asset1_returns, asset2_returns, quantile=0.05
    )

    print(f"Lower Tail Dependence (5%): {lower_dep:.4f}")
    print(f"Upper Tail Dependence (95%):{upper_dep:.4f}")

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
