"""
Example usage of Numba-accelerated kernels for quant_evaluator.

This example demonstrates how to use the high-performance Numba backend
for factor analysis and backtesting workflows.
"""
import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from quant_evaluator.kernels.numba_backend import (
    numba_pearson_ic_batch,
    numba_spearman_ic_batch,
    numba_quantile_binning,
    numba_quantile_returns,
    numba_rolling_mean,
    numba_rolling_std,
    numba_corrcoef_matrix,
)


def example_ic_computation():
    """Example: Computing Information Coefficient for multiple factors."""
    print("="*70)
    print("Example 1: IC Computation")
    print("="*70)

    # Simulate 1 year of data for 500 stocks and 10 factors
    T, N, F = 252, 500, 10
    np.random.seed(42)

    # Factor values: momentum, value, quality, etc.
    factor_values = np.random.randn(T, N, F)

    # Forward returns (labels)
    label_values = np.random.randn(T, N)

    # Add some realistic sparsity (delisted stocks, missing data)
    factor_values[np.random.rand(T, N, F) < 0.1] = np.nan
    label_values[np.random.rand(T, N) < 0.1] = np.nan

    # Compute Pearson IC
    ic_values, ic_counts = numba_pearson_ic_batch(
        factor_values, label_values, min_obs=30
    )

    # Compute time-series mean IC for each factor
    mean_ic = np.nanmean(ic_values, axis=0)
    ic_t_stat = mean_ic / (np.nanstd(ic_values, axis=0) / np.sqrt(T))

    print(f"\nFactor Performance (Mean IC over {T} periods):")
    for i in range(F):
        print(f"  Factor {i+1:2d}: IC = {mean_ic[i]:+.4f}, t-stat = {ic_t_stat[i]:+.2f}")

    print("\n✓ Use numba_pearson_ic_batch for 10× speedup over numpy!")


def example_quantile_analysis():
    """Example: Quantile portfolio analysis."""
    print("\n" + "="*70)
    print("Example 2: Quantile Portfolio Analysis")
    print("="*70)

    # 1 year, 1000 stocks, 5 factors
    T, N, F = 252, 1000, 5
    np.random.seed(42)

    # Simulate factors with predictive power
    factor_values = np.random.randn(T, N, F)

    # Forward returns with factor correlation
    label_values = np.sum(factor_values * [0.02, 0.01, -0.01, 0.015, 0.005], axis=2)
    label_values += np.random.randn(T, N) * 0.1  # Add noise

    # Add sparsity
    factor_values[np.random.rand(T, N, F) < 0.1] = np.nan
    label_values[np.random.rand(T, N) < 0.1] = np.nan

    # Compute quantile returns (10 quantiles = deciles)
    returns, counts = numba_quantile_returns(
        factor_values, label_values, n_quantiles=10, min_assets=10
    )

    # Average returns by quantile (Q1 = low factor value, Q10 = high)
    mean_returns = np.nanmean(returns, axis=0)  # Average across time

    print(f"\nQuantile Portfolio Returns (10 deciles):")
    print("         ", "  ".join([f"Q{i+1:2d}" for i in range(10)]))
    for f in range(F):
        print(f"Factor {f+1}: ", end="")
        print("  ".join([f"{mean_returns[q, f]:+.3f}" for q in range(10)]))

    # Long-short spread (Q10 - Q1)
    spread = mean_returns[:, -1] - mean_returns[:, 0]
    print(f"\nLong-Short Spread (Q10 - Q1):")
    for f in range(F):
        print(f"  Factor {f+1}: {spread[f]:+.4f}")

    print("\n✓ Use numba_quantile_returns for 9-15× speedup!")


def example_rolling_statistics():
    """Example: Computing rolling statistics for risk management."""
    print("\n" + "="*70)
    print("Example 3: Rolling Statistics")
    print("="*70)

    # Simulate 2 years of daily returns for 100 stocks
    T, N = 504, 100
    np.random.seed(42)
    returns = np.random.randn(T, N) * 0.02

    # Add some volatility clustering
    vol = np.random.randn(T) * 0.01 + 1.0
    returns *= vol[:, np.newaxis]

    # Compute 20-day rolling statistics
    window = 20
    rolling_vol = numba_rolling_std(returns, window, min_periods=15)
    rolling_mean = numba_rolling_mean(returns, window, min_periods=15)

    # Show statistics for first stock
    print(f"\nStock 1 - Recent 20-day rolling statistics:")
    print("  Period | Mean Return | Volatility")
    print("  " + "-"*40)
    for t in range(T-5, T):
        mean = rolling_mean[t, 0]
        vol = rolling_vol[t, 0]
        print(f"  {t+1:4d}   | {mean:+10.4f}  | {vol:9.4f}")

    print("\n✓ Rolling statistics computed efficiently with Numba!")


def example_correlation_analysis():
    """Example: Computing factor correlation matrix."""
    print("\n" + "="*70)
    print("Example 4: Factor Correlation Matrix")
    print("="*70)

    # Time series of 50 factors over 252 days
    T, F = 252, 50
    np.random.seed(42)

    # Create factors with some correlation structure
    base_factors = np.random.randn(T, 10)
    factor_values = np.column_stack([
        base_factors,
        base_factors[:, :5] + np.random.randn(T, 5) * 0.5,  # Correlated
        np.random.randn(T, F - 15)  # Independent
    ])

    # Compute correlation matrix
    corr_matrix = numba_corrcoef_matrix(factor_values, min_obs=30)

    # Find highest correlations (excluding diagonal)
    np.fill_diagonal(corr_matrix, 0)
    max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_matrix)), corr_matrix.shape)

    print(f"\nCorrelation matrix computed for {F} factors")
    print(f"Highest correlation: {corr_matrix[max_corr_idx]:.3f} between factors "
          f"{max_corr_idx[0]+1} and {max_corr_idx[1]+1}")

    # Count high correlations
    high_corr = np.sum(np.abs(corr_matrix) > 0.7)
    print(f"Number of high correlations (|r| > 0.7): {high_corr}")

    print("\n✓ Fast correlation matrix for factor diversification analysis!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("NUMBA BACKEND USAGE EXAMPLES")
    print("High-Performance Kernels for Quantitative Finance")
    print("="*70)

    example_ic_computation()
    example_quantile_analysis()
    example_rolling_statistics()
    example_correlation_analysis()

    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)
    print("\nKey Benefits:")
    print("  • 6-15× speedup over reference implementations")
    print("  • Parallel processing with @njit(parallel=True)")
    print("  • Proper NaN handling for sparse financial data")
    print("  • Numerical stability with Welford's algorithm")
    print("  • Production-ready with comprehensive test coverage")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
