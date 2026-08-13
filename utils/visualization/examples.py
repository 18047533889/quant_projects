"""
Example usage of visualization utilities for quantitative factor analysis.

Demonstrates all plotting functions with synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import visualization modules
import sys
sys.path.insert(0, '/home/shw/quant_projects')

from utils.visualization import (
    plot_ic_timeseries,
    plot_rolling_ic,
    plot_ic_distribution,
    plot_ic_heatmap,
    plot_factor_distribution,
    plot_quantile_returns,
    plot_turnover_analysis,
    plot_cumulative_returns,
    plot_multi_factor_ic,
    plot_factor_correlation_matrix,
    plot_ic_decay,
    plot_performance_comparison,
    plot_residual_analysis,
    plot_qq_plot,
    plot_autocorrelation,
    plot_heteroskedasticity,
)


def generate_sample_data():
    """Generate synthetic factor and return data for demonstrations."""
    np.random.seed(42)

    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start_date, periods=500, freq='D')

    # Generate IC series with trend and noise
    ic_base = 0.05 + 0.02 * np.sin(np.arange(500) * 2 * np.pi / 252)
    ic_noise = np.random.normal(0, 0.03, 500)
    ic_series = pd.Series(ic_base + ic_noise, index=dates, name='IC')

    # Generate multiple factor ICs
    n_factors = 5
    ic_df = pd.DataFrame(index=dates)
    for i in range(n_factors):
        base = 0.03 + i * 0.01
        trend = 0.015 * np.sin(np.arange(500) * 2 * np.pi / 252 + i)
        noise = np.random.normal(0, 0.025, 500)
        ic_df[f'Factor_{i+1}'] = base + trend + noise

    # Generate factor values (cross-sectional, flattened)
    n_stocks = 1000
    factor_values = pd.Series(
        np.random.normal(0, 1, n_stocks * 100).flatten(),
        name='factor_value'
    )

    # Generate quantile returns
    n_quantiles = 5
    quantile_returns = pd.DataFrame(index=dates[:400])
    for q in range(n_quantiles):
        base_return = 0.0005 * (q - 2)  # Long-short effect
        drift = base_return + np.random.normal(0, 0.01, 400)
        quantile_returns[f'Q{q+1}'] = drift

    # Generate turnover
    turnover = pd.Series(
        np.random.beta(2, 5, 500) * 0.5,
        index=dates,
        name='turnover'
    )

    # Generate returns
    returns = pd.Series(
        np.random.normal(0.0005, 0.015, 500),
        index=dates,
        name='returns'
    )
    benchmark = pd.Series(
        np.random.normal(0.0003, 0.012, 500),
        index=dates,
        name='benchmark'
    )

    # Generate IC decay
    lags = [1, 2, 3, 5, 10, 20]
    ic_decay = pd.DataFrame(index=lags)
    for i in range(n_factors):
        decay = [0.05 * np.exp(-lag * 0.15) * (1 + i * 0.1) for lag in lags]
        ic_decay[f'Factor_{i+1}'] = decay

    # Generate performance metrics
    performance_df = pd.DataFrame({
        'IC_mean': [0.045, 0.038, 0.052, 0.041, 0.055],
        'IR': [0.85, 0.72, 1.05, 0.78, 1.12],
        'Sharpe': [1.2, 0.9, 1.5, 1.0, 1.6],
        'Max_DD': [-0.15, -0.20, -0.12, -0.18, -0.10],
    }, index=[f'Factor_{i+1}' for i in range(n_factors)])

    # Generate residuals and fitted values for diagnostics
    n_obs = 1000
    fitted_values = pd.Series(np.random.uniform(-2, 2, n_obs), name='fitted')
    # Residuals with slight heteroskedasticity
    residuals = pd.Series(
        np.random.normal(0, 0.3 + 0.1 * np.abs(fitted_values), n_obs),
        name='residuals'
    )

    return {
        'ic_series': ic_series,
        'ic_df': ic_df,
        'factor_values': factor_values,
        'quantile_returns': quantile_returns,
        'turnover': turnover,
        'returns': returns,
        'benchmark': benchmark,
        'ic_decay': ic_decay,
        'performance_df': performance_df,
        'residuals': residuals,
        'fitted_values': fitted_values,
    }


def example_ic_plots(data):
    """Demonstrate IC visualization functions."""
    print("=" * 60)
    print("IC PLOTS EXAMPLES")
    print("=" * 60)

    # 1. IC Time Series
    print("\n1. Plotting IC time series...")
    fig = plot_ic_timeseries(
        data['ic_series'],
        title="Factor IC Time Series - Daily",
        show_mean=True,
        show_std_bands=True
    )
    plt.savefig('/home/shw/quant_projects/examples/ic_timeseries.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/ic_timeseries.png")

    # 2. Rolling IC
    print("\n2. Plotting rolling IC statistics...")
    fig = plot_rolling_ic(
        data['ic_series'],
        windows=[20, 60, 120],
        title="Rolling IC Analysis"
    )
    plt.savefig('/home/shw/quant_projects/examples/ic_rolling.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/ic_rolling.png")

    # 3. IC Distribution
    print("\n3. Plotting IC distribution...")
    fig = plot_ic_distribution(
        data['ic_series'],
        bins=50,
        show_stats=True
    )
    plt.savefig('/home/shw/quant_projects/examples/ic_distribution.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/ic_distribution.png")

    # 4. IC Heatmap
    print("\n4. Plotting IC heatmap...")
    # Resample to monthly for better visualization
    ic_monthly = data['ic_df'].resample('M').mean()
    fig = plot_ic_heatmap(
        ic_monthly,
        title="Monthly IC Heatmap Across Factors",
        annot=True
    )
    plt.savefig('/home/shw/quant_projects/examples/ic_heatmap.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/ic_heatmap.png")


def example_factor_plots(data):
    """Demonstrate factor performance visualization functions."""
    print("\n" + "=" * 60)
    print("FACTOR PLOTS EXAMPLES")
    print("=" * 60)

    # 1. Factor Distribution
    print("\n1. Plotting factor value distribution...")
    fig = plot_factor_distribution(
        data['factor_values'],
        bins=60,
        show_quantiles=True,
        quantiles=[0.2, 0.5, 0.8]
    )
    plt.savefig('/home/shw/quant_projects/examples/factor_distribution.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/factor_distribution.png")

    # 2. Quantile Returns
    print("\n2. Plotting quantile returns...")
    fig = plot_quantile_returns(
        data['quantile_returns'],
        show_spread=True
    )
    plt.savefig('/home/shw/quant_projects/examples/quantile_returns.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/quantile_returns.png")

    # 3. Turnover Analysis
    print("\n3. Plotting turnover analysis...")
    fig = plot_turnover_analysis(
        data['turnover'],
        rolling_window=20
    )
    plt.savefig('/home/shw/quant_projects/examples/turnover_analysis.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/turnover_analysis.png")

    # 4. Cumulative Returns
    print("\n4. Plotting cumulative returns...")
    fig = plot_cumulative_returns(
        data['returns'],
        benchmark=data['benchmark'],
        show_drawdown=True
    )
    plt.savefig('/home/shw/quant_projects/examples/cumulative_returns.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/cumulative_returns.png")


def example_comparison_plots(data):
    """Demonstrate multi-factor comparison visualization functions."""
    print("\n" + "=" * 60)
    print("COMPARISON PLOTS EXAMPLES")
    print("=" * 60)

    # 1. Multi-Factor IC
    print("\n1. Plotting multi-factor IC comparison...")
    fig = plot_multi_factor_ic(
        data['ic_df'],
        show_mean=True,
        rolling_window=20
    )
    plt.savefig('/home/shw/quant_projects/examples/multi_factor_ic.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/multi_factor_ic.png")

    # 2. Factor Correlation Matrix
    print("\n2. Plotting factor correlation matrix...")
    # Generate correlated factor data
    n_periods = 500
    factor_data = pd.DataFrame({
        'Factor_1': np.random.normal(0, 1, n_periods),
        'Factor_2': np.random.normal(0, 1, n_periods),
        'Factor_3': np.random.normal(0, 1, n_periods),
        'Factor_4': np.random.normal(0, 1, n_periods),
        'Factor_5': np.random.normal(0, 1, n_periods),
    })
    # Add some correlation
    factor_data['Factor_2'] += 0.3 * factor_data['Factor_1']
    factor_data['Factor_4'] += 0.5 * factor_data['Factor_3']

    fig = plot_factor_correlation_matrix(
        factor_data,
        method='pearson',
        annot=True
    )
    plt.savefig('/home/shw/quant_projects/examples/correlation_matrix.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/correlation_matrix.png")

    # 3. IC Decay
    print("\n3. Plotting IC decay...")
    fig = plot_ic_decay(data['ic_decay'])
    plt.savefig('/home/shw/quant_projects/examples/ic_decay.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/ic_decay.png")

    # 4. Performance Comparison
    print("\n4. Plotting performance comparison...")
    fig = plot_performance_comparison(
        data['performance_df'],
        metrics=['IC_mean', 'IR', 'Sharpe', 'Max_DD']
    )
    plt.savefig('/home/shw/quant_projects/examples/performance_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/performance_comparison.png")


def example_diagnostic_plots(data):
    """Demonstrate statistical diagnostic visualization functions."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC PLOTS EXAMPLES")
    print("=" * 60)

    # 1. Residual Analysis
    print("\n1. Plotting residual analysis...")
    fig = plot_residual_analysis(
        data['residuals'],
        fitted_values=data['fitted_values']
    )
    plt.savefig('/home/shw/quant_projects/examples/residual_analysis.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/residual_analysis.png")

    # 2. Q-Q Plot
    print("\n2. Plotting Q-Q plot...")
    fig = plot_qq_plot(
        data['residuals'],
        distribution='norm'
    )
    plt.savefig('/home/shw/quant_projects/examples/qq_plot.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/qq_plot.png")

    # 3. Autocorrelation
    print("\n3. Plotting autocorrelation...")
    # Use returns for autocorrelation
    fig = plot_autocorrelation(
        data['returns'],
        lags=40
    )
    plt.savefig('/home/shw/quant_projects/examples/autocorrelation.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/autocorrelation.png")

    # 4. Heteroskedasticity
    print("\n4. Plotting heteroskedasticity analysis...")
    fig = plot_heteroskedasticity(
        data['residuals'],
        data['fitted_values']
    )
    plt.savefig('/home/shw/quant_projects/examples/heteroskedasticity.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0D12')
    plt.close()
    print("   Saved: examples/heteroskedasticity.png")


def main():
    """Run all visualization examples."""
    print("\n" + "=" * 60)
    print("QUANTITATIVE FACTOR VISUALIZATION EXAMPLES")
    print("=" * 60)
    print("\nGenerating synthetic data...")

    # Generate sample data
    data = generate_sample_data()
    print("Data generated successfully.")

    # Ensure examples directory exists
    import os
    os.makedirs('/home/shw/quant_projects/examples', exist_ok=True)

    # Run all examples
    try:
        example_ic_plots(data)
        example_factor_plots(data)
        example_comparison_plots(data)
        example_diagnostic_plots(data)

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nAll plots saved to: /home/shw/quant_projects/examples/")
        print("\nGenerated plots:")
        print("  - ic_timeseries.png")
        print("  - ic_rolling.png")
        print("  - ic_distribution.png")
        print("  - ic_heatmap.png")
        print("  - factor_distribution.png")
        print("  - quantile_returns.png")
        print("  - turnover_analysis.png")
        print("  - cumulative_returns.png")
        print("  - multi_factor_ic.png")
        print("  - correlation_matrix.png")
        print("  - ic_decay.png")
        print("  - performance_comparison.png")
        print("  - residual_analysis.png")
        print("  - qq_plot.png")
        print("  - autocorrelation.png")
        print("  - heteroskedasticity.png")

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
