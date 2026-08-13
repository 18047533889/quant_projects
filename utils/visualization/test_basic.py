"""
Simple test script for visualization utilities without external dependencies.

Tests basic imports and function signatures.
"""

import sys
sys.path.insert(0, '/home/shw/quant_projects')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")

    try:
        from utils.visualization import ic_plots
        print("✓ ic_plots module imported")
    except ImportError as e:
        print(f"✗ ic_plots import failed: {e}")
        return False

    try:
        from utils.visualization import factor_plots
        print("✓ factor_plots module imported")
    except ImportError as e:
        print(f"✗ factor_plots import failed: {e}")
        return False

    try:
        from utils.visualization import comparison_plots
        print("✓ comparison_plots module imported")
    except ImportError as e:
        print(f"✗ comparison_plots import failed: {e}")
        return False

    try:
        from utils.visualization import diagnostic_plots
        print("✓ diagnostic_plots module imported")
    except ImportError as e:
        print(f"✗ diagnostic_plots import failed: {e}")
        return False

    return True


def test_function_signatures():
    """Test that all expected functions are available."""
    print("\nTesting function signatures...")

    from utils.visualization import ic_plots, factor_plots, comparison_plots, diagnostic_plots

    # IC plots
    expected_ic = ['plot_ic_timeseries', 'plot_rolling_ic', 'plot_ic_distribution', 'plot_ic_heatmap']
    for func_name in expected_ic:
        if hasattr(ic_plots, func_name):
            print(f"✓ ic_plots.{func_name} exists")
        else:
            print(f"✗ ic_plots.{func_name} missing")

    # Factor plots
    expected_factor = ['plot_factor_distribution', 'plot_quantile_returns',
                       'plot_turnover_analysis', 'plot_cumulative_returns']
    for func_name in expected_factor:
        if hasattr(factor_plots, func_name):
            print(f"✓ factor_plots.{func_name} exists")
        else:
            print(f"✗ factor_plots.{func_name} missing")

    # Comparison plots
    expected_comparison = ['plot_multi_factor_ic', 'plot_factor_correlation_matrix',
                          'plot_ic_decay', 'plot_performance_comparison']
    for func_name in expected_comparison:
        if hasattr(comparison_plots, func_name):
            print(f"✓ comparison_plots.{func_name} exists")
        else:
            print(f"✗ comparison_plots.{func_name} missing")

    # Diagnostic plots
    expected_diagnostic = ['plot_residual_analysis', 'plot_qq_plot',
                          'plot_autocorrelation', 'plot_heteroskedasticity']
    for func_name in expected_diagnostic:
        if hasattr(diagnostic_plots, func_name):
            print(f"✓ diagnostic_plots.{func_name} exists")
        else:
            print(f"✗ diagnostic_plots.{func_name} missing")

    return True


def test_package_init():
    """Test that __init__.py exports work."""
    print("\nTesting package exports...")

    try:
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
        print("✓ All functions exported from __init__.py")
        return True
    except ImportError as e:
        print(f"✗ Package export failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("VISUALIZATION UTILITIES TEST SUITE")
    print("=" * 60)

    results = []

    results.append(test_imports())
    if results[-1]:
        results.append(test_function_signatures())
        results.append(test_package_init())

    print("\n" + "=" * 60)
    if all(results):
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nNote: matplotlib and seaborn are required to run the plots.")
        print("Install with: pip install matplotlib seaborn scipy")
        print("\nTo generate example plots, run:")
        print("  python3 -m utils.visualization.examples")
    else:
        print("SOME TESTS FAILED")
        print("=" * 60)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
