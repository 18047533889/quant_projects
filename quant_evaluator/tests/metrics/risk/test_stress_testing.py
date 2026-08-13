"""
Tests for stress testing and scenario analysis.
"""

import pytest
import numpy as np

from quant_evaluator.metrics.risk.stress_testing import (
    apply_historical_scenario,
    apply_hypothetical_scenario,
    compute_scenario_impact,
    compute_correlation_breakdown,
    compute_worst_case_scenarios,
)


class TestHistoricalScenario:
    """Test historical scenario application."""

    def test_historical_scenario_direct(self):
        """Apply historical scenario directly."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        scenario_returns = np.array([-0.10, -0.08, -0.05])

        scenario_impact = apply_historical_scenario(
            returns, scenario_returns, scaling_method="direct"
        )

        # Should match scenario exactly
        assert np.allclose(scenario_impact, scenario_returns)

    def test_historical_scenario_scaled(self):
        """Apply historical scenario with volatility scaling."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02  # Vol ~0.02
        scenario_returns = np.array([-0.20, -0.15, -0.10])  # Higher vol

        scenario_impact = apply_historical_scenario(
            returns, scenario_returns, scaling_method="scaled"
        )

        # Scaled scenario should have lower magnitude
        assert np.abs(scenario_impact).mean() < np.abs(scenario_returns).mean()

    def test_historical_scenario_multi_factor(self):
        """Apply scenario to multiple factors."""
        np.random.seed(100)
        T, F = 200, 3
        returns = np.random.randn(T, F) * 0.02
        scenario_returns = np.array([-0.10, -0.08, -0.05])

        scenario_impact = apply_historical_scenario(
            returns, scenario_returns, scaling_method="direct"
        )

        assert scenario_impact.shape == (len(scenario_returns), F)

    def test_historical_scenario_invalid_method(self):
        """Invalid scaling method raises error."""
        returns = np.random.randn(100) * 0.02
        scenario_returns = np.array([-0.10, -0.05])

        with pytest.raises(ValueError):
            apply_historical_scenario(
                returns, scenario_returns, scaling_method="invalid"
            )


class TestHypotheticalScenario:
    """Test hypothetical scenario application."""

    def test_hypothetical_absolute_shock(self):
        """Apply absolute shock."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        shock_size = -0.10

        shocked = apply_hypothetical_scenario(
            returns, shock_size=shock_size, shock_type="absolute"
        )

        # All returns should be shifted by shock
        expected = returns + shock_size
        assert np.allclose(shocked, expected, rtol=1e-10, atol=1e-10)

    def test_hypothetical_volatility_shock(self):
        """Apply volatility-scaled shock."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        shock_size = -2.0  # -2 sigma shock

        shocked = apply_hypothetical_scenario(
            returns, shock_size=shock_size, shock_type="volatility"
        )

        # Shock magnitude should scale with volatility
        vol = np.std(returns, ddof=1)
        expected_shift = shock_size * vol
        mean_shift = np.mean(shocked - returns)
        assert abs(mean_shift - expected_shift) < 0.01

    def test_hypothetical_multi_factor(self):
        """Apply shock to multiple factors."""
        np.random.seed(100)
        T, F = 100, 3
        returns = np.random.randn(T, F) * 0.02
        shock_size = -0.05

        shocked = apply_hypothetical_scenario(
            returns, shock_size=shock_size, shock_type="absolute"
        )

        assert shocked.shape == returns.shape
        # All factors shocked
        assert np.all(shocked.mean(axis=0) < returns.mean(axis=0))

    def test_hypothetical_correlation_adjustment(self):
        """Apply correlation adjustment."""
        np.random.seed(200)
        T, F = 200, 3
        returns = np.random.randn(T, F) * 0.02

        # Increase correlation
        shocked = apply_hypothetical_scenario(
            returns,
            shock_size=-0.05,
            shock_type="absolute",
            correlation_adjustment=1.5,
        )

        assert shocked.shape == returns.shape

    def test_hypothetical_invalid_shock_type(self):
        """Invalid shock type raises error."""
        returns = np.random.randn(100) * 0.02

        with pytest.raises(ValueError):
            apply_hypothetical_scenario(
                returns, shock_size=-0.10, shock_type="invalid"
            )


class TestScenarioImpact:
    """Test scenario impact computation."""

    def test_scenario_impact_mean(self):
        """Compute mean return impact."""
        np.random.seed(42)
        baseline = np.random.randn(100) * 0.02 + 0.001
        scenario = np.random.randn(50) * 0.02 - 0.005  # Worse returns

        impact = compute_scenario_impact(
            baseline, scenario, metrics=["mean"], periods_per_year=252
        )

        assert "mean_baseline" in impact
        assert "mean_scenario" in impact
        assert "mean_change" in impact

        # Scenario should have lower mean
        assert impact["mean_scenario"] < impact["mean_baseline"]

    def test_scenario_impact_sharpe(self):
        """Compute Sharpe ratio impact."""
        np.random.seed(42)
        baseline = np.random.randn(100) * 0.02 + 0.002
        scenario = np.random.randn(50) * 0.04 + 0.001  # Higher vol, lower return

        impact = compute_scenario_impact(
            baseline, scenario, metrics=["sharpe"], periods_per_year=252
        )

        assert "sharpe_baseline" in impact
        assert "sharpe_scenario" in impact
        assert "sharpe_change" in impact

        # Scenario should have worse Sharpe
        assert impact["sharpe_scenario"] < impact["sharpe_baseline"]

    def test_scenario_impact_max_drawdown(self):
        """Compute max drawdown impact."""
        np.random.seed(42)
        baseline = np.random.randn(100) * 0.02
        # Scenario with large loss
        scenario = np.concatenate([[-0.20, -0.15], np.random.randn(48) * 0.02])

        impact = compute_scenario_impact(
            baseline, scenario, metrics=["max_drawdown"], periods_per_year=252
        )

        assert "max_drawdown_baseline" in impact
        assert "max_drawdown_scenario" in impact
        assert "max_drawdown_change" in impact

        # Both should have drawdowns
        assert impact["max_drawdown_scenario"] > 0
        assert impact["max_drawdown_baseline"] > 0

    def test_scenario_impact_var_cvar(self):
        """Compute VaR and CVaR impact."""
        np.random.seed(42)
        baseline = np.random.randn(100) * 0.02
        scenario = np.random.randn(50) * 0.04  # Higher volatility

        impact = compute_scenario_impact(
            baseline, scenario, metrics=["var", "cvar"], periods_per_year=252
        )

        assert "var_baseline" in impact
        assert "var_scenario" in impact
        assert "cvar_baseline" in impact
        assert "cvar_scenario" in impact

        # Higher vol = higher VaR/CVaR
        assert impact["var_scenario"] > impact["var_baseline"]
        assert impact["cvar_scenario"] > impact["cvar_baseline"]

    def test_scenario_impact_all_metrics(self):
        """Compute all metrics together."""
        np.random.seed(42)
        baseline = np.random.randn(100) * 0.02
        scenario = np.random.randn(50) * 0.03

        impact = compute_scenario_impact(
            baseline,
            scenario,
            metrics=["mean", "sharpe", "max_drawdown", "var", "cvar"],
            periods_per_year=252,
        )

        # All metric keys should be present
        expected_keys = [
            "mean_baseline", "mean_scenario", "mean_change",
            "sharpe_baseline", "sharpe_scenario", "sharpe_change",
            "max_drawdown_baseline", "max_drawdown_scenario", "max_drawdown_change",
            "var_baseline", "var_scenario", "var_change",
            "cvar_baseline", "cvar_scenario", "cvar_change",
        ]

        for key in expected_keys:
            assert key in impact

    def test_scenario_impact_multi_factor(self):
        """Scenario impact for multiple factors."""
        np.random.seed(42)
        T, F = 200, 3
        baseline = np.random.randn(T, F) * 0.02
        scenario = np.random.randn(100, F) * 0.03

        impact = compute_scenario_impact(
            baseline, scenario, metrics=["mean", "sharpe"], periods_per_year=252
        )

        # Results should be arrays for multi-factor
        assert isinstance(impact["mean_baseline"], np.ndarray)
        assert impact["mean_baseline"].shape == (F,)


class TestCorrelationBreakdown:
    """Test correlation breakdown analysis."""

    def test_correlation_breakdown_normal(self):
        """Correlation breakdown during normal vs stress."""
        np.random.seed(42)
        T = 1000

        returns_x = np.random.randn(T) * 0.02
        # Create correlation that increases during stress
        returns_y = np.where(
            returns_x < np.quantile(returns_x, 0.2),
            0.8 * returns_x + 0.2 * np.random.randn(T) * 0.02,  # High corr in stress
            0.3 * returns_x + 0.7 * np.random.randn(T) * 0.02,  # Low corr in normal
        )

        breakdown = compute_correlation_breakdown(
            returns_x, returns_y, stress_quantile=0.05
        )

        assert "correlation_normal" in breakdown
        assert "correlation_stress" in breakdown
        assert "correlation_all" in breakdown
        assert "correlation_breakdown" in breakdown

        # Both correlations should be computed
        assert np.isfinite(breakdown["correlation_stress"])
        assert np.isfinite(breakdown["correlation_normal"])

    def test_correlation_breakdown_constant_corr(self):
        """Correlation breakdown with constant correlation."""
        np.random.seed(42)
        T = 1000

        returns_x = np.random.randn(T) * 0.02
        returns_y = 0.5 * returns_x + 0.5 * np.random.randn(T) * 0.02

        breakdown = compute_correlation_breakdown(
            returns_x, returns_y, stress_quantile=0.05
        )

        # Both correlations should be computed (may vary due to sampling)
        assert np.isfinite(breakdown["correlation_stress"])
        assert np.isfinite(breakdown["correlation_normal"])

    def test_correlation_breakdown_independent(self):
        """Correlation breakdown with independent series."""
        np.random.seed(42)
        T = 1000

        returns_x = np.random.randn(T) * 0.02
        returns_y = np.random.randn(T) * 0.02

        breakdown = compute_correlation_breakdown(
            returns_x, returns_y, stress_quantile=0.05
        )

        # Overall correlation should be low
        assert abs(breakdown["correlation_all"]) < 0.3

    def test_correlation_breakdown_custom_quantile(self):
        """Correlation breakdown with custom stress quantile."""
        np.random.seed(42)
        T = 1000

        returns_x = np.random.randn(T) * 0.02
        returns_y = 0.5 * returns_x + 0.5 * np.random.randn(T) * 0.02

        breakdown_5 = compute_correlation_breakdown(
            returns_x, returns_y, stress_quantile=0.05
        )
        breakdown_10 = compute_correlation_breakdown(
            returns_x, returns_y, stress_quantile=0.10
        )

        # Both should be computed
        assert np.isfinite(breakdown_5["correlation_stress"])
        assert np.isfinite(breakdown_10["correlation_stress"])

    def test_correlation_breakdown_insufficient_data(self):
        """Correlation breakdown with insufficient data."""
        returns_x = np.random.randn(10)
        returns_y = np.random.randn(10)

        breakdown = compute_correlation_breakdown(
            returns_x, returns_y, min_periods=50
        )

        # Should return NaN
        assert np.isnan(breakdown["correlation_all"])

    def test_correlation_breakdown_length_mismatch(self):
        """Correlation breakdown with mismatched lengths."""
        returns_x = np.random.randn(100)
        returns_y = np.random.randn(50)

        with pytest.raises(ValueError):
            compute_correlation_breakdown(returns_x, returns_y)

    def test_correlation_breakdown_multidimensional_error(self):
        """Correlation breakdown requires 1D input."""
        returns_x = np.random.randn(100, 2)
        returns_y = np.random.randn(100, 2)

        with pytest.raises(ValueError):
            compute_correlation_breakdown(returns_x, returns_y)


class TestWorstCaseScenarios:
    """Test worst-case scenario identification."""

    def test_worst_case_cumulative(self):
        """Identify worst cumulative return periods."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        # Insert a bad period
        returns[50:70] = -0.05

        worst_scenarios = compute_worst_case_scenarios(
            returns, n_scenarios=5, window_size=20, metric="cumulative"
        )

        assert len(worst_scenarios) <= 5
        assert len(worst_scenarios) > 0

        # Each scenario has required fields
        for scenario in worst_scenarios:
            assert "start_idx" in scenario
            assert "end_idx" in scenario
            assert "metric_value" in scenario
            assert "returns" in scenario

        # First scenario should be worst (most negative cumulative)
        assert worst_scenarios[0]["metric_value"] <= worst_scenarios[-1]["metric_value"]

    def test_worst_case_volatility(self):
        """Identify highest volatility periods."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        # Insert high volatility period
        returns[100:120] = np.random.randn(20) * 0.10

        worst_scenarios = compute_worst_case_scenarios(
            returns, n_scenarios=5, window_size=20, metric="volatility"
        )

        assert len(worst_scenarios) > 0

        # First scenario should have highest volatility
        assert worst_scenarios[0]["metric_value"] >= worst_scenarios[-1]["metric_value"]

    def test_worst_case_multi_factor(self):
        """Identify worst scenarios for multiple factors."""
        np.random.seed(42)
        T, F = 200, 3
        returns = np.random.randn(T, F) * 0.02

        worst_scenarios = compute_worst_case_scenarios(
            returns, n_scenarios=3, window_size=20, metric="cumulative"
        )

        assert len(worst_scenarios) > 0

        # Returns should be multi-factor
        for scenario in worst_scenarios:
            assert scenario["returns"].shape == (20, F)

    def test_worst_case_small_window(self):
        """Worst scenarios with small window."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02

        worst_scenarios = compute_worst_case_scenarios(
            returns, n_scenarios=5, window_size=5, metric="cumulative"
        )

        assert len(worst_scenarios) > 0

        # Windows should be size 5
        for scenario in worst_scenarios:
            assert len(scenario["returns"]) == 5

    def test_worst_case_insufficient_data(self):
        """Worst scenarios with insufficient data."""
        returns = np.random.randn(10) * 0.02

        worst_scenarios = compute_worst_case_scenarios(
            returns, n_scenarios=5, window_size=20, metric="cumulative"
        )

        # Should return empty list
        assert len(worst_scenarios) == 0

    def test_worst_case_invalid_metric(self):
        """Invalid metric raises error."""
        returns = np.random.randn(100) * 0.02

        with pytest.raises(ValueError):
            compute_worst_case_scenarios(
                returns, n_scenarios=5, window_size=20, metric="invalid"
            )

    def test_worst_case_ordering(self):
        """Worst scenarios are properly ordered."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02

        worst_scenarios = compute_worst_case_scenarios(
            returns, n_scenarios=10, window_size=20, metric="cumulative"
        )

        # Should be ordered from worst to best
        for i in range(len(worst_scenarios) - 1):
            assert (
                worst_scenarios[i]["metric_value"]
                <= worst_scenarios[i + 1]["metric_value"]
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_scenario_empty_returns(self):
        """Scenario with empty returns."""
        returns = np.array([])
        scenario = np.array([-0.10, -0.05])

        impact = apply_historical_scenario(returns, scenario)

        assert len(impact) == len(scenario)

    def test_scenario_all_nan(self):
        """Scenario with all NaN returns."""
        returns = np.full(100, np.nan)
        scenario = np.array([-0.10, -0.05])

        impact = apply_historical_scenario(returns, scenario)

        # Should handle gracefully
        assert impact.shape == scenario.shape

    def test_correlation_breakdown_perfect_correlation(self):
        """Correlation breakdown with perfect correlation."""
        np.random.seed(42)
        T = 1000

        returns_x = np.random.randn(T) * 0.02
        returns_y = returns_x.copy()

        breakdown = compute_correlation_breakdown(returns_x, returns_y)

        # All correlations should be close to 1.0
        assert breakdown["correlation_all"] > 0.95
        assert breakdown["correlation_normal"] > 0.90
        assert breakdown["correlation_stress"] > 0.90

    def test_worst_case_constant_returns(self):
        """Worst case with constant returns."""
        returns = np.full(100, 0.01)

        worst_scenarios = compute_worst_case_scenarios(
            returns, n_scenarios=5, window_size=20, metric="cumulative"
        )

        # All scenarios should have same metric value
        if len(worst_scenarios) > 1:
            values = [s["metric_value"] for s in worst_scenarios]
            assert np.std(values) < 0.001
